from math import ceil

import cv2
import torch
from torch import nn

from .ops import ConvBNReLU, SegHead
from .backbones.backbone import Backbone
from .methods.unet import UNet
from .methods.pspnet import PSPNet
from .methods.upernet import UPerNet
from .methods.segformer import SegFormer
from .methods.deeplabv3 import DeepLabV3
from .methods.deeplabv3plus import DeepLabV3Plus
from .methods.cpt import CategoryPrototypeTransformer

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class Segmentor(nn.Module):
    def __init__(self,
                 model_config,
                 data_config,
                 loss_config,
                 criterion):
        super().__init__()

        if model_config["method"] in ["segformer", "unet", "upernet"]:
            assert model_config["output_stride"] == 32

        self.backbone = Backbone(backbone=model_config["backbone"],
                                in_channels=data_config["in_channels"],
                                output_stride=model_config["output_stride"],
                                image_size=data_config["crop_size"])

        if model_config["method"] == "deeplabv3":
            self.method = DeepLabV3(backbone_channels=self.backbone.out_channels,
                                    out_channels=model_config["out_channels"],
                                    atrous_rates=model_config["atrous_rates"],
                                    align_corners=model_config["align_corners"])
        elif model_config["method"] == "deeplabv3plus":
            self.method = DeepLabV3Plus(backbone_channels=self.backbone.out_channels,
                                        out_channels=model_config["out_channels"],
                                        shortcut_channels=model_config["shortcut_channels"],
                                        atrous_rates=model_config["atrous_rates"],
                                        align_corners=model_config["align_corners"])
        elif model_config["method"] == "pspnet":
            self.method = PSPNet(backbone_channels=self.backbone.out_channels,
                                 out_channels=model_config["out_channels"],
                                 bins=model_config["bins"],
                                 align_corners=model_config["align_corners"])
        elif model_config["method"] == "segformer":
            self.method = SegFormer(backbone_channels=self.backbone.out_channels[2:],
                                    out_channels=model_config["out_channels"],
                                    align_corners=model_config["align_corners"])
        elif model_config["method"] == "unet":
            self.method = UNet(backbone_channels=self.backbone.out_channels[1:],
                               out_channels=model_config["out_channels"],
                               align_corners=model_config["align_corners"])
        elif model_config["method"] == "upernet":
            self.method = UPerNet(backbone_channels=self.backbone.out_channels[2:],
                                  out_channels=model_config["out_channels"],
                                  bins=model_config["bins"],
                                  align_corners=model_config["align_corners"])
        elif model_config["method"] == "cpt":
            self.method = CategoryPrototypeTransformer(
                # backbone_channels=self.backbone.out_channels[2:],
                num_classes=data_config["num_classes"],
                align_corners=model_config["align_corners"],
                crop_size=data_config["crop_size"],
            )
        else:
            raise NotImplementedError

        self.head = SegHead(in_channels=model_config["out_channels"],
                            out_channels=data_config["num_classes"],
                            dropout=model_config["dropout"],
                            crop_size=data_config["crop_size"],
                            align_corners=model_config["align_corners"])

        self.aux = model_config["aux"]
        self.aux_conv = ConvBNReLU(in_channels=self.backbone.out_channels[-2],
                                   out_channels=model_config["aux_out_channels"],
                                   kernel_size=3,
                                   padding=1)
        self.aux_head = SegHead(in_channels=model_config["aux_out_channels"],
                                out_channels=data_config["num_classes"],
                                dropout=model_config["dropout"],
                                crop_size=data_config["crop_size"],
                                align_corners=model_config["align_corners"])

        self.main_weight = loss_config["main_weight"]
        self.aux_weight = loss_config["aux_weight"]

        self.criterion = criterion

        self.initialize()


    def initialize(self):
        self.initialize_decoder(self.method)
        self.initialize_head(self.head)
        if self.aux:
            self.initialize_decoder(self.aux_conv)
            self.initialize_head(self.aux_head)


    def initialize_decoder(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def initialize_head(self, module):
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def get_params_list(self, lr, multiplier):
        params_list = []
        params_list = self.group_weight(params_list, self.backbone, lr)
        params_list = self.group_weight(params_list, self.method, multiplier * lr)
        params_list = self.group_weight(params_list, self.head, multiplier * lr)
        if self.aux:
            params_list = self.group_weight(params_list, self.aux_conv, multiplier * lr)
            params_list = self.group_weight(params_list, self.aux_head, multiplier * lr)

        p = 0
        for params in params_list:
            p += len(params['params'])

        assert p == len(list(self.parameters()))

        return params_list


    def group_weight(self, params_list, module, lr):
        group_decay, group_no_decay = [], []

        for _, param in module.named_parameters():
            if param.requires_grad:
                if len(param.shape) == 1:
                    group_no_decay.append(param)
                else:
                    group_decay.append(param)

        params_list.append(dict(params=group_decay, lr=lr))
        params_list.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))

        assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)

        return params_list


    def forward(self, x):
        features = self.backbone(x)
        if isinstance(self.method, CategoryPrototypeTransformer):  # TODO: avoid hard-code
            # CategoryPrototypeTransformer does not use auxiliary head, which is implemented in the method itself.
            aux_logits, logits = self.method(*features)
        else:
            logits = self.head(self.method(*features))

            if self.aux:
                aux_logits = self.aux_head(self.aux_conv(features[-2]))
            else:
                aux_logits = None

        return logits, aux_logits


    def forward_loss(self, image, label, keep_mask=None):
        logits, aux_logits = self.forward(image)

        loss = self.main_weight * self.criterion(logits, label, keep_mask)

        if isinstance(self.method, CategoryPrototypeTransformer):
            for aux_logit in aux_logits:
                loss += self.aux_weight * self.criterion(aux_logit, label, keep_mask)
        else:
            if self.aux and self.aux_weight > 0:
                loss += self.aux_weight * self.criterion(aux_logits, label, keep_mask)

        return loss


    def multi_scale_predict(self, image, device, num_classes, crop_size, flip=False, ratios=[1], stride_rate=2/3, temperature=1.0):
        if len(ratios) == 1:
            return self.predict(image, device, num_classes, crop_size, stride_rate, temperature=temperature)

        image_np = image.squeeze(0).numpy().transpose(1, 2, 0)
        image_h, image_w, _ = image_np.shape
        crop_h, crop_w = crop_size

        prob = torch.zeros((1, num_classes, image_h, image_w))
        for r in ratios:
            h = int(image_h * r)
            w = int(image_w * r)

            if image_h % 2 and not h % 2:
                h += 1

            if image_w % 2 and not w % 2:
                w += 1

            image_resized = cv2.resize(image_np, (w, h), interpolation=cv2.INTER_LINEAR)

            pad_h = max(crop_h - h, 0)
            pad_w = max(crop_w - w, 0)
            pad_h_half = int(pad_h / 2)
            pad_w_half = int(pad_w / 2)
            if pad_h > 0 or pad_w > 0:
                mean = [0.485, 0.456, 0.406]
                image_resized = cv2.copyMakeBorder(image_resized,
                                                   pad_h_half,
                                                   pad_h - pad_h_half,
                                                   pad_w_half,
                                                   pad_w - pad_w_half,
                                                   cv2.BORDER_CONSTANT,
                                                   value=mean)

            image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).unsqueeze(0)

            if flip:
                image_tensor = torch.cat([image_tensor, image_tensor.flip(3)], 0)

            prob_tensor = self.predict(image_tensor, device, num_classes, crop_size, stride_rate)

            if flip:
                prob_tensor = (prob_tensor[0] + prob_tensor[1].flip(2)) / 2
            else:
                prob_tensor = prob_tensor.squeeze(0)

            prob_np = prob_tensor.numpy().transpose(1, 2, 0)
            prob_resized = cv2.resize(prob_np, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
            prob += torch.from_numpy(prob_resized.transpose(2, 0, 1)).unsqueeze(0)

        prob /= len(ratios)

        return prob


    def predict(self, image, device, num_classes, crop_size, stride_rate, temperature=1.0):
        image = image.to(device)
        batch_size, _, image_h, image_w = image.shape
        crop_h, crop_w = crop_size

        assert image_h >= crop_h and image_w >= crop_w

        if image_h == crop_h and image_w == crop_w:
            logits, _ = self.forward(image)
        else:
            stride_h = int(ceil(crop_h * stride_rate))
            stride_w = int(ceil(crop_w * stride_rate))
            grid_h = int(ceil(float(image_h - crop_h) / stride_h) + 1)
            grid_w = int(ceil(float(image_w - crop_w) / stride_w) + 1)

            logits_all = torch.zeros((batch_size, num_classes, image_h, image_w)).to(device)
            logits_count = torch.zeros((batch_size, 1, image_h, image_w)).to(device)

            for index_h in range(grid_h):
                for index_w in range(grid_w):
                    start_h = index_h * stride_h
                    start_w = index_w * stride_w
                    end_h = min(start_h + crop_h, image_h)
                    end_w = min(start_w + crop_w, image_w)
                    start_h = end_h - crop_h
                    start_w = end_w - crop_w
                    image_crop = image[:, :, start_h:end_h, start_w:end_w].clone()

                    logits, _ = self.forward(image_crop)
                    logits_all[:, :, start_h : end_h, start_w : end_w] += logits
                    logits_count[:, :, start_h : end_h, start_w : end_w] += 1

            assert (logits_count == 0).sum() == 0

            logits = (logits_all / logits_count)

        # prob = logits.log_softmax(dim=1).exp().cpu()
        prob = (logits / temperature).log_softmax(dim=1).exp().cpu()

        return prob
