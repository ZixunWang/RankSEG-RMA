import timm
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .mit import mix_transformer_encoders


class Backbone(nn.Module):
    def __init__(self, backbone, in_channels, output_stride, image_size):
        super().__init__()

        if backbone not in \
            ["resnet152d", "resnet101d", "resnet50d", "resnet34d", "resnet18d",
             "convnext_base.fb_in22k_ft_in1k_384", "xception65",
             "efficientnet_b0", "mobilenetv2_100", "mobilevitv2_100"
             "mit_b5", "mit_b4", "mit_b3", "mit_b3", "mit_b2", "mit_b1", "mit_b0",
             "swin_large_patch4_window12_384.ms_in22k"]:
            raise NotImplementedError

        if backbone in [f"mit_b{i}" for i in range(6)]:
            encoder = mix_transformer_encoders[backbone]["encoder"]
            params = mix_transformer_encoders[backbone]["params"]
            settings = mix_transformer_encoders[backbone]["pretrained_settings"]["imagenet"]

            self.backbone = encoder(**params)
            self.out_channels = self.backbone.out_channels
            self.backbone.load_state_dict(model_zoo.load_url(settings["url"]))
        elif backbone in ["swin_large_patch4_window12_384.ms_in22k"]:
            kwargs = dict(in_chans=in_channels,
                          output_stride=output_stride,
                          pretrained=True,
                          features_only=True,
                          img_size=image_size,)

            self.backbone = timm.create_model(backbone, **kwargs)
            self.out_channels = [in_channels,] + self.backbone.feature_info.channels()
        else:
            kwargs = dict(in_chans=in_channels,
                          output_stride=output_stride,
                          pretrained=True,
                          features_only=True)

            self.backbone = timm.create_model(backbone, **kwargs)
            self.out_channels = [in_channels,] + self.backbone.feature_info.channels()


    def forward(self, x):
        features = self.backbone(x)
        features = [x,] + features

        return features
