import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmcv import cnn


def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


class CFEmbedding(nn.Module):
    def __init__(self, embed_dims, num_classes, use_memory, init_memory=None, kv_bias=True,
                 num_groups=4):
        super(CFEmbedding, self).__init__()
        if use_memory:
            if init_memory is None:  # random init
                std = 1. / ((num_classes * embed_dims) ** 0.5)
                memory = torch.empty(1, num_classes, embed_dims).normal_(0, std)
            else:  # pretrained init
                memory = torch.tensor(np.load(init_memory), dtype=torch.float)[:, :embed_dims].unsqueeze(0)
            memory = F.normalize(memory, dim=2, p=2)
            self.register_buffer('memory', memory)
            # self.gamma = nn.Parameter(torch.zeros(1) + 0.1, requires_grad=True)  # init gamma=0.1
            # self.retrieve = MemoryRetrieveBlock(embed_dims, embed_dims, embed_dims, embed_dims, bias=True)

        self.mask_learner = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, 1, groups=num_groups, bias=False),
            nn.Conv2d(embed_dims, num_classes, 1, bias=False))
        self.align_conv = nn.Conv2d(embed_dims, embed_dims, 1, groups=num_groups, bias=False)
        self.cf_embed = nn.Linear(embed_dims, embed_dims * 2, bias=kv_bias)

    @torch.no_grad()
    def _update_memory(self, cf_feat, momentum=0.1):
        cf_feat = cf_feat.mean(dim=0, keepdim=True)
        cf_feat = reduce_mean(cf_feat)  # sync across GPUs
        cf_feat = F.normalize(cf_feat, dim=2, p=2)
        self.memory = (1.0 - momentum) * self.memory + momentum * cf_feat

    def forward(self, x, momentum=0.1):
        mask = self.mask_learner(x)
        outs = {'mask': mask}
        mask = mask.reshape(mask.size(0), mask.size(1), -1)  # B x L x N
        mask = F.softmax(mask, dim=-1)

        x = self.align_conv(x)
        x = x.reshape(x.size(0), x.size(1), -1)  # B x C x N
        cf_feat = mask @ x.transpose(-2, -1)  # category feature: B x L x C

        if hasattr(self, 'memory'):
            memory = self.memory.expand(cf_feat.size(0), -1, -1)
            if self.training:
                self._update_memory(cf_feat, momentum)
            # cf_feat = cf_feat + self.retrieve(cf_feat, memory) * self.gamma
            cf_feat = (1.0 - momentum) * cf_feat + momentum * memory

        out = self.cf_embed(cf_feat)
        outs.update({'out': out.transpose(-2, -1)})  # B x C x L, used as key/value
        return outs


class CFTransform(nn.Module):
    def __init__(self, embed_dims, num_heads, num_classes, attn_drop_rate=.0, drop_rate=.0, qkv_bias=True,
                 qk_scale=None, proj_bias=True, use_memory=False, init_memory=None):
        super(CFTransform, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims ** -0.5
        self.q = cnn.DepthwiseSeparableConvModule(embed_dims, embed_dims, 3, 1, 1,
                                                  act_cfg=None, bias=qkv_bias)
        self.kv = CFEmbedding(embed_dims, num_classes, use_memory, init_memory, qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Conv2d(embed_dims, embed_dims, 1, bias=proj_bias)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, query, key_value, momentum=0.1):
        B, _, H, W = query.shape
        q = self.q(query)
        outs = self.kv(key_value, momentum)
        k, v = torch.chunk(outs.pop('out'), chunks=2, dim=1)

        q = q.reshape(B, self.num_heads, self.head_dims, -1).permute(0, 1, 3, 2)
        k = k.reshape(B, self.num_heads, self.head_dims, -1).permute(0, 1, 3, 2)
        v = v.reshape(B, self.num_heads, self.head_dims, -1).permute(0, 1, 3, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.max(attn, -1, keepdim=True)[0].expand_as(attn) - attn  # stable training
        attn = F.softmax(attn, dim=-1)  # B(num_heads)(HW)L
        outs.update({'attn': torch.mean(attn, dim=1, keepdim=False)})
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(-2, -1).reshape(B, self.embed_dims, H, W)
        out = self.proj_drop(self.proj(out))
        outs.update({'out': out})
        return outs


class CFTBlock(nn.Module):
    def __init__(self, embed_dims, num_heads, num_classes, attn_drop_rate=.0, drop_rate=.0, qkv_bias=True,
                 mlp_ratio=4, use_memory=False, init_memory=None, norm_cfg=None):
        super(CFTBlock, self).__init__()
        norm_cfg = dict(type='LN', eps=1e-6) if not norm_cfg else norm_cfg
        _, self.norm_low = cnn.build_norm_layer(norm_cfg, num_features=embed_dims)
        _, self.norm_high = cnn.build_norm_layer(norm_cfg, num_features=embed_dims)
        self.cross_attn = CFTransform(embed_dims, num_heads, num_classes, attn_drop_rate,
                                      drop_rate, qkv_bias, use_memory=use_memory, init_memory=init_memory)

        _, self.norm_mlp = cnn.build_norm_layer(norm_cfg, num_features=embed_dims)
        ffn_channels = embed_dims * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dims, ffn_channels, 1, bias=True),
            nn.Conv2d(ffn_channels, ffn_channels, 3, 1, 1, groups=ffn_channels, bias=True),
            cnn.build_activation_layer(dict(type='GELU')),
            nn.Dropout(drop_rate),
            nn.Conv2d(ffn_channels, embed_dims, 1, bias=True),
            nn.Dropout(drop_rate))

    def forward(self, low, high, momentum=0.1):
        query = self.norm_low(low.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        key_value = self.norm_high(high.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        outs = self.cross_attn(query, key_value, momentum)

        out = outs.pop('out') + low
        out = self.mlp(self.norm_mlp(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)) + out
        outs.update({'out': out})
        return outs


class CategoryPrototypeTransformer(nn.Module):
    def __init__(
            self, 
            # backbone_channels, 
            num_classes,
            crop_size,
            num_heads=4,
            attn_drop_rate=0.,
            drop_rate=0.,
            qkv_bias=True,
            mlp_ratio=4,
            ln_norm_cfg={'type': 'LN', 'eps': 1e-6},
            norm_cfg={'type': 'SyncBN', 'requires_grad': True},
            use_memory=True,
            momentum_cfg={'start': 0.2, 'use_poly': False, 'total_iter': 160_000, 'power': 0.9, 'eta_min': 0.009},
            in_channels=(192, 384, 768, 1536),
            channels=256,
            out_channels=None,
            dropout_ratio=.1,
            act_cfg={'type': 'GELU'},
            in_index=(0, 1, 2, 3),
            align_corners=False,
            pool_channels=48,
            init_memory=None,
            loss_decode={'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0},
            loss_mask_decode={'type': 'MaskLoss', 'mask_weight': 5.0, 'dice_weight': 1.0, 'loss_weight': 1.0},
        ):
    # def __init__(self, ln_norm_cfg, loss_mask_decode, use_memory, momentum_cfg, init_memory=None,
    #              pool_channels=48, num_heads=4, attn_drop_rate=0.0, drop_rate=0.0, qkv_bias=True,
    #              mlp_ratio=4, **kwargs):
        super(CategoryPrototypeTransformer, self).__init__()
        self.num_classes = num_classes
        self.momentum_cfg = momentum_cfg
        self.norm_cfg = norm_cfg
        self.align_corners = align_corners
        self.act_cfg = act_cfg
        self.channels = channels
        self.in_channels = in_channels
        self.out_channels = num_classes
        self.in_index = in_index
        self.crop_size = crop_size

        loss_mask_decode.update({'num_classes': self.num_classes})

        self.img_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            cnn.ConvModule(self.in_channels[-1], pool_channels, 1,
                           norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
        self.conv_seg = nn.Conv2d(self.channels + pool_channels, self.out_channels, kernel_size=1)  # override

        self.lateral_convs = nn.ModuleList()
        self.cft_blocks = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i, in_channel in enumerate(self.in_channels):
            self.lateral_convs.append(nn.Conv2d(in_channel, self.channels, 1, bias=True))
            if i < len(self.in_channels) - 1:
                self.cft_blocks.append(CFTBlock(self.channels, num_heads, self.num_classes, attn_drop_rate,
                                                drop_rate, qkv_bias, mlp_ratio, use_memory, init_memory,
                                                ln_norm_cfg))
                self.fpn_convs.append(nn.Conv2d(self.channels, self.channels, 3, 1, 1))
        self.fusion_conv = cnn.ConvModule(len(self.in_channels) * self.channels, self.channels,
                                          kernel_size=3, stride=1, padding=1, norm_cfg=self.norm_cfg,
                                          act_cfg=self.act_cfg)

    def set_cur_iter(self, cur_iter):
        self.momentum_cfg.update(dict(cur_iter=cur_iter))

    def get_cur_momentum(self):
        start = self.momentum_cfg.get('start')
        if not self.momentum_cfg.get('use_poly'):
            return start
        cur_iter = self.momentum_cfg.get('cur_iter')
        total_iter = self.momentum_cfg.get('total_iter')
        power = self.momentum_cfg.get('power')
        eta_min = self.momentum_cfg.get('eta_min')
        cur_momentum = ((1 - cur_iter / total_iter) ** power) * (start - eta_min) + eta_min
        return cur_momentum

    def forward(self, *inputs):
        momentum = self.get_cur_momentum()
        # inputs = self._transform_inputs(inputs)
        inputs = [
            input.permute(0, 3, 1, 2) for input in inputs[-4:]
        ]  # match to in_channels: [192, 384, 768, 1536], and permute to [B, C, H, W]
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        # laterals = []
        # for i, lateral_conv in enumerate(self.lateral_convs):
        #     lateral = lateral_conv(inputs[i])
        #     laterals.append(lateral)
        #     print(f"{i}: input shape: {inputs[i].shape}, lateral shape: {lateral.shape}")

        high = laterals[-1]
        masks = []
        # attns = []
        fpn_outs = [high]
        for i in range(len(self.cft_blocks), 0, -1):
            low = laterals[i - 1]
            cft_outs = self.cft_blocks[i - 1](low, high, momentum=momentum)
            high = cft_outs['out']
            # masks.append(cft_outs['mask'])
            # attns.append(cft_outs['attn'])
            fpn_outs.append(self.fpn_convs[i - 1](high))

        h, w = fpn_outs[-1].shape[2:]
        # fpn_outs = [resize(fpn_out, size=(h, w), mode='bilinear',
        #                    align_corners=self.align_corners) for fpn_out in fpn_outs]
        fpn_outs = [F.interpolate(fpn_out, size=(h, w), mode='bilinear', align_corners=self.align_corners) for fpn_out in fpn_outs]
        out = torch.cat(fpn_outs, dim=1)
        out = self.fusion_conv(out)
        # pool_out = resize(self.img_pool(inputs[-1]), size=(h, w), mode='bilinear', align_corners=self.align_corners)
        pool_out = F.interpolate(self.img_pool(inputs[-1]), size=(h, w), mode='bilinear', align_corners=self.align_corners)
        out = torch.cat((out, pool_out), dim=1)
        out = self.conv_seg(out)
        
        out = F.interpolate(out, size=self.crop_size, mode='bilinear', align_corners=self.align_corners)
        masks = [F.interpolate(mask, size=self.crop_size, mode='bilinear', align_corners=self.align_corners) for mask in masks]

        return masks, out

