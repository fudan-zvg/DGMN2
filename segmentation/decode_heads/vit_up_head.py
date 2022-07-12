import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from mmcv.cnn import build_norm_layer
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@HEADS.register_module()
class VisionTransformerUpHead(BaseDecodeHead):
    def __init__(
        self, img_size=769, embed_dim=1024, norm_cfg=None, num_conv=1, **kwargs
    ):
        super(VisionTransformerUpHead, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.num_conv = num_conv

        if self.num_conv == 4:
            self.conv_0 = nn.Conv2d(embed_dim, 256, kernel_size=3, stride=1, padding=1)
            self.conv_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            
            self.syncbn_fc_0 = build_norm_layer(self.norm_cfg, 256)[1]
            self.syncbn_fc_1 = build_norm_layer(self.norm_cfg, 256)[1]
            self.syncbn_fc_2 = build_norm_layer(self.norm_cfg, 256)[1]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self._transform_inputs(x)

        if self.num_conv == 2:
            x = self.conv_seg(x)
            x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)
        elif self.num_conv == 4:
            x = self.conv_0(x)
            x = self.syncbn_fc_0(x)
            x = F.relu(x, inplace=True)
            x = F.interpolate(x, size=x.shape[-1] * 2, mode='bilinear', align_corners=self.align_corners)
            x = self.conv_1(x)
            x = self.syncbn_fc_1(x)
            x = F.relu(x, inplace=True)
            x = F.interpolate(x, size=x.shape[-1] * 2, mode='bilinear', align_corners=self.align_corners)
            x = self.conv_2(x)
            x = self.syncbn_fc_2(x)
            x = F.relu(x, inplace=True)
            x = F.interpolate(x, size=x.shape[-1] * 2, mode='bilinear', align_corners=self.align_corners)
            x = self.conv_seg(x)
            x = F.interpolate(x, size=x.shape[-1] * 2, mode='bilinear', align_corners=self.align_corners)

        return x
