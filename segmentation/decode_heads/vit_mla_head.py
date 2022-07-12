import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


class MLAHead(nn.Module):
    def __init__(self, mla_channels=256, mlahead_channels=128, norm_cfg=None):
        super(MLAHead, self).__init__()
        self.head2 = nn.Sequential(
            nn.Conv2d(mla_channels[0], mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1],
            nn.ReLU(),
            nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1],
            nn.ReLU()
        )
        self.head3 = nn.Sequential(
            nn.Conv2d(mla_channels[1], mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1],
            nn.ReLU(),
            nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1],
            nn.ReLU()
        )
        self.head4 = nn.Sequential(
            nn.Conv2d(mla_channels[2], mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1],
            nn.ReLU(),
            nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1],
            nn.ReLU()
        )
        self.head5 = nn.Sequential(
            nn.Conv2d(mla_channels[3], mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1],
            nn.ReLU(),
            nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1],
            nn.ReLU()
        )

    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):
        head2 = F.interpolate(self.head2(mla_p2), mla_p3.shape[-1] * 4, mode='bilinear', align_corners=True)
        head3 = F.interpolate(self.head3(mla_p3), mla_p3.shape[-1] * 4, mode='bilinear', align_corners=True)
        head4 = F.interpolate(self.head4(mla_p4), mla_p4.shape[-1] * 4, mode='bilinear', align_corners=True)
        head5 = F.interpolate(self.head5(mla_p5), mla_p5.shape[-1] * 4, mode='bilinear', align_corners=True)

        return torch.cat([head2, head3, head4, head5], dim=1)


@HEADS.register_module()
class VIT_MLAHead(BaseDecodeHead):
    def __init__(
        self, img_size=769, mla_channels=256, mlahead_channels=128, norm_cfg=None, **kwargs
    ):
        super(VIT_MLAHead, self).__init__(**kwargs)
        self.img_size = img_size
        self.mlahead = MLAHead(mla_channels=mla_channels, mlahead_channels=mlahead_channels, norm_cfg=norm_cfg)

    def forward(self, inputs):
        x = self.mlahead(inputs[0], inputs[1], inputs[2], inputs[3])
        x = self.conv_seg(x)
        x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)

        return x
