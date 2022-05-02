import torch.nn as nn
import torch.nn.functional as F
import math


from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@HEADS.register_module()
class VIT_MLA_AUXIHead(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=769, **kwargs):
        super(VIT_MLA_AUXIHead, self).__init__(**kwargs)
        self.img_size = img_size
        if self.in_channels == 1024:
            self.aux_0 = nn.Conv2d(self.in_channels, 256,
                                   kernel_size=1, bias=False)

    def to_2D(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def forward(self, x):
        x = self._transform_inputs(x)

        if self.in_channels == 1024:
            x = self.aux_0(x)
            x = self.conv_seg(x)
        elif self.in_channels == 256:
            x = self.conv_seg(x)
        else:
            x = self.conv_seg(x)
        x = F.interpolate(x, size=self.img_size, mode='bilinear',
                          align_corners=self.align_corners)
        return x
