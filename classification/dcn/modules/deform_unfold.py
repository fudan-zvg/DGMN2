import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from ..functions.deform_unfold import deform_unfold


class DeformUnfold(nn.Module):

    def __init__(self,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 deformable_groups=1,
                 bias=False):
        assert not bias
        super(DeformUnfold, self).__init__()

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups


    def forward(self, input, offset):
        return deform_unfold(input, offset, self.kernel_size, self.stride,
                           self.padding, self.dilation,
                           self.deformable_groups)

