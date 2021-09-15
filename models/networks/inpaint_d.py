import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torch.nn.functional import normalize
from models.networks.base_network import BaseNetwork
from models.networks.utils import dis_conv

class DeepFillDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super(DeepFillDiscriminator, self).__init__()
        cnum = 64
        self.conv1 = nn.utils.spectral_norm(dis_conv(4, cnum))
        self.conv2 = nn.utils.spectral_norm(dis_conv(cnum, cnum*2))
        self.conv3 = nn.utils.spectral_norm(dis_conv(cnum*2, cnum*4))
        self.conv4 = nn.utils.spectral_norm(dis_conv(cnum*4, cnum*4))
        self.conv5 = nn.utils.spectral_norm(dis_conv(cnum*4, cnum*4))
        self.conv6 = nn.utils.spectral_norm(dis_conv(cnum*4, cnum*4))

    def forward(self, x, mask=None):
        bsize, ch, height, width = x.shape
        if mask is None:
            ones_x = torch.ones(bsize, 1, height, width).to(x.device)
        else:
            ones_x = mask
        x = torch.cat([x, ones_x], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


if __name__ == "__main__":
    pass
