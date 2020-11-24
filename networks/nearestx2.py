import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from PIL import Image
from .utils import weight_init, gen_conv, gen_deconv, dis_conv

class InpaintGenerator(nn.Module):
    def __init__(self, cnum=48):
        super(InpaintGenerator, self).__init__()
        self.cnum = cnum
        # stage1
        self.conv1 = gen_conv(5, cnum, 5, 1)
        self.conv2_downsample = gen_conv(int(cnum/2), 2*cnum, 3, 2)
        self.conv3 = gen_conv(cnum, 2*cnum, 3, 1)
        self.conv4_downsample = gen_conv(cnum, 4*cnum, 3, 2)
        self.conv5 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.conv6 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.conv7_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=2)
        self.conv8_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=4)
        self.conv9_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=8)
        self.conv10_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=16)
        self.conv11 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.conv12 = gen_conv(2*cnum, 4*cnum, 3, 1)

        self.conv13_upsample_conv = gen_deconv(2*cnum, 2*cnum)
        self.conv14 = gen_conv(cnum, 2*cnum, 3, 1)
        self.conv15_upsample_conv = gen_deconv(cnum, cnum)
        self.conv16 = gen_conv(cnum//2, cnum//2, 3, 1)
        self.conv17 = gen_conv(cnum//4, 3, 3, 1, activation=None)

        # stage2
        self.xconv1 = gen_conv(3, cnum, 5, 1)
        self.xconv2_downsample = gen_conv(cnum//2, cnum, 3, 2)
        self.xconv3 = gen_conv(cnum//2, 2*cnum, 3, 1)
        self.xconv4_downsample = gen_conv(cnum, 2*cnum, 3, 2)
        self.xconv5 = gen_conv(cnum, 4*cnum, 3, 1)
        self.xconv6 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.xconv7_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=2)
        self.xconv8_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=4)
        self.xconv9_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=8)
        self.xconv10_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=16)

        self.pmconv1 = gen_conv(3, cnum, 5, 1)
        self.pmconv2_downsample = gen_conv(cnum//2, cnum, 3, 2)
        self.pmconv3 = gen_conv(cnum//2, 2*cnum, 3, 1)
        self.pmconv4_downsample = gen_conv(cnum, 4*cnum, 3, 2)
        self.pmconv5 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.pmconv6 = gen_conv(2*cnum, 4*cnum, 3, 1, 
                            activation=nn.ReLU())
        self.pmconv9 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.pmconv10 = gen_conv(2*cnum, 4*cnum, 3, 1)

        self.allconv11 = gen_conv(4*cnum, 4*cnum, 3, 1)
        self.allconv12 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.allconv13_upsample_conv = gen_deconv(2*cnum, 2*cnum)
        self.allconv14 = gen_conv(cnum, 2*cnum, 3, 1)
        self.allconv15_upsample_conv = gen_deconv(cnum, cnum)
        self.allconv16 = gen_conv(cnum//2, cnum//2, 3, 1)
        self.allconv17 = gen_conv(cnum//4, 3, 3, 1, activation=None)

        self.apply(weight_init)


    def forward(self, x, mask, guide=None, mask_obj=None):
        xin = x
        xs = F.interpolate(xin, scale_factor=0.5)
        mask_s = F.interpolate(mask, scale_factor=0.5)
        bsize, ch, height, width = xs.shape
        ones_x = torch.ones(bsize, 1, height, width).to(x.device)
        x = torch.cat([xs, ones_x, ones_x*mask_s], 1)

        # two stage network
        ## stage1
        x = self.conv1(x)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13_upsample_conv(x)
        x = self.conv14(x)
        x = self.conv15_upsample_conv(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = torch.tanh(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x_stage1 = x

        ## stage2, paste result as input
        x = x*mask + xin[:, 0:3, :, :]*(1.-mask)
        xnow = x
        ### atrous conv branch
        x = self.xconv1(xnow)
        x = self.xconv2_downsample(x)
        x = self.xconv3(x)
        x = self.xconv4_downsample(x)
        x = self.xconv5(x)
        x = self.xconv6(x)
        x = self.xconv7_atrous(x)
        x = self.xconv8_atrous(x)
        x = self.xconv9_atrous(x)
        x = self.xconv10_atrous(x)
        x_hallu = x
        ### attention branch
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], 1)

        x = self.allconv11(x)
        x = self.allconv12(x)
        x = self.allconv13_upsample_conv(x)
        x = self.allconv14(x)
        x = self.allconv15_upsample_conv(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage2 = torch.tanh(x)
        return x_stage1, x_stage2


if __name__ == "__main__":
    pretrained_dict = torch.load("../../../checkpoints/1120u/_5980000_net_ArrangeModel_twostagend_g.pth")
    baseg = InpaintGenerator()
    init_dict = baseg.state_dict()
    for k,v in pretrained_dict.items():
        _k = k.replace("module.", "")
        if _k.startswith("baseg"):
            __k = _k.replace("baseg.", "")
            init_dict[__k] = v
    pdb.set_trace()
    torch.save(init_dict, "../files/model_near512.pth")

