import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import gen_conv, gen_deconv, dis_conv
from .utils import batch_conv2d, batch_transposeconv2d, weight_init
from .utils import CP1, CP2
import pdb



class InpaintGenerator(nn.Module):
    """auxiliary network.
    """
    def __init__(self, cnum=48, nn_hard=False, baseg=None, rate=1):
        super(InpaintGenerator, self).__init__()
        self.trainbase=False
        self.baseg = baseg
        self.cnum = cnum
        # similarity encoder
        self.sconv1 = gen_conv(2*cnum, 4*cnum, 5, 1) # skip cnn out
        self.sconv2 = gen_conv(2*cnum, 4*cnum, 3, 1, activation=nn.ReLU()) # skip cnn out

        # auxiliary encoder
        self.bconv1 = gen_conv(3, cnum, 5, 1) # skip cnn out
        self.bconv2_downsample = gen_conv(int(cnum/2), 2*cnum, 3, 2)
        self.bconv3 = gen_conv(cnum, 2*cnum, 3, 1) # skip cnn out
        self.bconv4_downsample = gen_conv(cnum, 4*cnum, 3, 2)

        self.conv13_upsample_conv = gen_deconv(2*cnum, 2*cnum)
        self.conv14 = gen_conv(cnum*2, 2*cnum, 3, 1) # skip cnn in
        self.conv15_upsample_conv = gen_deconv(cnum, cnum)
        self.conv16 = gen_conv(cnum, cnum, 3, 1) # skip cnn in
        # auxiliary decoder
        self.conv16_2 = gen_conv(cnum//2, cnum, 3, 1)
        self.conv17 = gen_conv(cnum//2, 3, 3, 1, activation=None)
        self.cp_1 = CP1(nn_hard=nn_hard, ufstride=2*rate, 
                stride=2*rate, bkg_patch_size=4*rate, pd=1*rate)
        self.cp_2 = CP2(ufstride=2*4*rate, bkg_patch_size=16*rate, 
                stride=8*rate, pd=4*rate)
        self.apply(weight_init)

    def get_param_list(self, stage="all"):
        if stage=="all":
            list_param = [p for name, p in self.named_parameters()]
            return list_param
        elif stage=="base":
            list_param = [p for name, p in self.baseg.named_parameters()]
            return list_param
        else:
            raise NotImplementedError

    def forward(self, x, mask):
        _,_,hin,win = x.shape
        x_stage1, x_stage2, pm = self.baseg(x, mask)
        if (not self.training) or self.trainbase:
            return x_stage1, x_stage2, x_stage2

        # similarify
        xnow = x_stage2*mask + x*(1-mask)
        xs = self.sconv1(pm)
        x_similar = self.sconv2(xs)

        bsize, _, h, w = xs.size()
        mask_s = F.avg_pool2d(mask, kernel_size=4, stride=4)
        similar = self.cp_1(x_similar, x_similar, mask_s)

        xb = self.bconv1(xnow)
        x_skip1 = xb
        xb = self.bconv2_downsample(xb)
        xb = self.bconv3(xb)
        x_skip2 = xb
        xb = self.bconv4_downsample(xb)
        xb = self.conv13_upsample_conv(xb)
        xb = self.conv14(torch.cat((xb, x_skip2), 1))
        xb = self.conv15_upsample_conv(xb)
        xb = self.conv16(torch.cat((xb, x_skip1), 1))

        xb = self.cp_2(similar, xb, mask)

        xb = self.conv16_2(xb)
        xb = self.conv17(xb)
        xb = torch.tanh(xb)
        return x_stage1, x_stage2, xb


if __name__ == "__main__":
    pass
