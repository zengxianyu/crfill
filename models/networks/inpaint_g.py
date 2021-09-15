import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torch.nn.functional import normalize
from models.networks.base_network import BaseNetwork
from models.networks.utils import gen_conv, gen_deconv, dis_conv
from models.networks.splitcam import ReduceContextAttentionP1, ReduceContextAttentionP2
from util.util import find_class_in_module

class TwostagendGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--baseG', type=str, default='baseconv',
                            help="baseg")
        opt, unknown = parser.parse_known_args()
        network = find_class_in_module(opt.baseG+"generator",
                "models.networks.inpaint_g")
        network.modify_commandline_options(parser, is_train)
    def get_param_list(self, stage="all"):
        if stage=="all":
            list_param = [p for name, p in self.named_parameters()]
            return list_param
        else:
            raise NotImplementedError
    def __init__(self, opt):
        super(TwostagendGenerator, self).__init__()
        network = find_class_in_module(opt.baseG+"generator",
                "models.networks.inpaint_g")
        baseg = network(opt, return_pm=True)
        self.baseg = baseg
        rate = 1
        cnum = self.baseg.cnum
        self.cnum = cnum
        # similarity encoder
        self.sconv1 = gen_conv(2*cnum, 4*cnum, 5, 1) # skip cnn out
        self.sconv2 = gen_conv(2*cnum, 4*cnum, 3, 1, activation=nn.ReLU()) # skip cnn out

        # feature encoder
        self.bconv1 = gen_conv(3, cnum, 5, 1) # skip cnn out
        self.bconv2_downsample = gen_conv(int(cnum/2), 2*cnum, 3, 2)
        self.bconv3 = gen_conv(cnum, 2*cnum, 3, 1) # skip cnn out
        self.bconv4_downsample = gen_conv(cnum, 4*cnum, 3, 2)

        self.conv13_upsample_conv = gen_deconv(2*cnum, 2*cnum)
        self.conv14 = gen_conv(cnum*2, 2*cnum, 3, 1) # skip cnn in
        self.conv15_upsample_conv = gen_deconv(cnum, cnum)
        self.conv16 = gen_conv(cnum, cnum, 3, 1) # skip cnn in

        self.conv16_2 = gen_conv(cnum//2, cnum, 3, 1) # skip cnn in

        ##cnum//2
        self.conv17 = gen_conv(cnum//2, 3, 3, 1, activation=None)
        self.cam_1 = ReduceContextAttentionP1(nn_hard=False, ufstride=2*rate,
                stride=2*rate, bkg_patch_size=4*rate, pd=0,
                norm_type=opt.norm_type, is_th=opt.use_th, th=opt.th)
        self.cam_2 = ReduceContextAttentionP2(ufstride=2*4*rate, bkg_patch_size=16*rate,
                stride=8*rate, pd=0, mk=False)

    def forward(self, x, mask):
        # mask_obj: mask of avoid region
        _,_,hin,win = x.shape
        x_stage1, x_stage2, pm = self.baseg(x, mask)
        if not self.training:
            #print("network not forwarding refine")
            return x_stage1, x_stage2, x_stage2, {}

        # similarify
        xnow = x_stage2*mask + x*(1-mask)

        #xs = self.sconv1(pm)
        #x_similar = self.sconv2(xs)
        x_similar = pm

        bsize, _, h, w = pm.size()
        mask_s = F.avg_pool2d(mask, kernel_size=4, stride=4)
        similar = self.cam_1(x_similar, x_similar, mask_s)
        # feature
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

        xb, recon_aux = self.cam_2(similar, xb, mask, {'raw':x})
        xb = self.conv16_2(xb)
        xb = self.conv17(xb)
        xb = torch.tanh(xb)
        return x_stage1, x_stage2, xb, recon_aux

class DeepFillGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--use_cam', action='store_true',
                            help="use context attention module")
        parser.add_argument('--no_mk', action='store_true',
                            help="use context attention module")

    def __init__(self, opt, return_feat=False):
        super(DeepFillGenerator, self).__init__()
        self.return_feat = return_feat
        self.use_cam = opt.use_cam
        cnum = 48
        self.cnum = cnum
        rate = 1
        self.cam_1 = ReduceContextAttentionP1(nn_hard=False,
                ufstride=2*rate,
                stride=2*rate,
                bkg_patch_size=4*rate, pd=0,
                norm_type=opt.norm_type,
                is_th=opt.use_th, th=opt.th
                )
        mk = False if opt.no_mk else True
        self.cam_2 = ReduceContextAttentionP2(
                ufstride=2*rate,
                bkg_patch_size=4*rate,
                stride=2*rate, pd=0,mk=mk)
        #self.cam_1 = ReduceContextAttentionP1(nn_hard=False,
        #        ufstride=2*rate,
        #        stride=2*rate,
        #        bkg_patch_size=4*rate, pd=1*rate)
        #self.cam_2 = ReduceContextAttentionP2(
        #        ufstride=2*rate,
        #        bkg_patch_size=4*rate,
        #        stride=2*rate, pd=1*rate)
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
        self.conv10_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=16) #8
        self.conv11 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.conv12 = gen_conv(2*cnum, 4*cnum, 3, 1) #4
        self.conv13_upsample_conv = gen_deconv(2*cnum, 2*cnum)
        self.conv14 = gen_conv(cnum, 2*cnum, 3, 1) #2
        self.conv15_upsample_conv = gen_deconv(cnum, cnum)
        self.conv16 = gen_conv(cnum//2, cnum//2, 3, 1) #1
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

    def get_param_list(self, stage="all"):
        if stage=="all":
            list_param = [p for name, p in self.named_parameters()]
            return list_param
        elif stage=="coarse":
            list_param = [p for name, p in self.named_parameters() \
                    if (name.startswith("conv"))]
            return list_param
        elif stage=="fine":
            list_param = [p for name, p in self.named_parameters() \
                    if not (name.startswith("conv"))]
            return list_param
        else:
            raise NotImplementedError


    def forward(self, x, mask):
        xin = x
        bsize, ch, height, width = x.shape
        ones_x = torch.ones(bsize, 1, height, width).to(x.device)
        x = torch.cat([x, ones_x, ones_x*mask], 1)

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
        x_stage1 = x

        x = x*mask + xin[:, 0:3, :, :]*(1.-mask)
        xnow = x

        ###
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

        ###
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        if self.use_cam:
            mask_s = F.avg_pool2d(mask, kernel_size=4, stride=4)
            similar = self.cam_1(x, x, mask_s)
            x, recon_aux = self.cam_2(similar, x, mask_s, {})
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], 1)
        feat = x

        x = self.allconv11(x)
        x = self.allconv12(x)
        x = self.allconv13_upsample_conv(x)
        feat_x2 = x
        x = self.allconv14(x)
        x = self.allconv15_upsample_conv(x)
        feat_x4 = x
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage2 = torch.tanh(x)
        if self.return_feat:
            return x_stage1, x_stage2, [feat, feat_x2, feat_x4]
        else:
            return x_stage1, x_stage2


class BaseConvGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        pass

    def __init__(self, opt, return_feat=False, return_pm=False):
        super(BaseConvGenerator, self).__init__()
        self.return_feat = return_feat
        self.return_pm = return_pm
        cnum = 48
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

    def get_param_list(self, stage="all"):
        if stage=="all":
            list_param = [p for name, p in self.named_parameters()]
            return list_param
        elif stage=="coarse":
            list_param = [p for name, p in self.named_parameters() \
                    if (name.startswith("conv"))]
            return list_param
        elif stage=="fine":
            list_param = [p for name, p in self.named_parameters() \
                    if not (name.startswith("conv"))]
            return list_param
        else:
            raise NotImplementedError


    def forward(self, x, mask):
        xin = x
        bsize, ch, height, width = x.shape
        ones_x = torch.ones(bsize, 1, height, width).to(x.device)
        x = torch.cat([x, ones_x, ones_x*mask], 1)

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
        x_stage1 = x

        x = x*mask + xin[:, 0:3, :, :]*(1.-mask)
        xnow = x

        ###
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

        ###
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        pm_return = x

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
        if self.return_pm:
            return x_stage1, x_stage2, pm_return

        return x_stage1, x_stage2

if __name__ == "__main__":
    pass
