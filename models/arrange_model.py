import pdb
import torch
from models.inpaint_model import InpaintModel
import util.util as util

class ArrangeModel(InpaintModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        InpaintModel.modify_commandline_options(parser, is_train)
        parser.add_argument('--load_base_g', type=str, required=False, help='load baseg')
        parser.add_argument('--load_base_d', type=str, required=False, help='load based')
        parser.add_argument('--lambda_ref', type=float, default=1, help='weightloss')
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        _, self.netD_aux = self.initialize_networks(opt)
        if opt.continue_train:
            self.netD_aux = util.load_network(self.netD_aux, 'D_aux', opt.which_epoch, opt)
        if opt.load_base_g is not None:
            print(f"looad {opt.load_base_g}")
            self.netG.baseg = util.load_network_path(
                    self.netG.baseg, opt.load_base_g)
        if opt.load_base_d is not None:
            print(f"looad {opt.load_base_d}")
            self.netD = util.load_network_path(
                    self.netD, opt.load_base_d)

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        util.save_network(self.netD_aux, 'D_aux', epoch, self.opt)

    def create_optimizers(self, opt):
        G_params = self.netG.get_param_list(opt.update_part)
        #G_params = [p for name, p in self.netG.named_parameters() \
        #        if (not name.startswith("coarse"))]
        if opt.isTrain:
            D_params = list(self.netD.parameters()) + \
                    list(self.netD_aux.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def forward(self, data, mode):
        inputs, real_image, mask = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, composed_image, composed_image_aux, recon_aux =\
                    self.compute_generator_loss(inputs, real_image, mask)
            generated = {
                    'composed': composed_image,
                    'composed_aux': composed_image_aux,
                    **recon_aux
                    }
            return g_loss, inputs, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                inputs, real_image, mask)
            return d_loss, data['inputs']
        elif mode == 'inference':
            with torch.no_grad():
                coarse_image, fake_image, aux_image, recon_aux = self.generate_fake(
                        inputs, real_image, mask)
                composed_image = fake_image*mask + inputs*(1-mask)
            return composed_image, inputs
        else:
            raise ValueError("|mode| is invalid")

    def generate_fake(self, inputs, real_image, mask):
        coarse_image, fake_image, aux_image, recon_aux = self.netG(inputs, mask)

        return coarse_image, fake_image, aux_image, recon_aux

    def compute_discriminator_loss(self, inputs, real_image, mask):
        D_losses = {}
        if not self.opt.no_gan_loss:
            with torch.no_grad():
                coarse_image, fake_image, aux_image, recon_aux = self.generate_fake(
                        inputs, real_image, mask)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
            aux_image = aux_image.detach()
            aux_image.requires_grad_()
            composed_image = fake_image*mask + inputs*(1-mask)
            composed_aux = aux_image*mask + inputs*(1-mask)

            pred_fake, pred_real = self.discriminate(
                composed_image, real_image, mask)
            D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                                   for_discriminator=True)
            D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                                   for_discriminator=True)
            _netD = self.netD
            self.netD = self.netD_aux
            pred_fake, pred_real = self.discriminate(
                composed_aux, real_image, mask)
            D_losses['D_Fake_aux'] = self.criterionGAN(pred_fake, False,
                                                   for_discriminator=True)
            D_losses['D_real_aux'] = self.criterionGAN(pred_real, True,
                                                   for_discriminator=True)
            self.netD = _netD

        return D_losses

    def compute_generator_loss(self, inputs, real_image, mask):
        if not self.opt.no_ganFeat_loss:
            raise NotImplementedError
        if not self.opt.no_vgg_loss:
            raise NotImplementedError
        coarse_image, fake_image, aux_image, recon_aux = self.generate_fake(
                inputs, real_image, mask)
        composed_image = fake_image*mask + inputs*(1-mask)
        G_losses = self.g_image_loss(coarse_image, fake_image, composed_image, real_image, mask)

        composed_image_aux = aux_image*mask + inputs*(1-mask)
        _netD = self.netD
        self.netD = self.netD_aux
        G_losses_aux = self.g_image_loss(None, aux_image, composed_image_aux, real_image, mask)
        self.netD = _netD
        for k,v in G_losses_aux.items():
            G_losses[k+"_aux"] = v*self.opt.lambda_ref
        return G_losses, composed_image, composed_image_aux, recon_aux
