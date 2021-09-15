import pdb
import torch
from models.networks.sync_batchnorm import DataParallelWithCallback
import models
#from models.pix2pix_model import Pix2PixModel


class StyleGAN2Trainer():
    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = models.create_model(opt)
        if len(opt.gpu_ids) > 0:
            self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
                                                          device_ids=opt.gpu_ids)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = None
        self.inputs = None
        self.mean_path_length = torch.Tensor([0])
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

    def run_generator_one_step(self, data, i):
        self.optimizer_G.zero_grad()
        g_losses, inputs, generated = self.pix2pix_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated
        self.inputs = inputs
        g_regularize = (i % self.opt.g_reg_every == 0) and not (self.opt.no_g_reg)
        if g_regularize:
            self.optimizer_G.zero_grad()
            bsize = data['image'].size(0)
            data['mean_path_length'] = self.mean_path_length.expand(bsize)
            g_regs, self.mean_path_length \
                    = self.pix2pix_model(data, mode='g_reg')
            g_reg = sum(g_regs.values()).mean()
            g_reg.backward()
            self.optimizer_G.step()
            self.g_losses = {
                    **g_losses,
                    **g_regs}
        bsize = inputs.size(0)
        accum = 0.5 ** (bsize / (10 * 1000)) # 32
        self.pix2pix_model_on_one_gpu.accumulate(accum)

    def run_discriminator_one_step(self, data, i):
        self.optimizer_D.zero_grad()
        d_losses_real = self.pix2pix_model(data, mode='dreal')
        d_loss_real = sum(d_losses_real.values()).mean()
        d_loss_real.backward()
        d_losses_fake = self.pix2pix_model(data, mode='dfake')
        d_loss_fake = sum(d_losses_fake.values()).mean()
        d_loss_fake.backward()
        self.d_losses = {
                **d_losses_real,
                **d_losses_fake}
        self.optimizer_D.step()
        d_regularize = i % self.opt.d_reg_every == 0
        if d_regularize:
            self.optimizer_D.zero_grad()
            d_regs = self.pix2pix_model(data, mode='d_reg')
            d_reg = sum(d_regs.values()).mean()
            d_reg.backward()
            self.optimizer_D.step()
            self.d_losses = {
                    **self.d_losses,
                    **d_regs}

    def get_latest_losses(self):
        if not self.opt.freeze_D:
            return {**self.g_losses, **self.d_losses}
        else:
            return self.g_losses

    def get_latest_generated(self):
        return self.generated
    def get_latest_inputs(self):
        return self.inputs

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
