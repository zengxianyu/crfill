import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torch.nn.functional import normalize


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.ConvTranspose2d) and m.in_channels == m.out_channels:
        initial_weight = get_upsampling_weight(
            m.in_channels, m.out_channels, m.kernel_size[0])
        m.weight.data.copy_(initial_weight)


class gen_conv(nn.Conv2d):
    def __init__(self, cin, cout, ksize, stride=1, rate=1, activation=nn.ELU()):
        """Define conv for generator

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
            Stride: Convolution stride.
            rate: Rate for or dilated conv.
            activation: Activation function after convolution.
        """
        p = int(rate*(ksize-1)/2)
        super(gen_conv, self).__init__(in_channels=cin, out_channels=cout, kernel_size=ksize, stride=stride, padding=p, dilation=rate, groups=1, bias=True)
        self.activation = activation

    def forward(self, x):
        x = super(gen_conv, self).forward(x)
        if self.out_channels == 3 or self.activation is None:
            return x
        x, y = torch.split(x, int(self.out_channels/2), dim=1)
        x = self.activation(x)
        y = torch.sigmoid(y)
        x = x * y
        return x

class gen_deconv(gen_conv):
    def __init__(self, cin, cout):
        """Define deconv for generator.
        The deconv is defined to be a x2 resize_nearest_neighbor operation with
        additional gen_conv operation.

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
        """
        super(gen_deconv, self).__init__(cin, cout, ksize=3)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2)
        x = super(gen_deconv, self).forward(x)
        return x

class dis_conv(nn.Conv2d):
    def __init__(self, cin, cout, ksize=5, stride=2):
        """Define conv for discriminator.
        Activation is set to leaky_relu.

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
            Stride: Convolution stride.
        """
        p = int((ksize-1)/2)
        super(dis_conv, self).__init__(in_channels=cin, out_channels=cout, kernel_size=ksize, stride=stride, padding=p, dilation=1, groups=1, bias=True)

    def forward(self, x):
        x = super(dis_conv, self).forward(x)
        x = F.leaky_relu(x)
        return x

def batch_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """Define batch convolution to use different conv. kernels in a batch.

    Args:
        x: input feature maps of shape (batch, channel, height, width)
        weight: conv.kernels of shape (batch, out_channel, in_channels, kernel_size, kernel_size)
    """
    if bias is None:
        assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
    else:
        assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
            0], "dim=0 of bias must be equal in size to dim=0 of weight"

    b_i, c, h, w = x.shape
    b_i, out_channels, in_channels, kernel_height_size, kernel_width_size = weight.shape

    out = x[None, ...].view(1, b_i * c, h, w)
    weight = weight.contiguous().view(b_i * out_channels, in_channels, kernel_height_size, kernel_width_size)

    out = F.conv2d(out, weight=weight, bias=None, stride=stride, dilation=dilation, groups=b_i,
                   padding=padding)

    out = out.view(b_i, out_channels, out.shape[-2], out.shape[-1])

    if bias is not None:
        out = out + bias.unsqueeze(2).unsqueeze(3)

    return out


def batch_transposeconv2d(x, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1):
    """Define batch transposed convolution to use different conv. kernels in a batch.

    Args:
        x: input feature maps of shape (batch, channel, height, width)
        weight: conv.kernels of shape (batch, in_channel, out_channels, kernel_size, kernel_size)
    """
    if bias is None:
        assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
    else:
        assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
            0], "dim=0 of bias must be equal in size to dim=0 of weight"

    b_i, c, h, w = x.shape
    b_i, in_channels, out_channels, kernel_height_size, kernel_width_size = weight.shape

    out = x[None, ...].view(1, b_i * c, h, w)
    weight = weight.contiguous().view(in_channels*b_i, out_channels, kernel_height_size, kernel_width_size)

    out = F.conv_transpose2d(out, weight=weight, bias=None, stride=stride, dilation=dilation, groups=b_i,
                   padding=padding, output_padding=output_padding)

    out = out.view(b_i, out_channels, out.shape[-2], out.shape[-1])

    if bias is not None:
        out = out + bias.unsqueeze(2).unsqueeze(3)
    return out


if __name__ == "__main__":
    pass
