
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualStack


class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.res = ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)

        self.conv_stack_1 = nn.ConvTranspose2d(h_dim, h_dim,
                           kernel_size=kernel-1, stride=stride-1, padding=1)
        self.conv_stack_2 = nn.ConvTranspose2d(h_dim, h_dim//2, kernel_size=kernel,
                                               stride=stride, padding=1)
        self.conv_stack_3 = nn.ConvTranspose2d(h_dim//2, in_dim, kernel_size=kernel,
                                               stride=stride, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        a = self.res(x)
        b = self.conv_stack_1(a)
        b = self.relu(b)
        c = self.conv_stack_2(b)
        c = self.relu(c)
        d = self.conv_stack_3(c)
        return d

class Decoder1d(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder1d, self).__init__()
        kernel = 4
        stride = 2

        self.res = ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers, is2d=False)

        self.conv_stack_1 = nn.ConvTranspose1d(h_dim, h_dim,
                           kernel_size=kernel-1, stride=stride-1, padding=1)
        self.conv_stack_2 = nn.ConvTranspose1d(h_dim, h_dim//2, kernel_size=kernel,
                                               stride=stride, padding=1)
        self.conv_stack_3 = nn.ConvTranspose1d(h_dim//2, in_dim, kernel_size=kernel,
                                               stride=stride, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        a = self.res(x)
        b = self.conv_stack_1(a)
        b = self.relu(b)
        c = self.conv_stack_2(b)
        c = self.relu(c)
        d = self.conv_stack_3(c)
        return d

if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test decoder
    decoder = Decoder(40, 128, 3, 64)
    decoder_out = decoder(x)
    print('Dncoder out shape:', decoder_out.shape)
