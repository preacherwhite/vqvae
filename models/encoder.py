import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualStack


class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack_1 = nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1)
        self.conv_stack_2 = nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1)
        self.conv_stack_3 = nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1)
        self.res = ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        a = self.conv_stack_1(x)
        a = self.relu(a)
        b = self.conv_stack_2(a)
        b = self.relu(b)
        c = self.conv_stack_3(b)
        d = self.res(c)
        return d
class Encoder1d(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder1d, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack_1 = nn.Conv1d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1)
        self.conv_stack_2 = nn.Conv1d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1)
        self.conv_stack_3 = nn.Conv1d(h_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1)
        self.res = ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers, is2d=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        a = self.conv_stack_1(x)
        a = self.relu(a)
        b = self.conv_stack_2(a)
        b = self.relu(b)
        c = self.conv_stack_3(b)
        d = self.res(c)
        return d

if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(40, 128, 3, 64)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
