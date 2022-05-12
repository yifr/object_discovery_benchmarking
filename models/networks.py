import torch
import einops
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        n_layers,
        intermediate_size,
        out_features,
        nonlinearity=nn.ReLU,
    ):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.n_layers = n_layers
        self.out_features = out_features
        self.nonlinearity = nonlinearity
        layers = []
        for i in range(n_layers):
            if i == 0:
                layer = nn.Linear(in_features, intermediate_size)
            elif i == (n_layers - 1):
                layer = nn.Linear(intermediate_size, out_features)
            else:
                layer = nn.Linear(intermediate_size, intermediate_size)

            layers.append(layer)
            layers.append(nonlinearity())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def ConvNet(
    num_layers,
    in_channels,
    mid_channels,
    out_channels,
    kernel=4,
    stride=2,
    padding=1,
    dim=2,
):
    """
    :param num_layers: (int) number of layers in network
    :param in_channels: (int) number of input channels
    :param mid_channels: (int) number of channels in intermediate layers
    :param out_channels: (int) number of output channels
    :param kernel: (int) kernel size
    :param stride: (int) stride size
    :param padding: (int) padding size
    :param dim: (int) dimension of convolutions (ie; 1D, 2D, 3D)

    returns:
    --------
        Vanilla, fully convolutional network
    """

    layers = []
    if dim == 1:
        net = nn.Conv1d
    elif dim == 2:
        net = nn.Conv2d
    elif dim == 3:
        net = nn.Conv3d

    for i in range(int(num_layers)):
        if i == 0:
            conv = net(
                in_channels,
                mid_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
        elif i == num_layers - 1:
            conv = net(
                mid_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
        else:
            conv = net(
                mid_channels,
                mid_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
        layers.append(conv)
        layers.append(nn.ReLU())

    convnet = nn.Sequential(*layers)
    return convnet


class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y, z) during forward pass instead of (batchsize, x, y, z, ch)

        returns:
        --------
        Positional encoding layer for 3d tensors
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.pos_enc = PositionalEncoding3D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.pos_enc(tensor)
        return enc.permute(0, 4, 1, 2, 3)


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: (int) The number of channels to apply the position encoding to
        Forward pass expects data to be (batchsize, ch, x, y, z)

        returns:
        --------
        Positional encoding layer for 3d tensors
        """
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = (
            torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
            .unsqueeze(1)
            .unsqueeze(1)
        )
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        return emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
