import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm
from typing import List, Dict, Any
import utils

class WNConv1D(nn.Module):
    def __init__(self, *args, **kwargs):
        super(WNConv1D, self).__init__()
        self.weight_conv = weight_norm(nn.Conv1d(*args, **kwargs))

    def forward(self, x):

        return self.weight_conv(x)


class ResBlock(nn.Module):
    def __init__(self, params: Dict[str, Any], n_channels: int, kernel_size: int, dilation: List[int]):
        super(ResBlock, self).__init__()
        self.params = params
        assert len(dilation) == 3

        self.conv1_list = nn.ModuleList([])
        self.conv2_list = nn.ModuleList([])
        conv1_dils = dilation
        conv2_dils = [1, 1, 1]

        for i in range(len(dilation)):
            self.conv1_list.append(WNConv1D(
                n_channels, n_channels, kernel_size,
                stride=1, dilation=conv1_dils[i], padding='same'
            ))

            self.conv2_list.append(WNConv1D(
                n_channels, n_channels, kernel_size,
                stride=1, dilation=conv2_dils[i], padding='same'
            ))

        self.activation = utils.get_activation(params)

    def forward(self, x):
        for i in range(len(self.conv1_list)):
            res = x
            x = self.conv1_list[i](self.activation(x))
            x = self.conv2_list[i](self.activation(x))

            return x + res

    def remove_weight_norm(self):
        for l in self.conv1_list:
            remove_weight_norm(l)
        for l in self.conv2_list:
            remove_weight_norm(l)


class MRF(nn.Module):
    def __init__(self, params: Dict[str, Any], n_channels: int):
        super(MRF, self).__init__()

        kernel_sizes = params['resblock_kernel_sizes']
        dilation_sizes_list = params['resblock_dilation_sizes']
        assert len(kernel_sizes) == len(dilation_sizes_list)

        self.resblocks = nn.ModuleList([])
        for i in range(len(dilation_sizes_list)):
            self.resblocks.append(ResBlock(params, n_channels, kernel_sizes[i], dilation_sizes_list[i]))

    def forward(self, x):

        result = 0.
        for block in self.resblocks:
            result += block(x)

        return result / len(self.resblocks)

class WNConvTranspose1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(WNConvTranspose1d, self).__init__()
        self.weight_tr_conv = weight_norm(nn.ConvTranspose1d(*args, **kwargs))

    def forward(self, x):

        return self.weight_tr_conv(x)


class Generator(torch.nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.params = params

        up_n_channels = params['upsample_channel']
        self.pre_conv = WNConv1D(params['n_mels'], up_n_channels, kernel_size=7, stride=1, padding=3)

        self.up_layers = nn.ModuleList([])
        self.num_upsample_layers = len(params['upsample_strides'])

        for i in range(self.num_upsample_layers):

            kernel_size = params['upsample_kernel_sizes'][i]
            stride = params['upsample_strides'][i]
            padding = (kernel_size - stride) // 2

            self.up_layers.append(
                WNConvTranspose1d(
                    up_n_channels//(2**i), up_n_channels//(2**(i+1)),
                    kernel_size=kernel_size, stride=stride, padding=padding)
            )

        self.up_resblocks = nn.ModuleList([])

        for i in range(len(self.up_layers)):
            n_channels = up_n_channels //(2**(i+1))
            self.up_resblocks.append(MRF(params, n_channels))

        self.post_conv = WNConv1D(n_channels, 1, kernel_size=7, stride=1, padding=3)
        self.activation = utils.get_activation(params)
        # self.ups.apply(init_weights)
        # self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.pre_conv(x)
        for i in range(self.num_upsample_layers):
            x = self.activation(x)
            x = self.up_layers[i](x)
            x = self.up_resblocks[i](x)

        x = self.activation(x)
        x = self.post_conv(x)

        return torch.tanh(x)
