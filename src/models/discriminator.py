import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from utils import get_activation

class SNConv2D(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SNConv2D, self).__init__()
        self.spec_conv = spectral_norm(nn.Conv2d(*args, **kwargs))

    def forward(self, x):
        return self.spec_conv(x)

class WNConv2D(nn.Module):
    def __init__(self, *args, **kwargs):
        super(WNConv2D, self).__init__()
        self.weight_conv = weight_norm(nn.Conv2d(*args, **kwargs))

    def forward(self, x):
        return self.weight_conv(x)

class SNConv1D(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SNConv1D, self).__init__()
        self.spec_conv = spectral_norm(nn.Conv1d(*args, **kwargs))

    def forward(self, x):
        return self.spec_conv(x)

class WNConv1D(nn.Module):
    def __init__(self, *args, **kwargs):
        super(WNConv1D, self).__init__()
        self.weight_conv = weight_norm(nn.Conv1d(*args, **kwargs))

    def forward(self, x):
        return self.weight_conv(x)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, params, kernel_size=5, stride=3):
        super(DiscriminatorP, self).__init__()

        self.activation = get_activation(params)
        self.period = period

        conv_channels = [32, 128, 512, 1024, 1024]
        self.convs = nn.ModuleList([])

        for i in range(len(conv_channels)):
            in_channels = conv_channels[i - 1] if i != 0 else 1
            out_channels = conv_channels[i]
            cur_stride = (stride, 1) if i != len(conv_channels) - 1 else 1
            self.convs.append(WNConv2D(in_channels, out_channels, (kernel_size, 1), stride=cur_stride, padding=(2, 0)))

        self.conv_post = WNConv2D(conv_channels[-1], 1, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.flatten = nn.Flatten()


    def forward(self, x):

        b, c, t = x.shape
        pad_size = self.period - (t % self.period)
        t = t + pad_size

        x = F.pad(x, (0, pad_size), "reflect")
        x = x.view(b, c, t // self.period, self.period)

        features = []
        for conv in self.convs:
            x = conv(x)
            x = self.activation(x)
            features.append(x)

        x = self.conv_post(x)
        features.append(x)

        return self.flatten(x), features


class MPDiscriminator(torch.nn.Module):
    def __init__(self, params):
        super(MPDiscriminator, self).__init__()

        periods = [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList([])

        for period in periods:
            self.discriminators.append(DiscriminatorP(period, params))

    def forward(self, y):
        y_list = []
        f_list = []

        for discr in self.discriminators:
            cur_y, cur_f = discr(y)
            y_list.append(cur_y)
            f_list.append(cur_f)

        return y_list, f_list


class DiscriminatorS(torch.nn.Module):
    def __init__(self, params, use_spec_norm=False):
        super(DiscriminatorS, self).__init__()

        conv_layer = SNConv1D if use_spec_norm else WNConv1D

        self.activation = get_activation(params)
        self.convs = nn.ModuleList([
            conv_layer(1, 128, 15, 1, padding=7),
            conv_layer(128, 128, 41, 2, groups=4, padding=20),
            conv_layer(128, 256, 41, 2, groups=16, padding=20),
            conv_layer(256, 512, 41, 4, groups=16, padding=20),
            conv_layer(512, 1024, 41, 4, groups=16, padding=20),
            conv_layer(1024, 1024, 41, 1, groups=16, padding=20),
            conv_layer(1024, 1024, 5, 1, padding=2),
        ])

        self.conv_post = conv_layer(1024, 1, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()


    def forward(self, x):
        features = []

        for conv in self.convs:
            x = conv(x)
            x = self.activation(x)
            features.append(x)

        x = self.conv_post(x)
        features.append(x)

        return self.flatten(x), features


class MSDiscriminator(torch.nn.Module):
    def __init__(self, params):
        super(MSDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(params, use_spec_norm=True),
            DiscriminatorS(params),
            DiscriminatorS(params),
        ])
        self.avgpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

        assert len(self.discriminators) - 1 == len(self.avgpools)

    def forward(self, y):
        y_list = []
        f_list = []

        for i, discr in enumerate(self.discriminators):
            if i != 0:
                y = self.avgpools[i-1](y)

            cur_y, cur_f = discr(y)
            y_list.append(cur_y)
            f_list.append(cur_f)

        return y_list, f_list
