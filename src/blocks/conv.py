import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=False
    )


def conv1x1(in_channels, out_channels, stride=1, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias=False
    )


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, g=1):
        super().__init__()

        if k == 1:
            conv = conv1x1(in_ch, out_ch, s, g)
        else:
            conv = conv3x3(in_ch, out_ch, s, g)

        self.block = nn.Sequential(
            conv,
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DWConv(nn.Module):
    def __init__(self, channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(channels, channels, stride=stride, groups=channels),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return self.block(x)
