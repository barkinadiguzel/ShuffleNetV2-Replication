import torch
import torch.nn as nn
from blocks.conv import ConvBNReLU, DWConv
from blocks.shuffle import channel_shuffle


class DownsampleUnit(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        mid = out_ch // 2

        self.branch1 = nn.Sequential(
            DWConv(in_ch, stride=2),
            ConvBNReLU(in_ch, mid, k=1)
        )

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_ch, mid, k=1),
            DWConv(mid, stride=2)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x1, x2), dim=1)
        out = channel_shuffle(out)

        return out
