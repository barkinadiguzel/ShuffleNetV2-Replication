import torch.nn as nn
from blocks.conv import ConvBNReLU, DWConv
from blocks.split import channel_split
from blocks.shuffle import channel_shuffle


class ShuffleUnit(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        assert in_ch == out_ch

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_ch // 2, in_ch // 2, k=1),
            DWConv(in_ch // 2, stride=1),
            ConvBNReLU(in_ch // 2, in_ch // 2, k=1)
        )

    def forward(self, x):
        x1, x2 = channel_split(x)

        out = torch.cat((x1, self.branch2(x2)), dim=1)
        out = channel_shuffle(out)

        return out
