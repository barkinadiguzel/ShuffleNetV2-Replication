import torch


def channel_shuffle(x, groups=2):
    b, c, h, w = x.size()

    channels_per_group = c // groups

    x = x.view(b, groups, channels_per_group, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, c, h, w)

    return x
