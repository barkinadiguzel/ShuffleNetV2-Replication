import torch


def channel_split(x):
    c = x.size(1)
    c1 = c // 2
    return x[:, :c1, :, :], x[:, c1:, :, :]
