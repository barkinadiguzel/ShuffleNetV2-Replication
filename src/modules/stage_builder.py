import torch.nn as nn
from modules.shufflenet_unit import ShuffleUnit
from modules.downsample_unit import DownsampleUnit


class Stage(nn.Module):
    def __init__(self, in_ch, out_ch, repeats):
        super().__init__()

        layers = [DownsampleUnit(in_ch, out_ch)]

        for _ in range(repeats - 1):
            layers.append(ShuffleUnit(out_ch, out_ch))

        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)
