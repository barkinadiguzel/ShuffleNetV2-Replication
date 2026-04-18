import torch.nn as nn

from modules.stage_builder import Stage
from modules.downsample_unit import DownsampleUnit
from blocks.conv import ConvBNReLU
from head.classifier import Classifier


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=1000, stages_out_channels=(24, 116, 232, 464, 1024)):
        super().__init__()

        self.conv1 = ConvBNReLU(3, stages_out_channels[0], k=3, s=2)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.stage2 = Stage(stages_out_channels[0], stages_out_channels[1], repeats=4)
        self.stage3 = Stage(stages_out_channels[1], stages_out_channels[2], repeats=8)
        self.stage4 = Stage(stages_out_channels[2], stages_out_channels[3], repeats=4)

        self.conv5 = ConvBNReLU(stages_out_channels[3], stages_out_channels[4], k=1)

        self.classifier = Classifier(stages_out_channels[4], num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.conv5(x)
        x = self.classifier(x)

        return x
