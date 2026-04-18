import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, in_ch, num_classes=1000):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
