import torch
from capsule_layer import CapsuleConv2d
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.block1 = CapsuleConv2d(3, 64, 3, 1, 16, padding=1, similarity='tonimoto', squash=False)
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = CapsuleConv2d(64, 3, 3, 16, 1, padding=1, similarity='tonimoto', squash=False)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block1 + block2)
        block4 = self.block4(block3)

        return torch.sigmoid(block4)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.prelu2 = nn.PReLU()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu1(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.prelu2(residual)

        return x + residual

