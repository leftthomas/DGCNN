import math

import torch
from capsule_layer import CapsuleConv2d
from torch import nn


class Model(nn.Module):
    def __init__(self, upscale_factor=2):
        super(Model, self).__init__()

        upsample_block_num = int(math.log(upscale_factor, 2))
        if upscale_factor % 2 == 0:
            upscale_factor = 2
        self.block1 = CapsuleConv2d(3, 64, 3, 1, 16, padding=1, similarity='tonimoto', squash=False)
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        block4 = [UpsampleBlock(64, upscale_factor) for _ in range(upsample_block_num)]
        self.block4 = nn.Sequential(*block4)
        self.block5 = CapsuleConv2d(64, 3, 3, 16, 1, padding=1, similarity='tonimoto', squash=False)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block1 + block2)
        block4 = self.block4(block1 + block2 + block3)
        block5 = self.block5(block4)

        return torch.sigmoid(block5)


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


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, upscale_factor):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, in_channels, 3, upscale_factor, 1, upscale_factor - 1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x
