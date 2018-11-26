import math

import torch
from capsule_layer import CapsuleConvTranspose2d
from torch import nn


class Model(nn.Module):
    def __init__(self, upscale_factor=2):
        super(Model, self).__init__()

        upsample_block_num = int(math.log(upscale_factor, 2))
        if upscale_factor % 2 == 0:
            upscale_factor = 2
        self.block1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=9, padding=4), nn.PReLU())
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        block7 = [UpsampleBlock(64, upscale_factor) for _ in range(upsample_block_num)]
        self.block7 = nn.Sequential(*block7)
        self.block8 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=9, padding=4), nn.PReLU())

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block1 + block2)
        block4 = self.block4(block1 + block2 + block3)
        block5 = self.block5(block1 + block2 + block3 + block4)
        block6 = self.block6(block1 + block2 + block3 + block4 + block5)
        block7 = self.block7(block1 + block2 + block3 + block4 + block5 + block6)
        block8 = self.block8(block7)

        return torch.sigmoid(block8)


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
        self.conv = CapsuleConvTranspose2d(in_channels, in_channels, kernel_size=3, in_length=8, out_length=8,
                                           stride=upscale_factor, padding=1, output_padding=upscale_factor - 1,
                                           similarity='tonimoto')

    def forward(self, x):
        x = self.conv(x)
        return x
