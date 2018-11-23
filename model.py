import math

from capsule_layer import CapsuleConv2d
from torch import nn


class Model(nn.Module):
    def __init__(self, upscale_factor=2):
        super(Model, self).__init__()

        upsample_block_num = int(math.log(upscale_factor, 2))
        if upscale_factor % 2 == 0:
            upscale_factor = 2
        self.block1 = CapsuleConv2d(3, 64, 9, 1, 4, padding=4, similarity='tonimoto', squash=False)
        self.block2 = ResidualBlock(64, 4, 8)
        self.block3 = ResidualBlock(64, 8, 8)
        self.block4 = ResidualBlock(64, 8, 16)
        self.block5 = ResidualBlock(64, 16, 16)
        self.block6 = ResidualBlock(64, 16, 8)
        self.block7 = ResidualBlock(64, 8, 8)
        self.block8 = CapsuleConv2d(64, 64, 3, 8, 4, padding=1, similarity='tonimoto', squash=False)
        self.block9 = nn.Sequential(*[UpsampleBlock(64, upscale_factor, 4, 4) for _ in range(upsample_block_num)])
        self.block10 = CapsuleConv2d(64, 3, 9, 4, 1, padding=4, similarity='tonimoto')

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block7)
        block9 = self.block9(block1 + block8)
        block10 = self.block10(block9)
        return block10


class ResidualBlock(nn.Module):
    def __init__(self, channels, in_length, out_length):
        super(ResidualBlock, self).__init__()
        self.conv1 = CapsuleConv2d(channels, channels, 3, in_length, out_length, padding=1, similarity='tonimoto',
                                   squash=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = CapsuleConv2d(channels, channels, 3, out_length, out_length, padding=1, similarity='tonimoto',
                                   squash=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, upscale_factor, in_length, out_length):
        super(UpsampleBlock, self).__init__()
        self.conv = CapsuleConv2d(in_channels, in_channels * upscale_factor ** 2, 3, in_length, out_length, padding=1,
                                  similarity='tonimoto', squash=False)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x
