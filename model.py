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
        self.block2 = CapsuleConv2d(64, 64, 3, 4, 8, padding=1, similarity='tonimoto', squash=False)
        self.block3 = CapsuleConv2d(64, 64, 3, 8, 16, padding=1, similarity='tonimoto', squash=False)
        self.block4 = CapsuleConv2d(64, 64, 3, 16, 16, padding=1, similarity='tonimoto', squash=False)
        self.block5 = CapsuleConv2d(64, 64, 3, 16, 8, padding=1, similarity='tonimoto', squash=False)
        self.block6 = CapsuleConv2d(64, 64, 3, 8, 4, padding=1, similarity='tonimoto', squash=False)
        self.block7 = nn.Sequential(*[UpsampleBlock(64, upscale_factor, 4, 4) for _ in range(upsample_block_num)])
        self.block8 = CapsuleConv2d(64, 3, 9, 4, 1, padding=4, similarity='tonimoto')

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block1 + block6)
        block8 = self.block8(block7)
        return block8


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
