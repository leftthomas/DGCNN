from capsule_layer import CapsuleLinear
from torch import nn


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        return (x + 1) / 2


class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()

        self.in_c = InConv(3, 16)
        self.down1 = DownConv(16, 32)
        self.down2 = DownConv(32, 64)
        self.down3 = DownConv(64, 64)
        self.down4 = DownConv(64, 128)
        self.down5 = DownConv(128, 128)

        self.capsule_length = 32
        self.transform_t = CapsuleLinear(out_capsules=(input_size // (2 ** 5)) ** 2 * (128 // self.capsule_length),
                                         in_length=self.capsule_length, out_length=self.capsule_length,
                                         similarity='tonimoto')
        self.transform_r = CapsuleLinear(out_capsules=(input_size // (2 ** 5)) ** 2 * (128 // self.capsule_length),
                                         in_length=self.capsule_length, out_length=self.capsule_length,
                                         similarity='tonimoto')

        self.up5_t = UpConv(128, 128)
        self.up4_t = UpConv(128, 64)
        self.up3_t = UpConv(64, 64)
        self.up2_t = UpConv(64, 32)
        self.up1_t = UpConv(32, 16)
        self.out_t = OutConv(16, 3)

        self.up5_r = UpConv(128, 128)
        self.up4_r = UpConv(128, 64)
        self.up3_r = UpConv(64, 64)
        self.up2_r = UpConv(64, 32)
        self.up1_r = UpConv(32, 16)
        self.out_r = OutConv(16, 3)

    def forward(self, x):
        # encoder
        x_i = self.in_c(x)
        x_d1 = self.down1(x_i)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2)
        x_d4 = self.down4(x_d3)
        x_d5 = self.down5(x_d4)

        # transform
        batch_size, in_channel, in_height, in_width = x_d5.size()
        in_capsules = x_d5.permute(0, 2, 3, 1).contiguous()
        in_capsules = in_capsules.view(batch_size, -1, self.capsule_length)
        # for transmission
        out_capsules_t = self.transform_t(in_capsules)
        out_capsules_t = out_capsules_t.permute(0, 2, 1).contiguous()
        out_capsules_t = out_capsules_t.view(batch_size, self.capsule_length, -1, in_height // 2, in_width // 2)
        out_capsules_t = out_capsules_t.view(batch_size, -1, in_height // 2, in_width // 2)
        # for reflection
        out_capsules_r = self.transform_r(in_capsules)
        out_capsules_r = out_capsules_r.permute(0, 2, 1).contiguous()
        out_capsules_r = out_capsules_r.view(batch_size, self.capsule_length, -1, in_height // 2, in_width // 2)
        out_capsules_r = out_capsules_r.view(batch_size, -1, in_height // 2, in_width // 2)

        # decoder of transmission
        x_u5_t = self.up5_t(out_capsules_t)
        x_u4_t = self.up4_t(x_u5_t)
        x_u3_t = self.up3_t(x_u4_t)
        x_u2_t = self.up2_t(x_u3_t)
        x_u1_t = self.up1_t(x_u2_t)
        transmission = self.out_t(x_u1_t)

        # decoder of reflection
        x_u5_r = self.up5_r(out_capsules_r)
        x_u4_r = self.up4_r(x_u5_r)
        x_u3_r = self.up3_r(x_u4_r)
        x_u2_r = self.up2_r(x_u3_r)
        x_u1_r = self.up1_r(x_u2_r)
        reflection = self.out_r(x_u1_r)
        return transmission, reflection
