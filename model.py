import torch
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


class BasicModel(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(BasicModel, self).__init__()

        self.inc = InConv(in_ch, 32)
        self.down1 = DownConv(32, 64)
        self.down2 = DownConv(64, 128)
        self.down3 = DownConv(128, 256)
        self.down4 = DownConv(256, 512)
        self.up4 = UpConv(512, 256)
        self.up3 = UpConv(256, 128)
        self.up2 = UpConv(128, 64)
        self.up1 = UpConv(64, 32)
        self.outc = OutConv(32, out_ch)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.outc(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.g_0 = BasicModel(3, 3)
        self.h = BasicModel(6, 3)
        self.g_1 = BasicModel(6, 3)

    def forward(self, x):
        t_0 = self.g_0(x)
        r_p = self.h(torch.cat((t_0, x), dim=1))
        b_1 = self.g_1(torch.cat((r_p, x), dim=1))
        return t_0, r_p, b_1
