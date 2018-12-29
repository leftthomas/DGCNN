import torch
from capsule_layer import CapsuleConv2d, CapsuleConvTranspose2d
from torch import nn


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = CapsuleConv2d(in_ch, out_ch, 3, in_ch, out_ch, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownConv, self).__init__()
        self.conv1 = CapsuleConv2d(in_ch, in_ch, 3, in_ch, out_ch, padding=1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = CapsuleConv2d(in_ch, out_ch, 3, in_ch, out_ch, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu1(x)
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)
        return x1, x2


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.conv1 = CapsuleConvTranspose2d(in_ch, in_ch, 3, in_ch, out_ch, padding=1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = CapsuleConvTranspose2d(in_ch, out_ch, 3, in_ch, out_ch, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, output_size=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x, output_size)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = CapsuleConvTranspose2d(in_ch, out_ch, 3, in_ch, out_ch, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.tanh(x)
        return (x + 1) / 2


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.in_c = InConv(3, 32)
        self.down1 = DownConv(32, 64)
        self.down2 = DownConv(64, 128)
        self.down3 = DownConv(128, 256)
        self.down4 = DownConv(256, 512)

        self.up4 = UpConv(512, 256)
        self.up3 = UpConv(512, 128)
        self.up2 = UpConv(256, 64)
        self.up1 = UpConv(128, 32)
        self.out_t = OutConv(64, 3)

    def forward(self, x):
        # encoder
        x_i = self.in_c(x)
        x_ud1, x_d1 = self.down1(x_i)
        x_ud2, x_d2 = self.down2(x_d1)
        x_ud3, x_d3 = self.down3(x_d2)
        x_ud4, x_d4 = self.down4(x_d3)

        # decoder
        x_t = self.up4(x_d4, output_size=x_ud4.size())
        x_t = self.up3(torch.cat((x_t, x_ud4), dim=1), output_size=x_ud3.size())
        x_t = self.up2(torch.cat((x_t, x_ud3), dim=1), output_size=x_ud2.size())
        x_t = self.up1(torch.cat((x_t, x_ud2), dim=1), output_size=x_ud1.size())
        transmission = self.out_t(torch.cat((x_t, x_ud1), dim=1))
        return transmission
