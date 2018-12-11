import torch
from torch import nn


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownConv, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.PReLU()

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
        self.conv1 = nn.ConvTranspose2d(in_ch, in_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.PReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(64, out_ch, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.tanh(x)
        return (x + 1) / 2


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.in_c = InConv(3, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)
        self.down3 = DownConv(256, 512)
        self.down4 = DownConv(512, 1024)

        self.up4_t = UpConv(1024, 512)
        self.up3_t = UpConv(1024, 256)
        self.up2_t = UpConv(512, 128)
        self.up1_t = UpConv(256, 64)
        self.out_t = OutConv(128, 3)

        self.up4_r = UpConv(1024, 512)
        self.up3_r = UpConv(1024, 256)
        self.up2_r = UpConv(512, 128)
        self.up1_r = UpConv(256, 64)
        self.out_r = OutConv(128, 3)

    def forward(self, x):
        # encoder
        x_i = self.in_c(x)
        x_ud1, x_d1 = self.down1(x_i)
        x_ud2, x_d2 = self.down2(x_d1)
        x_ud3, x_d3 = self.down3(x_d2)
        x_ud4, x_d4 = self.down4(x_d3)

        # decoder of transmission
        x_t = self.up4_t(x_d4)
        x_t = self.up3_t(torch.cat((x_t, x_ud4), dim=1))
        x_t = self.up2_t(torch.cat((x_t, x_ud3), dim=1))
        x_t = self.up1_t(torch.cat((x_t, x_ud2), dim=1))
        transmission = self.out_t(torch.cat((x_t, x_ud1), dim=1))

        # decoder of reflection
        x_r = self.up4_r(x_d4)
        x_r = self.up3_r(torch.cat((x_r, x_ud4), dim=1))
        x_r = self.up2_r(torch.cat((x_r, x_ud3), dim=1))
        x_r = self.up1_r(torch.cat((x_r, x_ud2), dim=1))
        reflection = self.out_r(torch.cat((x_r, x_ud1), dim=1))
        return transmission, reflection
