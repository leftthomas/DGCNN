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
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
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


class Transform(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Transform, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.ConvTranspose2d(out_ch, out_ch, 3, stride=2, padding=1, output_padding=1)
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
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
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

        self.in_c = InConv(3, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)
        self.down3 = DownConv(256, 512)
        self.down4 = DownConv(512, 512)

        self.transform_t = Transform(512, 512)
        self.transform_r = Transform(512, 512)

        self.up4_t = UpConv(512, 512)
        self.up3_t = UpConv(512, 256)
        self.up2_t = UpConv(256, 128)
        self.up1_t = UpConv(128, 64)
        self.out_t = OutConv(64, 3)

        self.up4_r = UpConv(512, 512)
        self.up3_r = UpConv(512, 256)
        self.up2_r = UpConv(256, 128)
        self.up1_r = UpConv(128, 64)
        self.out_r = OutConv(64, 3)

    def forward(self, x):
        # encoder
        x_i = self.in_c(x)
        x_d1 = self.down1(x_i)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2)
        x_d4 = self.down4(x_d3)

        # transform
        out_t = self.transform_t(x_d4)
        out_r = self.transform_r(x_d4)

        # decoder of transmission
        x_u4_t = self.up4_t(out_t)
        x_u3_t = self.up3_t(x_u4_t)
        x_u2_t = self.up2_t(x_u3_t)
        x_u1_t = self.up1_t(x_u2_t)
        transmission = self.out_t(x_u1_t)

        # decoder of reflection
        x_u4_r = self.up4_r(out_r)
        x_u3_r = self.up3_r(x_u4_r)
        x_u2_r = self.up2_r(x_u3_r)
        x_u1_r = self.up1_r(x_u2_r)
        reflection = self.out_r(x_u1_r)
        return transmission, reflection
