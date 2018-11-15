import torch
from capsule_layer import CapsuleLinear
from torch import nn


class Model(nn.Module):
    def __init__(self, upscale_factor=2, num_iterations=3, **kwargs):
        super(Model, self).__init__()

        self.enc1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)

        self.transform = CapsuleLinear(out_capsules=None, in_length=32, out_length=8, num_iterations=num_iterations,
                                       **kwargs)

        self.dec3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.enc1_nl(e1))
        e3 = self.enc3(self.enc2_nl(e2))

        d3 = self.dec3(d4_c)
        d3_c = self.dec3_nl(torch.cat((d3, e3), dim=1))
        d2 = self.dec2(d3_c)
        d2_c = self.dec2_nl(torch.cat((d2, e2), dim=1))
        d1 = self.dec1(d2_c)
        d1_c = self.dec1_nl(torch.cat((d1, e1), dim=1))
        return d1_c
