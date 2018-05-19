from math import log10

from pytorch_ssim import ssim
from torchnet.meter import meter


class PSNRValueMeter(meter.Meter):
    def __init__(self):
        super(PSNRValueMeter, self).__init__()
        self.reset()

    def add(self, sr, hr):
        self.sum += 10 * log10(1 / ((sr - hr) ** 2).mean())
        self.n += 1

    def value(self):
        return self.sum / self.n

    def reset(self):
        self.n = 0
        self.sum = 0.0


class SSIMValueMeter(meter.Meter):
    def __init__(self):
        super(SSIMValueMeter, self).__init__()
        self.reset()

    def add(self, sr, hr):
        self.sum += ssim(sr, hr)[0]
        self.n += 1

    def value(self):
        return self.sum / self.n

    def reset(self):
        self.n = 0
        self.sum = 0.0

