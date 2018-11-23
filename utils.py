import os
from math import exp
from math import log10
from os.path import join

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchnet.meter import meter
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, Resize, \
    RandomHorizontalFlip, RandomVerticalFlip


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


# make sure the crop size can be divided by upscale_factor with no remainder
def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def hr_transform(crop_size):
    return Compose(
        [RandomCrop(crop_size, pad_if_needed=True), RandomHorizontalFlip(), RandomVerticalFlip(), ToTensor()])


def lr_transform(crop_size, upscale_factor):
    return Compose([ToPILImage(), Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC), ToTensor()])


class TrainValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainValDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = hr_transform(crop_size)
        self.lr_transform = lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in os.listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in os.listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        lr_image, hr_restore_img, hr_image = ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

        # make sure the gray image to be 3 channel
        if lr_image.size(0) == 1:
            lr_image = torch.cat((lr_image, lr_image, lr_image), dim=0)
        if hr_restore_img.size(0) == 1:
            hr_restore_img = torch.cat((hr_restore_img, hr_restore_img, hr_restore_img), dim=0)
        if hr_image.size(0) == 1:
            hr_image = torch.cat((hr_image, hr_image, hr_image), dim=0)

        return image_name, lr_image, hr_restore_img, hr_image

    def __len__(self):
        return len(self.lr_filenames)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class PSNRValueMeter(meter.Meter):
    def __init__(self):
        super(PSNRValueMeter, self).__init__()
        self.reset()

    def add(self, sr, hr):
        self.sum += 10 * log10(1 / ((sr - hr) ** 2).mean().item())
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
        self.sum += ssim(sr, hr).item()
        self.n += 1

    def value(self):
        return self.sum / self.n

    def reset(self):
        self.n = 0
        self.sum = 0.0

