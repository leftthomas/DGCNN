import os
from math import exp
from math import log10
from os.path import join

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data.dataset import Dataset
from torchnet.meter import meter
from torchvision.models.vgg import vgg16
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, Resize, \
    RandomHorizontalFlip, RandomVerticalFlip


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def hr_transform(crop_size):
    return Compose([RandomCrop(crop_size), RandomHorizontalFlip(), RandomVerticalFlip(), ToTensor()])


def lr_transform(crop_size):
    return Compose([ToPILImage(), Resize(crop_size, interpolation=Image.BICUBIC), ToTensor()])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]
        self.hr_transform = hr_transform(crop_size)
        self.lr_transform = lr_transform(crop_size)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, data_type='real'):
        super(TestDatasetFromFolder, self).__init__()
        blended_path = join(dataset_dir, data_type, 'blended')
        transmission_path = join(dataset_dir, data_type, 'transmission_layer')
        self.blended_images = [join(blended_path, x) for x in sorted(os.listdir(blended_path)) if is_image_file(x)]
        self.transmission_images = [join(transmission_path, x) for x in sorted(os.listdir(transmission_path)) if
                                    is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.blended_images[index].split('/')[-1]
        blended_image = ToTensor()(Image.open(self.blended_images[index]).convert('RGB'))
        transmission_image = ToTensor()(Image.open(self.transmission_images[index]).convert('RGB'))

        return image_name, blended_image, transmission_image

    def __len__(self):
        return len(self.blended_images)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
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


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class PSNRValueMeter(meter.Meter):
    def __init__(self):
        super(PSNRValueMeter, self).__init__()
        self.reset()

    def add(self, img1, img2):
        # make sure compute the PSNR on YCbCr color space and only on Y channel
        img1 = 0.299 * img1[:, 0, :, :] + 0.587 * img1[:, 1, :, :] + 0.114 * img1[:, 2, :, :]
        img2 = 0.299 * img2[:, 0, :, :] + 0.587 * img2[:, 1, :, :] + 0.114 * img2[:, 2, :, :]
        self.sum += 10 * log10(1 / ((img1 - img2) ** 2).mean().item())
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

    def add(self, img1, img2):
        # make sure compute the SSIM on YCbCr color space and only on Y channel
        img1 = 0.299 * img1[:, 0, :, :] + 0.587 * img1[:, 1, :, :] + 0.114 * img1[:, 2, :, :]
        img2 = 0.299 * img2[:, 0, :, :] + 0.587 * img2[:, 1, :, :] + 0.114 * img2[:, 2, :, :]
        self.sum += ssim(img1.unsqueeze(1), img2.unsqueeze(1)).item()
        self.n += 1

    def value(self):
        return self.sum / self.n

    def reset(self):
        self.n = 0
        self.sum = 0.0


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size, _, h_x, w_x = x.size()
        count_h, count_w = self.tensor_size(x[:, :, 1:, :]), self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.tv_loss = TVLoss()

    def forward(self, out_images, target_images):
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.l1_loss(out_images, target_images)
        # SSIM Loss
        ssim_loss = self.ssim_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + ssim_loss + perception_loss + tv_loss
