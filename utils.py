import argparse
import os
import random
from math import exp
from math import log10
from os.path import join

import torch
import torch.nn.functional as F
import torchvision.utils as utils
from PIL import Image
from torch import nn
from torch.utils.data.dataset import Dataset
from torchnet.meter import meter
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, \
    RandomVerticalFlip, RandomHorizontalFlip, RandomCrop


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def train_transform(crop_size):
    return Compose([RandomCrop(crop_size), RandomHorizontalFlip(), RandomVerticalFlip(), ToTensor()])


def test_transform(crop_size):
    return Compose([Resize(crop_size, interpolation=Image.BICUBIC), CenterCrop(crop_size), ToTensor()])


def gaussian(window_size, sigma, device='cpu'):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]).to(
        device)
    return gauss / gauss.sum()


def create_window(window_size, channel, sigma=1.5, device='cpu'):
    _1D_window = gaussian(window_size, sigma, device).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel, device=img1.device)

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


class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, data_type='train'):
        super(DatasetFromFolder, self).__init__()

        first_image_path = join(dataset_dir, 'first')
        self.first_images = [join(first_image_path, x) for x in sorted(os.listdir(first_image_path)) if
                             is_image_file(x)]
        second_image_path = join(dataset_dir, 'second')
        self.second_images = [join(second_image_path, x) for x in sorted(os.listdir(second_image_path)) if
                              is_image_file(x)]
        if data_type == 'train':
            self.transform = train_transform(crop_size)
        elif data_type == 'test':
            self.transform = test_transform(crop_size)
        else:
            raise NotImplementedError('the data_type must be train or test')
        self.data_type = data_type

    def __getitem__(self, index):
        first_image = self.transform(Image.open(self.first_images[index]).convert('RGB'))
        if self.data_type == 'train':
            second_image = self.transform(Image.open(random.choice(self.second_images)).convert('RGB'))
        else:
            second_image = self.transform(Image.open(self.second_images[index]).convert('RGB'))
        # synthetic image
        mixed_image = first_image + second_image
        if mixed_image.max() > 0:
            mixed_image = mixed_image / mixed_image.max()

        return mixed_image, first_image, second_image

    def __len__(self):
        return len(self.first_images)


class PSNRValueMeter(meter.Meter):
    def __init__(self):
        super(PSNRValueMeter, self).__init__()
        self.reset()

    def add(self, img1, img2):
        # make sure compute the PSNR on YCbCr color space and only on Y channel
        img1 = 0.299 * img1[:, 0, :, :] + 0.587 * img1[:, 1, :, :] + 0.114 * img1[:, 2, :, :]
        img2 = 0.299 * img2[:, 0, :, :] + 0.587 * img2[:, 1, :, :] + 0.114 * img2[:, 2, :, :]
        self.sum += 10 * log10(1 / ((img1 - img2) ** 2).mean().detach().cpu().item())
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
        self.sum += ssim(img1.unsqueeze(1), img2.unsqueeze(1)).detach().cpu().item()
        self.n += 1

    def value(self):
        return self.sum / self.n

    def reset(self):
        self.n = 0
        self.sum = 0.0


class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, first_predicted, second_predicted, first_image, second_image):
        # Image Loss
        first_loss = self.mse_loss(first_predicted, first_image)
        second_loss = self.mse_loss(second_predicted, second_image)

        return first_loss + second_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Mixed Image')
    parser.add_argument('--first_name', type=str, help='first image name')
    parser.add_argument('--second_name', type=str, help='second image name')
    parser.add_argument('--crop_size', default=512, type=int, help='image crop size')
    opt = parser.parse_args()

    FIRST_NAME = opt.first_name
    SECOND_NAME = opt.second_name
    CROP_SIZE = opt.crop_size

    first_image = test_transform(CROP_SIZE)(Image.open(FIRST_NAME).convert('RGB'))
    second_image = test_transform(CROP_SIZE)(Image.open(SECOND_NAME).convert('RGB'))
    if '/' in FIRST_NAME:
        FIRST_NAME = FIRST_NAME.split('/')[-1].split('.')[0]
    if '/' in SECOND_NAME:
        SECOND_NAME = SECOND_NAME.split('/')[-1].split('.')[0]
    saved_image_name = 'test_images/mixed_' + str(FIRST_NAME) + '_' + str(SECOND_NAME) + '.jpg'
    mixed_image = first_image + second_image
    if mixed_image.max() > 0:
        mixed_image = mixed_image / mixed_image.max()
    utils.save_image(mixed_image, saved_image_name, nrow=1)
