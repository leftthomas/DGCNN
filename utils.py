import os
import random
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
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def image_transform(crop_size):
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
    window = create_window(window_size, channel, device=img1.device)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def synthetic_image(transmission_image, reflection_image):
    transmission_image, reflection_image = transmission_image.unsqueeze(0), reflection_image.unsqueeze(0)
    (_, channel, _, _) = reflection_image.size()
    window = create_window(11, channel, sigma=random.uniform(2, 5) / 11, device=transmission_image.device)
    reflection_image = F.conv2d(reflection_image, window, padding=11 // 2, groups=channel)
    blended_image = transmission_image + reflection_image
    if blended_image.max() > 1:
        label_gt1 = torch.gt(blended_image, 1)
        reflection_image = reflection_image - torch.mean((blended_image - 1)[label_gt1]) * 1.3
        reflection_image = torch.clamp(reflection_image, 0, 1)
        blended_image = transmission_image + reflection_image
        blended_image = torch.clamp(blended_image, 0, 1)
    return blended_image.squeeze(0)


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size):
        super(TrainDatasetFromFolder, self).__init__()
        transmission_path = join(dataset_dir, 'transmission')
        self.transmission_images = [join(transmission_path, x) for x in sorted(os.listdir(transmission_path)) if
                                    is_image_file(x)]
        reflection_path = join(dataset_dir, 'reflection')
        self.reflection_images = [join(reflection_path, x) for x in sorted(os.listdir(reflection_path)) if
                                  is_image_file(x)]

        self.transform = image_transform(crop_size)

    def __getitem__(self, index):
        transmission_image = self.transform(Image.open(self.transmission_images[index]).convert('RGB'))
        reflection_image = self.transform(Image.open(self.reflection_images[index]).convert('RGB'))
        if torch.cuda.is_available():
            transmission_image, reflection_image = transmission_image.to('cuda'), reflection_image.to('cuda')
        # synthetic blended image
        blended_image = synthetic_image(transmission_image, reflection_image)
        # the reflection image have been changed after synthetic, so we compute it by B - T, because B = T + R
        return blended_image, transmission_image, blended_image - transmission_image

    def __len__(self):
        return len(self.transmission_images)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, data_type='real'):
        super(TestDatasetFromFolder, self).__init__()
        if data_type not in ['real', 'synthetic']:
            raise NotImplementedError('the data_type must be real or synthetic')

        blended_path = join(dataset_dir, data_type, 'blended')
        transmission_path = join(dataset_dir, data_type, 'transmission')
        self.blended_images = [join(blended_path, x) for x in sorted(os.listdir(blended_path)) if is_image_file(x)]
        self.transmission_images = [join(transmission_path, x) for x in sorted(os.listdir(transmission_path)) if
                                    is_image_file(x)]
        self.transform = image_transform(crop_size)

    def __getitem__(self, index):
        blended_image = self.transform(Image.open(self.blended_images[index]).convert('RGB'))
        transmission_image = self.transform(Image.open(self.transmission_images[index]).convert('RGB'))
        if torch.cuda.is_available():
            blended_image, transmission_image = blended_image.to('cuda'), transmission_image.to('cuda')
        # because the test dataset have not contain reflection image, so we just return B - T as R
        return blended_image, transmission_image, blended_image - transmission_image

    def __len__(self):
        return len(self.transmission_images)


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


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        window = create_window(self.window_size, channel, device=img1.device)
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


def compute_gradient(img):
    grad_x = img[:, :, 1:, :] - img[:, :, :-1, :]
    grad_y = img[:, :, :, 1:] - img[:, :, :, :-1]
    return grad_x, grad_y


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, img1, img2):
        grad_x1, grad_y1 = compute_gradient(img1)
        grad_x2, grad_y2 = compute_gradient(img2)
        grad_x_loss = F.l1_loss(grad_x1, grad_x2)
        grad_y_loss = F.l1_loss(grad_y1, grad_y2)
        return grad_x_loss + grad_y_loss


class ExclusionLoss(nn.Module):
    def __init__(self, level=3):
        super(ExclusionLoss, self).__init__()
        self.level = level

    def forward(self, img1, img2):
        grad_loss = []
        for l in range(self.level):
            grad_x1, grad_y1 = compute_gradient(img1)
            grad_x2, grad_y2 = compute_gradient(img2)
            grad_x1_norm = torch.sum(torch.abs(grad_x1) ** 2, dim=[2, 3], keepdim=True) ** 0.5
            grad_y1_norm = torch.sum(torch.abs(grad_y1) ** 2, dim=[2, 3], keepdim=True) ** 0.5
            grad_x2_norm = torch.sum(torch.abs(grad_x2) ** 2, dim=[2, 3], keepdim=True) ** 0.5
            grad_y2_norm = torch.sum(torch.abs(grad_y2) ** 2, dim=[2, 3], keepdim=True) ** 0.5
            lamda_x1 = (grad_x2_norm / grad_x1_norm) ** 0.5
            lamda_y1 = (grad_y2_norm / grad_y1_norm) ** 0.5
            lamda_x2 = (grad_x1_norm / grad_x2_norm) ** 0.5
            lamda_y2 = (grad_y1_norm / grad_y2_norm) ** 0.5

            grad_x1_s = torch.tanh(lamda_x1 * torch.abs(grad_x1))
            grad_y1_s = torch.tanh(lamda_y1 * torch.abs(grad_y1))
            grad_x2_s = torch.tanh(lamda_x2 * torch.abs(grad_x2))
            grad_y2_s = torch.tanh(lamda_y2 * torch.abs(grad_y2))

            grad_x_loss = torch.sum(torch.abs(grad_x1_s * grad_x2_s) ** 2, dim=[2, 3]) ** 0.5
            grad_y_loss = torch.sum(torch.abs(grad_y1_s * grad_y2_s) ** 2, dim=[2, 3]) ** 0.5

            grad_loss.append(torch.mean(grad_x_loss) + torch.mean(grad_y_loss))
            img1 = F.interpolate(img1, scale_factor=2, mode='bilinear', align_corners=True)
            img2 = F.interpolate(img2, scale_factor=2, mode='bilinear', align_corners=True)

        return torch.stack(grad_loss).mean()


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
        self.gradient_loss = GradientLoss()
        self.exclusion_loss = ExclusionLoss()

    def forward(self, transmission_predicted, reflection_predicted, transmission, reflection):
        # Image Loss
        transmission_image_loss = self.l1_loss(transmission_predicted, transmission)
        reflection_image_loss = self.l1_loss(reflection_predicted, reflection)
        # # Perception Loss
        # transmission_perception_loss = self.mse_loss(self.loss_network(transmission_predicted),
        #                                              self.loss_network(transmission))
        # # SSIM Loss
        # transmission_ssim_loss = self.ssim_loss(transmission_predicted, transmission)
        # # TV Loss
        # transmission_tv_loss = self.tv_loss(transmission_predicted)
        # # Gradient Loss
        # transmission_gradient_loss = self.gradient_loss(transmission_predicted, transmission)
        # # Exclusion Loss
        # exclusion_loss = self.exclusion_loss(transmission_predicted, reflection_predicted)

        return transmission_image_loss + reflection_image_loss
