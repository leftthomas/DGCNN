import os
import random
from math import exp
from math import log10
from os.path import join

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as vision_f
from PIL import Image
from torch import nn
from torch.utils.data.dataset import Dataset
from torchnet.meter import meter
from torchvision.models.vgg import vgg16
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, \
    RandomVerticalFlip, RandomHorizontalFlip, RandomCrop


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def get_params(img, output_size):
    """Get parameters for ``crop`` for a fixed crop.

    Args:
        img (PIL Image): Image to be cropped.
        output_size (tuple): Expected output size of the crop.

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for fixed crop.
    """
    w, h = img.size
    th, tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w

    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw


class FixedCrop(object):
    """Crop the given PIL Image at a fixed location."""

    def __call__(self, img, i, j, h, w):
        return vision_f.crop(img, i, j, h, w)


def train_synthetic_transform(crop_size):
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
    def __init__(self, dataset_dir, crop_size, data_type='real'):
        super(TrainDatasetFromFolder, self).__init__()
        if data_type not in ['real', 'synthetic']:
            raise NotImplementedError('the data_type must be real or synthetic')

        transmission_path = join(dataset_dir, data_type, 'transmission')
        self.transmission_images = [join(transmission_path, x) for x in sorted(os.listdir(transmission_path)) if
                                    is_image_file(x)]
        if data_type == 'synthetic':
            reflection_path = join(dataset_dir, data_type, 'reflection')
            self.reflection_images = [join(reflection_path, x) for x in sorted(os.listdir(reflection_path)) if
                                      is_image_file(x)]
            self.transform = train_synthetic_transform(crop_size)
        else:
            blended_path = join(dataset_dir, data_type, 'blended')
            self.blended_images = [join(blended_path, x) for x in sorted(os.listdir(blended_path)) if is_image_file(x)]
            self.crop_size = (crop_size, crop_size)

        self.data_type = data_type

    def __getitem__(self, index):
        if self.data_type == 'synthetic':
            transmission_image = self.transform(Image.open(self.transmission_images[index]).convert('RGB'))
            reflection_image = self.transform(Image.open(random.choice(self.reflection_images)).convert('RGB'))
            if torch.cuda.is_available():
                transmission_image, reflection_image = transmission_image.to('cuda'), reflection_image.to('cuda')
            # synthetic blended image
            blended_image = synthetic_image(transmission_image, reflection_image)
        else:
            transmission_image = Image.open(self.transmission_images[index]).convert('RGB')
            blended_image = Image.open(self.blended_images[index]).convert('RGB')
            i, j, th, tw = get_params(transmission_image, output_size=self.crop_size)
            transmission_image = ToTensor()(FixedCrop()(transmission_image, i, j, th, tw))
            blended_image = ToTensor()(FixedCrop()(blended_image, i, j, th, tw))
            if torch.cuda.is_available():
                transmission_image, blended_image = transmission_image.to('cuda'), blended_image.to('cuda')

        # the reflection image have been changed after synthetic, so we compute it by B - T, because B = T + R
        # pay attention, B - T may be product negative value, so we need do clamp operation
        reflection_image = torch.clamp(blended_image - transmission_image, 0, 1)
        return blended_image, transmission_image, reflection_image

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
        self.transform = test_transform(crop_size)

    def __getitem__(self, index):
        blended_image = self.transform(Image.open(self.blended_images[index]).convert('RGB'))
        transmission_image = self.transform(Image.open(self.transmission_images[index]).convert('RGB'))
        if torch.cuda.is_available():
            blended_image, transmission_image = blended_image.to('cuda'), transmission_image.to('cuda')
        # because the test dataset have not contain reflection image, so we just return B - T as R
        reflection_image = torch.clamp(blended_image - transmission_image, 0, 1)
        return blended_image, transmission_image, reflection_image

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


def compute_gradient(img):
    grad_x = img[:, :, 1:, :] - img[:, :, :-1, :]
    grad_y = img[:, :, :, 1:] - img[:, :, :, :-1]
    return grad_x, grad_y


class GradientDiffLoss(nn.Module):
    def __init__(self):
        super(GradientDiffLoss, self).__init__()

    def forward(self, img1, img2):
        grad_x1, grad_y1 = compute_gradient(img1)
        grad_x2, grad_y2 = compute_gradient(img2)
        grad_x_loss = (1 + F.cosine_similarity(grad_x1.view(grad_x1.size(0), -1), grad_x2.view(grad_x2.size(0), -1),
                                               dim=1)) / 2
        grad_y_loss = (1 + F.cosine_similarity(grad_y1.view(grad_y1.size(0), -1), grad_y2.view(grad_y2.size(0), -1),
                                               dim=1)) / 2
        return grad_x_loss.mean() + grad_y_loss.mean()


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
        self.gradient_loss = GradientDiffLoss()

    def forward(self, transmission_predicted, reflection_predicted, transmission, reflection):
        # Image Loss
        transmission_image_loss = self.l1_loss(transmission_predicted, transmission)
        reflection_image_loss = self.l1_loss(reflection_predicted, reflection)
        # Perception Loss
        transmission_perception_loss = self.mse_loss(self.loss_network(transmission_predicted),
                                                     self.loss_network(transmission))
        # Gradient Loss
        # gradient_loss = self.gradient_loss(transmission_predicted, reflection_predicted)

        return transmission_image_loss + reflection_image_loss + transmission_perception_loss
