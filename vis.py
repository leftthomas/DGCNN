import argparse
import glob
import os
import time
from math import log10

import torch
import torchvision.utils as utils
from PIL import Image
from torchvision.transforms import ToTensor

from model import Model
from utils import image_transform, ssim

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--blended_name', type=str, help='test blended image name')
parser.add_argument('--transmission_name', default='', type=str, help='test transmission image name')
parser.add_argument('--crop_size', default=None, type=int, help='image crop size')
parser.add_argument('--model_name', default='model.pth', type=str, help='reflection removal model name')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
opt = parser.parse_args()

BLENDED_NAME = opt.blended_name
TRANSMISSION_NAME = opt.transmission_name
CROP_SIZE = opt.crop_size
MODEL_NAME = opt.model_name
USE_CUDA = True if opt.test_mode == 'GPU' else False

blended_image = Image.open(BLENDED_NAME).convert('RGB')
if CROP_SIZE is None:
    blended_image = ToTensor()(blended_image).unsqueeze(0)
else:
    blended_image = image_transform(CROP_SIZE)(blended_image).unsqueeze(0)
if TRANSMISSION_NAME is not '':
    transmission_image = Image.open(TRANSMISSION_NAME).convert('RGB')
    if CROP_SIZE is None:
        transmission_image = ToTensor()(transmission_image).unsqueeze(0)
    else:
        transmission_image = image_transform(CROP_SIZE)(transmission_image).unsqueeze(0)
else:
    transmission_image = None

model = Model()
if USE_CUDA:
    model, blended_image = model.to('cuda'), blended_image.to('cuda')
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
    if transmission_image is not None:
        transmission_image = transmission_image.to('cuda')
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location='cpu'))

print('generating reflection removed image...')
if '/' in BLENDED_NAME:
    saved_image_name = BLENDED_NAME.split('/')[-1]
else:
    saved_image_name = BLENDED_NAME
with torch.no_grad():
    start = time.clock()
    out = model(blended_image)
    elapsed = (time.clock() - start)
    print('cost %.4f ' % elapsed + 's')
    if transmission_image is not None:
        # only compute the PSNR and SSIM on YCbCr color space and only on Y channel
        transmission_image_l = 0.299 * transmission_image[:, 0, :, :] + 0.587 * transmission_image[:, 1, :, :] \
                               + 0.114 * transmission_image[:, 2, :, :]
        out_image_l = 0.299 * out[:, 0, :, :] + 0.587 * out[:, 1, :, :] + 0.114 * out[:, 2, :, :]
        mse = ((transmission_image_l - out_image_l) ** 2).mean().detach().cpu().item()
        psnr_value = 10 * log10(1 / mse)
        ssim_value = ssim(transmission_image_l.unsqueeze(1), out_image_l.unsqueeze(1)).detach().cpu().item()

        image = torch.stack([blended_image.detach().cpu().squeeze(0), out.detach().cpu().squeeze(0),
                             transmission_image.detach().cpu().squeeze(0)])
        saved_image_name = 'results/out_' + saved_image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' \
                           % (psnr_value, ssim_value) + saved_image_name.split('.')[-1]
    else:
        image = torch.stack([blended_image.detach().cpu().squeeze(0), out.detach().cpu().squeeze(0)])
        saved_image_name = 'results/out_' + saved_image_name

    # make sure it only save once
    existed_files = glob.glob('results/out_' + saved_image_name.split('/')[-1].split('_')[1] + '*')
    if len(existed_files) != 0:
        for file in existed_files:
            if ('psnr' in file and 'psnr' in saved_image_name) or \
                    ('psnr' not in file and 'psnr' not in saved_image_name):
                os.remove(file)
    if transmission_image is not None:
        utils.save_image(image, saved_image_name, nrow=3, padding=5, pad_value=255)
    else:
        utils.save_image(image, saved_image_name, nrow=2, padding=5, pad_value=255)
