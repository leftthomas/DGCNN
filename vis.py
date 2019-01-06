import argparse
import time

import torch
import torchvision.utils as utils
from PIL import Image
from torchvision.transforms import ToTensor

from model import Model
from utils import test_transform

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--mixed_name', type=str, help='test mixed image name')
parser.add_argument('--crop_size', default=None, type=int, help='image crop size')
parser.add_argument('--model_name', default='model.pth', type=str, help='mixed image separation model name')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
opt = parser.parse_args()

MIXED_NAME = opt.mixed_name
CROP_SIZE = opt.crop_size
MODEL_NAME = opt.model_name
USE_CUDA = True if opt.test_mode == 'GPU' else False

mixed_image = Image.open(MIXED_NAME).convert('RGB')
if CROP_SIZE is None:
    mixed_image = ToTensor()(mixed_image).unsqueeze(0)
else:
    mixed_image = test_transform(CROP_SIZE)(mixed_image).unsqueeze(0)

model = Model()
if USE_CUDA:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
    model, mixed_image = model.to('cuda'), mixed_image.to('cuda')
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location='cpu'))

print('separating images...')
if '/' in MIXED_NAME:
    saved_image_name = MIXED_NAME.split('/')[-1]
else:
    saved_image_name = MIXED_NAME
with torch.no_grad():
    start = time.process_time()
    first_predicted, second_predicted = model(mixed_image)
    elapsed = (time.process_time() - start)
    print('cost %.4f ' % elapsed + 's')
    image = torch.stack([mixed_image.detach().cpu().squeeze(0), first_predicted.detach().cpu().squeeze(0),
                         second_predicted.detach().cpu().squeeze(0)])
    saved_image_name = 'results/out_' + saved_image_name
    utils.save_image(image, saved_image_name, nrow=3, padding=5, pad_value=255)
