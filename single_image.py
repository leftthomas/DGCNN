import argparse
import time

import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from model import Model

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 3, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='upscale_4.pth', type=str, help='super resolution model name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

image = Image.open(IMAGE_NAME)
image = ToTensor()(image).unsqueeze(0)

model = Model(UPSCALE_FACTOR).eval()
if torch.cuda.is_available():
    model, image = model.to('cuda'), image.to('cuda')
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME), map_location='cpu')

print('generating super resolution image...')
start = time.clock()
out = model(image)
elapsed = (time.clock() - start)
print('cost' + str(elapsed) + 's')
out_img = ToPILImage()(out.detach().cpu().squeeze(0))
out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)
