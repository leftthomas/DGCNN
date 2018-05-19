import argparse

import pandas as pd
import torch
import torch.nn as nn
import torchnet as tnt
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger
from tqdm import tqdm

from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder
from model import Model
from utils import PSNRValueMeter, SSIMValueMeter


def processor(sample):
    data, labels, training = sample

    if torch.cuda.is_available():
        data = data.cuda()
        labels = labels.cuda()
    data = Variable(data)
    labels = Variable(labels)

    model.train(training)

    classes = model(data)
    loss = loss_criterion(classes, labels)
    return loss, classes


def on_sample(state):
    state['sample'].append(state['train'])


def reset_meters():
    meter_loss.reset()
    meter_psnr.reset()
    meter_ssim.reset()


def on_forward(state):
    meter_loss.add(state['loss'].data[0])
    meter_psnr.add(state['output'].data, state['sample'][1])
    meter_ssim.add(state['output'].data, state['sample'][1])


def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    print('[Epoch %d] Training Loss: %.4f Training PSNR: %.4f dB Training SSIM: %.4f' % (
        state['epoch'], meter_loss.value()[0], meter_psnr.value()[0], meter_ssim.value()[0]))

    train_loss_logger.log(state['epoch'], meter_loss.value()[0])
    train_psnr_logger.log(state['epoch'], meter_psnr.value()[0])
    train_ssim_logger.log(state['epoch'], meter_ssim.value()[0])
    results['train_loss'].append(meter_loss.value()[0])
    results['train_psnr'].append(meter_psnr.value()[0])
    results['train_ssim'].append(meter_ssim.value()[0])

    reset_meters()

    engine.test(processor, val_loader)

    val_loss_logger.log(state['epoch'], meter_loss.value()[0])
    val_psnr_logger.log(state['epoch'], meter_psnr.value()[0])
    val_ssim_logger.log(state['epoch'], meter_ssim.value()[0])
    results['val_loss'].append(meter_loss.value()[0])
    results['val_psnr'].append(meter_psnr.value()[0])
    results['val_ssim'].append(meter_ssim.value()[0])

    print('[Epoch %d] Valing Loss: %.4f Valing PSNR: %.4f dB Valing SSIM: %.4f' % (
        state['epoch'], meter_loss.value()[0], meter_psnr.value()[0], meter_ssim.value()[0]))

    # save model
    torch.save(model.state_dict(), 'epochs/upscale_%d_epoch_%d.pth' % (UPSCALE_FACTOR, state['epoch']))
    # save statistics at every 10 epochs
    if state['epoch'] % 10 == 0:
        data_frame = pd.DataFrame(data={'train_loss': results['train_loss'], 'train_psnr': results['train_psnr'],
                                        'train_ssim': results['train_ssim'], 'val_loss': results['val_loss'],
                                        'val_psnr': results['val_psnr'], 'val_ssim': results['val_ssim']},
                                  index=range(1, state['epoch'] + 1))
        data_frame.to_csv('statistics/upscale_' + UPSCALE_FACTOR + '_results.csv', index_label='epoch')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
    parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                        help='super resolution upscale factor')
    parser.add_argument('--batch_size', default=50, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')

    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs

    train_set = TrainDatasetFromFolder('data/VOC2012/train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('data/VOC2012/val', upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)

    model = Model(UPSCALE_FACTOR)
    loss_criterion = nn.MSELoss()
    if torch.cuda.is_available():
        model.cuda()
        loss_criterion.cuda()
    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters())

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_psnr = PSNRValueMeter()
    meter_ssim = SSIMValueMeter()
    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    val_loss_logger = VisdomPlotLogger('line', opts={'title': 'Val Loss'})
    train_psnr_logger = VisdomPlotLogger('line', opts={'title': 'Train PSNR'})
    val_psnr_logger = VisdomPlotLogger('line', opts={'title': 'Val PSNR'})
    train_ssim_logger = VisdomPlotLogger('line', opts={'title': 'Train SSIM'})
    val_ssim_logger = VisdomPlotLogger('line', opts={'title': 'Val SSIM'})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    results = {'train_loss': [], 'train_psnr': [], 'train_ssim': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': []}

    engine.train(processor, train_loader, maxepoch=NUM_EPOCHS, optimizer=optimizer)

