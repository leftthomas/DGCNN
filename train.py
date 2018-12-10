import argparse

import pandas as pd
import torch
import torchnet as tnt
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger
from tqdm import tqdm

from model import Model
from utils import PSNRValueMeter, SSIMValueMeter, TrainDatasetFromFolder, TotalLoss, TestDatasetFromFolder


def processor(sample):
    blended, transmission, reflection, training = sample

    if torch.cuda.is_available():
        blended, transmission = blended.to('cuda'), transmission.to('cuda')
        if type(reflection) is not int:
            reflection = reflection.to('cuda')

    model.train(training)

    transmission_predicted, reflection_predicted = model(blended)
    loss = loss_criterion(transmission_predicted, reflection_predicted, transmission, reflection)
    return loss, transmission_predicted


def on_sample(state):
    state['sample'].append(state['train'])


def reset_meters():
    meter_loss.reset()
    meter_psnr.reset()
    meter_ssim.reset()


def on_forward(state):
    meter_loss.add(state['loss'].detach().cpu().item())
    meter_psnr.add(state['output'], state['sample'][1])
    meter_ssim.add(state['output'], state['sample'][1])


def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    print('[Epoch %d] Training Loss: %.4f Training PSNR: %.4f dB Training SSIM: %.4f' % (
        state['epoch'], meter_loss.value()[0], meter_psnr.value(), meter_ssim.value()))

    train_loss_logger.log(state['epoch'], meter_loss.value()[0])
    train_psnr_logger.log(state['epoch'], meter_psnr.value())
    train_ssim_logger.log(state['epoch'], meter_ssim.value())
    results['train_loss'].append(meter_loss.value()[0])
    results['train_psnr'].append(meter_psnr.value())
    results['train_ssim'].append(meter_ssim.value())

    # save best model
    global best_psnr, best_ssim
    if meter_psnr.value() > best_psnr and meter_ssim.value() > best_ssim:
        torch.save(model.state_dict(), 'epochs/model.pth')
        best_psnr, best_ssim = meter_psnr.value(), meter_ssim.value()

    reset_meters()

    engine.test(processor, test_real_loader)

    test_real_loss_logger.log(state['epoch'], meter_loss.value()[0])
    test_real_psnr_logger.log(state['epoch'], meter_psnr.value())
    test_real_ssim_logger.log(state['epoch'], meter_ssim.value())
    results['test_real_loss'].append(meter_loss.value()[0])
    results['test_real_psnr'].append(meter_psnr.value())
    results['test_real_ssim'].append(meter_ssim.value())

    print('[Epoch %d] Testing Real Loss: %.4f Testing Real PSNR: %.4f dB Testing Real SSIM: %.4f' % (
        state['epoch'], meter_loss.value()[0], meter_psnr.value(), meter_ssim.value()))

    reset_meters()

    engine.test(processor, test_synthetic_loader)

    test_synthetic_loss_logger.log(state['epoch'], meter_loss.value()[0])
    test_synthetic_psnr_logger.log(state['epoch'], meter_psnr.value())
    test_synthetic_ssim_logger.log(state['epoch'], meter_ssim.value())
    results['test_synthetic_loss'].append(meter_loss.value()[0])
    results['test_synthetic_psnr'].append(meter_psnr.value())
    results['test_synthetic_ssim'].append(meter_ssim.value())

    print('[Epoch %d] Testing Synthetic Loss: %.4f Testing Synthetic PSNR: %.4f dB Testing Synthetic SSIM: %.4f' % (
        state['epoch'], meter_loss.value()[0], meter_psnr.value(), meter_ssim.value()))

    # save statistics at every 10 epochs
    if state['epoch'] % 10 == 0:
        data_frame = pd.DataFrame(data={'train_loss': results['train_loss'], 'train_psnr': results['train_psnr'],
                                        'train_ssim': results['train_ssim'],
                                        'test_real_loss': results['test_real_loss'],
                                        'test_real_psnr': results['test_real_psnr'],
                                        'test_real_ssim': results['test_real_ssim'],
                                        'test_synthetic_loss': results['test_synthetic_loss'],
                                        'test_synthetic_psnr': results['test_synthetic_psnr'],
                                        'test_synthetic_ssim': results['test_synthetic_ssim']},
                                  index=range(1, state['epoch'] + 1))
        data_frame.to_csv('statistics/results.csv', index_label='epoch')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Reflection Removal Model')
    parser.add_argument('--crop_size', default=224, type=int, help='image crop size')
    parser.add_argument('--batch_size', default=4, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
    parser.add_argument('--train_path', default='data/train', type=str, help='train image data path')
    parser.add_argument('--test_path', default='data/test', type=str, help='test image data path')

    opt = parser.parse_args()
    CROP_SIZE = opt.crop_size
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs
    TRAIN_PATH = opt.train_path
    TEST_PATH = opt.test_path

    results = {'train_loss': [], 'train_psnr': [], 'train_ssim': [], 'test_real_loss': [], 'test_real_psnr': [],
               'test_real_ssim': [], 'test_synthetic_loss': [], 'test_synthetic_psnr': [], 'test_synthetic_ssim': []}
    # record current best measures
    best_psnr, best_ssim = 0, 0

    train_set = TrainDatasetFromFolder(TRAIN_PATH, crop_size=CROP_SIZE)
    test_real_set = TestDatasetFromFolder(TEST_PATH, crop_size=CROP_SIZE, data_type='real')
    test_synthetic_set = TestDatasetFromFolder(TEST_PATH, crop_size=CROP_SIZE, data_type='synthetic')
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    test_real_loader = DataLoader(dataset=test_real_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
    test_synthetic_loader = DataLoader(dataset=test_synthetic_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)

    model = Model(CROP_SIZE)
    loss_criterion = TotalLoss()
    if torch.cuda.is_available():
        model = model.to('cuda')
        loss_criterion = loss_criterion.to('cuda')
    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters())

    engine = Engine()
    meter_loss, meter_psnr, meter_ssim = tnt.meter.AverageValueMeter(), PSNRValueMeter(), SSIMValueMeter()
    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    test_real_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Real Loss'})
    test_synthetic_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Synthetic Loss'})
    train_psnr_logger = VisdomPlotLogger('line', opts={'title': 'Train PSNR'})
    test_real_psnr_logger = VisdomPlotLogger('line', opts={'title': 'Test Real PSNR'})
    test_synthetic_psnr_logger = VisdomPlotLogger('line', opts={'title': 'Test Synthetic PSNR'})
    train_ssim_logger = VisdomPlotLogger('line', opts={'title': 'Train SSIM'})
    test_real_ssim_logger = VisdomPlotLogger('line', opts={'title': 'Test Real SSIM'})
    test_synthetic_ssim_logger = VisdomPlotLogger('line', opts={'title': 'Test Synthetic SSIM'})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, train_loader, maxepoch=NUM_EPOCHS, optimizer=optimizer)
