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
from utils import PSNRValueMeter, SSIMValueMeter, DatasetFromFolder, TotalLoss


def processor(sample):
    mixed, first, second, training = sample
    if torch.cuda.is_available():
        mixed, first, second = mixed.to('cuda'), first.to('cuda'), second.to('cuda')

    model.train(training)

    first_predicted, second_predicted = model(mixed)
    loss = loss_criterion(first_predicted, second_predicted, first, second)
    return loss, [first_predicted, second_predicted]


def on_sample(state):
    state['sample'].append(state['train'])


def reset_meters():
    meter_loss.reset()
    meter_first_psnr.reset()
    meter_first_ssim.reset()
    meter_second_psnr.reset()
    meter_second_ssim.reset()


def on_forward(state):
    meter_loss.add(state['loss'].detach().cpu().item())
    meter_first_psnr.add(state['output'][0].detach().cpu(), state['sample'][1])
    meter_first_ssim.add(state['output'][0].detach().cpu(), state['sample'][1])
    meter_second_psnr.add(state['output'][1].detach().cpu(), state['sample'][2])
    meter_second_ssim.add(state['output'][1].detach().cpu(), state['sample'][2])


def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    print('[Epoch %d] Training Loss: %.4f Training First PSNR: %.4f dB Training First SSIM: %.4f Training Second PSNR:'
          ' %.4f dB Training Second SSIM: %.4f' % (state['epoch'], meter_loss.value()[0],
                                                   meter_first_psnr.value(), meter_first_ssim.value(),
                                                   meter_second_psnr.value(), meter_second_ssim.value()))

    train_loss_logger.log(state['epoch'], meter_loss.value()[0])
    train_first_psnr_logger.log(state['epoch'], meter_first_psnr.value())
    train_first_ssim_logger.log(state['epoch'], meter_first_ssim.value())
    train_second_psnr_logger.log(state['epoch'], meter_second_psnr.value())
    train_second_ssim_logger.log(state['epoch'], meter_second_ssim.value())
    results['train_loss'].append(meter_loss.value()[0])
    results['train_first_psnr'].append(meter_first_psnr.value())
    results['train_first_ssim'].append(meter_first_ssim.value())
    results['train_second_psnr'].append(meter_second_psnr.value())
    results['train_second_ssim'].append(meter_second_ssim.value())

    # save best model
    global best_first_psnr, best_first_ssim, best_second_psnr, best_second_ssim
    if meter_first_psnr.value() > best_first_psnr and meter_first_ssim.value() > best_first_ssim and meter_second_psnr.value() > best_second_psnr and meter_second_ssim.value() > best_second_ssim:
        torch.save(model.state_dict(), 'epochs/model.pth')
        best_first_psnr, best_first_ssim, best_second_psnr, best_second_ssim = meter_first_psnr.value(), meter_first_ssim.value(), meter_second_psnr.value(), meter_second_ssim.value()

    reset_meters()

    with torch.no_grad():
        engine.test(processor, test_loader)

    test_loss_logger.log(state['epoch'], meter_loss.value()[0])
    test_first_psnr_logger.log(state['epoch'], meter_first_psnr.value())
    test_first_ssim_logger.log(state['epoch'], meter_first_ssim.value())
    test_second_psnr_logger.log(state['epoch'], meter_second_psnr.value())
    test_second_ssim_logger.log(state['epoch'], meter_second_ssim.value())
    results['test_loss'].append(meter_loss.value()[0])
    results['test_first_psnr'].append(meter_first_psnr.value())
    results['test_first_ssim'].append(meter_first_ssim.value())
    results['test_second_psnr'].append(meter_second_psnr.value())
    results['test_second_ssim'].append(meter_second_ssim.value())

    print('[Epoch %d] Testing Loss: %.4f Testing First PSNR: %.4f dB Testing First SSIM: %.4f Testing Second PSNR:'
          ' %.4f dB Testing Second SSIM: %.4f' % (state['epoch'], meter_loss.value()[0], meter_first_psnr.value(),
                                                  meter_first_ssim.value(), meter_second_psnr.value(),
                                                  meter_second_ssim.value()))

    # save statistics at every 10 epochs
    if state['epoch'] % 10 == 0:
        data_frame = pd.DataFrame(data={'train_loss': results['train_loss'],
                                        'train_first_psnr': results['train_first_psnr'],
                                        'train_first_ssim': results['train_first_ssim'],
                                        'train_second_psnr': results['train_second_psnr'],
                                        'train_second_ssim': results['train_second_ssim'],
                                        'test_loss': results['test_loss'],
                                        'test_first_psnr': results['test_first_psnr'],
                                        'test_first_ssim': results['test_first_ssim'],
                                        'test_second_psnr': results['test_second_psnr'],
                                        'test_second_ssim': results['test_second_ssim']},
                                  index=range(1, state['epoch'] + 1))
        data_frame.to_csv('statistics/results.csv', index_label='epoch')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Mixed Image Separation Model')
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

    results = {'train_loss': [], 'train_first_psnr': [], 'train_first_ssim': [], 'train_second_psnr': [],
               'train_second_ssim': [], 'test_loss': [], 'test_first_psnr': [], 'test_first_ssim': [],
               'test_second_psnr': [], 'test_second_ssim': []}
    # record current best measures
    best_first_psnr, best_first_ssim, best_second_psnr, best_second_ssim = 0, 0, 0, 0

    train_set = DatasetFromFolder(TRAIN_PATH, crop_size=CROP_SIZE, data_type='train')
    test_set = DatasetFromFolder(TEST_PATH, crop_size=224, data_type='test')
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)

    model = Model()
    loss_criterion = TotalLoss()
    if torch.cuda.is_available():
        model = model.to('cuda')
        loss_criterion = loss_criterion.to('cuda')
    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters())

    engine = Engine()
    meter_loss, meter_first_psnr, meter_first_ssim = tnt.meter.AverageValueMeter(), PSNRValueMeter(), SSIMValueMeter()
    meter_second_psnr, meter_second_ssim = PSNRValueMeter(), SSIMValueMeter()
    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'})
    train_first_psnr_logger = VisdomPlotLogger('line', opts={'title': 'Train First PSNR'})
    test_first_psnr_logger = VisdomPlotLogger('line', opts={'title': 'Test First PSNR'})
    train_first_ssim_logger = VisdomPlotLogger('line', opts={'title': 'Train First SSIM'})
    test_first_ssim_logger = VisdomPlotLogger('line', opts={'title': 'Test First SSIM'})
    train_second_psnr_logger = VisdomPlotLogger('line', opts={'title': 'Train Second PSNR'})
    test_second_psnr_logger = VisdomPlotLogger('line', opts={'title': 'Test Second PSNR'})
    train_second_ssim_logger = VisdomPlotLogger('line', opts={'title': 'Train Second SSIM'})
    test_second_ssim_logger = VisdomPlotLogger('line', opts={'title': 'Test Second SSIM'})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, train_loader, maxepoch=NUM_EPOCHS, optimizer=optimizer)
