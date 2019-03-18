import argparse

import numpy as np
import pandas as pd
import torch
import torchnet as tnt
from sklearn.model_selection import RepeatedStratifiedKFold
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger
from tqdm import tqdm

from model import Model
from utils import MarginLoss, Indegree

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def processor(sample):
    data, training = sample
    labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=data.y)

    if torch.cuda.is_available():
        data, labels = data.to('cuda'), labels.to('cuda')

    model.train(training)

    classes = model(data)
    loss = loss_criterion(classes, labels)
    return loss, classes


def on_sample(state):
    state['sample'] = state['sample'], state['train']


def reset_meters():
    meter_loss.reset()
    meter_accuracy.reset()


def on_forward(state):
    meter_loss.add(state['loss'].detach().cpu().item())
    meter_accuracy.add(state['output'].detach().cpu(), state['sample'][0].y)


def on_start_epoch(state):
    reset_meters()


def on_end_epoch(state):
    train_loss_logger.log(state['epoch'], meter_loss.value()[0], name='fold_' + str(fold_number))
    train_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0], name='fold_' + str(fold_number))
    fold_results['train_loss'].append(meter_loss.value()[0])
    fold_results['train_accuracy'].append(meter_accuracy.value()[0])

    reset_meters()
    with torch.no_grad():
        engine.test(processor, test_loader)

    test_loss_logger.log(state['epoch'], meter_loss.value()[0], name='fold_' + str(fold_number))
    test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0], name='fold_' + str(fold_number))
    fold_results['test_loss'].append(meter_loss.value()[0])
    fold_results['test_accuracy'].append(meter_accuracy.value()[0])

    # save best model at every fold
    global best_accuracy
    if meter_accuracy.value()[0] > best_accuracy:
        torch.save(model.state_dict(), 'epochs/%s_%d.pth' % (DATA_TYPE, fold_number))
        best_accuracy = meter_accuracy.value()[0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--data_type', default='DD', type=str,
                        choices=['DD', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K', 'PTC_MR', 'NCI1',
                                 'NCI109', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'MUTAG', 'ENZYMES', 'COLLAB'],
                        help='dataset type')
    parser.add_argument('--num_iterations', default=3, type=int, help='routing iterations number')
    parser.add_argument('--batch_size', default=20, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')

    opt = parser.parse_args()

    DATA_TYPE = opt.data_type
    NUM_ITERATIONS = opt.num_iterations
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs

    data_set = TUDataset('data/%s' % DATA_TYPE, DATA_TYPE, pre_transform=Indegree(), use_node_attr=True)
    NUM_FEATURES, NUM_CLASSES = data_set.num_features, data_set.num_classes

    over_results = {'train_accuracy': [], 'test_accuracy': []}
    # record current best measures
    best_accuracy = 0

    model = Model(NUM_FEATURES, NUM_CLASSES, NUM_ITERATIONS)
    loss_criterion = MarginLoss()
    if torch.cuda.is_available():
        model = model.to('cuda')
        loss_criterion = loss_criterion.to('cuda')

    print('# model parameters:', sum(param.numel() for param in model.parameters()))

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    train_loss_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Train Loss'})
    train_accuracy_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Train Accuracy'})
    test_loss_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Test Loss'})
    test_accuracy_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Test Accuracy'})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    # create a 10 times 10-fold cross validation
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)
    fold_number = 1
    train_iter = tqdm(rskf.split(data_set, data_set.data.y), desc='Training Model......')
    for train_index, test_index in train_iter:
        # 90/10 train/test split
        train_index = torch.zeros(len(data_set)).index_fill(0, torch.as_tensor(train_index), 1).byte()
        test_index = torch.zeros(len(data_set)).index_fill(0, torch.as_tensor(test_index), 1).byte()
        train_set, test_set = data_set[train_index], data_set[test_index]
        train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)

        fold_results = {'train_loss': [], 'test_loss': [], 'train_accuracy': [], 'test_accuracy': []}

        optimizer = Adam(model.parameters())

        engine.train(processor, train_loader, maxepoch=NUM_EPOCHS, optimizer=optimizer)
        # save statistics at every fold
        fold_data_frame = pd.DataFrame(
            data={'train_loss': fold_results['train_loss'], 'test_loss': fold_results['test_loss'],
                  'train_accuracy': fold_results['train_accuracy'],
                  'test_accuracy': fold_results['test_accuracy']},
            index=range(1, NUM_EPOCHS + 1))
        fold_data_frame.to_csv('statistics/%s_results_%d.csv' % (DATA_TYPE, fold_number), index_label='epoch')

        over_results['train_accuracy'].append(np.array(fold_results['train_accuracy']).max())
        over_results['test_accuracy'].append(np.array(fold_results['test_accuracy']).max())

        train_iter.set_description('[Fold %d] Training Accuracy: %.2f%% Testing Accuracy: %.2f%%' % (
            fold_number, np.array(fold_results['train_accuracy']).max(), np.array(fold_results['test_accuracy']).max()))

        fold_number += 1
        # reset them for each fold
        best_accuracy = 0
        model = Model(NUM_FEATURES, NUM_CLASSES, NUM_ITERATIONS)
        if torch.cuda.is_available():
            model = model.to('cuda')

    # save statistics at all fold
    data_frame = pd.DataFrame(
        data={'train_accuracy': over_results['train_accuracy'], 'test_accuracy': over_results['test_accuracy']},
        index=range(1, fold_number))
    data_frame.to_csv('statistics/%s_results_overall.csv' % DATA_TYPE, index_label='fold')

    print('Overall Training Accuracy: %.2f%% (std: %.2f) Testing Accuracy: %.2f%% (std: %.2f)' %
          (np.array(over_results['train_accuracy']).mean(), np.array(over_results['train_accuracy']).std(),
           np.array(over_results['test_accuracy']).mean(), np.array(over_results['test_accuracy']).std()))
