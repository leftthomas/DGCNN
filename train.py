import argparse

import numpy as np
import pandas as pd
import torch
import visdom
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from tqdm import tqdm

from model import Model
from utils import Indegree
from set_determ import set_determ

def get_args():
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--data_type', default='DD', type=str,
                        choices=['DD', 'PTC_MR', 'NCI1', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'MUTAG', 'COLLAB'],
                        help='dataset type')
    parser.add_argument('--batch_size', default=50, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')
    parser.add_argument('--seed', default=324, type=int, help='random seed')
    return parser.parse_args()

def train(dataloader, model, loss_fn, optimizer, device):
    """Training in one epoch. Return loss and accuracy*100."""

    model.train()
    num_batches = len(dataloader)
    num_samples = len(dataloader.dataset)
    running_loss, correct = 0, 0

    for sample in dataloader:
        data, y = sample.to(device), sample.y.to(device)
        pred = model(data)

        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        correct += (pred.argmax(dim=1) == y).sum().item()

    return running_loss/num_batches, correct/num_samples*100

def test(dataloader, model, loss_fn, device):
    """Test in one epoch. Return loss and accuracy*100."""

    model.eval()
    num_batches = len(dataloader)
    num_samples = len(dataloader.dataset)
    running_loss, correct = 0, 0

    with torch.no_grad():
        for sample in dataloader:
            data, y = sample.to(device), sample.y.to(device)
            pred = model(data)
            loss = loss_fn(pred, y)

            running_loss += loss.item()
            correct += (pred.argmax(dim=1) == y).sum().item()
    
    return running_loss/num_batches, correct/num_samples*100


if __name__ == '__main__':
    
    # ─── Initialization ───────────────────────────────────────────────────

    opt = get_args()
    set_determ(opt.seed)
    device = (
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    vis = visdom.Visdom(env=opt.data_type)  # To plot loss and accuracy
    data_set = TUDataset(
        f'data/{opt.data_type}',
        opt.data_type,
        pre_transform=Indegree(),
        use_node_attr=True,
    )
    print(f'{data_set.num_features=}, {data_set.num_classes=}')

    # ─── 10-fold Cross Validation ─────────────────────────────────────────

    over_results = {'train_accuracy': [], 'test_accuracy': []}
    train_iter = tqdm(range(1, 11), desc='Training Model......')
    for fold_number in train_iter:

        # ─── Model Definition ─────────────────────────────────────────

        model = Model(data_set.num_features, data_set.num_classes).to(device)
        loss_criterion = nn.NLLLoss()  # Set loss criterion to negative log likelihood loss
        optimizer = Adam(model.parameters()) # Create Adam optimizer for model parameters

        # ─── Dataset Split ────────────────────────────────────────────

        train_idxes = torch.as_tensor(np.loadtxt('data/%s/10fold_idx/train_idx-%d.txt' % (opt.data_type, fold_number),
                                                 dtype=np.int32), dtype=torch.long)
        test_idxes = torch.as_tensor(np.loadtxt('data/%s/10fold_idx/test_idx-%d.txt' % (opt.data_type, fold_number),
                                                dtype=np.int32), dtype=torch.long)
        train_set, test_set = data_set[train_idxes], data_set[test_idxes]
        train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=opt.batch_size, shuffle=False)

        # ─── Training Loop ────────────────────────────────────────────
        
        fold_results = {'train_loss': [], 'test_loss': [], 'train_accuracy': [], 'test_accuracy': []}
        for epoch in range(1, opt.num_epochs+1):
            train_loss, train_acc = train(train_loader, model, loss_criterion, optimizer, device)
            test_loss, test_acc = test(test_loader, model, loss_criterion, device)
            
            fold_results['train_loss'].append(train_loss)
            fold_results['train_accuracy'].append(train_acc)
            fold_results['test_loss'].append(test_loss)
            fold_results['test_accuracy'].append(test_acc)
            vis.line(torch.tensor([train_loss]), torch.tensor([epoch]), win='Train Loss', update='append', name=f'Fold_{fold_number}', opts={'title':'Train Loss', 'xlabel':'Epoch', 'ylabel':'NLL Loss'})
            vis.line(torch.tensor([train_acc]), torch.tensor([epoch]), win='Train Accuracy', update='append', name=f'Fold_{fold_number}', opts={'title':'Train Accuracy', 'xlabel':'Epoch', 'ylabel':'%'})
            vis.line(torch.tensor([test_loss]), torch.tensor([epoch]), win='Test Loss', update='append', name=f'Fold_{fold_number}', opts={'title':'Test Loss', 'xlabel':'Epoch', 'ylabel':'NLL Loss'})
            vis.line(torch.tensor([test_acc]), torch.tensor([epoch]), win='Test Accuracy', update='append', name=f'Fold_{fold_number}', opts={'title':'Test Accuracy', 'xlabel':'Epoch', 'ylabel':'%'})

        # ─── Save To Files ────────────────────────────────────────────

        torch.save(model.state_dict(), f'epochs/{opt.data_type}_{fold_number}.pth')
        pd.DataFrame(data=fold_results, index=range(1, opt.num_epochs + 1)).to_csv(
            f'statistics/{opt.data_type}_results_{fold_number}.csv', index_label='epoch')
        
        # ─── Save Overall Results ─────────────────────────────────────

        over_results['train_accuracy'].append(fold_results['train_accuracy'][-1])
        over_results['test_accuracy'].append(fold_results['test_accuracy'][-1])

        # ─── Print Progress Bar ───────────────────────────────────────

        train_iter.set_description(f'[{fold_number}] Train Acc: {fold_results["train_accuracy"][-1]:.2f}% Test Acc: {fold_results["test_accuracy"][-1]:.2f}%')

    # ─── Save And Print Overall Result ────────────────────────────────────

    pd.DataFrame(data=over_results,index=range(1, 11)).to_csv(
        f'statistics/{opt.data_type}_results_overall.csv', index_label='fold')
    print('Overall Training Accuracy: %.2f%% (std: %.2f) Testing Accuracy: %.2f%% (std: %.2f)' %
          (np.array(over_results['train_accuracy']).mean(), np.array(over_results['train_accuracy']).std(),
           np.array(over_results['test_accuracy']).mean(), np.array(over_results['test_accuracy']).std()))
