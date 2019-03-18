import torch
import torch.functional as F
from torch import nn
from torch.nn import Conv1d, MaxPool1d, Linear, Dropout
from torch_geometric.nn import GCNConv, global_sort_pool


class Model(nn.Module):
    def __init__(self, num_features, num_classes):
        super(Model, self).__init__()

        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 1)
        self.conv5 = Conv1d(97, 16, 97, 97)
        self.conv6 = Conv1d(16, 32, 5, 1)
        self.pool = MaxPool1d(2, 2)
        self.classifier = Linear(128, num_classes)
        self.drop_out = Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_1 = torch.tanh(self.conv1(x, edge_index))
        x_2 = torch.tanh(self.conv2(x_1, edge_index))
        x_3 = torch.tanh(self.conv3(x_2, edge_index))
        x_4 = torch.tanh(self.conv4(x_3, edge_index))
        x = torch.cat([x_1, x_2, x_3, x_4], dim=-1)
        x = global_sort_pool(x, batch, k=50)
        x = x.view(*x.size(), 1)
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        x = self.relu(self.conv6(x))
        x = x.view(x.size(0), -1)
        out = F.softmax(self.classifier(x), dim=-1)
        classes = self.drop_out(out)

        return classes
