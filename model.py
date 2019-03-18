import torch
from capsule_layer import CapsuleLinear
from torch import nn
from torch_geometric.nn import GCNConv

from utils import global_sort_pool


class Model(nn.Module):
    def __init__(self, num_features, num_classes, num_iterations=3):
        super(Model, self).__init__()

        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 1)
        self.classifier = CapsuleLinear(out_capsules=num_classes, in_length=97, out_length=16, in_capsules=None,
                                        share_weight=True, routing_type='k_means', similarity='tonimoto',
                                        num_iterations=num_iterations)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_1 = torch.tanh(self.conv1(x, edge_index))
        x_2 = torch.tanh(self.conv2(x_1, edge_index))
        x_3 = torch.tanh(self.conv3(x_2, edge_index))
        x_4 = torch.tanh(self.conv4(x_3, edge_index))
        x = torch.cat([x_1, x_2, x_3, x_4], dim=-1)
        x = global_sort_pool(x, batch)
        out = self.classifier(x)
        classes = out.norm(dim=-1)

        return classes
