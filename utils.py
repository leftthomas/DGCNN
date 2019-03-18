import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import to_dense_batch, degree


def global_sort_pool(x, batch, k=None):
    r"""where node features are first sorted individually and then sorted in
    descending order based on their last features. The first :math:`k` nodes
    form the output of the layer, if k equals None, than don't select nodes.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        k (int): The number of nodes to hold for each graph.

    :rtype: :class:`Tensor`
    """
    x, _ = x.sort(dim=-1)

    fill_value = x.min().item() - 1
    batch_x, num_nodes = to_dense_batch(x, batch, fill_value)
    B, N, D = batch_x.size()

    _, perm = batch_x[:, :, -1].sort(dim=-1, descending=True)
    arange = torch.arange(B, dtype=torch.long, device=perm.device) * N
    perm = perm + arange.view(-1, 1)

    batch_x = batch_x.view(B * N, D)
    batch_x = batch_x[perm]
    batch_x = batch_x.view(B, N, D)

    if k is not None and N >= k:
        batch_x = batch_x[:, :k].contiguous()
    elif k is not None:
        expand_batch_x = batch_x.new_full((B, k - N, D), fill_value)
        batch_x = torch.cat([batch_x, expand_batch_x], dim=1)

    batch_x[batch_x == fill_value] = 0

    return batch_x


class Indegree(object):
    r"""Adds the globally normalized node degree to the node features.

    Args:
        cat (bool, optional): If set to :obj:`False`, all existing node
            features will be replaced. (default: :obj:`True`)
    """

    def __init__(self, norm=True, max_value=None, cat=True):
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def __call__(self, data):
        col, x = data.edge_index[1], data.x
        deg = degree(col, data.num_nodes)

        if self.norm:
            deg = deg / (deg.max() if self.max is None else self.max)

        deg = deg.view(-1, 1)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data.x = deg

        return data

    def __repr__(self):
        return '{}(norm={}, max_value={})'.format(self.__class__.__name__, self.norm, self.max)


class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, classes, labels):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        loss = labels * left + 0.5 * (1 - labels) * right
        return loss.sum(dim=-1).mean()
