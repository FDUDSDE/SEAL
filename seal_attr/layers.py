import torch
from torch import nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def swish(x):
    return x * torch.sigmoid(x)


class SelfAttnPooling(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.score_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        # x: [*, n, d]
        w = self.score_layer(x)
        w = torch.softmax(w, -2)  # [*, n, 1]
        return (x * w).sum(-2, keepdim=True)  # [*, 1, d]


def make_linear_block(in_size, out_size, act_cls=None, norm_type=None, bias=True, residual=True, dropout=0.):
    # layers = []
    # if norm_type == 'batch_norm':
    #     layers.append(nn.BatchNorm1d(in_size))
    # elif norm_type == 'layer_norm':
    #     layers.append(nn.LayerNorm(in_size))
    # elif norm_type is not None:
    #     raise NotImplementedError
    # if act_cls is not None:
    #     layers.append(act_cls())
    # layers.append(nn.Dropout(dropout))
    # layers.append(nn.Linear(in_size, out_size, bias))
    # return nn.Sequential(*layers)
    return LinearBlock(in_size, out_size, act_cls, norm_type, bias, residual, dropout)


class LinearBlock(nn.Module):

    def __init__(self, in_size, out_size, act_cls=None, norm_type=None, bias=True, residual=True, dropout=0.):
        super().__init__()
        self.residual = residual and (in_size == out_size)
        layers = []
        if norm_type == 'batch_norm':
            layers.append(nn.BatchNorm1d(in_size))
        elif norm_type == 'layer_norm':
            layers.append(nn.LayerNorm(in_size))
        elif norm_type is not None:
            raise NotImplementedError
        if act_cls is not None:
            layers.append(act_cls())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_size, out_size, bias))
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        z = self.f(x)
        if self.residual:
            z += x
        return z

