import numpy as np
from scipy import sparse as sp

import torch
from torch import nn

from .graph import Graph
from .layers import swish, Swish, make_linear_block, SelfAttnPooling


class GraphConv:

    def __init__(self, graph: Graph, k: int = 3, alpha: float = 0.85):
        self.graph = graph
        self.k = k
        self.alpha = alpha
        self.normlized_adj_mat = self._normalize_adj(graph.adj_mat).astype(np.float32)

    def __repr__(self):
        return f'Conv_{self.k}_{self.alpha}'

    def __str__(self):
        return self.__repr__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: sp.spmatrix):
        init_val = x
        for _ in range(self.k):
            x = self.alpha * (self.normlized_adj_mat @ x) + (1 - self.alpha) * init_val
        return x

    @staticmethod
    def _normalize_adj(adj: sp.spmatrix) -> sp.spmatrix:
        """Symmetrically normalize adjacency matrix."""
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


class Agent(nn.Module):

    def __init__(self, hidden_size, with_attr=False, norm_type=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.with_attr = with_attr
        if with_attr:
            self.attr_embedding = nn.Linear(hidden_size, hidden_size)
        self.seed_embedding = nn.Linear(1, hidden_size, bias=False)
        self.node_embedding = nn.Linear(1, hidden_size, bias=False)
        self.input_mapping = nn.Sequential(
            make_linear_block(hidden_size, hidden_size, Swish, norm_type),
            make_linear_block(hidden_size, hidden_size, Swish, norm_type)
        )
        self.pooling_layer = nn.Sequential(SelfAttnPooling(hidden_size), Swish())
        self.value_layer = nn.Linear(hidden_size, 1)
        self.node_score_layer = nn.Linear(hidden_size, 1, bias=False)
        self.stopping_score_layer = nn.Linear(hidden_size, 2, bias=False)

        nn.init.zeros_(self.value_layer.weight.data)
        nn.init.zeros_(self.node_score_layer.weight.data)
        nn.init.zeros_(self.stopping_score_layer.weight.data)
        # nn.init.constant_(self.stopping_score_layer.weight.data[0], 1.)

    def forward(self, x_attrs, x_seeds, x_nodes, indptr):
        # x_attrs: [n, d]
        # x_seeds: [n]
        # x_nodes: [n]
        # indpt: [ [ start, end, candidate_end ] ]
        h = self.seed_embedding(x_seeds.unsqueeze(1)) + self.node_embedding(x_nodes.unsqueeze(1))
        if self.with_attr:
            h += self.attr_embedding(x_attrs)
        else:
            assert x_attrs is None
        h = self.input_mapping(h)  # [*, d]
        node_scores = self.node_score_layer(h).squeeze(1)  # [*]
        batch = []
        for startpoint, endpoint, candidate_endpoint in indptr:
            if startpoint == endpoint:
                raise ValueError('Finished Episode!')
            else:
                global_z = self.pooling_layer(h[startpoint:endpoint])  # [1, d]
                value = self.value_layer(global_z).squeeze()
                node_logits = torch.log_softmax(node_scores[startpoint:candidate_endpoint], 0)  # [r-l]
                stopping_logits = torch.log_softmax(self.stopping_score_layer(global_z), 1).squeeze(0)  # [2]
                logits = torch.cat([node_logits + stopping_logits[0], stopping_logits[1:]], dim=0)  # [*+1]
                batch.append([logits, value])
        batch_logits, values = zip(*batch)
        values = torch.stack(values)
        return batch_logits, values
