import torch
from torch import nn
import dgl
from dgl.nn.pytorch import GINConv


from .layers import Swish, swish, make_linear_block


class GINClassifier(nn.Module):

    def __init__(self, hidden_size, n_layers, n_states=2, dropout=0.0, feat_dropout=0.5, norm_type=None, agg_type='sum', with_attr=False):
        super().__init__()
        self.agg_type = agg_type
        self.with_attr = with_attr
        self.state_embedding = nn.Embedding(n_states, hidden_size)
        if with_attr:
            self.feat_mapping = make_linear_block(hidden_size, hidden_size,
                                                  residual=False, dropout=feat_dropout, bias=False)
        self.gconv_layers = nn.ModuleList([GINConv(None, agg_type) for _ in range(n_layers)])
        self.fc_layers = nn.ModuleList([make_linear_block(hidden_size, hidden_size, act_cls=Swish,
                                                          norm_type=norm_type, dropout=dropout)
                                        for _ in range(n_layers)])
        self.scoring_layer = make_linear_block(hidden_size * n_layers, 2,
                                               norm_type=norm_type, dropout=dropout, act_cls=Swish)

    def forward(self, g: dgl.DGLGraph):
        state = g.ndata['state']
        h = self.state_embedding(state)
        if self.with_attr:
            h += self.feat_mapping(g.ndata['feats'])
        hs = []
        if g.number_of_edges() == 0:
            for gn, fn in zip(self.gconv_layers, self.fc_layers):
                h = fn(h) * (1 + gn.eps)
                hs.append(h)
        else:
            for gn, fn in zip(self.gconv_layers, self.fc_layers):
                h = gn(g, fn(h))
                hs.append(h)
        g.ndata['h'] = torch.cat(hs, dim=1)
        if self.agg_type == 'sum':
            z = dgl.sum_nodes(g, 'h')
        elif self.agg_type == 'mean':
            z = dgl.mean_nodes(g, 'h')
        else:
            raise NotImplementedError
        logits = torch.log_softmax(self.scoring_layer(z), -1)
        return logits

