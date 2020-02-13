from typing import List, Set, Dict, Optional, Union
import itertools
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
import dgl


class Discriminator:

    def __init__(self, graph, model, optimizer, device=None, log_reward=False, max_boundary_size=200, nodefeats=None):
        self.model = model
        self.optimizer = optimizer
        self.max_boundary_size = max_boundary_size
        self.graph = graph
        self.dgl_graph :dgl.DGLGraph = dgl.DGLGraph(self.graph.adj_mat)
        if nodefeats is not None:
            self.dgl_graph.ndata['feats'] = torch.from_numpy(nodefeats)
            self.with_attr = True
        else:
            self.with_attr = False
        self.log_reward = log_reward
        self.n_nodes = self.graph.n_nodes
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
        self.dgl_graph.to(self.device)

    def score_comms(self, nodes: List[List[int]]):
        if self.with_attr:
            batch_g = dgl.BatchedDGLGraph([self.prepare_graph(x) for x in nodes], ['state', 'feats'], None)
        else:
            batch_g = dgl.BatchedDGLGraph([self.prepare_graph(x) for x in nodes], 'state', None)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(batch_g)
            p = torch.exp(logits[:, 1]).cpu().numpy()
        if self.log_reward:
            r = -np.log(1 - p + 1e-9)
            r = np.clip(r, 0, 5.)
            return r
        else:
            return p

    def prepare_graph(self, nodes):
        nodes = list(nodes)
        if self.max_boundary_size:
            boundary = self.graph.outer_boundary(nodes)
            if len(boundary) > self.max_boundary_size:
                boundary = np.random.choice(list(boundary), size=self.max_boundary_size, replace=False)
            state = torch.zeros(len(nodes) + len(boundary), dtype=torch.long, device=self.device)
            state[:len(nodes)] = 1
            nodes = nodes + list(boundary)
        else:
            state = torch.ones(len(nodes), dtype=torch.long, device=self.device)
        subg = self.dgl_graph.subgraph(nodes)
        subg.copy_from_parent()
        # subg = dgl.DGLGraph(self.graph.adj_mat[nodes][:, nodes])
        subg.ndata['state'] = state
        return subg

    def train_step(self, pos_comms, neg_comms):
        subgs = []
        for nodes in itertools.chain(pos_comms):
            subg = self.prepare_graph(nodes)
            subgs.append(subg)
        for nodes in itertools.chain(neg_comms):
            subg = self.prepare_graph(nodes)
            subgs.append(subg)
        if self.with_attr:
            batch_graph = dgl.BatchedDGLGraph(subgs, ['state', 'feats'], None)
        else:
            batch_graph = dgl.BatchedDGLGraph(subgs, 'state', None)
        labels = torch.cat([torch.ones(len(pos_comms)), torch.zeros(len(neg_comms))]).long().to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(batch_graph)
        loss = F.nll_loss(logits, labels)
        loss.backward()
        self.optimizer.step()
        acc = (torch.argmax(logits, 1) == labels).float().mean().item()
        return loss.item(), {'acc': acc}
