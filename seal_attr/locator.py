from typing import List, Set, Dict, Optional, Union
import itertools
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
import dgl


class Locator:

    def __init__(self, graph, model, optimizer, device=None, max_boundary_size=200, nodefeats=None):
        self.model = model
        self.optimizer = optimizer
        self.max_boundary_size = max_boundary_size
        self.graph = graph
        self.dgl_graph = dgl.DGLGraph(self.graph.adj_mat)
        self.n_nodes = self.graph.n_nodes
        if nodefeats is not None:
            self.dgl_graph.ndata['feats'] = torch.from_numpy(nodefeats)
            self.with_attr = True
        else:
            self.with_attr = False
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
        self.dgl_graph.to(self.device)

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
        subg.ndata['state'] = state
        return subg

    def score_comms(self, nodes: List[List[int]]):
        comms = [list(x) for x in nodes]
        subgs = [self.prepare_graph(x) for x in comms]
        if self.with_attr:
            batched_graph = dgl.BatchedDGLGraph(subgs, ['state', 'feats'], None)
        else:
            batched_graph = dgl.BatchedDGLGraph(subgs, 'state', None)
        self.model.eval()
        all_logits, values = self.model(batched_graph)
        offset = 0
        rewards = []
        for comm, n_nodes in zip(comms, batched_graph.batch_num_nodes):
            scores = all_logits[offset:offset+len(comm)]
            offset += n_nodes
            rewards.append((scores <= scores[0]).float().mean().item())
        rewards = np.array(rewards)
        return rewards

    def train_step(self, comms, fn):
        comms = [list(x) for x in comms]
        subgs = [self.prepare_graph(x) for x in comms]
        if self.with_attr:
            batched_graph = dgl.BatchedDGLGraph(subgs, ['state', 'feats'], None)
        else:
            batched_graph = dgl.BatchedDGLGraph(subgs, 'state', None)
        self.model.train()
        self.optimizer.zero_grad()
        all_logits, values = self.model(batched_graph)
        offset = 0
        seeds = []
        logps = []
        for comm, n_nodes in zip(comms, batched_graph.batch_num_nodes):
            logits = torch.log_softmax(all_logits[offset:offset+len(comm)], 0)
            offset += n_nodes
            ps = torch.exp(logits.detach())
            seed_idx = torch.multinomial(ps, 1).item()
            seeds.append(comm[seed_idx])
            logps.append(logits[seed_idx])
        logps = torch.stack(logps)
        # dual learning
        generated_comms = [x[:-1] if x[-1] == 'EOS' else x for x in fn(seeds)]
        rewards = []
        for x, y in zip(comms, generated_comms):
            a = set(x)
            b = set(y)
            rewards.append(len(a & b) / len(a | b))
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        # advantages = rewards - values.detach()
        value_loss = ((values - rewards) ** 2).mean()
        # policy_loss = -(advantages * logps).mean()
        policy_loss = -(rewards * logps).mean()
        loss = policy_loss + .5 * value_loss
        loss.backward()
        self.optimizer.step()
        return policy_loss.item(), value_loss.item(), rewards.mean().item()
