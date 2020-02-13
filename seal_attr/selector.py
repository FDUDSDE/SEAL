import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from .generator import Generator
from .layers import Swish, LinearBlock


class Selector:

    def __init__(self, generator: Generator, score_fn, n_train, n_valid, seen_nodes, savedir, dropout=0.5):
        self.savedir = savedir
        self.seen_nodes = seen_nodes
        self.n_train, self.n_valid = n_train, n_valid
        self.g = generator
        self.score_fn = score_fn
        self.device = generator.device
        self.nodefeats = self.prepare_features()
        self.n_feats = self.nodefeats.shape[1]
        self.dropout = dropout

    def prepare_features(self):
        graph = self.g.graph
        feat_mat = self.local_degree_profile(graph)
        feat_mat = np.concatenate([feat_mat, np.ones([graph.n_nodes, 1])], 1)
        feat_mat = self.g.conv(feat_mat).astype(np.float32)
        if self.g.nodefeats is not None:
            feat_mat = np.concatenate([feat_mat, self.g.nodefeats], 1)
        return feat_mat

    def prepare_dataloaders(self):
        graph = self.g.graph
        seeds = np.random.choice(graph.n_nodes, self.n_train + self.n_valid, replace=False)
        comms = self.generate_communities(seeds, self.g.generate)
        ys = self.score_communities(comms, self.score_fn)
        # Train
        train_nodes = seeds[:self.n_train]
        train_X = self.nodefeats[train_nodes]
        train_y = ys[:self.n_train]
        train_ds = TensorDataset(torch.from_numpy(train_X).float(), torch.from_numpy(train_y).float())
        train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
        # Valid
        valid_nodes = seeds[self.n_train:]
        valid_X = self.nodefeats[valid_nodes]
        valid_y = ys[self.n_train:]
        valid_ds = TensorDataset(torch.from_numpy(valid_X).float(), torch.from_numpy(valid_y).float())
        valid_dl = DataLoader(valid_ds, batch_size=512, shuffle=False)
        return train_dl, valid_dl

    def train(self, train_dl, valid_dl, it):
        mlp_model = RegModel(self.n_feats, 64, 3, norm_type='batch_norm', dropout=self.dropout).to(self.device)
        mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        best_val = 1e10
        best_idx = -1
        counter = 0
        for i_epoch in range(500):
            epoch_loss = 0.
            mlp_model.train()
            for x, y in train_dl:
                x, y = x.to(self.device), y.to(self.device)
                mlp_optimizer.zero_grad()
                pred_y = mlp_model(x).squeeze()
                loss = criterion(pred_y, y)
                loss.backward()
                mlp_optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(train_dl)
            if (i_epoch + 1) % 10 == 0:
                valid_loss = 0.
                mlp_model.eval()
                with torch.no_grad():
                    for x, y in valid_dl:
                        x, y = x.to(self.device), y.to(self.device)
                        pred_y = mlp_model(x).squeeze()
                        loss = criterion(pred_y, y)
                        valid_loss += loss.item()
                    valid_loss /= len(valid_dl)
                if valid_loss < best_val:
                    best_val = valid_loss
                    best_idx = i_epoch + 1
                    counter = 0
                    torch.save(mlp_model.state_dict(), self.savedir / f'selector-{it:0>4d}.pth')
                else:
                    counter += 1
                print(f'[Epoch {i_epoch + 1:3d}] TrainLoss={epoch_loss:.4f} ValidLoss={valid_loss:.4f}')
                if counter >= 5:
                    print('Early Stopping.')
                    break
        print(f'Load the best model at Epoch {best_idx}.')
        mlp_model.load_state_dict(torch.load(self.savedir / f'selector-{it:0>4d}.pth'))
        return mlp_model

    def sort_nodes(self, it):
        print('Preparing datasets for the seed selector.')
        train_dl, valid_dl = self.prepare_dataloaders()
        model = self.train(train_dl, valid_dl, it)
        test_ds = TensorDataset(torch.from_numpy(self.nodefeats).float())
        test_dl = DataLoader(test_ds, batch_size=512, shuffle=False)
        pred_ys = []
        model.eval()
        with torch.no_grad():
            for x, *_ in test_dl:
                x = x.to(self.device)
                y = model(x).cpu().numpy()
                pred_ys.extend(y)
        pred_ys = np.array(pred_ys)
        idx = np.argsort(pred_ys)[::-1]
        return idx, pred_ys[idx]

    def make_predicition(self, n_outputs, it):
        idx, _ = self.sort_nodes(it)
        idx = [i for i in idx if i not in self.seen_nodes][:n_outputs]
        pred_comms = self.generate_communities(idx, self.g.generate)
        return pred_comms

    @staticmethod
    def local_degree_profile(graph):
        feat_mat = np.zeros([graph.n_nodes, 5], dtype=np.float32)
        feat_mat[:, 0] = np.array(graph.adj_mat.sum(1)).squeeze()
        for i in range(graph.n_nodes):
            neighbor_degs = feat_mat[list(graph.neighbors[i]), 0]
            feat_mat[i, 1:] = neighbor_degs.min(), neighbor_degs.max(), neighbor_degs.mean(), neighbor_degs.std()
        feat_mat = (feat_mat - feat_mat.mean(0, keepdims=True)) / (feat_mat.std(0, keepdims=True) + 1e-9)
        return feat_mat

    @staticmethod
    def score_communities(cs, score_fn):
        chunk_size = 100
        all_scores = []
        for i in range((len(cs) // chunk_size) + 1):
            chunk = [cs[k] for k in range(chunk_size * i, chunk_size * (i+1)) if k < len(cs)]
            if len(chunk):
                all_scores.extend(score_fn(chunk))
        return np.array(all_scores)

    @staticmethod
    def generate_communities(seeds, g_fn):
        chunk_size = 100
        all_comms = []
        for i in range((len(seeds) // chunk_size) + 1):
            chunk = seeds[i*chunk_size:(i+1)*chunk_size]
            if len(chunk):
                all_comms.extend(g_fn(chunk))
        all_comms = [x[:-1] if x[-1] == 'EOS' else x for x in all_comms]
        return all_comms


class RegModel(nn.Module):

    def __init__(self, in_feats, hidden_size, n_layers, norm_type='batch_norm', dropout=0.):
        super().__init__()
        layers = [LinearBlock(in_feats, hidden_size, Swish, norm_type=norm_type, dropout=dropout)]
        for _ in range(n_layers - 1):
            layers.append(LinearBlock(hidden_size, hidden_size, Swish, norm_type=norm_type, dropout=dropout))
        layers.append(LinearBlock(hidden_size, 1, act_cls=None, norm_type=norm_type, dropout=dropout))
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x).squeeze()

