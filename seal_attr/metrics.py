import collections

import numpy as np


def compare_comm(pred_comm, true_comm):
    intersect = set(true_comm) & set(pred_comm)
    p = len(intersect) / len(pred_comm)
    r = len(intersect) / len(true_comm)
    f = 2 * p * r / (p + r + 1e-9)
    j = len(intersect) / (len(pred_comm) + len(true_comm) - len(intersect))
    return p, r, f, j


def eval_comms_double_sparse(x_comms, y_comms):
    x_node_comm = collections.defaultdict(set)
    y_node_comm = collections.defaultdict(set)
    for i, nodes in enumerate(x_comms):
        for u in nodes:
            x_node_comm[u].add(i)
    for i, nodes in enumerate(y_comms):
        for u in nodes:
            y_node_comm[u].add(i)
    x_neighbors = collections.defaultdict(set)
    y_neighbors = collections.defaultdict(set)
    for u in x_node_comm.keys() & y_node_comm.keys():
        x_idx = x_node_comm[u]
        y_idx = y_node_comm[u]
        for xid in x_idx:
            x_neighbors[xid].update(y_idx)
        for yid in y_idx:
            y_neighbors[yid].update(x_idx)
    cache = {}
    x_metrics = np.zeros([len(x_comms), 4])
    y_metrics = np.zeros([len(y_comms), 4])
    for i, neighbor_js in x_neighbors.items():
        x_metrics[i] = np.max([cache.setdefault((i, j), compare_comm(x_comms[i], y_comms[j])) for j in neighbor_js], 0)
    for j, neighbor_is in y_neighbors.items():
        y_metrics[j] = np.max([cache[(i, j)] for i in neighbor_is], 0)
    y_metrics[:, :2] = y_metrics[:, [1, 0]]
    # print(len(cache), '/', len(x_comms) * len(y_comms))
    return x_metrics, y_metrics
