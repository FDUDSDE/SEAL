from typing import Union, Optional, List, Set, Dict
import collections

import numpy as np
from scipy import sparse as sp
import random


class Graph:

    def __init__(self, edges):
        self.neighbors, self.n_nodes, self.adj_mat = self._init_from_edges(edges)

    @staticmethod
    def _init_from_edges(edges: np.ndarray) -> (Dict[int, Set[int]], int, sp.spmatrix):
        neighbors = collections.defaultdict(set)
        max_id = -1
        for u, v in edges:
            max_id = max(max_id, u, v)
            if u != v:
                neighbors[u].add(v)
                neighbors[v].add(u)
        n_nodes = len(neighbors)
        if (max_id + 1) != n_nodes:
            raise ValueError('Please re-label nodes first!')
        adj_mat = sp.csr_matrix((np.ones(len(edges)), edges.T), shape=(n_nodes, n_nodes))
        adj_mat += adj_mat.T
        return neighbors, n_nodes, adj_mat

    def outer_boundary(self, nodes: Union[List, Set]) -> Set[int]:
        boundary = set()
        for u in nodes:
            boundary |= self.neighbors[u]
        boundary.difference_update(nodes)
        return boundary

    def k_ego(self, nodes: Union[List, Set], k: int) -> Set[int]:
        ego_nodes = set(nodes)
        current_boundary = set(nodes)
        for _ in range(k):
            current_boundary = self.outer_boundary(current_boundary) - ego_nodes
            ego_nodes |= current_boundary
        return ego_nodes

    def check_valid_expansion(self, expansion: List[int]) -> bool:
        if len(expansion) != len(set(expansion)):
            return False
        for i in range(1, len(expansion) - 1):
            boundary = self.outer_boundary(expansion[:i])
            if expansion[i] not in boundary:
                return False
        return True

    def sample_expansion_from_community(self, comm_nodes: Union[List, Set],
                                        seed: Optional[int] = None) -> List[int]:
        if seed is None:
            seed = random.choice(tuple(comm_nodes))
        remaining = set(comm_nodes) - {seed}
        boundary = self.neighbors[seed].copy()
        walk = [seed]
        while len(remaining):
            candidates = tuple(boundary & remaining)
            new_node = random.choice(candidates)
            remaining.remove(new_node)
            boundary |= self.neighbors[new_node]
            walk.append(new_node)
        return walk

    def sample_expansion(self, max_size: int, seed: Optional[int] = None) -> List[int]:
        if seed is None:
            seed = random.randint(0, self.n_nodes - 1)
        walk = [seed]
        boundary = self.neighbors[seed].copy()
        for i in range(max_size - 1):
            candidates = boundary - set(walk)
            if len(candidates) == 0:
                break
            new_node = random.choice(tuple(candidates))
            boundary |= self.neighbors[new_node]
            walk.append(new_node)
        return walk

    def connected_components(self, nodes):
        remaining = set(nodes)
        ccs = []
        cc = set()
        queue = collections.deque()
        while len(remaining) or len(queue):
            # print(queue, remaining)
            if len(queue) == 0:
                if len(cc):
                    ccs.append(cc)
                v = remaining.pop()
                cc = {v}
                queue.extend(self.neighbors[v] & remaining)
                remaining -= {v}
                remaining -= self.neighbors[v]
            else:
                v = queue.popleft()
                queue.extend(self.neighbors[v] & remaining)
                cc |= (self.neighbors[v] & remaining) | {v}
                remaining -= self.neighbors[v]
        if len(cc):
            ccs.append(cc)
        return ccs
