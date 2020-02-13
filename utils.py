import numpy as np
import torch
import random
import dgl

# from models import eval_comms

from typing import List, Dict, Set, Union, Any, Iterable


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)


def split_comms(comms: Iterable[List[int]], n_train: int, seed: int,
                max_size: int = 0) -> (List[List[int]], List[List[int]]):
    if max_size:
        comms = [x for x in comms if 3 <= len(x) <= max_size]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(comms))
    train_comms = [comms[i] for i in idx[:n_train]]
    test_comms = [comms[i] for i in idx[n_train:]]
    return train_comms, test_comms


def print_results(score: Iterable[float], prefix: str = 'Metric'):
    p, r, f, j = score
    print(f'{prefix}: Precision:{p:.2f} Recall:{r:.2f} F1:{f:.2f} Jaccard:{j:.2f}', flush=True)


def write_comms_to_file(comms: List[List[int]], fname: str):
    with open(fname, 'w') as fh:
        content = '\n'.join([' '.join([str(i) for i in comm]) for comm in comms])
        fh.write(content)


def read_comms_from_file(fname: str) -> List[List[int]]:
    with open(fname) as fh:
        content = fh.read().strip().split('\n')
        comms = [[int(i) for i in line.strip().split(' ')] for line in content]
    return comms


# def evaluate(pred_comms: List[List[int]], test_comms: List[List[int]]) -> (np.ndarray, np.ndarray, np.ndarray):
#     scores_1, scores_2 = eval_comms(pred_comms, test_comms)
#     mean_score_1 = scores_1.mean(0)
#     print_results(mean_score_1, 'AvgOverPredictions')
#     mean_score_2 = scores_2.mean(0)
#     print_results(mean_score_2, 'AvgOverGroundTruth')
#     mean_score_all = (mean_score_1 + mean_score_2) / 2.
#     print_results(mean_score_all, 'AvgOverBidirection')
#     return mean_score_1, mean_score_2, mean_score_all

