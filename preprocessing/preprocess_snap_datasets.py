import collections
import gzip
import pathlib
import tarfile
import shutil
import os
import numpy as np
import tqdm

from graph import Graph


def load_ego_edges(u, data_dir):
    """Load edges in ego-net datasets."""
    with open(data_dir / f'{u}.edges') as fh:
        edges = fh.read().strip().split('\n')
    edges = [[int(i) for i in x.split()] for x in edges]
    edges = np.array([[a, b] if a < b else [b, a] for a, b in edges if a != b])
    return edges


def load_ego_feats(u, data_dir):
    """Load features in ego-net datasets."""
    with open(data_dir / f'{u}.egofeat') as fh:
        egofeat = np.array([int(i) for i in fh.read().strip().split()])
    with open(data_dir / f'{u}.feat') as fh:
        feats = fh.read().strip().split('\n')
        feats = np.array([[int(i) for i in x.split()] for x in feats])
    with open(data_dir / f'{u}.featnames') as fh:
        featnames = fh.read().strip().split('\n')
        featnames = [' '.join(x.split(' ')[1:]) for x in featnames]
    nodelist = [u] + list(feats[:, 0])
    feats = np.concatenate([egofeat[None, :], feats[:, 1:]], axis=0)
    return nodelist, feats, featnames


def load_ego_circles(u, data_dir):
    """Load circles in ego-net datasets."""
    with open(data_dir / f'{u}.circles') as fh:
        circles = fh.read().strip().split('\n')
        circles = [[int(i) for i in x.split()[1:]] for x in circles]
    return circles


def stack_ego_nets(data_dir):
    """stack ego-nets in snap ego-net datasets."""
    ego_users = {int(x.name.split('.')[0]) for x in data_dir.iterdir() if x.suffix == '.edges'}
    ego_users = sorted(ego_users)
    n_nets = len(ego_users)
    all_edges = []
    all_circles = []
    node_feats = {}
    offset = 0
    for u in tqdm.tqdm(ego_users, position=0):
        edges = load_ego_edges(u, data_dir)
        nl, fs, fns = load_ego_feats(u, data_dir)
        circles = load_ego_circles(u, data_dir)
        circles = [x for x in circles if len(x) >= 3]
        if len(edges) == 0 or len(circles) == 0:
            continue
        # encode
        nodes = set(edges.ravel())
        mapping = {u: i + offset for i, u in enumerate(sorted(nodes))}
        offset += len(nodes)
        edges = np.asarray([[mapping[u], mapping[v]] for u, v in edges])
        circles = [[mapping[i] for i in x if i in mapping] for x in circles]
        for i, feat in zip(nl, fs):
            if i in mapping:
                node_feats[mapping[i]] = {fns[k] for k in np.where(feat)[0]}
        all_edges.append(edges)
        all_circles.extend(circles)
    all_edges = np.concatenate(all_edges)
    all_edges = {(a, b) if a < b else (b, a) for a, b in all_edges}
    all_edges = np.asarray(list(all_edges))
    all_circles = [x for x in all_circles if len(x) >= 3]
    # encode feats
    feat_counter = collections.Counter()
    feat_counter.update([f for x in node_feats.values() for f in x])
    valid_feats = {f for f, v in feat_counter.items() if v >= 10}
    feat_mapping = {f: i for i, f in enumerate(valid_feats)}
    node_feats = {k: sorted({feat_mapping[f] for f in fs if f in feat_mapping})
                  for k, fs in node_feats.items()}
    return all_edges, all_circles, node_feats, n_nets


def save_ego_nets(edges, comms, nodefeats, name, data_folder):
    """Write the ego-net dataset into 3 files."""
    root = pathlib.Path(data_folder) / name
    with open(root / f'com-{name}.ungraph.txt', 'w') as fh:
        s = '\n'.join([f'{a} {b}' for a, b in edges])
        fh.write(s)
    with open(root / f'com-{name}.cmty.txt', 'w') as fh:
        s = '\n'.join([' '.join([str(i) for i in x]) for x in comms])
        fh.write(s)
    with open(root / f'com-{name}.nodefeat.txt', 'w') as fh:
        nodefeats = sorted(nodefeats.items(), key=lambda x: x[0])
        s = '\n'.join([f'{u} {" ".join([str(i) for i in fs])}' for u, fs in nodefeats])
        fh.write(s)


def preprocess_ego_nets(name, data_folder):
    """Extract ego-nets from the tar file and stack the nets."""
    root = pathlib.Path(data_folder)
    data_dir = root / name / name
    if not data_dir.exists():
        # untar
        with tarfile.open(root / f'{name}/{name}.tar.gz', 'r:gz') as fh:
            fh.extractall(root / name)
    edges, circles, nodefeats, _ = stack_ego_nets(data_dir)
    save_ego_nets(edges, circles, nodefeats, name, data_folder)


def load_snap_ego_data(name, data_folder):
    """Load the pre-processed snap ego-net datasets."""
    root = pathlib.Path(data_folder)
    with open(root / f'{name}/com-{name}.ungraph.txt') as fh:
        edges = fh.read().strip().split('\n')
        edges = np.array([[int(i) for i in e.split()] for e in edges])
    with open(root / f'{name}/com-{name}.cmty.txt') as fh:
        comms = fh.read().strip().split('\n')
        comms = [[int(i) for i in x.split()] for x in comms]
    with open(root / f'{name}/com-{name}.nodefeat.txt') as fh:
        nodefeats = fh.read().strip().split('\n')
        nodefeats = [[int(i) for i in x.split()] for x in nodefeats]
        nodefeats = {k: list(v) for k, *v in nodefeats}
    return edges, comms, nodefeats


def load_snap_comm_data(name, data_folder):
    """Load the snap comm datasets."""
    root = pathlib.Path(data_folder)
    with gzip.open(root / f'{name}/com-{name}.ungraph.txt.gz', 'rt') as fh:
        edges = fh.read().strip().split('\n')[4:]
    edges = [[int(i) for i in e.split()] for e in edges]
    edges = [[u, v] if u < v else [v, u] for u, v in edges if u != v]
    nodes = {i for x in edges for i in x}
    mapping = {u: i for i, u in enumerate(sorted(nodes))}
    edges = np.asarray([[mapping[u], mapping[v]] for u, v in edges])
    with gzip.open(root / f'{name}/com-{name}.top5000.cmty.txt.gz', 'rt') as fh:
        comms = fh.readlines()
    comms = [[mapping[int(i)] for i in x.split()] for x in comms]
    return edges, comms, mapping


def extract_subdataset(edges, comms, nodes, nodefeats=None):
    """Extract the sub-dataset from the full net."""
    nodes = set(nodes)
    edges = np.array([[a, b] for a, b in edges if a in nodes and b in nodes])
    mapping = {u: i for i, u in enumerate(np.unique(edges.ravel()))}
    edges = [[mapping[a], mapping[b]] for a, b in edges]
    edges = np.array(edges)
    comms = [{mapping[i] for i in x if i in mapping} for x in comms]
    comms = [x for x in comms if len(x) >= 3]
    if nodefeats is None:
        return edges, comms, None, mapping
    else:
        node_feats = {mapping[k]: v for k, v in nodefeats.items() if k in mapping}
        feat_counter = collections.Counter()
        feat_counter.update([f for x in node_feats.values() for f in x])
        valid_feats = {f for f, v in feat_counter.items() if v >= 10}
        feat_mapping = {f: i for i, f in enumerate(valid_feats)}
        node_feats = {k: sorted({feat_mapping[f] for f in fs if f in feat_mapping})
                      for k, fs in node_feats.items()}
        return edges, comms, node_feats, mapping


def save_dataset(edges, comms, name, data_folder='.', nodefeats=None):
    """Write the comm data into files."""
    root = pathlib.Path(data_folder) / 'processed'
    root.mkdir(exist_ok=True, parents=True)
    with open(root / f'{name}.ungraph.txt', 'w') as fh:
        s = '\n'.join([f'{a} {b}' for a, b in edges])
        fh.write(s)
    with open(root / f'{name}.cmty.txt', 'w') as fh:
        s = '\n'.join([' '.join([str(i) for i in x]) for x in comms])
        fh.write(s)
    if nodefeats is not None:
        with open(root / f'{name}.nodefeat.txt', 'w') as fh:
            nodefeats = sorted(nodefeats.items(), key=lambda x: x[0])
            s = '\n'.join([f'{u} {" ".join([str(i) for i in fs])}' for u, fs in nodefeats])
            fh.write(s)


def process_snap(name, p=95, data_folder='raw_data', seed=42):
    """Extract sub-graphs from snap datasets."""
    max_size_p90 = dict(dblp=16, amazon=30, youtube=25, facebook=72, twitter=34)
    max_size_p95 = dict(dblp=21, amazon=42, youtube=48, facebook=129, twitter=48)
    info_str = f'[{name}]\n'
    if name in {'facebook', 'twitter'}:
        edges, comms, nodefeats = load_snap_ego_data(name, data_folder)
    else:
        edges, comms, _ = load_snap_comm_data(name, data_folder)
        nodefeats = None
    graph = Graph(edges)
    n_nodes, n_edges = edges.max() + 1, len(edges)
    info_str += f'# nodes = {n_nodes:,}\n# edges = {n_edges:,}\n'
    if nodefeats is not None:
        n_feats = len({i for x in nodefeats.values() for i in x})
        info_str += f'# feats = {n_feats:,}\n'
    if p == 90:
        th = max_size_p90[name]
    elif p == 95:
        th = max_size_p95[name]
    else:
        raise ValueError('invalid percentile')
    comms = [sorted(x) for x in comms if len(x) <= th]
    rng = np.random.RandomState(seed)
    comms = rng.permutation(comms)
    info_str += f'# comms(filtered <= {th}) = {len(comms):,}\n'
    # 1-ego
    info_str += '=' * 20 + '\n'
    info_str += f'[{name}-1: subgraph with nodes in comms\' 1-ego net]\n'
    nodes_in_ego1 = graph.k_ego(nodes_in_comms, 1)
    sub_edges, sub_comms, sub_nf, _ = extract_subdataset(edges, comms, nodes_in_ego1, nodefeats)
    save_dataset(sub_edges, sub_comms, f'{name}-1.{p}', data_folder, sub_nf)
    info_str += f'# nodes = {sub_edges.max() + 1:,}\n# edges = {len(sub_edges):,}\n'
    if sub_nf is not None:
        info_str += f'# feats = {len({i for x in sub_nf.values() for i in x}):,}\n'
    root = pathlib.Path(data_folder) / 'processed'
    with open(root / f'{name}.{p}.info.txt', 'w') as fh:
        fh.write(info_str)


if __name__ == '__main__':
    data_folder = 'raw_data'
    preprocess_ego_nets('facebook', data_folder)
    preprocess_ego_nets('twitter', data_folder)
    for name in ['dblp', 'amazon', 'youtube', 'facebook', 'twitter']:
        print(name)
        process_snap(name, p=90, data_folder=data_folder)
    print('Done!')
    print('Move raw_data/processed/ -> datasets/')
    shutil.move('raw_data/processed', './')
    os.rename('processed', 'datasets')
