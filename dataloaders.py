import json
from copy import deepcopy
from itertools import compress

import networkx as nx
import numpy as np
import torch
import torch_geometric.utils as tgu
from scipy.sparse import csr_matrix, vstack, diags
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader, Data
from tqdm import tqdm


def read_label(path):
    labels = []
    node_id_mapping = dict()
    fin_labels = open(path + 'labels.txt')
    # relabel node from 0 and save the mapping to node_id_mapping
    for new_id, line in enumerate(fin_labels.readlines()):
        old_id, label = line.strip().split()
        labels.append(int(label))
        node_id_mapping[old_id] = new_id
    fin_labels.close()
    return labels, node_id_mapping


def read_edges(path, node_id_mapping):
    edges = []
    fin_edges = open(path + 'edges.txt')
    for line in fin_edges.readlines():
        node1, node2 = line.strip().split()[:2]
        edges.append([node_id_mapping[node1], node_id_mapping[node2]])
    fin_edges.close()
    return edges


def read_features(path, node_id_mapping):
    # load raw features from json as a dictionary
    try:
        with open(path + 'features.json', 'r') as load_f:
            dict_features = json.load(load_f)
    except:
        raise FileNotFoundError
    features = [dict_features[w] for w in sorted(node_id_mapping, key=node_id_mapping.get)]
    features = np.asarray(features)
    return features


def read_file(args, logger):
    dataset = args.dataset
    di_flag = args.directed
    if dataset in ['brazil-airports', 'europe-airports', 'usa-airports', 'foodweb', 'karate', 'chameleon', 'Actor',
                   'Squirrel', 'cornell', 'texas', 'wisconsin', 'cora', 'citeseer', 'pubmed']:
        task = 'node_classification'
    else:
        raise ValueError('dataset not found')

    directory = f'./data/{task}/{dataset}/'
    if dataset in ['Actor', 'Squirrel']:
        graph_adjacency_list_file_path = directory + 'out1_graph_edges.txt'
        graph_node_features_and_labels_file_path = directory + 'out1_node_feature_label.txt'

        # read node features and labels
        node_features = []
        labels = []
        node_id_mapping = {}
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for new_id, line in enumerate(graph_node_features_and_labels_file):
                line = line.rstrip().split('\t')
                old_id = line[0]
                assert (len(line) == 3)
                assert (old_id not in node_id_mapping.keys())
                node_id_mapping[old_id] = new_id
                if dataset == 'Actor':
                    node_feature = np.zeros(932, dtype=np.uint8)
                    node_feature[np.array(line[1].split(','), dtype=np.uint16)] = 1
                else:
                    node_feature = np.array(line[1].split(','), dtype=np.uint8)

                node_features.append(node_feature)
                label = int(line[2])
                labels.append(label)

            node_features = np.stack(node_features, axis=0)

        # read edges
        edges = []
        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                edges.append([node_id_mapping[line[0]], node_id_mapping[line[1]]])

        # load dataset and save as type of nx.Graph
        if not di_flag:
            G = nx.Graph(edges)
        else:
            G = nx.DiGraph(edges)
        attributes = np.zeros((G.number_of_nodes(), 1), dtype=np.float32)

        # degree features
        if not args.no_degree:
            attributes += np.expand_dims(np.log(get_degrees(G) + 1), 1).astype(np.float32)

        # node raw features
        if args.use_raw != 'None':
            if args.use_raw == 'init':
                if not args.no_degree:
                    attributes = np.concatenate([attributes, node_features], axis=1)
                else:
                    attributes = node_features
            elif args.use_raw == 'concat':
                node_features = torch.tensor(node_features, dtype=torch.float32)
            else:
                raise NotImplementedError
        else:
            node_features = None

    else:
        labels, node_id_mapping = read_label(directory)
        edges = read_edges(directory, node_id_mapping)

        # load dataset and save as type of nx.Graph
        if not di_flag:
            G = nx.Graph(edges)
        else:
            G = nx.DiGraph(edges)
        attributes = np.zeros((G.number_of_nodes(), 1), dtype=np.float32)

        # degree features
        if not args.no_degree:
            attributes += np.expand_dims(np.log(get_degrees(G) + 1), 1).astype(np.float32)

        # node raw features
        if args.use_raw != 'None':
            node_features = read_features(directory, node_id_mapping)
            if args.use_raw == 'init':
                if not args.no_degree:
                    attributes = np.concatenate([attributes, node_features], axis=1)
                else:
                    attributes = node_features
            elif args.use_raw == 'concat':
                node_features = torch.tensor(node_features, dtype=torch.float32)
            else:
                raise NotImplementedError
        else:
            node_features = None

    G.graph['attributes'] = attributes
    logger.info(
        f'Read in {dataset} for {task} - number of nodes: {G.number_of_nodes()}, number of edges: {G.number_of_edges()}, '
        f'number of labels: {len(labels) if labels is not None else 0}. Directed: {di_flag}')
    labels = np.array(labels) if labels is not None else None
    return (G, labels), node_features


def get_data(G, args, labels, logger):
    G = deepcopy(G)  # to make sure original G is unchanged
    if args.use_de:
        feature_flags = ('sp' in args.de_feature, 'rw' in args.de_feature)
    else:
        feature_flags = (False, False)

    G, labels, set_indices, (train_mask, val_test_mask) = generate_samples(G, labels, args, logger)

    # in order to get the correct degree normalization for the subgraph, num_hop should add 1
    data_list = extract_subgaphs(G, labels, set_indices, num_hop=args.num_hop + 1, feature_flags=feature_flags,
                                 max_sprw=(args.max_sp, args.max_rw), mask=train_mask, logger=logger)
    return data_list, labels


def generate_samples(G, labels, args, logger):
    if labels is None:
        raise Exception('Labels unavailable.')
    else:
        # training on nodes with labels
        logger.info('Labels provided (node-level task).')
        assert (G.number_of_nodes() == labels.shape[0])
        n_samples = int(round(labels.shape[0] * args.data_usage))
        set_indices = np.random.choice(G.number_of_nodes(), n_samples, replace=False)
        labels = labels[set_indices]
        set_indices = np.expand_dims(set_indices, 1)
        train_mask, val_test_mask = split_dataset(set_indices.shape[0], test_ratio=2 * args.test_ratio, stratify=labels)
    logger.info(f'Generate {set_indices.shape[0]} train+val+test instances in total. data_usage: {args.data_usage}.')
    return G, labels, set_indices, (train_mask, val_test_mask)


def extract_subgaphs(G, labels, set_indices, num_hop, feature_flags, max_sprw, mask, logger):
    # deal with adj and features
    logger.info('Encode positions ... ')
    data_list = []
    n_samples = set_indices.shape[0]

    # inductive settings -> use mask to induce subgraph
    # prepare edges for subgraph extraction and treated as undirected
    G_edge_idx = torch.tensor(list(G.edges)).long().t().contiguous()
    G_edge_idx = torch.cat([G_edge_idx, G_edge_idx[[1, 0]]], dim=-1)

    for sample_i in tqdm(range(n_samples)):
        data = get_data_sample(G, set_indices[sample_i], G_edge_idx, num_hop, feature_flags, max_sprw,
                               label=labels[sample_i] if labels is not None else None)
        data_list.append(data)
    return data_list


def get_data_sample(G, set_index, edge_index, num_hop, feature_flags, max_sprw, label):
    set_index = list(set_index)
    sp_flag, rw_flag = feature_flags
    max_sp, max_rw = max_sprw

    # extract subgraph from the root node with num_hop; for node classification, len(set_index)=1
    subgraph_node_old_index, new_edge_index, new_set_index, edge_mask = tgu.k_hop_subgraph(
        torch.tensor(set_index).long(), num_hop, edge_index, num_nodes=G.number_of_nodes(), relabel_nodes=True)

    # reconstruct networkx graph object for the extracted subgraph
    num_nodes = subgraph_node_old_index.size(0)
    new_G = nx.from_edgelist(new_edge_index.t().numpy().astype(dtype=np.int32), create_using=type(G))
    new_G.add_nodes_from(np.arange(num_nodes, dtype=np.int32))  # to add disconnected nodes
    assert (new_G.number_of_nodes() == num_nodes)

    # assemble x from features to x_list
    x_list = []
    attributes = G.graph['attributes']
    if attributes is not None:
        new_attributes = torch.tensor(attributes, dtype=torch.float32)[subgraph_node_old_index]
        if new_attributes.dim() < 2:
            new_attributes.unsqueeze_(1)
        x_list.append(new_attributes)

    if sp_flag:
        features_sp_sample = gen_sp_features(new_G, new_set_index.numpy(), max_sp=max_sp)
        features_sp_sample = torch.from_numpy(features_sp_sample).float()
        x_list.append(features_sp_sample)

    if rw_flag:
        # use sparse matrix for computing the landing probabilities [n_nodes, n_nodes]
        adj = nx.adjacency_matrix(new_G, nodelist=np.arange(new_G.number_of_nodes(), dtype=np.int32))
        features_rw_sample = gen_rw_features(adj, new_set_index.numpy(), rw_depth=max_rw)
        features_rw_sample = torch.from_numpy(features_rw_sample).float()
        x_list.append(features_rw_sample)

    x = torch.cat(x_list, dim=-1)
    y = torch.tensor([label], dtype=torch.long) if label is not None else torch.tensor([0], dtype=torch.long)
    new_set_index = new_set_index.long().unsqueeze(0)

    return Data(x=x, edge_index=new_edge_index, y=y, set_indices=new_set_index,
                old_set_indices=torch.tensor(set_index).long().unsqueeze(0))


def gen_sp_features(G, node_set, max_sp):
    dim = max_sp + 2
    set_size = len(node_set)
    sp_length = np.ones((G.number_of_nodes(), set_size), dtype=np.int32) * -1
    for i, node in enumerate(node_set):
        for node_ngh, length in nx.shortest_path_length(G, source=node).items():
            sp_length[node_ngh, i] = length
    sp_length = np.minimum(sp_length, max_sp)
    onehot_encoding = np.eye(dim, dtype=np.float64)  # [n_features, n_features]
    features_sp = onehot_encoding[sp_length].sum(axis=1)
    return features_sp


def gen_rw_features(adj, root, rw_depth):
    epsilon = 1e-6
    norm = diags(1 / (adj.sum(axis=1) + epsilon).A.ravel())
    # W = A*D^-1
    adj_n = norm * adj
    list_rw = [csr_matrix(np.identity(adj_n.shape[0])[root])]
    for _ in range(rw_depth):
        rw = list_rw[-1].dot(adj_n)
        list_rw.append(rw)
    if len(root) < 2:
        features_rw = vstack(list_rw).T.todense()
    else:
        pooling = [csr_matrix(m.sum(axis=0)) for m in list_rw]
        features_rw = vstack(pooling).T.todense()
    return features_rw.astype(np.float32)


def gen_dataloader(datalist, test_ratio, bs, logger, labels=None, splits_file_path = None):
    n_samples = len(datalist)

    if splits_file_path:
        print('Using fixed split')
        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']
            val_index = np.where(val_mask == 1)[0]
            test_index = np.where(test_mask == 1)[0]
            val_ratio = len(val_index) / n_samples
            test_ratio = len(test_index) / n_samples

    else:
        val_ratio = test_ratio
        train_indices, val_test_indices = split_dataset(list(range(n_samples)), test_ratio=2 * test_ratio, stratify=labels)

        val_test_labels = np.array(labels)[val_test_indices]
        val_indices, test_indices = split_dataset(val_test_indices, test_ratio=int(0.5 * len(val_test_indices)),
                                                  stratify=val_test_labels)

        train_mask = get_mask(train_indices, n_samples)
        val_mask = get_mask(val_indices, n_samples)
        test_mask = get_mask(test_indices, n_samples)

    assert sum(train_mask) + sum(val_mask) + sum(test_mask) == n_samples

    train_set = list(compress(datalist, train_mask))
    val_set = list(compress(datalist, val_mask))
    test_set = list(compress(datalist, test_mask))

    train_loader, val_loader, test_loader = load_datasets(train_set, val_set, test_set, bs)

    logger.info(f'Train size :{len(train_set)}, val size: {len(val_set)}, test size: {len(test_set)}, '
                f'val ratio: {val_ratio}, test ratio: {test_ratio}')

    # return {'train': train_loader, 'val': val_loader, 'test': test_loader}
    return train_loader, val_loader, test_loader


def split_dataset(n_samples, test_ratio, stratify=None):
    flag = isinstance(n_samples, int)
    node_indices = list(range(n_samples)) if flag else n_samples

    try:
        train_indices, test_indices = train_test_split(node_indices, test_size=test_ratio, stratify=stratify)
    except:
        print('Dataset split changed to stratify = None')
        train_indices, test_indices = train_test_split(node_indices, test_size=test_ratio)

    if flag:
        train_mask = get_mask(train_indices, n_samples)
        test_mask = get_mask(test_indices, n_samples)
        return train_mask, test_mask
    else:
        return train_indices, test_indices


def load_datasets(train_set, val_set, test_set, bs, num_workers=0):
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def get_degrees(G):
    num_nodes = G.number_of_nodes()
    return np.array([G.degree[i] for i in range(num_nodes)])


def get_mask(idx, length):
    mask = np.zeros(length)
    mask[idx] = 1
    return np.array(mask, dtype=np.int8)


def retain_partial(indices, ratio):
    sample_i = np.random.choice(indices.shape[0], int(ratio * indices.shape[0]), replace=False)
    return indices[sample_i], sample_i
