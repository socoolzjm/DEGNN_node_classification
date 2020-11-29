import os
import random
import sys

import numpy as np

from models.models import *


def arg_check(args):
    if args.directed:
        raise Warning(f'dataset {args.dataset} is a directed network but currently treated as undirected.')

    num_prop = int(args.prop_depth * args.layers)
    if args.num_hop < 0:
        args.num_hop = num_prop
    elif args.num_hop < num_prop:
        raise Warning(
            f'maximum hop of subgraph (num_hop={args.num_hop}) is less than max number of propagation '
            f'(prop_depth*layer={num_prop}), which may deteriorate model capability.')


def set_random_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(args):
    return torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')


def get_optimizer(model, args):
    optim = args.optimizer
    if optim == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif optim == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise NotImplementedError


def get_model(layers, in_features, out_features, raw_features, prop_depth, args, logger):
    model_name = args.model
    if model_name in ['DE-SAGE', 'DE-GNN', 'GCN', 'GIN', 'GAT']:
        model = GNNModel(layers=layers, in_features=in_features, hidden_features=args.hidden_dim,
                         out_features=out_features, raw_features=(args.use_raw, raw_features),
                         use_de=args.use_de, prop_depth=prop_depth, dropout=args.dropout, model_name=model_name)
    else:
        return NotImplementedError
    logger.info(model.short_summary())
    return model


def estimate_storage(dataloaders, names, logger):
    total_gb = 0
    for dataloader, name in zip(dataloaders, names):
        dataset = dataloader.dataset
        storage = 0
        total_length = len(dataset)
        sample_size = 100
        for i in np.random.choice(total_length, sample_size):
            storage += (sys.getsizeof(dataset[i].x.storage()) + sys.getsizeof(dataset[i].edge_index.storage()) +
                        sys.getsizeof(dataset[i].y.storage())) + sys.getsizeof(dataset[i].set_indices.storage())
        gb = storage * total_length / sample_size / 1e9
        total_gb += gb
    logger.info(f'Data roughly takes {total_gb:.4f} GB in total')
    return total_gb
