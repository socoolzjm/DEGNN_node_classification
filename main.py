import argparse

from dataloaders import *
from log import *
from train import *


def main():
    parser = argparse.ArgumentParser('Interface for DE-GNN framework (Node Classification)')

    # general model and training setting
    parser.add_argument('--dataset', type=str, default='cora', help='dataset name')
    parser.add_argument('--model', type=str, default='DE-SAGE', help='base model to use',
                        choices=['DE-SAGE', 'DE-GNN', 'GCN', 'GAT', 'GIN'])
    parser.add_argument('--layers', type=int, default=2, help='number of layers')
    parser.add_argument('--hidden_dim', type=int, default=32, help='hidden dimension')
    parser.add_argument('--data_usage', type=float, default=1.0, help='use partial dataset')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='ratio of the test / valid against whole')
    parser.add_argument('--splits_file_path', help="Please give a splits file path")
    parser.add_argument('--metric', type=str, default='acc', help='metric for evaluating performance',
                        choices=['acc', 'auc'])
    parser.add_argument('--seed', type=int, default=0, help='seed to initialize all the random modules')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--directed', type=bool, default=False,
                        help='(Currently unavailable) whether to treat the graph as directed')

    # features and positional encoding
    parser.add_argument('--prop_depth', type=int, default=1, help='propagation depth (number of hops) for one layer')
    parser.add_argument('--num_hop', type=int, default=-1, help='total number of hops for subgraph extraction')
    parser.add_argument('--no_degree', default=False, action="store_true",
                        help='whether to use node degree as node feature')
    parser.add_argument('--use_raw', type=str, default='None', help='which way to use node attributes')
    parser.add_argument('--use_de', default=False, action="store_true",
                        help='whether to use distance encoding as node feature')
    parser.add_argument('--de_feature', type=str, default='sp',
                        help='distance encoding category: shortest path or random walk (landing probabilities)')
    # sp (shortest path) or rw (random walk)
    parser.add_argument('--max_rw', type=int, default=3, help='maximum steps for random walk feature')
    parser.add_argument('--max_sp', type=int, default=3, help='greatest distance for shortest path feature')

    # model training
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use')
    parser.add_argument('--epoch', type=int, default=500, help='number of epochs to train')
    parser.add_argument('--bs', type=int, default=32, help='minibatch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--l2', type=float, default=0., help='l2 regularization (weight decay)')
    parser.add_argument('--patience', type=int, default=50, help='early stopping steps')
    parser.add_argument('--repeat', type=int, default=0, help='number of training instances to repeat')

    # logging & debug
    parser.add_argument('--log_dir', type=str, default='./log/', help='log directory')
    parser.add_argument('--summary_file', type=str, default='result_summary.log',
                        help='brief summary of training results')
    parser.add_argument('--debug', default=False, action='store_true', help='whether to use debug mode')

    sys_argv = sys.argv
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    # check parsed-in parameters and set up logger
    arg_check(args)
    logger = set_up_log(args, sys_argv)
    set_random_seed(args)

    # load dataset from files
    (G, labels), raw_features = read_file(args, logger)
    # convert to Pytorch_geometric Data format
    datalist, labels = get_data(G, labels=labels, args=args, logger=logger)
    dim_in, dim_out = datalist[0].x.shape[-1], len(np.unique(labels))
    model = get_model(layers=args.layers, in_features=dim_in, out_features=dim_out,
                      raw_features=raw_features, prop_depth=args.prop_depth, args=args, logger=logger)

    if args.repeat > 0:
        dic_res = []
        for r in range(args.repeat):
            # randomly split dataset and construct dataloaders
            data = gen_dataloader(datalist, test_ratio=args.test_ratio, bs=args.bs, logger=logger, labels=labels,
                                  splits_file_path = args.splits_file_path)
            results = train_model(model, data, args, logger, repeat=r)
            dic_res.append([results[0] * 100, results[1] * 100])
            save_performance_result(args, logger, results, repeat=r)

        mean_acc = np.mean(dic_res, axis=0)
        std_acc = np.std(dic_res, axis=0)
        logger.info(f'{args.metric} ({args.repeat} Results, [w/, w/o]): '
                    f'mean {mean_acc[0]:.2f}, {mean_acc[1]:.2f}; std {std_acc[0]:.2f}, {std_acc[1]:.2f};')
    else:
        data = gen_dataloader(datalist, test_ratio=args.test_ratio, bs=args.bs, logger=logger, labels=labels)
        results = train_model(model, data, args, logger, repeat=args.repeat)
        save_performance_result(args, logger, results, repeat=args.repeat)


if __name__ == '__main__':
    main()
