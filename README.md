

# Distance Encoding for GNN on Node Classification

This repository is a subbranch of the official PyTorch implementation of [*DEGNN*](https://github.com/snap-stanford/distance-encoding) for classification task of nodes with community-type labels and structural-type labels. This version introduces the following new features:

- the framework is able to repeatedly train instances of models without regenerating graph samples (param `repeat`);
- the procedure of subgraph extraction and the number of GNN layers are decoupled. The model now can run on subgraphs with arbitrary number of hops, but without the limit of binding the extraction hops with the total number of layer propagation in GNNs (param `num_hop`);
- sparse matrices are utilized to accelerate the computation of DE, especially for random-walk based features.

## Installation
Requirements: Python >= 3.8, [Anaconda3](https://www.anaconda.com/)

- Update conda:
```bash
conda update -n base -c defaults conda
```

- Install basic dependencies to virtual environment and activate it: 
```bash
conda env create -f environment.yml
conda activate dlg-env
```

- Example commends of installation for PyTorch (>= 1.7.0) and torch-geometric (>=1.6.0) with CUDA 10.1:
```bash
conda install pytorch=1.7.0 torchvision cudatoolkit=10.1 -c pytorch
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-geometric
```
For more details, please refer to the [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). The code of this repository is lately tested with Python 3.8.2 + PyTorch 1.7.0 + torch-geometric 1.6.1.

## Quick Start

- To train **Model-RW** (random-walk variant) for node classification on Chameleon (s-type labels) :
```bash
python main.py --model DE-SAGE --dataset chameleon --use_de --de_feature rw --max_rw 3 --use_raw init --layer 1 --hidden_dim 32 --epoch 300 --bs 32 --dropout 0.4 --test_ratio 0.2 --repeat 5
```

This model has 1 hidden layer and 32-dimensional hidden features. The input is consist of raw features (added in the first layer, `init`) and structural features (node degrees and DE-RW with maximum step of 3; for DE_SPD, specify `de_feature` as sp and set the value of `max_sp` ). It will train for 300 epochs with the dropout rate 0.4, and repeat 5 times to obtain the average. Note that `test_ratio`  here represents the ratio for both validation set and the test set. In this case, the split of train/val/test set is  60/20/20.

- To train **Model-Base** (without DE) for node classification on Cora (c-type labels):
```bash
python main.py --dataset cora --use_raw init --layer 2 --hidden_dim 32 --prop_depth 1 --num_hop 2 --epoch 500 --bs 64 --l2 1e-4 --patience 50
```

This model has 2 hidden layers and 32-dimensional hidden features. The input includes node degrees and raw features. It will train for 500 epochs with the weight decay of 1e-4. `num_hop` is number of hops for subgraph extraction. In general, its value should be set  `>=prop_depth*layers`; otherwise, it may deteriorate model capability. `patience` is the indicator for early stopping if the best metric on validation set has not been improved for certain epochs.

- Based on the combinations of `use_raw(none, init, concat)`, `no_degree`, `use_de`, and `feature(sp, rw)`, we can construct different type of variants for DE-GNN.

- All detailed training logs can be found at `<log_dir>/<dataset>/<training-time>.log`. For each instance, one-line summary will be reported to `<log_dir>/result_summary.log` after model training.

## Usage Summary
```
usage: Interface for DE-GNN framework (Node Classification)
       [-h] [--dataset DATASET] [--model {DE-SAGE,DE-GNN,GCN,GAT,GIN}]
       [--layers LAYERS] [--hidden_dim HIDDEN_DIM] [--data_usage DATA_USAGE]
       [--test_ratio TEST_RATIO] [--metric {acc,auc}] [--seed SEED]
       [--gpu GPU] [--directed DIRECTED] [--prop_depth PROP_DEPTH]
       [--num_hop NUM_HOP] [--no_degree] [--use_raw USE_RAW] [--use_de]
       [--de_feature DE_FEATURE] [--max_rw MAX_RW] [--max_sp MAX_SP]
       [--optimizer OPTIMIZER] [--epoch EPOCH] [--bs BS] [--lr LR]
       [--dropout DROPOUT] [--l2 L2] [--patience PATIENCE] [--repeat REPEAT]
       [--log_dir LOG_DIR] [--summary_file SUMMARY_FILE] [--debug]
```

## Optional Arguments
```
    -h, --help            show this help message and exit
  --dataset DATASET     dataset name
  --model {DE-SAGE,DE-GNN,GCN,GAT,GIN}
                        base model to use
  --layers LAYERS       number of layers
  --hidden_dim HIDDEN_DIM
                        hidden dimension
  --data_usage DATA_USAGE
                        use partial dataset
  --test_ratio TEST_RATIO
                        ratio of the test / valid against whole
  --metric {acc,auc}    metric for evaluating performance
  --seed SEED           seed to initialize all the random modules
  --gpu GPU             gpu id
  --directed DIRECTED   (Currently unavailable) whether to treat the graph as
                        directed
  --prop_depth PROP_DEPTH
                        propagation depth (number of hops) for one layer
  --num_hop NUM_HOP     total number of hops for subgraph extraction
  --no_degree           whether to use node degree as node feature
  --use_raw USE_RAW     which way to use node attributes
  --use_de              whether to use distance encoding as node feature
  --de_feature DE_FEATURE
                        distance encoding category: shortest path or random
                        walk (landing probabilities)
  --max_rw MAX_RW       maximum steps for random walk feature
  --max_sp MAX_SP       greatest distance for shortest path feature
  --optimizer OPTIMIZER
                        optimizer to use
  --epoch EPOCH         number of epochs to train
  --bs BS               minibatch size
  --lr LR               learning rate
  --dropout DROPOUT     dropout rate
  --l2 L2               l2 regularization (weight decay)
  --patience PATIENCE   early stopping steps
  --repeat REPEAT       number of training instances to repeat
  --log_dir LOG_DIR     log directory
  --summary_file SUMMARY_FILE
                        brief summary of training results
  --debug               whether to use debug mode
```

