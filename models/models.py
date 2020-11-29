from itertools import combinations

import torch
import torch.nn as nn
from models.mlp import MLP
from torch_geometric.nn import GCNConv, TAGConv, SAGEConv, GINConv, GATConv


class GNNModel(nn.Module):
    def __init__(self, layers, in_features, hidden_features, out_features, raw_features, use_de, prop_depth,
                 dropout=0.0, model_name='DE-SAGE'):
        super(GNNModel, self).__init__()
        self.num_layers = layers
        self.in_features, self.hidden_features, self.out_features = in_features, hidden_features, out_features
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.use_de = use_de
        self.use_raw = raw_features[0]
        self.raw_features = raw_features[-1]
        self.model_name = model_name

        Layer = self.get_layer_class()
        self.layers = nn.ModuleList()

        if self.model_name == 'DE-GNN':
            self.layers.append(Layer(in_channels=in_features, out_channels=hidden_features, K=prop_depth))
        elif self.model_name == 'GIN':
            self.layers.append(
                Layer(MLP(num_layers=2, input_dim=in_features, hidden_dim=hidden_features, output_dim=hidden_features)))
        else:
            self.layers.append(Layer(in_channels=in_features, out_channels=hidden_features))

        if layers > 1:
            for i in range(layers - 1):
                if self.model_name == 'DE-GNN':
                    self.layers.append(Layer(in_channels=hidden_features, out_channels=hidden_features, K=prop_depth))
                elif self.model_name == 'GIN':
                    self.layers.append(Layer(MLP(num_layers=2, input_dim=hidden_features, hidden_dim=hidden_features,
                                                 output_dim=hidden_features)))
                elif self.model_name == 'GAT':
                    self.layers.append(Layer(in_channels=hidden_features, out_channels=hidden_features, heads=8))
                else:
                    # for GCN and SAGE
                    self.layers.append(Layer(in_channels=hidden_features, out_channels=hidden_features))
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_features) for _ in range(layers)])
        self.concat_norm = nn.LayerNorm(2 * hidden_features)
        self.merger = nn.Linear(3 * hidden_features, hidden_features)

        if self.use_raw == 'concat':
            self.mlp_layer = MLP(num_layers=2, input_dim=self.raw_features.shape[1], hidden_dim=hidden_features,
                                 output_dim=hidden_features)
            self.feed_forward = FeedForwardNetwork(2 * hidden_features, out_features)
        else:
            self.feed_forward = FeedForwardNetwork(hidden_features, out_features)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        for i, layer in enumerate(self.layers):
            if self.model_name in {'GIN', 'GAT', 'DE-SAGE'}:
                x = layer(x, edge_index)
            else:
                x = layer(x, edge_index, edge_weight=None)
            x = self.act(x)
            x = self.dropout(x)  # [n_nodes, mini_batch, input_dim]
            if self.model_name in {'DE-GNN'}:
                x = self.layer_norms[i](x)
        x = self.get_minibatch_embeddings(x, batch)

        if self.use_raw == 'concat':
            old_set_indices = batch.old_set_indices
            node_features = self.raw_features[old_set_indices].squeeze().to(x.device)
            x_raw = self.mlp_layer(node_features)
            x = self.concat_norm(torch.cat((x, x_raw), dim=1))

        x = self.feed_forward(x)
        return x

    def get_minibatch_embeddings(self, x, batch):
        device = x.device
        set_indices, batch, num_graphs = batch.set_indices, batch.batch, batch.num_graphs
        num_nodes = torch.eye(num_graphs)[batch].to(device).sum(dim=0)
        zero = torch.tensor([0], dtype=torch.long).to(device)
        index_bases = torch.cat([zero, torch.cumsum(num_nodes, dim=0, dtype=torch.long)[:-1]])
        index_bases = index_bases.unsqueeze(1).expand(-1, set_indices.size(-1))
        assert (index_bases.size(0) == set_indices.size(0))
        set_indices_batch = index_bases + set_indices
        # print('set_indices shape', set_indices.shape, 'index_bases shape', index_bases.shape, 'x shape:', x.shape)
        x = x[set_indices_batch]  # shape [B, set_size, F]
        x = self.pool(x)
        return x

    def pool(self, x):
        # for node classification, the size of set S is always 1
        if x.size(1) == 1:
            return torch.squeeze(x, dim=1)
        # use mean/diff/max to pool each set's representations, for link prediction
        x_diff = torch.zeros_like(x[:, 0, :], device=x.device)
        for i, j in combinations(range(x.size(1)), 2):
            x_diff += torch.abs(x[:, i, :] - x[:, j, :])
        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1)[0]
        x = self.merger(torch.cat([x_diff, x_mean, x_max], dim=-1))
        return x

    def get_layer_class(self):
        layer_dict = {'DE-GNN': TAGConv, 'DE-SAGE': SAGEConv, 'GCN': GCNConv, 'GAT': GATConv, 'GIN': GINConv}
        # TAGConv essentially sums up GCN layerwise outputs, can use GCN instead
        Layer = layer_dict.get(self.model_name)
        if Layer is None:
            raise NotImplementedError(f'Unknown model name: {self.model_name}')
        return Layer

    def short_summary(self):
        return f'Model: {self.model_name}, #layers: {self.layers}, in_features: {self.in_features}, ' \
               f'hidden_features: {self.hidden_features}, out_features: {self.out_features}'


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_features, out_features, act=nn.ReLU(), dropout=0.0):
        super(FeedForwardNetwork, self).__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.layer1 = nn.Sequential(nn.Linear(in_features, in_features), self.act, self.dropout)
        self.layer2 = nn.Sequential(nn.Linear(in_features, out_features), nn.LogSoftmax(dim=-1))

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return x

    def reset_parameters(self):
        self.layer1[0].reset_parameters()
        self.layer2[0].reset_parameters()


class MLPModel(nn.Module):
    def __init__(self, layers, in_features, hidden_features, out_features, dropout=0.0, model_name='MLP'):
        super(MLPModel, self).__init__()
        self.model_name = model_name
        self.num_layers = layers
        self.in_features, self.hidden_features, self.out_features = in_features, hidden_features, out_features
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.Sequential(MLP(layers, in_features, hidden_features, out_features), nn.LogSoftmax(dim=-1))

    def forward(self, inputs):
        x = self.layers[0](inputs)
        return x

    def reset_parameters(self):
        self.layers[0].reset_parameters()

    def short_summary(self):
        return f'Model: {self.model_name}, #layers: {self.layers}, in_features: {self.in_features}, ' \
               f'hidden_features: {self.hidden_features}, out_features: {self.out_features}'
