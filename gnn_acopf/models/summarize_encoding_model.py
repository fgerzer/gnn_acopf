import torch
from torch import nn
import torch_geometric.nn as geom_nn
from gnn_acopf.models.power_base_model import PowerBaseModel, GraphAwareBatchNorm

from torch_scatter import scatter_mean, scatter_add
import copy


class EdgeModel(torch.nn.Module):
    def __init__(self, n_features, n_edge_features, hiddens, n_targets, residuals):
        super().__init__()
        self.residuals = residuals
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * n_features + n_edge_features, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, n_targets),
        )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = torch.cat([src, dest, edge_attr], 1)
        out = self.edge_mlp(out)
        if self.residuals:
            out = out + edge_attr
        return out


class NodeModel(torch.nn.Module):
    def __init__(self, n_features, n_edge_features, hiddens, n_targets, residuals):
        super(NodeModel, self).__init__()
        self.residuals = residuals
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(n_features + n_edge_features, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, n_targets),
        )
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(hiddens + n_features, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, n_targets),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[col], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        out =  self.node_mlp_2(out)
        if self.residuals:
            out = out + x
        return out

class SummarizedEncodingModel(PowerBaseModel):
    def __init__(self, n_features, n_targets, n_hiddens, n_edge_features, n_targets_edge, n_layers, batchnorm, residuals):
        super().__init__()
        self.build_model(n_features, n_targets, n_edge_features=n_edge_features,
                         n_hiddens=n_hiddens, n_targets_edge=n_targets_edge,
                         n_layers=n_layers, batchnorm=batchnorm, residuals=residuals)
        self.build_optimizer_scheduler()

    def build_layer(self, n_hiddens, batchnorm, residuals):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(n_hiddens, n_hiddens, n_hiddens, n_hiddens, residuals=residuals),
            node_model=NodeModel(n_hiddens, n_hiddens, n_hiddens, n_hiddens, residuals=residuals),
        )

    def build_output_layer(self, n_features, hiddens, n_targets):
        return nn.Sequential(
            nn.Linear(n_features, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, n_targets)
        )

    def build_model(self, n_features, n_targets, n_targets_edge, n_edge_features, n_hiddens, n_layers, batchnorm, residuals):
        self.encode_layers = self.build_encoding_layers(n_features, n_hiddens)
        self.encode_branch_attr = self._build_encode_layer(n_edge_features, n_hiddens)
        self.summarize_all_features = self._build_encode_layer(4 * n_hiddens, n_hiddens)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                self.build_layer(n_hiddens, batchnorm=batchnorm, residuals=residuals)
            )
            if batchnorm:
                self.layers.append(GraphAwareBatchNorm(n_hiddens))

        self.bus_output_layer = self.build_output_layer(n_hiddens, n_hiddens, n_targets["bus"])
        self.gen_output_layer = self.build_output_layer(n_hiddens + n_hiddens, n_hiddens, n_targets["gen"])
        self.branch_output_layer = self.build_output_layer(n_hiddens, n_hiddens, n_targets_edge)

    def build_encoding_layers(self, n_features, n_hiddens):
        encode_layers = nn.ModuleDict()
        for name, n_feat in n_features.items():
            encode_layers[name] = self._build_encode_layer(n_feat, n_hiddens)
        return encode_layers

    def _build_encode_layer(self, n_feat, n_hiddens):
        return nn.Sequential(
            nn.Linear(n_feat, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_hiddens)
        )

    def forward(self, data):
        data = copy.deepcopy(data)
        encoded_bus = self.encode_layers["bus"](data.bus)
        encoded_gen = self.encode_layers["gen"](data.gen)
        encoded_load = self.encode_layers["load"](data.load)
        encoded_shunt = self.encode_layers["shunt"](data.shunt)

        gen_as_bus = scatter_add(encoded_gen, data.gen_to_bus, dim=0, dim_size=data.bus.shape[0])
        load_as_bus = scatter_add(encoded_load, data.load_to_bus, dim=0, dim_size=data.bus.shape[0])
        shunt_as_bus = scatter_add(encoded_shunt, data.shunt_to_bus, dim=0, dim_size=data.bus.shape[0])

        x = torch.cat([encoded_bus, gen_as_bus, load_as_bus, shunt_as_bus], dim=1)
        x = self.summarize_all_features(x)
        edge_index = data.branch_index
        edge_attr = data.branch_attr
        edge_attr = self.encode_branch_attr(edge_attr)
        for i, layers in enumerate(self.layers):
            if isinstance(layers, GraphAwareBatchNorm):
                x = layers(x)
            else:
                x, edge_attr, _ = layers(x, edge_index, edge_attr=edge_attr)
        # Now, aggregate such that we get to undirected edges - simple take the mean of both
        # branch_attr = (data.branch_attr[:data.branch_attr.shape[0] // 2] + data.branch_attr[data.branch_attr.shape[0] // 2:]) / 2

        bus = x
        gen = torch.cat([x[data.gen_to_bus], encoded_gen], dim=1)
        branch_attr = edge_attr
        return {
            "target_bus": self.bus_output_layer(bus),
            "target_gen": self.gen_output_layer(gen),
            "target_branch": self.branch_output_layer(branch_attr)
        }
