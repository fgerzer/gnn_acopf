import torch
from torch import nn
import torch_geometric.nn as geom_nn
from torch_scatter import scatter_mean, scatter_add
import copy
from gnn_acopf.models.power_base_model import PowerBaseModel, GraphAwareBatchNorm
import numpy as np

class PowerNetLayer(nn.Module):
    def __init__(self, n_features, n_edge_features, hiddens, n_targets, n_edge_targets, batchnorm, residuals):
        super().__init__()
        subnet_def = {
            "bus": [n_features["bus"] + 3 * hiddens, hiddens],
            "shunt": [n_features["bus"] + n_features["shunt"], n_targets["shunt"]],
            "load": [n_features["bus"] + n_features["load"], n_targets["load"]],
            "gen": [n_features["bus"] + n_features["gen"], n_targets["gen"]],
            "branch": [2 * hiddens + n_edge_features, n_edge_targets],
            "bus_and_branch": [hiddens + n_edge_features, hiddens],
            "bus_and_neighbours": [2 * hiddens, n_targets["bus"]],
        }
        self.residuals = residuals
        self.batchnorm = batchnorm
        self.subnets = nn.ModuleDict({k: self.build_subnet(v[0], hiddens, v[1]) for k, v in subnet_def.items()})

    def build_subnet(self, n_features, hiddens, n_targets):
        layers = []
        layers.append(nn.Linear(n_features, hiddens))
        if self.batchnorm:
            # layers.append(torch.nn.BatchNorm1d(hiddens))
            layers.append(GraphAwareBatchNorm(hiddens))

        layers.append(nn.ReLU())
        layers.append(nn.Linear(hiddens, n_targets))
        if self.batchnorm:
            # layers.append(torch.nn.BatchNorm1d(n_targets))
            layers.append(GraphAwareBatchNorm(hiddens))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _scatter_items(self, orig_items, index, dim_size):
        new_matrix = scatter_add(orig_items, index, dim_size=dim_size, dim=0)
        return new_matrix

    def forward(self, data):
        bus = data.bus
        shunt = self.subnets["shunt"](torch.cat([data.shunt, bus[data.shunt_to_bus]], dim=-1))
        gen = self.subnets["gen"](torch.cat([data.gen, bus[data.gen_to_bus]], dim=-1))
        load = self.subnets["load"](torch.cat([data.load, bus[data.load_to_bus]], dim=-1))
        bus = self.subnets["bus"](torch.cat([
            bus,
            self._scatter_items(shunt, data.shunt_to_bus, bus.shape[0]),
            self._scatter_items(gen, data.gen_to_bus, bus.shape[0]),
            self._scatter_items(load, data.load_to_bus, bus.shape[0]),
        ], dim=-1))
        src, dest = data.branch_index
        branch = self.subnets["branch"](torch.cat([bus[src], data.branch_attr, bus[dest]], dim=-1))
        bus_neighbours = self.subnets["bus_and_branch"](torch.cat([bus[dest], data.branch_attr], dim=-1))
        bus_neighbours = scatter_add(bus_neighbours, src, dim=0, dim_size=bus.shape[0])
        bus = self.subnets["bus_and_neighbours"](torch.cat([bus_neighbours, bus], dim=-1))
        if self.residuals:
            bus = bus + data.bus
            shunt = shunt + data.shunt
            gen = gen + data.gen
            load = load + data.load
            branch = branch + data.branch_attr
        data.bus = bus
        data.shunt = shunt
        data.gen = gen
        data.load = load
        data.branch_attr = branch
        return data


class SimpleModel(PowerBaseModel):
    def __init__(self, n_features, n_targets, n_hiddens, n_edge_features, n_targets_edge, n_layers, batchnorm, residuals):
        super().__init__()
        self.build_model(n_features, n_targets, n_edge_features=n_edge_features,
                         n_hiddens=n_hiddens, n_targets_edge=n_targets_edge,
                         n_layers=n_layers, batchnorm=batchnorm, residuals=residuals)
        self.build_optimizer_scheduler()

    def build_layer(self, n_features, n_edge_features, hiddens, n_targets, n_edge_targets, batchnorm, residuals):
        return PowerNetLayer(n_features, n_edge_features, hiddens, n_targets, n_edge_targets, batchnorm=batchnorm,
                             residuals=residuals)

    def build_output_layer(self, n_features, hiddens, n_targets):
        return nn.Linear(n_features, n_targets)

    def build_model(self, n_features, n_targets, n_targets_edge, n_edge_features, n_hiddens, n_layers, batchnorm, residuals):
        last_features = {k: v for k, v in n_features.items()}
        last_edge_features = n_edge_features

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            new_features = {k: n_hiddens for k in last_features}
            new_edge_features = n_hiddens
            self.layers.append(
                self.build_layer(last_features, last_edge_features, n_hiddens, new_features, n_edge_targets=new_edge_features,
                                 batchnorm=batchnorm, residuals=i > 0 and residuals)
            )
            last_features = new_features
            last_edge_features = new_edge_features
        self.output_layers = nn.ModuleDict({
            t: self.build_output_layer(*[last_features[t], n_hiddens, n_targets[t]]) for t in n_targets
        })
        self.output_layers["branch"] = self.build_output_layer(last_edge_features + 2 * n_hiddens, n_hiddens, n_targets_edge)


    def forward(self, data):
        data = copy.deepcopy(data)
        for i, layers in enumerate(self.layers):
            data = layers(data)
        # Now, aggregate such that we get to undirected edges - simple take the mean of both
        # branch_attr = (data.branch_attr[:data.branch_attr.shape[0] // 2] + data.branch_attr[data.branch_attr.shape[0] // 2:]) / 2
        branch_attr = data.branch_attr
        src, dest = data.branch_index
        return {
            "target_bus": self.output_layers["bus"](data.bus),
            "target_gen": self.output_layers["gen"](data.gen),
            "target_branch": self.output_layers["branch"](
                torch.cat([data.bus[src], data.branch_attr, data.bus[dest]], dim=-1)
            )
        }


class PhonyModel(PowerBaseModel):
    def update_step(self, data, target):
        output = self(data)
        loss = 0
        return output, loss

    def forward(self, data):
        output_dict = {
            "target_bus": copy.deepcopy(data["target_bus"]),
            #"target_bus": torch.zeros_like(data["target_bus"]),
            "target_gen": copy.deepcopy(data["target_gen"]),
            # "target_gen": torch.zeros_like(data["target_gen"]),
            # "target_branch": copy.deepcopy(data["target_branch"])
            "target_branch": torch.zeros_like(data["target_branch"])
        }
        return output_dict

    def compute_loss(self, data, target):
        return 0


class MeanModel(PowerBaseModel):
    def __init__(self):
        self.stored_target_bus = []
        self.stored_target_gen = []
        self.stored_target_branch = []
        self.cache = None
        super().__init__()

    def update_step(self, data, target):
        self.stored_target_branch.append(target["target_branch"])
        self.stored_target_bus.append(target["target_bus"])
        self.stored_target_gen.append(target["target_gen"])
        self.cache = None
        output = self(data)
        loss = self.compute_loss(data, target)
        return output, loss

    def forward(self, data):
        if self.cache is None:
            busses = torch.mean(torch.cat(self.stored_target_bus, dim=0), dim=0)
            gens = torch.cat(self.stored_target_gen, dim=0)
            mean_val = np.nanmean(gens.cpu().numpy())
            gens[torch.isnan(gens)] = gens.new_tensor(mean_val)
            gens = torch.mean(gens, dim=0)
            branch = torch.mean(torch.cat(self.stored_target_branch, dim=0), dim=0)
            self.cache = busses, gens, branch
        else:
            busses, gens, branch = self.cache
        output_dict = {
            "target_bus": busses[None, :].expand(data["bus"].shape[0], -1).to(data["bus"].device),
            #"target_bus": torch.zeros_like(data["target_bus"]),
            "target_gen": gens[None, :].expand(data["gen"].shape[0], -1).to(data["gen"].device),
            # "target_gen": torch.zeros_like(data["target_gen"]),
            # "target_branch": copy.deepcopy(data["target_branch"])
            "target_branch": branch[None, :].expand(data["branch_attr"].shape[0], -1).to(data["branch_attr"].device),
        }
        return output_dict

    def compute_loss(self, data, target):
        return torch.tensor(0)


class EdgeModel(torch.nn.Module):
    def __init__(self, n_features, n_edge_features, hiddens, n_targets):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * n_features + n_edge_features, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, n_targets),
        )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)


class NodeModel(torch.nn.Module):
    def __init__(self, n_features, n_edge_features, hiddens, n_targets):
        super(NodeModel, self).__init__()
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
        return self.node_mlp_2(out)


class GlobalModel(torch.nn.Module):
    def __init__(self, n_node_features, n_global_features, hiddens, n_targets):
        super().__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(n_global_features + n_node_features, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, n_targets),
        )

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        if u is None:
            out = scatter_mean(x, batch, dim=0)
        else:
            out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        return self.global_mlp(out)


class IncrementalModel(PowerBaseModel):
    def __init__(self, n_features, n_targets, n_hiddens, n_edge_features, n_targets_edge, n_layers):
        super().__init__()
        self.n_hiddens = n_hiddens
        self.build_model(n_features, n_targets, n_edge_features=n_edge_features,
                         n_hiddens=n_hiddens, n_targets_edge=n_targets_edge,
                         n_layers=n_layers)
        self.build_optimizer_scheduler()

    def build_layer(self, n_features, n_edge_features, n_global_features, hiddens, n_targets):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(n_features, n_edge_features, hiddens, n_targets),
            node_model=NodeModel(n_features, n_targets, hiddens, n_targets),
            global_model=GlobalModel(n_node_features=hiddens, n_global_features=n_global_features, hiddens=hiddens, n_targets=hiddens)
        )

    def build_converter(self, n_features, n_targets):
        return nn.Linear(n_features, n_targets)



    def build_model(self, n_features, n_targets, n_targets_edge, n_edge_features, n_hiddens, n_layers):
        self.converters = nn.ModuleDict(
            {"gen": self.build_converter(n_features["gen"], n_hiddens),
             "load": self.build_converter(n_features["load"], n_hiddens),
             "shunt": self.build_converter(n_features["shunt"], n_hiddens),
             "bus": self.build_converter(n_features["bus"], n_hiddens),
             "branch": self.build_converter(n_edge_features, n_hiddens)
             }
        )

        self.output_layers = nn.ModuleDict({
            "gen": self.build_converter(2 * n_hiddens, n_targets["gen"]),
            "bus": self.build_converter(n_hiddens, n_targets["bus"]),
            "branch": self.build_converter(n_hiddens, n_targets_edge),
        })

        last_features = 4 * n_hiddens
        self.layers = nn.ModuleList()

        n_global_features = 0
        for i in range(n_layers):
            self.layers.append(self.build_layer(
                n_features=last_features,
                n_edge_features=n_hiddens,
                n_global_features=n_global_features,
                hiddens=n_hiddens,
                n_targets=n_hiddens
            ))
            n_global_features = n_hiddens
            last_features = n_hiddens


    def _scatter_items(self, orig_items, index, dim_size):
        new_matrix = scatter_add(orig_items, index, dim_size=dim_size, dim=0)
        return new_matrix


    def forward(self, data):
        data = copy.deepcopy(data)

        gen = self.converters["gen"](data.gen)
        load = self.converters["load"](data.load)
        shunt = self.converters["shunt"](data.shunt)
        bus = self.converters["bus"](data.bus)
        branch_attr = self.converters["branch"](data.branch_attr)

        x = torch.cat([
            bus,
            self._scatter_items(shunt, data.shunt_to_bus, bus.shape[0]),
            self._scatter_items(gen, data.gen_to_bus, bus.shape[0]),
            self._scatter_items(load, data.load_to_bus, bus.shape[0]),
        ], dim=-1)
        u = None
        for i, layers in enumerate(self.layers):
            x, branch_attr, u = layers(x, data.branch_index, edge_attr=branch_attr, u=u, batch=data.batch)

        gen_features = torch.cat([x[data.gen_to_bus], gen], dim=-1)

        return {
            "target_bus": self.output_layers["bus"](x),
            "target_gen": self.output_layers["gen"](gen_features),
            "target_branch": self.output_layers["branch"](branch_attr)
        }
