from torch import nn
import copy
from gnn_acopf.models.power_base_model import PowerBaseModel


class FFModel(PowerBaseModel):
    def __init__(self, n_features, n_targets, n_hiddens, n_edge_features, n_targets_edge, n_layers, batchnorm):
        super().__init__()
        self.build_model(n_features, n_targets, n_edge_features=n_edge_features,
                         n_hiddens=n_hiddens, n_targets_edge=n_targets_edge,
                         n_layers=n_layers, batchnorm=batchnorm)
        self.build_optimizer_scheduler()

    def build_subnets(self, n_features, n_hiddens, n_targets, n_layers, batchnorm):
        layers = []
        last_features = n_features
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(last_features, n_hiddens))
            if batchnorm:
                layers.append(nn.BatchNorm1d(n_hiddens))
            last_features = n_hiddens
            layers.append(nn.ReLU())
        layers.append(nn.Linear(n_hiddens, n_targets))
        return nn.Sequential(*layers)

    def build_model(self, n_features, n_targets, n_targets_edge, n_edge_features, n_hiddens, n_layers, batchnorm):
        self.models = nn.ModuleDict({
            "bus": self.build_subnets(n_features["bus"], n_hiddens, n_targets["bus"], n_layers, batchnorm=batchnorm),
            # "load": self.build_subnets(n_features["load"], n_hiddens, n_targets["load"], n_layers, batchnorm=batchnorm),
            # "shunt": self.build_subnets(n_features["shunt"], n_hiddens, n_targets["shunt"], n_layers, batchnorm=batchnorm),
            "gen": self.build_subnets(n_features["gen"], n_hiddens, n_targets["gen"], n_layers, batchnorm=batchnorm),
            "branch_attr": self.build_subnets(n_edge_features, n_hiddens, n_targets_edge, n_layers, batchnorm=batchnorm)
        })

    def forward(self, data):
        data = copy.deepcopy(data)

        return {
            "target_bus": self.models["bus"](data.bus),
            "target_gen": self.models["gen"](data.gen),
            "target_branch": self.models["branch_attr"](data.branch_attr)
        }
