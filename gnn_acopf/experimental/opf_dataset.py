from pathlib import Path
from gnn_acopf.julia_interface import JuliaInterface
from gnn_acopf.utils.observers import DefaultObserver
from gnn_acopf.utils.power_net import PowerNetwork
from gnn_acopf.training.dataloader import Dataset
import numpy as np
import itertools
import torch
from torch_geometric.data import Data


class PowerData(Data):
    def __inc__(self, key, value):
        increasing_funcs = ["load_to_bus", "shunt_to_bus", "branch_index", "gen_to_bus"]
        if key in increasing_funcs:
            return len(self["bus"])
        else:
            return 0


class SingleOPFDataset:
    def __init__(self, scenario_ids, parent_datset):
        self.scenario_ids = scenario_ids
        self.parent_dataset = parent_datset

    def __len__(self):
        return len(self.scenario_ids)

    def __getitem__(self, item):
        scenario_id = self.scenario_ids[item]
        case_dict, area_scale = self.parent_dataset.powernet.get_scenario_area_scale(scenario_id)
        observation = self.parent_dataset.observer.to_geometric_representation(case_dict, area_scale, scenario_id)
        data = PowerData(
            **{k: torch.tensor(v) for k, v in observation.items()}
        )
        data.x = data.bus
        return data

    def get_orig_data(self, scenario_id):
        case_dict = self.parent_dataset.powernet.create_scenario_case_dict(scenario_id)
        return case_dict

    @property
    def n_features(self):
        return self[0]["x"].shape[1]

    @property
    def n_targets(self):
        return self[0]["y"].shape[1]

    @property
    def n_edge_features(self):
        return self[0]["edge_attr"].shape[1]

    @property
    def n_targets_edge(self):
        return self[0]["y_branch"].shape[1]


class OPFDataset(Dataset):
    def __init__(self, powernet, observer):
        self.powernet = powernet
        all_scenario_ids = sorted(powernet.varying_load.keys())
        train_scenario_ids, val_scenario_ids, test_scenario_ids = self.split_by_day(all_scenario_ids)
        self.observer = observer
        self.train_dataset = SingleOPFDataset(train_scenario_ids, self)
        self.val_dataset = SingleOPFDataset(val_scenario_ids, self)
        self.test_dataset = SingleOPFDataset(test_scenario_ids, self)

    def split_by_day(self, scenario_ids):
        days = []
        for i in range(0, len(scenario_ids), 24):
            days.append(scenario_ids[i:i+24])
        np.random.RandomState(seed=0).shuffle(days)
        n_train_examples = int(len(days) * 0.5)
        n_val_examples = int(len(days) * 0.25)
        train = list(itertools.chain(*days[:n_train_examples]))
        val = list(itertools.chain(*days[n_train_examples:n_train_examples+n_val_examples]))
        test = list(itertools.chain(*days[n_train_examples+n_val_examples:]))
        return train, val, test

    @property
    def n_features(self):
        return self.train.n_features

    @property
    def n_targets(self):
        return self.train.n_targets

    @property
    def n_edge_features(self):
        return self.train.n_edge_features

    @property
    def n_targets_edge(self):
        return self.train.n_targets_edge



def main(casename, area_name):
    pn = PowerNetwork.from_pickle(Path(f"../../data/case_{casename}.pickle"), area_name=area_name)
    pn.load_scenarios_file(Path(f"../../casefiles/scenarios_{casename}.m"))
    obs = DefaultObserver(jl=JuliaInterface(), solution_cache_file=Path(f"../../data/generated_solutions/case_{casename}_solutions.pickle"))
    pdl = OPFDataset(pn, obs)
    for i in range(100):
        print(pdl.train[i])


if __name__ == "__main__":
    casename, areaname = "ACTIVSg200", "zone"
    main(casename, areaname)

