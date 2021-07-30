from collections import defaultdict
import pandas as pd
from gnn_acopf.training.checkpointing import Serializable

def denumpy_ify(original):
    if isinstance(original, dict):
        converted = {}
        for k in original:
            converted[k] = denumpy_ify(original[k])
        return converted
    if isinstance(original, list):
        converted = []
        for entry in original:
            converted.append(denumpy_ify(entry))
        return converted
    try:
        return float(original)
    except ValueError:
        return original


class HistoryStep(Serializable):
    def __init__(self):
        self.values = defaultdict(dict)

    def add_entry(self, dataset, **kwargs):
        for k in kwargs:
            self.values[k][dataset] = kwargs[k]

    def known_names(self):
        return list(self.values.keys())

    def known_datasets(self):
        datasets_list = [v.keys() for v in self.values.values()]
        datasets_list = [item for sublist in datasets_list for item in sublist]
        return set(datasets_list)

    def state_dict(self):
        return dict(denumpy_ify(self.values))

    def load_state_dict(self, state_dict):
        self.values = defaultdict(dict, state_dict)

    def __getitem__(self, item):
        return self.values[item]


class History(Serializable):
    def __init__(self):
        self.df = pd.DataFrame(columns=["step", "dataset"])

    def add_entry(self, step, dataset, **kwargs):
        # we first find the index to set, which includes both the correct dataset and step.
        idx = self.df.loc[(self.df["step"] == step) & (self.df["dataset"] == dataset)].index
        if idx.empty:
            self.df = self.df.append({"step": step, "dataset": dataset}, ignore_index=True)
            idx = self.df.index[-1]
        else:
            assert len(idx) == 1    # we should only find one
            idx = idx[0]
        for metric_name, metric_val in kwargs.items():
            self.df.at[idx, metric_name] = metric_val

    def state_dict(self):
        state_dict = {
            "history": self.df.to_dict()
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.df = self.df.from_dict(state_dict["history"])

    def known_names(self):
        return [c for c in self.df.columns if c not in ["step", "dataset"]]

    def latest_step(self) -> (int, HistoryStep):
        max_step = self.df["step"].max()
        latest_step_info = self.df.loc[(self.df["step"] == max_step)]
        return max_step, latest_step_info

    def latest_value(self, key, dataset):
        _, latest_step = self.latest_step()
        return latest_step.loc[latest_step.dataset == dataset][key]

    def filter_by_dataset(self, dataset):
        val_df = self.df.loc[self.df.dataset == dataset].drop("dataset", axis=1)
        val_df.set_index("step", inplace=True)
        return val_df

class OldHistory(Serializable):
    def __init__(self):
        self.steps = defaultdict(HistoryStep)

    def add_entry(self, step, dataset, **kwargs):
        self.steps[step].add_entry(dataset, **kwargs)

    def state_dict(self):
        state_dict = {
            "history": {k: v.state_dict() for k, v in self.steps.items()}
        }
        return state_dict

    def load_state_dict(self, state_dict):
        hist = state_dict["history"]
        for step in hist:
            self.steps[step].load_state_dict(hist[step])

    def known_names(self):
        all_known_names = [s.known_names() for s in self.steps.values()]
        return {item for sublist in all_known_names for item in sublist}

    def known_datasets(self):
        all_known_datasets = [s.known_datasets() for s in self.steps.values()]
        return {item for sublist in all_known_datasets for item in sublist}

    def latest_step(self) -> (int, HistoryStep):
        latest_key = sorted(self.steps.keys())[-1]
        return latest_key, self[latest_key]

    def latest_value(self, key, dataset):
        latest_key = sorted(self.steps.keys())[-1]
        latest_step = self[latest_key]
        return latest_step[key][dataset]

    def __getitem__(self, item) -> HistoryStep:
        return self.steps[item]

    def key_history(self, key):
        key_history = {dataset: {} for dataset in self.known_datasets()}
        for step in self.steps:
            for dataset in self.steps[step][key]:
                key_history[dataset][step] = self.steps[step][key][dataset]
        return key_history
