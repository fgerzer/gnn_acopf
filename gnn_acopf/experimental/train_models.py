from collections import namedtuple

from gnn_acopf.training.training_run import SupervisedTrainingRun
import torch.nn.functional as F
from gnn_acopf.models.single_graph_model import SingleGraphModel
from gnn_acopf.models.simple_model import SimpleModel, PhonyModel, IncrementalModel, MeanModel
from gnn_acopf.models.ff_model import FFModel
from gnn_acopf.models.summarized_model import SummarizedModel
from gnn_acopf.models.summarize_encoding_model import SummarizedEncodingModel

from gnn_acopf.experimental.opf_dataset import OPFDataset
from gnn_acopf.models.power_base_model import GeomOPFLoss
from pathlib import Path
from gnn_acopf.utils.observers import DefaultObserver, Scaler
from gnn_acopf.utils.power_net import PowerNetwork
import torch
import numpy as np


QualityMetric = namedtuple("QualityMetric", "name compare_func")


class GradientNorm(torch.nn.Module):
    def forward(self, prediction, target, data=None, model=None):
        if model is None:
            return 0
        grads = [torch.norm(p.grad).cpu().numpy() for p in model.parameters()]
        return np.mean(grads)


class MetricMean():
    def __init__(self,
                 eval_function,
                 weighted_by_count=True,
                 numerator=0,
                 denominator=0,
                 item=False
                 ):
        self.numerator = numerator
        self.denominator = denominator
        self.weighted_by_count = weighted_by_count
        self.eval_function = eval_function
        self.item = item

    def add_eval(self, prediction, target, **kwargs):
        value = self.eval_function(prediction, target, **kwargs)
        if self.item:
            value = value.item()
        self.add_values(value)

    def add_values(self, v):
        if self.weighted_by_count:
            self.numerator += np.sum(v)
            self.denominator += np.size(v)
        else:
            self.numerator += np.mean(v)
            self.denominator += 1

    def result(self):
        if self.denominator == 0:
            return float("NaN")
        return self.numerator / self.denominator

    def reset(self):
        self.numerator = 0
        self.denominator = 0




class OPFTrainingRun(SupervisedTrainingRun):
    def _new_metrics(self):
        metric_dict = {
            "mse": MetricMean(F.mse_loss, item=True),
            # "grad_norm": MetricMean(GradientNorm())
        }
        return metric_dict


class OPFTrainAndEval(SupervisedTrainingRun):
    def _build_training_run(self,
                            dataset,
                            results_path,
                            checkpoint_path):
        tr = OPFTrainingRun(model=self.model,
                            dataset=dataset,
                            results_path=results_path,
                            checkpoint_path=checkpoint_path,
                            quality_metric=self.quality_metric)
        return tr

    def eval_on(self, data_loader, device):
        """
        Evaluates the model on a certain data loader.

        Parameters
        ----------
        data_loader : data loader
            The data loader to evaluate on
        device : str
            The torch device to evaluate on

        Returns
        -------
        eval_dict : dict
            Dict with the same keys as self.metrics getting the corresponding results as value.
        """
        self.model.eval()
        self.model.to(device)
        metrics = self._new_metrics()
        with torch.no_grad():
            for data in data_loader:
                data.to(device)
                target = {
                    "target_branch": data["target_branch"],
                    "target_bus": data["target_bus"],
                    "target_gen": data["target_gen"]
                }
                output = self.model(data)

                for metric in metrics.values():
                    metric.add_eval(output, target, data=data)

        eval_dict = {l: m.result() for l, m in metrics.items()}
        return eval_dict

    def train_one_epoch(self,
                        device,
                        batch_size,
                        num_workers=0):
        """
        Trains for a single epoch.

        This takes care of callbacks, training steps, checkpointing, etc.

        Parameters
        ----------
        device : str
            Device to run with (cuda or cpu)
        batch_size : int
            Batch size for training
        num_workers : int
            Number of workers; see pytorch dataloader doc
        """
        data_loaders = self.datasetloader.from_dataset(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        self.model.train()
        self.model.to(device)
        metrics = self._new_metrics()
        self.callbacks["progress"].start_epoch()
        for batch_idx, data in enumerate(data_loaders.train):
            data.to(device)
            target = {
                "target_branch": data["target_branch"],
                "target_bus": data["target_bus"],
                "target_gen": data["target_gen"]
            }
            output, loss = self.model.update_step(data, target)

            with torch.no_grad():
                for metric in metrics.values():
                    metric.add_eval(output, target, data=data, model=self.model)
            self.callbacks["progress"].batch_end(epoch=self.epochs_trained,
                                                 batch_idx=batch_idx,
                                                 n_samples=data.solved.shape[0],
                                                 n_total_samples=len(self.dataset.train),
                                                 loss=loss.item())
        self.epochs_trained += 1
        self.model.epoch_finished()
        self.history.add_entry(self.epochs_trained, dataset="train",
                               **{l: m.result() for l, m in metrics.items()})
        self.history.add_entry(self.epochs_trained,
                               dataset="val",
                               **self.eval_on(data_loaders.val, device))
        self.callbacks["progress"].epoch_end(history=self.history)
        self.callbacks["history"].save_history(self.history)
        self.callbacks["history"].save_plots(self.history)
        self.model.to("cpu")
        self.callbacks["checkpoint"].save(self)
        val_results = self.history.filter_by_dataset("val")[self.quality_metric.name]
        comp_func = f"idx{self.quality_metric.compare_func}"
        if val_results.__getattr__(comp_func)(val_results) == len(val_results):
            self.callbacks["checkpoint"].save_best(self.model)
        return self.history.latest_value(self.quality_metric.name, "val")

    def _new_metrics(self):
        metric_dict = {
            "mse": MetricMean(GeomOPFLoss(F.mse_loss), item=True),
            "mae": MetricMean(GeomOPFLoss(F.l1_loss), item=True),
            "mae_branch": MetricMean(GeomOPFLoss(F.l1_loss, vals_to_summarize=["target_branch"]), item=True),
            "mae_bus": MetricMean(GeomOPFLoss(F.l1_loss, vals_to_summarize=["target_bus"]), item=True),
            "mae_gen": MetricMean(GeomOPFLoss(F.l1_loss, vals_to_summarize=["target_gen"]), item=True),
            "grad_norm": MetricMean(GradientNorm())
        }
        return metric_dict


def main(casename, area_name, pgmin_to_zero, cluster, device, scaler):
    print(device)
    if cluster:
        datapath = Path("/experiment/data")
        results_path = Path("/experiment/results")
        checkpoint_path = Path("/experiment/checkpoints")
    else:
        datapath = Path("../../data")
        results_path = Path("../../experiment/results")
        checkpoint_path = Path("../../experiment/checkpoints")

    pn = PowerNetwork.from_pickle(datapath / f"case_{casename}.pickle", area_name=area_name,
                                  pgmin_to_zero=pgmin_to_zero)
    pn.load_scenarios_file(datapath / f"scenarios_{casename}.m")

    obs = DefaultObserver(
        jl=None,
        solution_cache_dir=datapath / f"generated_solutions/case_{casename}_solutions",
        area_name=area_name,
        scaler=scaler
    )
    dataset = OPFDataset(pn, obs)
    batchnorm = True
    residuals = True
    quality_metric = QualityMetric(name="mse", compare_func="min")
    # model = SimpleModel(n_features=obs.n_node_features, n_targets=obs.n_node_targets,
    #                  n_hiddens=32, n_targets_edge=obs.n_branch_targets,
    #                  n_edge_features=obs.n_branch_features,
    #                  n_layers=8, batchnorm=batchnorm, residuals=residuals)
    # model = MeanModel()
    # model = SummarizedModel(n_features=obs.n_node_features, n_targets=obs.n_node_targets,
    #                  n_hiddens=48, n_targets_edge=obs.n_branch_targets,
    #                  n_edge_features=obs.n_branch_features,
    #                  n_layers=8, batchnorm=batchnorm, residuals=residuals)
    model = SummarizedEncodingModel(n_features=obs.n_node_features, n_targets=obs.n_node_targets,
                       n_hiddens=48, n_targets_edge=obs.n_branch_targets,
                       n_edge_features=obs.n_branch_features,
                       n_layers=8, batchnorm=batchnorm, residuals=residuals)

    #model = SingleGraphModel(n_features=obs.n_node_features, n_targets=obs.n_node_targets,
    #                    n_hiddens=48, n_targets_edge=obs.n_branch_targets,
    #                    n_edge_features=obs.n_branch_features,
    #                    n_layers=9, batchnorm=batchnorm, residuals=residuals)
    # model = IncrementalModel(n_features=obs.n_node_features, n_targets=obs.n_node_targets,
    #                     n_hiddens=32, n_targets_edge=obs.n_branch_targets,
    #                     n_edge_features=obs.n_branch_features,
    #                     n_layers=4)

    # model = FFModel(n_features=obs.n_node_features, n_targets=obs.n_node_targets,
    #                    n_hiddens=105, n_targets_edge=obs.n_branch_targets,
    #                    n_edge_features=obs.n_branch_features,
    #                    n_layers=8, batchnorm=batchnorm)

    print("Training on model", model.__class__.__name__)
    train_and_eval = OPFTrainAndEval(model, dataset, results_path, checkpoint_path, quality_metric)
    train_and_eval.train_until(epochs_until=500, device=device, batch_size=32, num_workers=4)
    train_and_eval.callbacks["checkpoint"].save(train_and_eval)
    train_and_eval.load_best_model()
    train_and_eval.callbacks["checkpoint"].save_best(train_and_eval.model)
    data_loaders = train_and_eval.datasetloader.from_dataset(
        train_and_eval.dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    eval_device = device
    print("train Results: ", train_and_eval.eval_on(data_loaders.train, device=eval_device))
    print("val  Results: ", train_and_eval.eval_on(data_loaders.val, device=eval_device))
    print("Test Results: ", train_and_eval.eval_on(data_loaders.test, device=eval_device))

    data_loaders = train_and_eval.datasetloader.from_dataset(
       train_and_eval.dataset,
       batch_size=1,
       shuffle=False,
       num_workers=0
    )
    # eval_opf = EvalauteOPF(model)
    # eval_opf.eval_opf(data_loaders.test, device=eval_device, observer=obs, print_level=print_level)


def with_parsed_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--run_local", action="store_true")
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    scaler_200 = Scaler(scales={
        "bus": (np.array([13.8, 0.85, 1.05], dtype=np.float32), np.array([230, 0.95, 1.15], dtype=np.float32)),
        "load": (np.array([0, 0], dtype=np.float32), np.array([0.76, 0.22], dtype=np.float32)),
        "gen": (np.array([-0.55, 0, 0, 0, 0, 0], dtype=np.float32), np.array([0, 0.13, 1.1, 0.28, 0.09, 33.5], dtype=np.float32)),
        "shunt": (np.array([0.3], dtype=np.float32), np.array([0.8], dtype=np.float32)),
        "branch_attr": (np.array([0, 0], dtype=np.float32), np.array([0.1, 2.28], dtype=np.float32)),
    })

    scaler_2000 = Scaler(scales={
        "bus": (np.array([13.2, 0.85, 1.05], dtype=np.float32), np.array([500, 0.95, 1.15], dtype=np.float32)),
        "load": (np.array([0, 0], dtype=np.float32), np.array([1.3, 0.37], dtype=np.float32)),
        "gen": (np.array([-0.108, 0, 0, 0, 0, 0], dtype=np.float32), np.array([0, 0.16, 1, 0.75, 0.225, 90], dtype=np.float32)),
        "shunt": (np.array([-2], dtype=np.float32), np.array([4.9], dtype=np.float32)),
        "branch_attr": (np.array([0, 0], dtype=np.float32), np.array([0.1, 0.8], dtype=np.float32)),
    })

    cases = {
        "ACTIVSg200": ["zone", False, scaler_200],
        "ACTIVSg2000": ["area", True, scaler_2000]
    }
    casename = args.dataset
    areaname, pgmin_to_zero, scaler = cases[args.dataset]
    main(casename, areaname, pgmin_to_zero, cluster=not args.run_local, device=args.device, scaler=scaler)


if __name__ == "__main__":
    with_parsed_args()



