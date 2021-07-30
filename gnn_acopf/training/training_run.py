"""Contains prototype for a training run."""
from collections import namedtuple
import torch
from gnn_acopf.training.checkpointing import ChildrenSerializable, cur_rng_state, load_rng_state
from gnn_acopf.training import callbacks as cb
from gnn_acopf.training.history import History
from gnn_acopf.training.dataloader import GeometricDatasetLoaders
QualityMetric = namedtuple("QualityMetric", "name compare_func")


class BaseTrainingRun(ChildrenSerializable):
    def __init__(self,
                 model,
                 dataset,
                 results_path,
                 checkpoint_path,
                 quality_metric: QualityMetric,
                 try_load=True):
        """
        Initializes the training run

        Parameters
        ----------
        model : Model
            Callable pytorch model.
        dataset : Dataset
            The dataset to train on
        results_path, checkpoint_path : Path
            Path for checkpoints and results

        try_load : bool, optional
            Whether to load and continue the run if possible. Default is True.
        """
        self.dataset = dataset
        self.results_path = results_path
        self.checkpoint_path = checkpoint_path
        self.model = model
        self.quality_metric = quality_metric
        self.callbacks = self._create_callbacks()
        self.epochs_trained = 0
        self.history = History()
        if try_load:
            try:
                self.callbacks["checkpoint"].load_latest_from_folder(self,
                                                                 enforce_load=True)
            except FileNotFoundError as e:
                print(e)

    def _create_callbacks(self):
        callbacks = {
            "progress": cb.ProgressCallback(every_n_seconds=120),
            "history": cb.WriteHistoryCallback(self.results_path,
                                               wait_before_plotting=10),
            "checkpoint": cb.CheckpointCallback(self.checkpoint_path,
                                                self.results_path),
            "num_params": cb.ModelParamCountCallback()
        }
        return callbacks

    def train(self, epochs, device, batch_size):
        """
        Runs training for several epochs.

        Parameters
        ----------
        epochs : int
            Epochs to run for
        device : str
            Device to run with (cuda or cpu)
        batch_size : int
            Batch size for training
        """
        for _ in range(epochs):
            self.train_one_epoch(device, batch_size)
        self.callbacks["history"].save_all(self.history, force_plot=True)

    def train_until(self,
                    epochs_until,
                    device,
                    batch_size,
                    patience=None,
                    num_workers=0):
        """
        Runs training until a certain epoch number has been reached.

        Parameters
        ----------
        epochs_until : int
            Epoch to run until.
        device : str
            Device to run with (cuda or cpu)
        batch_size : int
            Batch size for training
        num_workers : int
            Number of workers; see pytorch dataloader doc
        """
        self.callbacks["num_params"].print_param_count(self.model)
        while self.epochs_trained < epochs_until:
            if patience is not None and len(self.history.filter_by_dataset("val")) > 0:
                val_results = self.history.filter_by_dataset("val")[self.quality_metric.name]
                if len(val_results) > patience:
                    # construct the comparison function name, which is either idxmax or idxmin.
                    comp_func = f"idx{self.quality_metric.compare_func}"
                    if val_results.__getattr__(comp_func)() <= len(val_results) - patience:
                        # the best result is longer than patience steps back; stop
                        return self

            self.train_one_epoch(
                device,
                batch_size,
                num_workers=num_workers
            )
        self.callbacks["history"].save_all(self.history, force_plot=True)
        return self

    def load_best_model(self):
        self.callbacks["checkpoint"].load_best(self.model)

    def load_latest_model(self):
        self.callbacks["checkpoint"].load_latest(self.model)

    def _new_metrics(self):
        """
        (Re)Creates the metrics to use for this training.

        Returns
        -------
        metrics : Dict
            dict containing string keys and interesting metrics as value.
        """
        raise NotImplementedError

    def key_to_component(self):
        """Subcomponent dictionary for save/return"""
        components_dict = {
            "model": self.model,
            "checkpoint_callback": self.callbacks["checkpoint"],
            "progress_callback": self.callbacks["progress"],
            "history": self.history
        }
        return components_dict

    def own_state_dict(self):
        """State dict for save/return"""
        state_dict = {
            "epochs_trained": self.epochs_trained,
            "rng_state": cur_rng_state()
        }
        return state_dict

    def load_own_state_dict(self, state_dict):
        """State dict loading for save/return"""
        self.epochs_trained = state_dict["epochs_trained"]
        load_rng_state(state_dict["rng_state"])


class SupervisedTrainingRun(BaseTrainingRun):
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
        data_loaders = GeometricDatasetLoaders.from_dataset(
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
            target = data.y.to(device)
            output, loss = self.model.update_step(data, target)

            with torch.no_grad():
                for metric in metrics.values():
                    metric.add_eval(output, target, data=data, model=self.model)
            self.callbacks["progress"].batch_end(epoch=self.epochs_trained,
                                                 batch_idx=batch_idx,
                                                 n_samples=torch.max(data.batch).item(),
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
        self.callbacks["checkpoint"].save(self)
        val_results = self.history.filter_by_dataset("val")[self.quality_metric.name]
        comp_func = f"idx{self.quality_metric.compare_func}"
        if val_results.__getattr__(comp_func)(val_results) == len(val_results):
            self.callbacks["checkpoint"].save_best(self.model)
        return self.history.latest_value(self.quality_metric.name, "val")


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
                target = data.y.to(device)
                output = self.model(data)

                for metric in metrics.values():
                    metric.add_eval(output, target, data=data)

        eval_dict = {l: m.result() for l, m in metrics.items()}
        return eval_dict
