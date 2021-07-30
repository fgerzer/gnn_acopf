import time
from pathlib import Path
import pickle
import datetime
from datetime import timedelta

import yaml
import torch
import torch.onnx
import matplotlib.pyplot as plt
import dill

from gnn_acopf.training.checkpointing import Serializable, EmptySerializable

try:
    import seaborn as sns
    sns.set(style='ticks', palette='Set2')
except ImportError:
    pass


class ModelParamCountCallback(EmptySerializable):
    @staticmethod
    def param_count(model):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return trainable_params

    def print_param_count(self, model):
        print("Trainable parameters for model {}".format(self.param_count(model)))


class SaveBestModelCallback(Serializable):
    def __init__(self, store_path: Path):
        self.store_path = store_path.absolute()
        self.best_result = None

        # Monkey-patching from https://github.com/pytorch/pytorch/pull/10296
        from torch.nn.parameter import Parameter
        def patched___reduce_ex__(self, proto):     # pylint: disable=unused-argument
            return Parameter, (self.data, self.requires_grad)

        torch.nn.parameter.Parameter.__reduce_ex__ = patched___reduce_ex__

    def state_dict(self):
        state_dict = {
            "best_result": self.best_result
        }
        return state_dict

    def try_save_model(self, model, cur_result, inpt):

        if self.best_result is None or self.best_result > cur_result:
            with (self.store_path / "best_model.dill").open("wb") as dill_file:
                # torch.save(model, f, pickle_module=dill)
                dill.dump(model, dill_file, byref=False, protocol=2)
            onnx_path = str(self.store_path / "best_model.onnx")
            torch.onnx.export(model, inpt, onnx_path, export_params=True)
            with (self.store_path / "best_model.pth").open("wb") as pth_file:
                torch.save(model.state_dict(), pth_file)
            self.best_result = cur_result

    def load_state_dict(self, state_dict):
        self.best_result = state_dict["best_result"]


class ProgressCallback(EmptySerializable):
    def __init__(self, every_n_batches=None, every_n_seconds=None):
        self.every_n_batches = every_n_batches
        self.every_n_seconds = every_n_seconds
        self.last_printed = time.time()
        self.epoch_started = None
        self.print_header = True
        self.padding_necessary = None

    def start_epoch(self):
        self.epoch_started = time.time()

    def batch_end(self, epoch, batch_idx, n_samples, n_total_samples, loss):
        if self._test_for_batch(batch_idx) or self._test_for_time():
            self.last_printed = time.time()

            out_str = " ".join([
                "Train Epoch: {epoch}".format(epoch=epoch),
                "[{cur_samples}/{tot_samples} ({perc:.0f})%]".format(
                    cur_samples=batch_idx * n_samples,
                    tot_samples=n_total_samples,
                    perc=100 * batch_idx * n_samples / n_total_samples),
                "\tLoss: {loss:.6f}".format(loss=loss)
            ])
            if self.epoch_started is not None:
                time_taken = time.time() - self.epoch_started
                perc_done = batch_idx * n_samples / n_total_samples
                if perc_done > 0:
                    pred_total = time_taken / perc_done
                    eta = pred_total * (1 - perc_done)
                    time_str = " ".join([
                        "\t\tTime: {}".format(str(timedelta(seconds=round(time_taken)))),
                        "ETA: {}".format(str(timedelta(seconds=round(eta)))),
                        "Total: {}".format(str(timedelta(seconds=round(pred_total))))
                    ])
                    out_str += time_str
            print(out_str)
            self.print_header = True

    def epoch_end(self, history):

        step, latest_step = history.latest_step()
        intro_str = "Step {step:4d}".format(step=step)
        if self.epoch_started:
            time_taken = timedelta(seconds=(time.time() - self.epoch_started))
            intro_str += "\t{time_taken}s".format(time_taken=str(time_taken))
        datasets = latest_step.dataset.unique()
        metrics = [c for c in latest_step.columns if c not in ["step", "dataset"]]

        if self.padding_necessary is None or self.print_header:
            header = []
            self.padding_necessary = []
            for m in metrics:
                for d in datasets:
                    header.append(f"{m}_{d}")
                    self.padding_necessary.append(len(header[-1]))
            if self.print_header:
                self.print_header = False
                print(" " * len(intro_str) + "\t\t" + "\t".join(header))
        results_strings = []
        i = 0
        for m in metrics:
            for d in datasets:
                v = latest_step[latest_step.dataset == d][m].iloc[0]
                results_strings.append(f"{v:{self.padding_necessary[i]}.4f}")
                i += 1
        print(intro_str + "\t\t" + "\t".join(results_strings))
        self.reset()

    def _test_for_batch(self, batch_idx):
        return self.every_n_batches is not None and batch_idx % self.every_n_batches == 0

    def _test_for_time(self):
        return self.every_n_seconds is not None and \
               time.time() - self.last_printed > self.every_n_seconds

    def reset(self):
        self.last_printed = time.time()


class CheckpointCallback(Serializable):
    def __init__(self,
                 directory: Path,
                 result_path: Path,
                 keep_recent: int = 3,
                 checkpoint_paths=None):
        self.directory = directory
        self.keep_recent = keep_recent
        self.best_model_path = result_path / "best_model.pth"
        self.checkpoint_paths = checkpoint_paths or []

    def save_best(self, to_save):
        self._save_single(to_save, self.best_model_path)
        return self.best_model_path

    def _save_single(self, to_save, savepath):
        if not self.directory.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
        with savepath.open("wb") as pickle_file:
            pickle.dump(to_save.state_dict(), pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    def save(self, to_save):
        cur_time = datetime.datetime.utcnow().isoformat()
        new_name = "ckpt_%s.checkpoint" % cur_time.replace(":", "-").replace(".", "-")
        savepath = self.directory / new_name
        self._save_single(to_save, savepath)
        self.checkpoint_paths.append(savepath)
        if len(self.checkpoint_paths) > self.keep_recent:
            del self.checkpoint_paths[0]
            list_of_files = self.directory.glob("ckpt*")
            for ckpt_file in list_of_files:
                if ckpt_file not in self.checkpoint_paths:
                    ckpt_file.unlink()
        return savepath

    def load_best(self, model):
        if self.best_model_path.exists():
            self.load(model, self.best_model_path)

    @staticmethod
    def load(model: torch.nn.Module, model_file: Path, subkey=None):
        with model_file.open("rb") as state_file:
            state_dict = pickle.load(state_file)
        if subkey is not None:
            state_dict = state_dict[subkey]
        model.load_state_dict(state_dict)

    def load_latest(self, model):
        model_file = self.checkpoint_paths[-1]
        with model_file.open("rb") as state_file:
            state_dict = pickle.load(state_file)
        model.load_state_dict(state_dict["model"])

    def load_latest_from_folder(self, to_load, index=-1, enforce_load=True, subkey=None):
        if not enforce_load and not self.directory.exists():
            return
        checkpoints_by_time = sorted(self.directory.iterdir(), key=lambda f: f.stat().st_mtime)
        checkpoints_by_time = [c for c in checkpoints_by_time if
                               c.is_file() and c.suffix == ".checkpoint"]
        if not checkpoints_by_time:
            if enforce_load:
                raise FileNotFoundError("No checkpoint files at {}".format(self.directory))
        else:
            checkpoint = checkpoints_by_time[index]
            self.load(to_load, checkpoint, subkey)

    def state_dict(self):
        state_dict = {
            "checkpoint_paths": self.checkpoint_paths
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.checkpoint_paths = state_dict.get("checkpoint_paths", [])


class WriteHistoryCallback:
    def __init__(self, directory: Path, wait_before_plotting=None):
        self.directory = directory
        self.wait_before_plotting = wait_before_plotting
        self.directory.mkdir(exist_ok=True, parents=True)
        self.last_saved = time.time()

    def save_all(self, history, force_plot=False):
        self.save_plots(history, force_plot=force_plot)
        self.save_history(history)

    def save_plots(self, history, force_plot=False):
        if (not self.wait_before_plotting or force_plot or
                time.time() - self.last_saved > self.wait_before_plotting):
            for name in history.known_names():
                fig = self.create_plot(name, history.df.filter(["step", "dataset", name]))
                fig_path = self.directory / (name + ".svg")
                fig.savefig(str(fig_path), bbox_inches="tight")
                plt.close(fig)
            self.last_saved = time.time()

    @staticmethod
    def create_plot(key, key_history):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(key, y=1.05)
        ax.set_xlabel("Steps")
        ax.set_ylabel(key)
        for dataset in key_history["dataset"].unique():
            chosen = key_history.loc[key_history.dataset == dataset]
            samples = chosen["step"].values
            vals = chosen[key].values
            if samples.size > 0:
                ax.plot(samples, vals, label=dataset)
        if len(key_history) > 1:
            ax.legend()
        try:
            sns.despine()
        except NameError:
            pass
        return fig

    def save_history(self, history):
        with open(str(self.directory / "result.yaml"), "w") as yaml_file:
            yaml.dump(history.state_dict(), yaml_file)
