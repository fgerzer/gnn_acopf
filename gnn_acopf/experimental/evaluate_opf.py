import pandas as pd
import os
import portalocker
import contextlib
import yaml
import subprocess
from gnn_acopf.experimental.opf_dataset import OPFDataset
from gnn_acopf.training.training_run import QualityMetric
from gnn_acopf.utils.timer import Timer
from pathlib import Path
import copy
from gnn_acopf.utils.observers import DefaultObserver, Scaler
from gnn_acopf.utils.power_net import PowerNetwork
from gnn_acopf.models.summarize_encoding_model import SummarizedEncodingModel

from gnn_acopf.julia_interface import JuliaInterface
import torch
import numpy as np


class EvaluateOPF:
    def __init__(self, model, results_path):
        self.model = model
        self.results_path = results_path
        self.results_fp = results_path / "results.csv"
        self.run_fp = results_path / "runs.yaml"
        self.scenarios = None
        with self.synced_results():
            pass

    @contextlib.contextmanager
    def synced_runs(self):
        with portalocker.Lock(self.results_path / ".runs.lock", timeout=120) as lockfile:
            lockfile.flush()
            os.fsync(lockfile.fileno())
            try:
                with self.run_fp.open("r") as run_file:
                    exp_states = yaml.load(run_file, Loader=yaml.FullLoader)
            except FileNotFoundError:
                exp_states = {}
            yield exp_states
            with self.run_fp.open("w") as run_file:
                yaml.dump(exp_states, run_file)
            lockfile.flush()
            os.fsync(lockfile.fileno())

    def get_slurm_id(self):
        # returns either the slurm ID or "running" if no slurm ID can be found.
        try:
            slurm_id = os.environ["SLURM_JOB_ID"]
        except KeyError:
            # no slurm available
            slurm_id = "running"
        return slurm_id

    def is_running(self, exp_state):
        try:
            slurm_id = os.environ["SLURM_JOB_ID"]
        except KeyError:
            slurm_id = None
        if exp_state in [None, "stopped", slurm_id]:
            return False
        try:
            result = subprocess.check_output("squeue -hO jobid:15", shell=True,
                                             stderr=subprocess.DEVNULL).decode("utf-8").strip()
            result = result.split("\n")
            result = [int(line.strip()) for line in result]
            return exp_state in result
        except subprocess.CalledProcessError:
            return True

    @contextlib.contextmanager
    def synced_results(self):
        with portalocker.Lock(self.results_path / ".results.lock", timeout=120) as lockfile:
            lockfile.flush()
            os.fsync(lockfile.fileno())
            try:
                self.results = pd.read_csv(self.results_fp)
            except (pd.errors.EmptyDataError, FileNotFoundError):
                self.results = pd.DataFrame(columns=[
                   "scenario_id",
                   "opf_method",
                   "time_taken",
                   "solved",
                   "power_generated",
                   "power_loss"
                ])
            yield
            self.results.to_csv(self.results_fp, index=False)
            lockfile.flush()
            os.fsync(lockfile.fileno())

    def eval_method(self, eval_func, case_dict, jl, data, observer):
        with Timer() as optimization_timer:
            solution = eval_func(case_dict, jl, data=data, observer=observer)
        ac_pf_result, _ = jl.run_pf(case_dict, method="ac",
                                      previous_result=solution, print_level=0,
                                    max_iter=1)
        solved = "SOLVED" in ac_pf_result["termination_status"]
        power_demand = sum([g["pd"] for g in case_dict["load"].values() if g["pd"] is not None])
        power_generated = sum([g["pg"] for g in ac_pf_result["solution"]["gen"].values() if g["pg"] is not None])
        power_loss = power_generated / power_demand
        return {
            "time_taken": optimization_timer.interval,
            "solved": solved,
            "power_generated": power_generated,
            "power_loss": power_loss
        }

    def set_run_state(self, exp_name, state):
        with self.synced_runs() as exp_states:
            exp_states[exp_name] = state

    def eval_ac_opf(self, case_dict, jl, print_level=0, **kwargs):
        ac_opf_result, _ = jl.run_opf(case_dict, method="ac", print_level=print_level)
        return ac_opf_result

    def eval_dc_opf(self, case_dict, jl, print_level=0, **kwargs):
        ac_opf_result, _ = jl.run_opf(case_dict, method="dc", print_level=print_level)
        return ac_opf_result

    def eval_dcac_opf(self, case_dict, jl, print_level=0, **kwargs):
        ac_opf_result, _ = jl.run_opf(case_dict, method="dcac", print_level=print_level)
        return ac_opf_result

    def eval_model_and_ac_opf(self, case_dict, jl, data, observer, print_level=0, **kwargs):
        output = self.model(data)
        model_output_dict = observer.translate_output_to_results_dict(
                                          data, output, case_dict
                                      )
        ac_opf_result, _ = jl.run_opf(case_dict, method="ac",
                                      previous_result=model_output_dict, print_level=print_level)
        return ac_opf_result

    def eval_model_and_ac_opf_nobranch(self, case_dict, jl, data, observer, print_level=0, **kwargs):
        output = self.model(data)
        model_output_dict = observer.translate_output_to_results_dict(
                                          data, output, case_dict, keys_to_consider=["bus", "gen"]
                                      )
        ac_opf_result, _ = jl.run_opf(case_dict, method="ac",
                                  previous_result=model_output_dict, print_level=print_level)
        return ac_opf_result

    def eval_model_pf_ac_opf(self, case_dict, jl, data, observer, print_level=0, **kwargs):
        output = self.model(data)
        model_output_dict = observer.translate_output_to_results_dict(
                                          data, output, case_dict
                                      )
        pf_result, _ = jl.run_pf(case_dict, method="ac", previous_result=model_output_dict,
                              print_level=print_level)
        # TODO: Maybe need to combine it.
        ac_opf_result, _ = jl.run_opf(case_dict, method="ac",
                                      previous_result=pf_result, print_level=print_level)
        return ac_opf_result

    def eval_model_pf(self, case_dict, jl, data, observer, print_level=0, **kwargs):
        output = self.model(data)
        model_output_dict = observer.translate_output_to_results_dict(
                                          data, output, case_dict
                                      )
        pf_result, _ = jl.run_pf(case_dict, method="ac", previous_result=model_output_dict,
                          print_level=print_level)
        return pf_result

    def eval_model(self, case_dict, jl, data, observer, print_level=0, **kwargs):
        output = self.model(data)
        output_dict = observer.translate_output_to_results_dict(
            data, output, case_dict
        )
        return output_dict

    def eval_model_feasibility_check_opf(self, case_dict, jl, data, observer, print_level=0, **kwargs):
        output = self.model(data)
        output_dict = observer.translate_output_to_results_dict(
            data, output, case_dict
        )
        ac_pf_result, _ = jl.run_pf(case_dict, method="ac",
                                      previous_result=output_dict, print_level=0)
        solved = "SOLVED" in ac_pf_result["termination_status"]
        if not solved:
            output_dict, _ = jl.run_opf(case_dict, method="ac",
                                  previous_result=output_dict, print_level=print_level)

        return output_dict


    def statistical_summary(self, filter_by_solved):
        group_by = ["opf_method"]
        results = self.results
        if filter_by_solved:
            results = results[results["solved"] == 1]
        grouped_results = results.groupby(group_by)
        agg_dict = {c: ["mean", "std"] for c in list(self.results.columns.values)
                    if c not in group_by + ["scenario_id", "solved"]}
        agg_dict["solved"] = ["mean", "sum"]
        statistics_df = grouped_results.agg(agg_dict)
        # statistics_df = statistics_df.unstack(level=[1]).reorder_levels([2, 0, 1], axis=1)
        # sort whole group according to test acc
        statistics_df = statistics_df.sort_values(by=[("time_taken", "mean")], ascending=True)
        return statistics_df


    def pretty_statistics(self, filter_by_solved):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                               "display.width", 200):
            pretty_statistic_string = str(self.statistical_summary(filter_by_solved))
        return pretty_statistic_string

    def pprint_results(self, results_by_method, n_evaluated):
        for method, results in results_by_method.items():
            print(method)
            n_solved = len([r for r in results["solved"] if r])
            print(f"\tSolved:             {n_solved}")
            print(f"\tTime (solved):      {np.mean([results['time_taken'][i] for i in range(n_evaluated) if results['solved'][i]])}")
            print(f"\tTime (all):         {np.mean(results['time_taken'])}")
            print(f"\tCost (solved):      {np.mean([results['cost'][i] for i in range(n_evaluated) if results['solved'][i]])}")
            print(f"\tPower Gen (solved): {np.mean([results['power_generated'][i] for i in range(n_evaluated) if results['solved'][i]])}")

    def claim_scenario_idx(self, scenario_id):
        with self.synced_runs() as exp_states:
            scen_state = exp_states.get(scenario_id, None)
            if not self.is_running(scen_state):
                exp_states[scenario_id] = self.get_slurm_id()
                return True
        return False

    def eval_opf(self, dataloader, observer, device, print_level=0):
        jl = JuliaInterface()
        self.model.eval()
        self.model.to(device)
        methods = {
            "ac_opf": self.eval_ac_opf,
            "dcac_opf": self.eval_dcac_opf,
            "dc_opf": self.eval_dc_opf,
            "model_ac_opf": self.eval_model_and_ac_opf,
            "model": self.eval_model,
            # "model_feasibility_acopf": self.eval_model_feasibility_check_opf
            # "model_pf_ac_opf": self.eval_model_pf_ac_opf,
            # "model_ac_opf_nobranch": self.eval_model_and_ac_opf_nobranch,
            # "model_pf": self.eval_model_pf
        }
        results_by_method = {m: {"solved": [], "time_taken": [], "cost": []} for m in methods}
        n_evaluated = 0

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                scenario_idx = data["scenario_idx"].item()
                if not self.claim_scenario_idx(scenario_idx):
                    continue
                scenario_df = pd.DataFrame()
                data.to(device)
                n_evaluated += 1
                base_case_dict = dataloader.dataset.get_orig_data(data.scenario_idx.item())
                for m, eval_m in methods.items():
                    case_dict = copy.deepcopy(base_case_dict)
                    result_dict = self.eval_method(
                        eval_m, case_dict, jl, data=data, observer=observer
                    )
                    result_dict["scenario_id"] = scenario_idx
                    result_dict["opf_method"] = m
                    result_dict["solved"] = float(result_dict["solved"])
                    single_results_df = pd.DataFrame.from_dict({k: [v] for k, v in result_dict.items()})
                    scenario_df = pd.concat([scenario_df, single_results_df], ignore_index=True, axis=0, sort=False)
                with self.synced_results():
                    self.results = pd.concat([self.results, scenario_df], ignore_index=True, axis=0, sort=False)

                print(f"Finished {self.results.scenario_id.nunique()} scenarios")
                print("CURSTATE: ALL SCENARIOS")
                print(self.pretty_statistics(filter_by_solved=False))
                print("CURSTATE: SOLVED SCENARIOS")
                print(self.pretty_statistics(filter_by_solved=True))
                with self.synced_runs() as exp_states:
                    exp_states[scenario_idx] = "finished"

        print("FINALSTATE: ALL SCENARIOS")
        print(self.pretty_statistics(filter_by_solved=False))
        print("FINALSTATE: SOLVED SCENARIOS")
        print(self.pretty_statistics(filter_by_solved=True))


def main(casename, area_name, pgmin_to_zero, cluster, scaler, model_folder):
    device = "cpu"
    from gnn_acopf.experimental.train_models import OPFTrainAndEval
    if cluster:
        datapath = Path("/experiment/data")
        trained_models_path = Path("/experiment/trained_models")
        model_results_path = trained_models_path / model_folder / "results"
        model_checkpoint_path = trained_models_path / model_folder / "checkpoints"
        results_path = Path("/experiment/results")
        # checkpoint_path = Path("/experiment/checkpoints")
        print_level = 0
    else:
        datapath = Path("../../data")
        results_path = Path("../../experiment")
        model_results_path = results_path
        model_checkpoint_path = Path("../../experiment")
        print_level = 5

    pn = PowerNetwork.from_pickle(datapath / f"case_{casename}.pickle", area_name=area_name,
                                  pgmin_to_zero=pgmin_to_zero)
    pn.load_scenarios_file(datapath / f"scenarios_{casename}.m")

    obs = DefaultObserver(
        jl=JuliaInterface(),
        solution_cache_dir=datapath / f"generated_solutions/case_{casename}_solutions",
        # solution_cache_dir=Path("/tmp/power_results"),
        area_name=area_name,
        scaler=scaler
    )
    dataset = OPFDataset(pn, obs)
    batchnorm = True
    quality_metric = QualityMetric(name="mse", compare_func="min")
    # model = SimpleModel(n_features=obs.n_node_features, n_targets=obs.n_node_targets,
    #                  n_hiddens=32, n_targets_edge=obs.n_branch_targets,
    #                  n_edge_features=obs.n_branch_features,
    #                  n_layers=8, batchnorm=batchnorm, residuals=True)
    model = SummarizedEncodingModel(n_features=obs.n_node_features, n_targets=obs.n_node_targets,
                      n_hiddens=48, n_targets_edge=obs.n_branch_targets,
                      n_edge_features=obs.n_branch_features,
                      n_layers=8, batchnorm=batchnorm, residuals=True)
    train_and_eval = OPFTrainAndEval(model, dataset, model_results_path, model_checkpoint_path, quality_metric,
                                     # datasetloader=PowerDataSetLoader
                                     )
    train_and_eval.load_best_model()

    data_loaders = train_and_eval.datasetloader.from_dataset(
       train_and_eval.dataset,
       batch_size=1,
       shuffle=False,
       num_workers=0
    )
    eval_opf = EvaluateOPF(model, results_path)
    eval_opf.eval_opf(data_loaders.test, device=device, observer=obs, print_level=print_level)


def with_parsed_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--run_local", action="store_true")
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("--model_to_use", default=None)
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
    model_folders = {
        "ACTIVSg200": "2019-08-21T13-06-35-752927_ACTIVSg200_newdata",
        "ACTIVSg2000": "2019-08-28T12-05-52-426883_ACTIVSg2k_SummarizedEmbeddingsBig500epochs"
    }
    if args.model_to_use is None:
        model_folder = model_folders[args.dataset]
    else:
        model_folder = model_folders[args.model_to_use]
    main(casename, areaname, pgmin_to_zero, cluster=not args.run_local, scaler=scaler, model_folder=model_folder)


if __name__ == '__main__':
    with_parsed_args()
