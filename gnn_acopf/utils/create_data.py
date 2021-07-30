import portalocker
from gnn_acopf.utils.power_net import PowerNetwork
from gnn_acopf.utils.timer import Timer
from pathlib import Path
import pickle
import os
from gnn_acopf.julia_interface import JuliaInterface


CASEFILES_PATH = Path("/experiment/data/")
TARGET_PATH = Path("/experiment/results/")


def get_unused_scenario_id(pn, target_path, n_scenarios=1, finished=None):
    with portalocker.Lock(target_path / ".results.lock", timeout=120) as lockfile:
        lockfile.flush()
        os.fsync(lockfile.fileno())
        run_path = (target_path / "runs.pickle_sync")
        if not run_path.exists():
            runs = {scenario_idx: None for scenario_idx in pn.varying_load.keys()}
        else:
            with run_path.open("rb") as run_file:
                runs = pickle.load(run_file)
        open_scenarios = sorted([k for k in runs if runs[k] is None])
        unfinished_scenarios = sorted([k for k in runs if runs[k] is not "done"])
        if len(open_scenarios) > 0:
            new_scenario_ids = open_scenarios[:n_scenarios]
            for scen_id in new_scenario_ids:
                runs[scen_id] = "running"
        else:
            new_scenario_ids = None
        if finished is not None:
            for f in finished:
                runs[f] = "done"
        with run_path.open("wb") as run_file:
            pickle.dump(runs, run_file)
        lockfile.flush()
        os.fsync(lockfile.fileno())
    all_finished = len(unfinished_scenarios) == 0
    return new_scenario_ids, all_finished

def create_and_save_observations(target_path, casename, area_name, pgmin_to_zero):
    pn = PowerNetwork.from_casefile(CASEFILES_PATH / f"case_{casename}.m",
                                  area_name=area_name,
                                  pgmin_to_zero=pgmin_to_zero)
    pn.load_scenarios_file(Path(CASEFILES_PATH / f"scenarios_{casename}.m"))
    target_path.mkdir(exist_ok=True)
    n_solved = 0
    scenario_ids, all_finished = get_unused_scenario_id(pn, target_path, n_scenarios=10)
    # for scenario_id in sorted(pn.varying_load.keys()):
    while scenario_ids is not None:
        for scenario_id in scenario_ids:
            with Timer() as timer:
                case_dict = pn.create_scenario_case_dict(scenario_id)
                result, _ = JuliaInterface.get_instance().run_opf(case_dict, "ac")
            if "SOLVED" in result["termination_status"]:
                n_solved += 1
            print(f"Solved {n_solved} of {scenario_id} in {timer.interval} seconds for {n_solved / scenario_id}")
            with (target_path / f"{pn.name}_{scenario_id:06d}_solution.pickle").open("wb") as pickle_file:
                pickle.dump(result, pickle_file)
        scenario_ids, all_finished = get_unused_scenario_id(pn, target_path, n_scenarios=10,
                                                            finished=scenario_ids)
    return all_finished

def summarize_cases(casename, casepath):
    """Takes the output of create_and_save_observation and summarizes it into a single
    pickle file."""
    solutions = {}
    n_solution_files = len(list(casepath.glob("*.pickle")))
    for i in range(1, n_solution_files + 1):
        with (casepath / f"case_{casename}_{i:06d}_solution.pickle").open("rb") as pickle_file:
            solutions[i] = pickle.load(pickle_file)
        print(i)

    with (casepath / f"case_{casename}_solutions.pickle").open("wb") as pickle_file:
        pickle.dump(solutions, pickle_file)

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    if args.case == "ACTIVSg200":
        casename, area_name, pgmin_to_zero = "ACTIVSg200", "zone", False
    elif args.case == "ACTIVSg2000":
        casename, area_name, pgmin_to_zero = "ACTIVSg2000", "area", True
    else:
        raise RuntimeError(f"Don't know the dataset {args.case}.")
    all_finished = create_and_save_observations(TARGET_PATH, casename, area_name, pgmin_to_zero)
    #if all_finished:
    # summarize_cases(casename, TARGET_PATH)


if __name__ == "__main__":
    main()
