from pathlib import Path
from gnn_acopf.julia_interface import JuliaInterface
from gnn_acopf.utils.observers import DefaultObserver
from ruamel import yaml
import numpy as np
import copy
import pickle
from collections import defaultdict
import shlex


class PowerNetwork:
    def __init__(self, name, case_dict, area_name="area", pgmin_to_zero=False):
        self.name = name
        self.case_dict = case_dict
        self.area_name = area_name
        self.pgmin_to_zero = pgmin_to_zero

    @classmethod
    def from_casefile(cls, casepath: Path, **kwargs):
        case_dict = JuliaInterface.get_instance().load_from_julia(casepath)
        return cls(casepath.stem, case_dict, **kwargs)

    def load_by_area(self):
        load_by_area = defaultdict(int)
        for load in self.case_dict["load"].values():
            load_by_area[self.case_dict["bus"][str(load["load_bus"])][self.area_name]] += load["pd"]
        return load_by_area

    def load_scenarios_file(self, scenario_filepath):
        is_reading_chgtab = False
        default_load_by_area = self.load_by_area()
        self.varying_load = {}
        with scenario_filepath.open("r") as scenario_file:
            for l in scenario_file:
                if is_reading_chgtab:
                    if l.strip().startswith("];"):
                        return self.varying_load
                    change_info = l.strip().strip(";").split("\t")
                    # change label is
                    # label	prob	table	row	col	chgtype	newval
                    label = int(change_info[0])
                    assert change_info[2] == "CT_TAREALOAD"
                    assert change_info[4] == "CT_LOAD_ALL_P"
                    assert change_info[5] == "CT_REP"
                    area = int(change_info[3])
                    p = float(change_info[6])
                    cur_d = self.varying_load.get(label, {})
                    cur_d[area] = p / default_load_by_area[area] / 100
                    self.varying_load[label] = cur_d
                elif l.strip().startswith("chgtab = ["):
                    is_reading_chgtab = True

    def load_coordinates(self, gic_filepath):
        is_reading_coordinates = False
        all_bus_names = {b["name"]: k for k, b in self.case_dict["bus"].items()}
        # create alternate names
        old_bus_names = list(all_bus_names.keys())
        for b in old_bus_names:
            assert b[:-2] not in old_bus_names
            all_bus_names[b[:-2]] = all_bus_names[b]
        self.coordinates = {}
        with gic_filepath.open("r") as scenario_file:
            for l in scenario_file:
                if is_reading_coordinates:
                    if l.strip().startswith("0 /"):
                        return self.coordinates
                    #    1 'ODESSA 2' 0  31.9067 -102.2623    0.090
                    line = shlex.split(l)
                    # change label is
                    # label	prob	table	row	col	chgtype	newval
                    name = line[1] + " " + line[2]
                    long = float(line[3])
                    lat = float(line[4])
                    try:
                        bus_name = self.case_dict["bus"][all_bus_names[name]]["name"]
                    except KeyError:
                        bus_name = name
                    self.coordinates[bus_name] = np.array([long, lat])
                elif l.strip().startswith("GICFILEVRSN=3"):
                    is_reading_coordinates = True

    def get_coordinates(self, bus_name):
        name_rules = [bus_name, bus_name[:-2], " ".join((bus_name.split()[:-1])) + " 0"]
        for n in name_rules:
            if n in self.coordinates:
                return self.coordinates[n]
        if bus_name == "May-00":
            return self.coordinates["MAY 0"]
        raise ValueError(f"{bus_name} not found.")

    @classmethod
    def from_pickle(cls, picklepath, **kwargs):
        with picklepath.open("rb") as picklefile:
            pickle_dict = pickle.load(picklefile)
        return cls(name=picklepath.stem, **pickle_dict, **kwargs)

    @property
    def _descr_dict(self):
        descr_dict = {
            "case_dict": self.case_dict,
        }
        return descr_dict

    def to_pickle(self, picklepath):

        with picklepath.open("wb") as picklefile:
            pickle.dump(self._descr_dict, picklefile)

    @classmethod
    def from_yaml(cls, yamlpath):
        with yamlpath.open("r") as yamlfile:
            case_dict = yaml.safe_load(yamlfile)
        return cls(**case_dict)

    def to_yaml(self, yamlpath):
        with yamlpath.open("w") as yamlfile:
            yaml.dump(self._descr_dict, yamlfile)

    def get_scenario_area_scale(self, scenario_idx):
        area_scale = self.varying_load[scenario_idx]
        return self.case_dict, area_scale

    def create_scenario_case_dict(self, scenario_idx):
        area_scale = self.varying_load[scenario_idx]
        case_dict = copy.deepcopy(self.case_dict)
        for l in case_dict["load"]:
            load_area = case_dict["bus"][str(case_dict["load"][l]["load_bus"])][self.area_name]
            load_factor = area_scale[load_area]
            case_dict["load"][l]["pd"] *= load_factor
            case_dict["load"][l]["qd"] *= load_factor
        # TODO: This is necessary for the 2000 bus case
        if self.pgmin_to_zero:
            for g in case_dict["gen"]:
                case_dict["gen"][g]["pmin"] = 0
        case_dict["scenario_idx"] = scenario_idx
        return case_dict


def main(casename, area_name):
    #pn = PowerNetwork.from_casefile(Path(f"../../casefiles/case_{casename}.m"))
    #pn.to_pickle(Path(f"../../data/case_{casename}.pickle"))

    pn = PowerNetwork.from_pickle(Path(f"../../data/case_{casename}.pickle"), area_name=area_name)
    pn.load_scenarios_file(Path(f"../../casefiles/scenarios_{casename}.m"))
    obs = DefaultObserver(jl=JuliaInterface(),
                          solution_cache_file=Path(f"../../data/generated_solutions/case_{casename}_solutions.pickle"))
    # result = obs.to_geometric_representation(pn.case_dict)
    n_solved = 0
    all_case_dicts = {}
    for scenario_id in sorted(pn.varying_load.keys()):
        scen_casedict = pn.create_scenario_case_dict(scenario_id)
        result = obs.to_geometric_representation(scen_casedict)
        if result["solved"]:
            n_solved += 1
        all_case_dicts[scenario_id] = result
        print(f"Solved {n_solved} of {scenario_id} for {n_solved / scenario_id}")
    with (Path(".") / f"case_{casename}_data.pickle").open("wb") as pickle_file:
        pickle.dump(all_case_dicts, pickle_file)
    #


if __name__ == "__main__":
    casename, areaname = "ACTIVSg200", "zone"
    # casename, areaname = "ACTIVSg2000", "area"
    main(casename, areaname)
