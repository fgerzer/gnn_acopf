import numpy as np
import pickle
import copy


class Scaler:
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, data):
        for k in self.scales:
            min_val, max_val = self.scales[k]
            data[k] = ((data[k] - min_val) / (max_val - min_val)).astype(np.float32)
        return data


class DefaultObserver:
    node_params = {
        "bus": ["base_kv", "vmin", "vmax"],
        "shunt": ["bs"],
        "load": ["pd", "qd"],
        "gen": ["qmin", "qmax", "vg", "pmax", "pmin"],
    }

    branch_params = ["br_r", "br_x"]

    node_targets = {
        "gen": ["pg", "qg"],
        "bus": ["va", "vm"],
    }

    branch_targets = ["pf", "pt", "qf", "qt"]

    def __init__(self, jl, area_name="area", solution_cache_dir=None, scaler=None):
        self.jl = jl
        self.target_cache = {}
        self.scaler = scaler
        if solution_cache_dir is not None:
            self.solution_cache = {}
            for solution_cache_file in sorted(solution_cache_dir.glob("*.pickle")):
                scenario_id = int(solution_cache_file.stem.split("_")[2])
                self.solution_cache[scenario_id] = solution_cache_file
        else:
            self.solution_cache = None
        self.cache = None
        self.area_name = area_name

    def _create_edges(self, case_dict):
        bus_order = sorted(case_dict["bus"].keys())
        bus_mapping = {int(b): i for i, b in enumerate(bus_order)}
        branch = case_dict["branch"]
        branch_params = self.branch_params
        branch_order = sorted(branch.keys())
        n_branches = len(branch)
        np_edge_index = np.zeros(shape=(2, n_branches * 2), dtype=np.long)
        np_edge_attr = np.zeros(shape=(2 * n_branches, len(branch_params)), dtype=np.float32)
        # add the real branches
        for i, b in enumerate(branch_order):
            np_edge_index[:, i] = [bus_mapping[branch[b]["f_bus"]], bus_mapping[branch[b]["t_bus"]]]
            np_edge_attr[i] = [branch[b][p] for p in branch_params]

            np_edge_index[:, i + n_branches] = [bus_mapping[branch[b]["t_bus"]], bus_mapping[branch[b]["f_bus"]]]
            np_edge_attr[i + n_branches] = [branch[b][p] for p in branch_params]

        return {
            "branch_index": np_edge_index,
            "branch_attr": np_edge_attr
        }

    def _convert_nodetype(self, case_dict, key):
        nodes = case_dict[key]
        params = self.node_params[key]
        node_order = sorted(nodes.keys())
        np_nodes = np.zeros(shape=(len(nodes), len(params)), dtype=np.float32)
        for i, b in enumerate(node_order):
            np_nodes[i] = [nodes[b][p] for p in params]
        return {
            key: np_nodes,
        }

    def _convert_nodetarget(self, solution_dict, key):
        nodes = solution_dict[key]
        params = self.node_targets[key]
        node_order = sorted(nodes.keys())
        np_nodes = np.zeros(shape=(len(nodes), len(params)), dtype=np.float32)
        for i, b in enumerate(node_order):
            np_nodes[i] = [nodes[b][p] for p in params]
        return np_nodes

    def _convert_branch_target(self, solution_dict):
        branch = solution_dict["branch"]
        branch_order = sorted(branch.keys())
        branch_params = self.branch_targets
        n_branches = len(branch)
        np_tgt = np.zeros(shape=(2 * n_branches, len(branch_params)), dtype=np.float32)
        for i, b in enumerate(branch_order):
            np_tgt[i] = [branch[b][p] for p in branch_params]
            np_tgt[i + n_branches] = [branch[b][p] for p in branch_params]
        return np_tgt

    def _build_target(self, case_dict, scenario_idx=None):
        if self.solution_cache is None or scenario_idx is None:
            result, _ = self.jl.run_opf(case_dict, "ac")
        else:
            with self.solution_cache[scenario_idx].open("rb") as pickle_file:
                result = pickle.load(pickle_file)
        tgt_dict = {}
        solution = result["solution"]
        solved = "SOLVED" in result["termination_status"]
        tgt_dict["target_bus"] = self._convert_nodetarget(solution, "bus")
        tgt_dict["target_gen"] = self._convert_nodetarget(solution, "gen")

        tgt_dict["target_branch"] = self._convert_branch_target(solution)
        tgt_dict["solved"] = [solved]
        return tgt_dict

    def _extract_areas(self, case_dict):
        loads = case_dict["load"]
        load_order = sorted(loads.keys())
        np_areas = np.zeros(shape=[len(load_order)], dtype=np.long)
        for i, b in enumerate(load_order):
            corresponding_bus = case_dict["bus"][str(loads[b]["load_bus"])]
            area = corresponding_bus[self.area_name]
            np_areas[i] = area
        return np_areas

    def _create_node_to_bus(self, case_dict, node_type):
        nodes = case_dict[node_type]
        bus_order = sorted(case_dict["bus"].keys())
        params = self.node_params[node_type]
        node_order = sorted(nodes.keys())
        np_nodes = np.zeros(shape=(len(nodes), len(params)), dtype=np.float32)
        bus_mapping = {int(b): i for i, b in enumerate(bus_order)}
        node_to_bus = np.zeros(shape=(len(node_order)), dtype=np.long)

        for i, b in enumerate(node_order):
            np_nodes[i] = [nodes[b][p] for p in params]
            node_to_bus[i] = bus_mapping[nodes[b][f"{node_type}_bus"]]
        return {node_type: np_nodes, f"{node_type}_to_bus": node_to_bus}


    def _scale_load_by_area(self, data_dict, orig_load, load_to_area, area_scale):
        load = copy.deepcopy(orig_load)
        pd_index = self.node_params["load"].index("pd")
        area_multiplier = np.ones(shape=[int(np.max(load_to_area) + 1)])
        for area in area_scale:
            area_multiplier[area] = area_scale[area]

        load[:, pd_index] *= area_multiplier[load_to_area.astype(np.int)]
        qd_index = self.node_params["load"].index("qd")
        load[:, qd_index] *= area_multiplier[load_to_area.astype(np.int)]
        data_dict["load"] = load
        return data_dict

    def to_geometric_representation(self, case_dict, area_scale, scenario_idx):
        if self.cache is None:
            data_dict = {}
            data_dict.update(self._convert_nodetype(case_dict, "bus"))
            data_dict.update(self._create_node_to_bus(case_dict, "load"))
            data_dict.update(self._create_node_to_bus(case_dict, "gen"))
            data_dict.update(self._create_node_to_bus(case_dict, "shunt"))

            load_to_area = self._extract_areas(case_dict)
            data_dict.update(self._create_edges(case_dict))
            orig_load = data_dict["load"]
            self.cache = [load_to_area, orig_load, data_dict]
        else:
            load_to_area, orig_load, data_dict = self.cache
        # apply area multiplier
        if scenario_idx not in self.target_cache:
            target_dict = self._build_target(
                case_dict,
                scenario_idx=scenario_idx,
            )
            self.target_cache[scenario_idx] = target_dict
        target_dict = copy.deepcopy(self.target_cache[scenario_idx])
        data_dict.update(target_dict)
        data_dict = self._scale_load_by_area(data_dict, orig_load, load_to_area, area_scale)
        data_dict["scenario_idx"] = [scenario_idx]
        if self.scaler:
            data_dict = self.scaler(copy.deepcopy(data_dict))
        return data_dict

    def translate_output_to_results_dict(self, data, output, case_dict, keys_to_consider=None):
        result_dict = {"solution": {}}
        if keys_to_consider is None:
            keys_to_consider = {"gen", "branch", "bus"}
        if "target_gen" in output and "gen" in keys_to_consider:
            gen_order = sorted(case_dict["gen"].keys())
            gen = {}
            for i, gen_idx in enumerate(gen_order):
                gen[gen_idx] = {
                    tgt: float(output["target_gen"][i, j]) for j, tgt in enumerate(self.node_targets["gen"])
                }
            result_dict["solution"]["gen"] = gen

        if "target_branch" in output and "branch" in keys_to_consider:
            branch_order = sorted(case_dict["branch"].keys())
            branch = {}
            for i, branch_idx in enumerate(branch_order):
                branch[branch_idx] = {
                    tgt: float(output["target_branch"][i, j]) for j, tgt in enumerate(self.branch_targets)
                }
            result_dict["solution"]["branch"] = branch

        if "target_bus" in output and "bus" in keys_to_consider:
            bus_order = sorted(case_dict["bus"].keys())
            bus = {}
            for i, bus_idx in enumerate(bus_order):
                bus[bus_idx] = {
                    tgt: float(output["target_bus"][i, j]) for j, tgt in enumerate(self.node_targets["bus"])
                }
            result_dict["solution"]["bus"] = bus

        return result_dict

    @property
    def n_node_features(self):
        return {k: len(v) for k, v in self.node_params.items()}

    @property
    def n_node_targets(self):
        return {k: len(v) for k, v in self.node_targets.items()}

    @property
    def n_branch_features(self):
        return len(self.branch_params)

    @property
    def n_branch_targets(self):
        return len(self.branch_targets)
