"""Interface for PowerModels.jl using the scripts under julia_code"""

import json
from pathlib import Path

_ji = None

class JuliaInterface:
    def __init__(self):
        from julia import Main
        julia_scripts_path = Path(__file__).parent / "julia_code"
        self.julia_opf = Main.include(str(julia_scripts_path / "solve_opf.jl"))
        self.julia_pf = Main.include(str(julia_scripts_path / "solve_pf.jl"))
        self.julia_load = Main.include(str(julia_scripts_path / "loadcase.jl"))

        # compile
        net = self.load_from_julia(Path(julia_scripts_path / "case5.m"))
        self.run_opf(net, "dcac", print_level=0)

    @classmethod
    def get_instance(cls):
        global _ji
        if _ji is None:
            _ji = cls()
        return _ji

    def load_from_julia(self, casefile_path):
        loaded = self.julia_load(str(casefile_path))
        return json.loads(loaded)

    def _json_opf(self, net, method, print_level=0):
        net = json.dumps(net)
        result, net = self.julia_opf(net, method, print_level)
        return json.loads(result), json.loads(net)

    def _json_pf(self, net, method, print_level=0, max_iter=None):
        net = json.dumps(net)
        result, net = self.julia_pf(net, method, print_level, max_iter)
        return json.loads(result), json.loads(net)

    def run_opf(self, net, method, previous_result=None, print_level=0):

        if method in ["dc", "ac"]:
            if previous_result is not None:
                net = self.integrate_result_into_net(net, previous_result)
            result, net = self._json_opf(net, method, print_level)
        elif method == "dcac":
            assert previous_result is None
            result, net = self._json_opf(net, "dc", print_level)
            time_ac = result["solve_time"]
            net = self.integrate_result_into_net(net, result)
            result, net = self._json_opf(net, "ac", print_level)
            result["solve_time"] += time_ac
        elif method == "acac":
            assert previous_result is None
            result, net = self._json_opf(net, "ac", print_level)
            net = self.integrate_result_into_net(net, result)
            result, net = self._json_opf(net, "ac", print_level)
        else:
            raise NotImplementedError(f"Don't know method {method}")
        return result, net

    def run_pf(self, net, method, previous_result=None, print_level=0, max_iter=None):
        assert method in ["dc", "ac"]
        if previous_result is not None:
            net = self.integrate_result_into_net(net, previous_result)
        result, net = self._json_pf(net, method, print_level, max_iter=max_iter)
        return result, net

    def integrate_result_into_net(self, net, result):
        if "bus" in result["solution"]:
            for bus_idx, bus_result in result["solution"]["bus"].items():
                for k, v in bus_result.items():
                    if v is not None:
                        net["bus"][bus_idx][k + "_start"] = v
                        net["bus"][bus_idx][k] = v
        if "gen" in result["solution"]:
            for gen_idx, gen_result in result["solution"]["gen"].items():
                for k, v in gen_result.items():
                    if v is not None:
                        net["gen"][gen_idx][k + "_start"] = v
                        net["gen"][gen_idx][k] = v
        if "branch" in result["solution"]:
            for branch_idx, branch_result in result["solution"]["branch"].items():
                for k, v in branch_result.items():
                    if v is not None:
                        net["branch"][branch_idx][k + "_start"] = v
                        net["branch"][branch_idx][k] = v
        return net
