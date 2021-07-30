from gnn_acopf.utils.observers import DefaultObserver
import numpy as np
from gnn_acopf.julia_interface import JuliaInterface

def _create_observer(create_jl=False):
    if create_jl:
        jl = JuliaInterface()
    else:
        jl = None
    return DefaultObserver(jl)

def _example_casedict():
    bus = {
        "1": {
            "area": 1,
            "base_kv": 101,
            "vmin": 0.901,
            "vmax": 1.101
        },
        "15": {
            "area": 1,
            "base_kv": 115,
            "vmin": 0.915,
            "vmax": 1.115
        },
        "3": {
            "area": 2,
            "base_kv": 103,
            "vmin": 0.903,
            "vmax": 1.103
        },
        "80": {
            "area": 3,
            "base_kv": 180,
            "vmin": 0.980,
            "vmax": 1.180
        }
    }
    load = {
        "3": {
            "load_bus": 80,
            "pd": 3,
            "qd": 3.1
        },
        "35": {
            "load_bus": 3,
            "pd": 35,
            "qd": 35.1
        },
        "9": {
            "load_bus": 1,
            "pd": 9,
            "qd": 9.1
        }
    }
    gen = {
        "2": {
            "gen_bus": 15,
            "qmin": 2,
            "qmax": 20,
            "vg": 2,
            "pmax": 20,
            "pmin": 2
        },
        "3": {
            "gen_bus": 15,
            "qmin": 3,
            "qmax": 30,
            "vg": 3,
            "pmax": 30,
            "pmin": 3
        },
        "20": {
            "gen_bus": 80,
            "qmin": 0,
            "qmax": 200,
            "vg": 20,
            "pmax": 200,
            "pmin": 0
        }
    }
    shunt = {
        "3": {
            "shunt_bus": 80,
            "gs": 3,
            "bs": 3
        }
    }
    branch = {
        "1": {
            "f_bus": 1,
            "t_bus": 15,
            "br_r": 115,
            "br_x": 115
        },
        "2": {
            "f_bus": 1,
            "t_bus": 3,
            "br_r": 13,
            "br_x": 13
        },
        "3": {
            "f_bus": 15,
            "t_bus": 80,
            "br_r": 1580,
            "br_x": 1580
        },
        "4": {
            "f_bus": 3,
            "t_bus": 80,
            "br_r": 380,
            "br_x": 380
        },
        "5": {
            "f_bus": 1,
            "t_bus": 80,
            "br_r": 180,
            "br_x": 180
        }
    }
    casedict = {
        "bus": bus,
        "load": load,
        "gen": gen,
        "shunt": shunt,
        "branch": branch
    }
    return casedict

def create_solutioN():
    solution = {

    }

def test_convert_bus():
    case_dict = _example_casedict()
    converted_dict = _create_observer()._convert_nodetype(case_dict, "bus")
    assert np.allclose(converted_dict["bus"],
                       np.array([
                           [101, 0.901, 1.101],
                           [115, 0.915, 1.115],
                           [103, 0.903, 1.103],
                           [180, 0.98, 1.18]
                       ]))


def test_convert_load():
    case_dict = _example_casedict()
    converted_dict = _create_observer()._create_node_to_bus(case_dict, "load")
    assert np.allclose(converted_dict["load"],
                       np.array([
                           [3, 3.1],
                           [35, 35.1],
                           [9, 9.1],
                       ]))
    assert np.allclose(converted_dict["load_to_bus"],
                       np.array([3, 2, 0]))


def test_convert_gen():
    case_dict = _example_casedict()
    converted_dict = _create_observer()._create_node_to_bus(case_dict, "gen")
    assert np.allclose(converted_dict["gen"],
                       np.array([
                           [2.0, 20, 2, 20, 2],
                           [0, 200, 20, 200, 0],
                           [3, 30, 3, 30, 3],
                       ]))
    assert np.allclose(converted_dict["gen_to_bus"],
                       np.array([1, 3, 1]))


def test_convert_shunt():
    case_dict = _example_casedict()
    converted_dict = _create_observer()._create_node_to_bus(case_dict, "shunt")
    assert np.allclose(converted_dict["shunt"],
                       np.array([
                           [3, 3],
                       ]))
    assert np.allclose(converted_dict["shunt_to_bus"],
                       np.array([3]))


def test_load_to_area():
    case_dict = _example_casedict()
    load_to_area = _create_observer()._extract_areas(case_dict)
    assert np.allclose(load_to_area, np.array(
        [3, 2, 1]
    ))


def test_create_edges():
    case_dict = _example_casedict()
    converted_dict = _create_observer()._create_edges(case_dict)
    branch_index = converted_dict["branch_index"]
    branch_attr = converted_dict["branch_attr"]
    assert branch_index.shape == (2, 10)
    assert np.allclose(branch_index, np.array([
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [0, 3],
        [1, 0],
        [2, 0],
        [3, 1],
        [3, 2],
        [3, 0],
    ]).T)

    assert np.allclose(branch_attr, np.array([
        [115, 115],
        [13, 13],
        [1580, 1580],
        [380, 380],
        [180, 180],
        [115, 115],
        [13, 13],
        [1580, 1580],
        [380, 380],
        [180, 180]
    ]))

def test_scale_load_by_area():
    obs = _create_observer()
    case_dict = _example_casedict()
    data_dict = obs._create_node_to_bus(case_dict, "load")
    orig_load = data_dict["load"]
    load_to_area = obs._extract_areas(case_dict)
    area_scale = {
        1: 2,
        2: 3,
        3: 4
    }
    data_dict = obs._scale_load_by_area(data_dict, orig_load, load_to_area, area_scale)
    assert np.allclose(data_dict["load"], np.array(np.array([
                           [12, 12.4],
                           [105, 105.3],
                           [18, 18.2],
                       ])))
    area_scale = {
        1: 1,
        2: 1,
        3: 1
    }
    # testing multiple to make sure we don't end up with leaking state
    data_dict = obs._scale_load_by_area(data_dict, orig_load, load_to_area, area_scale)
    assert np.allclose(data_dict["load"], np.array(np.array([
                           [3, 3.1],
                           [35, 35.1],
                           [9, 9.1],
                       ])))
    area_scale = {
        1: 2,
        2: 3,
        3: 4
    }
    data_dict = obs._scale_load_by_area(data_dict, orig_load, load_to_area, area_scale)
    assert np.allclose(data_dict["load"], np.array(np.array([
                           [12, 12.4],
                           [105, 105.3],
                           [18, 18.2],
                       ])))

def _create_result_dict_and_numpy():
    opf_result = {
        "solution": {
            "bus": {
                "1": {"va": 1.0, "vm": 1.0},
                "15": {"va": 15.0, "vm": 15.0},
                "3": {"va": 3.0, "vm": 3.0},
                "80": {"va": 80.0, "vm": 80.0}
            },
            "branch": {
                "1": {"pf": 1.0, "pt": 1.0, "qf": 1.0, "qt": 1.0},
                "2": {"pf": 2.0, "pt": 2.0, "qf": 2.0, "qt": 2.0},
                "3": {"pf": 3.0, "pt": 3.0, "qf": 3.0, "qt": 3.0},
                "4": {"pf": 4.0, "pt": 4.0, "qf": 4.0, "qt": 4.0},
                "5": {"pf": 5.0, "pt": 5.0, "qf": 5.0, "qt": 5.0},
            },
            "gen": {
                "2": {"pg": 2.0, "qg": 2.0},
                "3": {"pg": 3.0, "qg": 3.0},
                "20": {"pg": 20.0, "qg": 20.0}
            }
        },
        "termination_status": "SOLVED"
    }

    target_dict = {
        "target_bus": np.array(np.array([
            [1, 1],
            [15, 15],
            [3, 3],
            [80, 80]
        ])),
        "target_branch": np.array(np.array([
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5]
        ])),
        "target_gen":  np.array(np.array([
            [2, 2],
            [20, 20],
            [3, 3],
        ]))
    }
    return opf_result, target_dict


def test_convert_bus_target():
    solution_dict, real_target_dict = _create_result_dict_and_numpy()
    class MockJuliaInterface:
        def run_opf(self, case_dict, model):
            return solution_dict, None

    obs = _create_observer()
    obs.jl = MockJuliaInterface()
    target_dict = obs._build_target(case_dict=None)

    assert np.allclose(target_dict["target_bus"], real_target_dict["target_bus"])

    assert np.allclose(target_dict["target_branch"], real_target_dict["target_branch"])

    assert np.allclose(target_dict["target_gen"], real_target_dict["target_gen"])

    assert target_dict["solved"]


def test_translate_output_to_results_dict():
    real_solution_dict, model_output = _create_result_dict_and_numpy()
    case_dict = _example_casedict()
    obs = _create_observer()
    created_results = obs.translate_output_to_results_dict(data=None, output=model_output, case_dict=case_dict)
    for key in ["bus", "branch", "gen"]:
        assert created_results["solution"][key] == real_solution_dict["solution"][key]
