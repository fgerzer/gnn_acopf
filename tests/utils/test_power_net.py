from gnn_acopf.utils.power_net import PowerNetwork
from tests.test_observer import _example_casedict

def test_create_scenario_case_dict():
    pn = PowerNetwork(
        name="test_net",
        case_dict=_example_casedict(),
    )
    pn.varying_load = {
        1: {1: 0.5, 2: 0.25, 3: 0.125},
        2: {1: 2, 2: 3, 3: 4}
    }

    conv_case_dict = pn.create_scenario_case_dict(1)
    assert conv_case_dict["load"]["3"]["pd"] == 3 * 0.125
    assert conv_case_dict["load"]["3"]["qd"] == 3.1 * 0.125
    assert conv_case_dict["load"]["35"]["pd"] == 35 * 0.25
    assert conv_case_dict["load"]["35"]["qd"] == 35.1 * 0.25
    assert conv_case_dict["load"]["9"]["pd"] == 9 * 0.5
    assert conv_case_dict["load"]["9"]["qd"] == 9.1 * 0.5

    conv_case_dict = pn.create_scenario_case_dict(1)
    assert conv_case_dict["load"]["3"]["pd"] == 3 * 0.125
    assert conv_case_dict["load"]["3"]["qd"] == 3.1 * 0.125
    assert conv_case_dict["load"]["35"]["pd"] == 35 * 0.25
    assert conv_case_dict["load"]["35"]["qd"] == 35.1 * 0.25
    assert conv_case_dict["load"]["9"]["pd"] == 9 * 0.5
    assert conv_case_dict["load"]["9"]["qd"] == 9.1 * 0.5

    conv_case_dict = pn.create_scenario_case_dict(2)
    assert conv_case_dict["load"]["3"]["pd"] == 3 * 4
    assert conv_case_dict["load"]["3"]["qd"] == 3.1 * 4
    assert conv_case_dict["load"]["35"]["pd"] == 35 * 3
    assert conv_case_dict["load"]["35"]["qd"] == 35.1 * 3
    assert conv_case_dict["load"]["9"]["pd"] == 9 * 2
    assert conv_case_dict["load"]["9"]["qd"] == 9.1 * 2
