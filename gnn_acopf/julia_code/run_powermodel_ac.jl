# Test code for JULIA power files

using PowerModels
using Ipopt
using JuMP
using JSON
# using .PP2PM

function run_powermodels(json_path)
    # pm = PP2PM.load_pm_from_json(json_path)
    ipopt_solver = JuMP.with_optimizer(Ipopt.Optimizer, print_level=5)
    # data = PowerModels.parse_file("../../casefiles/case_SyntheticUSA.m")
    # data = PowerModels.parse_file("../../casefiles/case_ACTIVSg2000.m")
    data = PowerModels.parse_file("../../casefiles/case9target.m")
    net = JSON.parse(JSON.json(data))
    result = PowerModels.run_ac_opf(net, ipopt_solver)


    print(json(data["bus"], 4))
    #print(keys(data))
    #result = run_ac_opf(data, with_optimizer(Ipopt.Optimizer))
    #result = PowerModels.run_ac_opf(pm, ipopt_solver,
    #                                setting = Dict("output" => Dict("branch_flows" => true)))
    return result
end

run_powermodels(false)
