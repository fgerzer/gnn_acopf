import PowerModels
import Ipopt
import JuMP
import JSON
# using .PP2PM

function run_opf(json_net, method, print_level=5)
    # pm = PP2PM.load_pm_from_json(json_path)
    solver = JuMP.with_optimizer(Ipopt.Optimizer, print_level=print_level)
    # solver = JuMP.with_optimizer(GLPK.Optimizer, tm_lim = 60.0, msg_lev = GLPK.OFF)

    data = JSON.parse(json_net)
    if method == "ac"
        result = PowerModels.run_ac_opf(data, solver, setting = Dict("output" => Dict("branch_flows" => true)))
    elseif method == "dc"
        result = PowerModels.run_dc_opf(data, solver, setting = Dict("output" => Dict("branch_flows" => true)))
    else
        throw(DomainError(method), "method not recognized")
    end
    return JSON.json(result), JSON.json(data)
end
