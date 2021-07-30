import PowerModels
import Ipopt
import JuMP
import JSON
# using .PP2PM

function run_pf(json_net, method, print_level=5, max_iter=Nothing)
    # pm = PP2PM.load_pm_from_json(json_path)
    if max_iter == Nothing
        solver = JuMP.with_optimizer(Ipopt.Optimizer, print_level=print_level)
    else
        solver = JuMP.with_optimizer(Ipopt.Optimizer, print_level=print_level)
    end
    # solver = JuMP.with_optimizer(GLPK.Optimizer, tm_lim = 60.0, msg_lev = GLPK.OFF)
    data = JSON.parse(json_net)
    if method == "ac"
        result = PowerModels.run_ac_pf(data, solver, setting = Dict("output" => Dict("branch_flows" => true)))
    elseif method == "dc"
        result = PowerModels.run_dc_pf(data, solver, setting = Dict("output" => Dict("branch_flows" => true)))
    else
        throw(DomainError(method), "method not recognized")
    end
    return JSON.json(result), JSON.json(data)
end
