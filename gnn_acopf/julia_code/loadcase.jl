import PowerModels
import JSON
import PyCall

function _load_case(casefile::String)
    data = PowerModels.parse_file(casefile)
    net = JSON.json(data)
    return net
end

return load_case = PyCall.pyfunctionret(_load_case, String, String)
