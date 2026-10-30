# Helpers for reading the per-W result files written by run_ising_Wbank.jl
#
#   f = jldopen(file)            # lazy; nothing loaded yet
#   f["0.5/tri/E_selfish"]       # one variable
#   f["meta/W"]                  # the W used
#   close(f)

using JLD2

"""All 8 saved variables for one (γ_name, K_name) condition as a Dict."""
function load_condition(file, γ_name, K_name)
    jldopen(file, "r") do f
        g = f["$(γ_name)/$(K_name)"]
        Dict(k => g[k] for k in keys(g))
    end
end

"""
Collect one variable across many W files, e.g.
    collect_variable(readdir(dir; join=true), "0.5", "tri", "avg_tissue_frustration")
Returns Dict(W_name => value).
"""
function collect_variable(files, γ_name, K_name, varname)
    out = Dict{String,Any}()
    for file in filter(endswith(".jld2"), files)
        jldopen(file, "r") do f
            out[f["meta/W_name"]] = f["$(γ_name)/$(K_name)/$(varname)"]
        end
    end
    return out
end
