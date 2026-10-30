####################################################################
# run_ising_Wbank_opti.jl  (was run_ising_Wbank_V3.jl)
#
# Runs the γ-range / GRN study over a bank of pre-defined W matrices,
# multithreaded, writing ONE .jld2 file per W (all γ, all GRNs, all variables).
#
# V3: uses ising_core_V3.jl (|ΔE| < ZERO_TOL treated as a tie, see its header),
# writes to .../nonrcep_casestudy/gamma_data_V3 by default, records
# meta/core_version = "V3" and meta/zero_tol, and refuses to merge into a file
# that was not written by V3. V1 results (ising_core.jl) are left untouched.
#
# Usage (positional, like the old script):
#   julia -t <threads> run_ising_Wbank_opti.jl \
#       n_cells n_genes n_sims max_sweeps β β_name σ_name \
#       "γ1,γ2,..." "γname1,γname2,..." \
#       W_bank W_start W_end [W_signs="1"] [seed=1234]
#
#   W_bank   : directory containing W_sym9x9_<i>.csv   (W_FILE_PATTERN below)
#              or a .jld2 file with key "W" = n_genes x n_genes x n_W array
#   W_start / W_end : inclusive range of W indices this job processes
#   W_signs  : "1", "-1" or "1,-1"  (-1 gives the old "-W<i>" runs)
#   seed     : base seed; every (W, sign, γ, GRN, sim) gets its own derived stream,
#              so results are reproducible and independent of thread count
#
# Parallelism: every (W, sign, γ, GRN, sim) is a task on a shared queue
# consumed by Threads.nthreads() workers. When the last task of a W finishes,
# that W's file is written.
#
# Merging: if a W file already exists, only the (γ, GRN) groups it does not
# contain yet are run, and they are MERGED into that file (old groups kept,
# meta/gamma_* and meta/K* updated). So a new set of γ values can be run with
# the same command and lands in the same per-W files. Before running, every
# existing file is checked against this run (n_cells, n_genes, n_sims,
# max_sweeps, β, σ_name, seed, sim_kwargs, W, initial states, GRN matrices,
# γ name <-> value); any mismatch is an error, nothing is mixed silently.
# A killed / timed-out job can simply be resubmitted.
# Do not run two jobs on the same W range at the same time.
####################################################################

include(joinpath(@__DIR__, "..", "functions", "ising_core_opti.jl"))
const CORE_VERSION = "V3"

using DelimitedFiles
using Random
using JLD2
using Statistics
using LinearAlgebra

LinearAlgebra.BLAS.set_num_threads(1)

const base_path = get(ENV, "ISING_OUT_DIR",
    "/data/scratch/projects/punim0638/woodward/nonrcep_casestudy/gamma_data_V3")
const W_FILE_PATTERN = i -> "W_sym9x9_$(i).csv"

# ----------- arguments ----------- #

length(ARGS) >= 12 || error("expected at least 12 arguments, got $(length(ARGS)); see header of run_ising_Wbank_opti.jl")

const n_cells    = parse(Int, ARGS[1])
const n_genes    = parse(Int, ARGS[2])
const n_sims     = parse(Int, ARGS[3])
const max_sweeps = parse(Int, ARGS[4])

const β_test = parse(Float64, ARGS[5])
const β_name = ARGS[6]
const σ_name = ARGS[7]

const γ_values = parse.(Float64, strip.(split(ARGS[8], ",")))
const γ_names  = String.(strip.(split(ARGS[9], ",")))
length(γ_values) == length(γ_names) || error("γ_values and γ_names must have the same length")

const W_bank  = ARGS[10]
const W_start = parse(Int, ARGS[11])
const W_end   = parse(Int, ARGS[12])
const W_signs = length(ARGS) >= 13 ? parse.(Int, strip.(split(ARGS[13], ","))) : [1]
const base_seed = length(ARGS) >= 14 ? parse(Int, ARGS[14]) : 1234
all(in((1, -1)), W_signs) || error("W_signs must be 1 and/or -1")

const sim_kwargs = (equilibrium_window = 30,
                    flip_tol           = 5,
                    oscillation_window = 100,
                    flip_range_tol     = 5)

mkpath(base_path)

# ----------- GRNs ----------- #

K_tree = Float64.([
    0  1  1  0  0  0  0  0  0;
    0  0  0  1  0  0  0  0  0;
    0  0  0  0  1  1  0  0  0;
    0  0  0  0  0  0 -1  0  0;
    0  0  0  0  0  0  0 -1  0;
    0  0  0  0  0  0  0  0 -1;
    0  0  0  0  0  0  0  0  0;
    0  0  0  0  0  0  0  0  0;
    0  0  0  0  0  0  0  0  0
])

K_negtri = Float64.([
    0   1  0  0  0  0  0  0  0;
    0   0  1  1  0  0  0  0  0;
   -1   0  0  0  0  1  0  0  0;
    0   0  0  0  0  0 -1  0  0;
    0   0  0  0  0  0  0 -1  0;
    0   0  0  0  0  0  0  0 -1;
    0   0  0  0  0  0  0  0  0;
    0   0  0  0  0  0  0  0  0;
    0   0  0  0  0  0  0  0  0
])

K_ICFF = Float64.([
    0   1  -1  0  0  0  0  0  0;
    0   0   1  1  0  0  0  0  0;
    0   0   0  0  0  1  0  0  0;
    0   0   0  0  0  0 -1  0  0;
    0   0   0  0  0  0  0 -1  0;
    0   0   0  0  0  0  0  0 -1;
    0   0   0  0  0  0  0  0  0;
    0   0   0  0  0  0  0  0  0;
    0   0   0  0  0  0  0  0  0
])

K_CFF = Float64.([
    0   1   1  0  0  0  0  0  0;
    0   0   1  1  0  0  0  0  0;
    0   0   0  0  0  1  0  0  0;
    0   0   0  0  0  0 -1  0  0;
    0   0   0  0  0  0  0 -1  0;
    0   0   0  0  0  0  0  0 -1;
    0   0   0  0  0  0  0  0  0;
    0   0   0  0  0  0  0  0  0;
    0   0   0  0  0  0  0  0  0
])

K_negtritri = Float64.([
    0   -1  0  0  0  0  0  0  0;
    0   0  -1  1  0  0  0  0  0;
   -1   0  0  0  0  1  0  0  0;
    0   0  0  0  0  0 -1  0  0;
    0   0  0  0  0  0  0 -1  0;
    0   0  0  0  0  0  0  0 -1;
    0   0  0  0  0  0  0  0  0;
    0   0  0  0  0  0  0  0  0;
    0   0  0  0  0  0  0  0  0
])

K_negtritwin = Float64.([
    0   -1  0  0  0  0  0  0  0;
    0   0  1  1  0  0  0  0  0;
   -1   0  0  0  0  1  0  0  0;
    0   0  0  0  0  0 -1  0  0;
    0   0  0  0  0  0  0 -1  0;
    0   0  0  0  0  0  0  0 -1;
    0   0  0  0  0  0  0  0  0;
    0   0  0  0  0  0  0  0  0;
    0   0  0  0  0  0  0  0  0
])

const GRNs = [
    ("tree",   K_tree),
    ("negtri", K_negtri),
    ("ICFF",   K_ICFF),
    ("CFF",    K_CFF),
    ("negtritri", K_negtritri),
    ("negtritwin", K_negtritwin)
]

all(size(K) == (n_genes, n_genes) for (_, K) in GRNs) || error("GRNs must be $(n_genes)x$(n_genes)")

# ----------- W bank ----------- #

function load_W_bank(bank::AbstractString, ids)
    if endswith(bank, ".jld2")
        Wall = JLD2.load(bank, "W")
        return Dict(i => Matrix{Float64}(Wall[:, :, i]) for i in ids)
    else
        return Dict(i => Matrix{Float64}(readdlm(joinpath(bank, W_FILE_PATTERN(i)), ',')) for i in ids)
    end
end

W_name(i, sgn) = sgn == 1 ? "W$(i)" : "-W$(i)"

outfile(i, sgn) = joinpath(base_path,
    "$(W_name(i, sgn))_$(β_name)_$(σ_name)_$(n_sims)_max$(max_sweeps)sweeps_flipequ.jld2")

# deterministic, independent RNG stream per task, keyed by the γ VALUE (not its position
# in the γ list) so a result does not depend on which γ values were run together.
# Groups written before this change were keyed by list position (seed_scheme = "gamma_index").
const SEED_SCHEME = "gamma_value"
task_rng(i, sgn, γ, ki, sim) = Xoshiro(UInt64[base_seed, i, sgn == 1 ? 1 : 2, reinterpret(UInt64, γ), ki, sim])

function make_σ_init(i, sim)
    if σ_name == "σrand"
        # a different random start for every sim (and every W); sim k of a given W uses the
        # same start for all signs / γ / GRNs, so comparisons between conditions are paired.
        # (third seed entry 0 never collides with task_rng, which uses 1 or 2 there)
        return rand(Xoshiro(UInt64[base_seed, i, 0, 0, 0, sim]), [-1, 1], n_cells, n_genes)
    elseif σ_name == "σneg1"
        return -1.0 .* ones(n_cells, n_genes)
    elseif σ_name == "σpos1"
        return 1.0 .* ones(n_cells, n_genes)
    else
        error("Unknown σ_name: $σ_name")
    end
end

# ----------- existing files ----------- #

"""
existing_state(file, W, σ_inits)

Read an existing per-W file, check it is compatible with this run, and return
`(done_groups, gamma_map, K_map)`: the set of "γname/Kname" groups already in
the file, and its γ name => value and K name => matrix maps. Errors on any mismatch.
"""
function existing_state(file, W, σ_inits)
    jldopen(file, "r") do f
        m(k) = f["meta/$k"]
        bad(what) = error("$(basename(file)): existing file has a different $what than this run; " *
                          "use a different ISING_OUT_DIR or matching arguments")

        (haskey(f, "meta/core_version") && m("core_version") == CORE_VERSION) ||
            bad("core version (file was not written by run_ising_Wbank_$(CORE_VERSION).jl)")
        isequal(m("zero_tol"), ZERO_TOL) || bad("zero_tol ($(m("zero_tol")))")
        m("n_cells")    == n_cells    || bad("n_cells ($(m("n_cells")))")
        m("n_genes")    == n_genes    || bad("n_genes ($(m("n_genes")))")
        m("n_sims")     == n_sims     || bad("n_sims ($(m("n_sims")))")
        m("max_sweeps") == max_sweeps || bad("max_sweeps ($(m("max_sweeps")))")
        isequal(m("beta"), β_test)    || bad("β ($(m("beta")))")
        m("sigma_name") == σ_name     || bad("σ_name ($(m("sigma_name")))")
        m("seed")       == base_seed  || bad("seed ($(m("seed")))")
        m("sim_kwargs") == Dict(pairs(sim_kwargs)) || bad("sim_kwargs ($(m("sim_kwargs")))")
        m("W")          == W          || bad("W matrix")
        m("sigma_inits") == σ_inits   || bad("set of initial states")

        gamma_map = Dict{String,Float64}(zip(m("gamma_names"), m("gamma_values")))
        K_map     = Dict{String,Matrix{Float64}}(m("K"))

        for (γ, γn) in zip(γ_values, γ_names)
            haskey(gamma_map, γn) && !isequal(gamma_map[γn], γ) &&
                bad("value for γ name $γn ($(gamma_map[γn]) vs $γ)")
            for (γn_old, γ_old) in gamma_map
                isequal(γ_old, γ) && γn_old != γn && bad("name for γ = $γ ($γn_old vs $γn)")
            end
        end
        for (Kn, K) in GRNs
            haskey(K_map, Kn) && K_map[Kn] != K && bad("matrix for GRN $Kn")
        end

        done = Set{String}()
        for γn in keys(gamma_map), Kn in keys(K_map)
            γn in keys(f) && Kn in keys(f[γn]) && push!(done, "$(γn)/$(Kn)")
        end
        return done, gamma_map, K_map
    end
end

"""Read every (γ, GRN) group of an existing file: Dict("γname/Kname" => Dict(var => value))."""
function read_groups(file)
    jldopen(file, "r") do f
        out = Dict{String,Dict{String,Any}}()
        for γn in keys(f)
            γn == "meta" && continue
            for Kn in keys(f[γn])
                g = f["$(γn)/$(Kn)"]
                out["$(γn)/$(Kn)"] = Dict{String,Any}(v => g[v] for v in keys(g))
            end
        end
        return out
    end
end

# ----------- work units ----------- #

struct Unit
    W_id::Int
    sgn::Int
    W::Matrix{Float64}            # signed, unscaled
    σ_inits::Vector{Matrix}       # one start per sim
    pairs::Vector{Tuple{Int,Int}} # (γ index, GRN index) still to run for this W
    gamma_map::Dict{String,Float64}          # γ already in the file (empty if new file)
    K_map::Dict{String,Matrix{Float64}}      # GRNs already in the file
    slots::Vector{Any}            # per-sim results, index = slot(pi, sim)
    remaining::Threads.Atomic{Int}
end

const nγ = length(γ_values)
const nK = length(GRNs)
slot(pi, sim) = (pi - 1) * n_sims + sim

const write_lock = ReentrantLock()

function write_unit(u::Unit)
    file = outfile(u.W_id, u.sgn)
    tmp  = file * ".tmp"

    new_groups = Dict{String,Any}()
    for (pi, (gi, ki)) in enumerate(u.pairs)
        sims = [u.slots[slot(pi, sim)] for sim in 1:n_sims]
        new_groups["$(γ_names[gi])/$(GRNs[ki][1])"] = aggregate_sims(sims, max_sweeps)
    end

    # merged γ list, sorted by value so meta/gamma_values can be used directly as an x axis
    gamma_map = merge(u.gamma_map, Dict(γ_names[gi] => γ_values[gi] for (gi, _) in u.pairs))
    γ_sorted  = sort(collect(gamma_map); by = last)
    K_map     = merge(u.K_map, Dict{String,Matrix{Float64}}(GRNs[ki] for (_, ki) in u.pairs))
    K_names   = vcat([Kn for (Kn, _) in GRNs if haskey(K_map, Kn)],
                     sort([Kn for Kn in keys(K_map) if !any(==(Kn) ∘ first, GRNs)]))

    lock(write_lock) do
        old_groups = isfile(file) ? read_groups(file) : Dict{String,Dict{String,Any}}()

        # compress: NaN padding in E_selfish etc. shrinks ~100x (built-in Deflate, JLD2 >= 0.5)
        jldopen(tmp, "w"; compress = true) do f
            f["meta/W"]          = u.W
            f["meta/W_name"]     = W_name(u.W_id, u.sgn)
            f["meta/sigma_inits"] = u.σ_inits      # Vector, index = sim
            f["meta/gamma_values"] = last.(γ_sorted)
            f["meta/gamma_names"]  = first.(γ_sorted)
            f["meta/K_names"]    = K_names
            f["meta/K"]          = K_map
            f["meta/beta"]       = β_test
            f["meta/beta_name"]  = β_name
            f["meta/sigma_name"] = σ_name
            f["meta/n_cells"]    = n_cells
            f["meta/n_genes"]    = n_genes
            f["meta/n_sims"]     = n_sims
            f["meta/max_sweeps"] = max_sweeps
            f["meta/seed"]       = base_seed
            f["meta/sim_kwargs"] = Dict(pairs(sim_kwargs))
            f["meta/core_version"] = CORE_VERSION
            f["meta/zero_tol"]   = ZERO_TOL
            for (g, vars) in old_groups, (name, val) in vars
                f["$(g)/$(name)"] = val
            end
            for (g, res) in new_groups
                for (name, val) in pairs(res)
                    f["$(g)/$(name)"] = val
                end
                f["$(g)/seed_scheme"] = SEED_SCHEME
            end
        end
        mv(tmp, file; force = true)
    end
    empty!(u.slots)   # free memory
    return file
end

# ----------- main ----------- #

function main()
    t0 = time()
    lat = Lattice(n_cells; n_neighbors = 8)

    candidates = [(i, s) for i in W_start:W_end for s in W_signs]
    ids = unique(first.(candidates))
    Ws = load_W_bank(W_bank, ids)
    σ_inits = Dict(i => Matrix[make_σ_init(i, sim) for sim in 1:n_sims] for i in ids)

    # check every existing file first, so a mismatch stops the job before any compute
    units = Unit[]
    n_skip = 0
    n_merge = 0
    for (i, s) in candidates
        W = s .* Ws[i]
        file = outfile(i, s)
        done, gamma_map, K_map = isfile(file) ? existing_state(file, W, σ_inits[i]) :
            (Set{String}(), Dict{String,Float64}(), Dict{String,Matrix{Float64}}())
        todo_pairs = [(gi, ki) for gi in 1:nγ, ki in 1:nK
                      if !("$(γ_names[gi])/$(GRNs[ki][1])" in done)] |> vec
        if isempty(todo_pairs)
            n_skip += 1
            continue
        end
        isfile(file) && (n_merge += 1)
        n_tasks = length(todo_pairs) * n_sims
        push!(units, Unit(i, s, W, σ_inits[i], todo_pairs, gamma_map, K_map,
                          Vector{Any}(nothing, n_tasks), Threads.Atomic{Int}(n_tasks)))
    end

    println("Threads: $(Threads.nthreads()) | W $(W_start):$(W_end) signs=$(W_signs) | ",
            "$(length(units)) W-runs to do ($(n_merge) merging into existing files), ",
            "$(n_skip) already complete | up to $(nγ) γ × $(nK) GRNs × $(n_sims) sims each")
    flush(stdout)
    isempty(units) && return

    # queue ordered by W so files complete (and memory is released) progressively
    jobs = Channel{NTuple{3,Int}}(Inf)
    for ui in eachindex(units), pi in eachindex(units[ui].pairs), sim in 1:n_sims
        put!(jobs, (ui, pi, sim))
    end
    close(jobs)

    n_done = Threads.Atomic{Int}(0)

    worker() = for (ui, pi, sim) in jobs
        u = units[ui]
        gi, ki = u.pairs[pi]
        m = Model(lat, γ_values[gi] .* u.W, GRNs[ki][2])
        rng = task_rng(u.W_id, u.sgn, γ_values[gi], ki, sim)
        u.slots[slot(pi, sim)] = run_single_sim(copy(u.σ_inits[sim]), m, β_test, max_sweeps;
                                                rng = rng, sim_kwargs...)
        if Threads.atomic_sub!(u.remaining, 1) == 1      # last task of this W
            file = write_unit(u)
            k = Threads.atomic_add!(n_done, 1) + 1
            println("[$(round(time() - t0, digits=1)) s] $k/$(length(units)) done: $(basename(file))")
            flush(stdout)
        end
    end

    workers = [Threads.@spawn worker() for _ in 1:Threads.nthreads()]
    foreach(wait, workers)   # rethrows if any worker failed

    println("All done in $(round(time() - t0, digits=1)) s")
end

main()
