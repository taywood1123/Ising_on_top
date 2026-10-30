using Plots
using Random
using Statistics
using JLD2
using CodecZlib   # older JLD2 versions need this to read compressed files


"""
fast_tissue_frustration(σ, sW, lat, nsum, cell_frus)

Same value as tissue_frustration(σ, Model(lat, γ .* W, K)) for γ > 0, computed with
whole-number arithmetic. For a coupling J·γW_AB·σ_A·σ_B only its sign matters, and γ > 0
does not change it, so with p = sign(W_AB)·σ_A·σ_B ∈ {±1} and Z = number of nonzero W_AB:
    n_unsat(c) = (n_nonzero(c) - Σ p) / 2,   Σ p = σ_cᵀ · sign(W) · (Σ_{neighbours c2} σ_c2),
    n_nonzero(c) = (number of neighbours) · Z.
`sW = sign.(γ .* W)` (n_genes × n_genes); `nsum` (n_genes × n_cells, Int) and `cell_frus`
(n_cells) are work buffers. Per-cell values and their mean are formed exactly as in
tissue_frustration!, so the result is bit-identical.
For γ = 0 every coupling is zero and the frustration is 0, as in the original.
"""
function fast_tissue_frustration(σ::AbstractMatrix, sW::AbstractMatrix, lat,
                                 nsum::Matrix{Int}, cell_frus::Vector{Float64})
    n_cells, G = size(σ)
    Z = count(!iszero, sW)
    Z == 0 && return 0.0
    fill!(nsum, 0)
    @inbounds for c in 1:n_cells
        nb = lat.frus_nbr[c]; nJ = lat.frus_J[c]
        n_nb = 0
        for k in eachindex(nb)
            iszero(nJ[k]) && continue
            n_nb += 1
            c2 = nb[k]
            for B in 1:G
                nsum[B, c] += Int(σ[c2, B])
            end
        end
        Σp = 0
        for A in 1:G
            σA = Int(σ[c, A])
            s = 0
            for B in 1:G
                s += Int(sW[A, B]) * nsum[B, c]
            end
            Σp += σA * s
        end
        n_nz    = n_nb * Z
        n_unsat = (n_nz - Σp) ÷ 2
        cell_frus[c] = n_nz == 0 ? 0.0 : n_unsat / n_nz
    end
    return mean(cell_frus)
end

####################################################################
# Intracellular and intercellular frustration of every cell, from the blocks of
#   J_sys = I_N ⊗ K + J_cell ⊗ (γW)
####################################################################
"""
grn_pairs(K)

The cell's own GRN interactions (A, B, sign(K[A,B])) with A ≠ B and K[A,B] ≠ 0: the
nonzero entries of a diagonal (I_N ⊗ K) block of J_sys.
"""
grn_pairs(K) = [(A, B, Int(sign(K[A, B]))) for A in axes(K, 1), B in axes(K, 2) if A != B && !iszero(K[A, B])]

"""
intracellular_cell_unsat!(unsat, σ, K_pairs)

INTRACELLULAR frustration of every cell, from its own diagonal block of J_sys (I_N ⊗ K):
`unsat[c]` = number of the cell's GRN interactions with K[A,B] σ_A σ_B < 0. The cell's intracellular
frustration density is unsat[c] / length(K_pairs) (same as intracellular_frustration_density).
"""
function intracellular_cell_unsat!(unsat::Vector{Int}, σ::AbstractMatrix, K_pairs)
    @inbounds for c in axes(σ, 1)
        n = 0
        for (A, B, s) in K_pairs
            n += s * Int(σ[c, A]) * Int(σ[c, B]) < 0
        end
        unsat[c] = n
    end
    return unsat
end

"""
intercellular_cell_unsat!(unsat, nnz, σ, sW, lat, nsum)

INTERCELLULAR frustration of every cell, from the off-diagonal (J_cell ⊗ γW) blocks of J_sys between
the cell and its 8 neighbours: `unsat[c]` of the `nnz[c]` nonzero couplings are negative. The
cell's intercellular frustration density is unsat[c] / nnz[c] (same as intercellular_frustration_density).
`sW = sign.(γ .* W)`. At γ = 0 every coupling is 0, so nnz = 0 and the density is 0, as in
intercellular_frustration_density. Whole-number arithmetic, as in fast_tissue_frustration.
"""
function intercellular_cell_unsat!(unsat::Vector{Int}, nnz::Vector{Int}, σ::AbstractMatrix,
                            sW::AbstractMatrix, lat, nsum::Matrix{Int})
    n_cells, G = size(σ)
    Z = count(!iszero, sW)
    fill!(nsum, 0)
    @inbounds for c in 1:n_cells
        nb = lat.frus_nbr[c]; nJ = lat.frus_J[c]
        n_nb = 0
        for k in eachindex(nb)
            iszero(nJ[k]) && continue
            n_nb += 1
            c2 = nb[k]
            for B in 1:G
                nsum[B, c] += Int(σ[c2, B])
            end
        end
        Σp = 0
        for A in 1:G
            s = 0
            for B in 1:G
                s += Int(sW[A, B]) * nsum[B, c]
            end
            Σp += Int(σ[c, A]) * s
        end
        nnz[c]   = n_nb * Z
        unsat[c] = (nnz[c] - Σp) ÷ 2
    end
    return unsat, nnz
end

new_counts(nγ, nK) = [Dict{Rational{Int},Int}() for _ in 1:nγ, _ in 1:nK]

"""
summarize_W_file(file, gamma_names, grns, lat; n_sims, max_sweeps)

Read ONE per-W result file and keep only what the analysis needs. The big
arrays (E_selfish, the per-sweep curves, the final tissues) are read one
(γ, GRN) group at a time and dropped straight after.

Per sim (n_sims × nγ × nK):
- `final_E`      : lattice energy at that sim's last sweep
- `sweeps_used`, `equilibrated`
- `final_frus`   : tissue frustration of that sim's final tissue (recomputed from σ_final,
                   with fast_tissue_frustration: same value as tissue_frustration, ~4× faster)
- `n_profiles`   : number of unique gene-expression profiles in that sim's final tissue
- `intracellular_frus`   : mean INTRACELLULAR frustration density of that sim's final tissue (V3)

Into `acc` (V3): for every (γ, GRN), the exact number of cells at each intracellular and intercellular
frustration value over all final tissues (`intracellular_counts`, `intercellular_counts`, keyed by the value as
an exact fraction), and the number of tissues counted (`n_lattices`).

Per W (nγ × nK), the original "last sweep" metrics:
- `last_flips`, `last_frus` : avg_flip_counts / avg_tissue_frustration at the
  last recorded sweep (= value of the slowest sim(s) only)

Plus the per-sweep curves (max_sweeps × nγ × nK), added into `acc` for the W average.
"""
function summarize_W_file(file, gamma_names, grns, lat, acc; n_sims, max_sweeps)
    nγ, nK = length(gamma_names), length(grns)

    final_E      = fill(NaN, n_sims, nγ, nK)
    sweeps_used  = zeros(Int, n_sims, nγ, nK)
    equilibrated = zeros(Bool, n_sims, nγ, nK)
    final_frus   = fill(NaN, n_sims, nγ, nK)
    n_profiles   = zeros(Int, n_sims, nγ, nK)
    intracellular_frus   = fill(NaN, n_sims, nγ, nK)
    intracellular_counts  = new_counts(nγ, nK)
    intercellular_counts = new_counts(nγ, nK)
    last_flips   = fill(NaN, nγ, nK)
    last_frus    = fill(NaN, nγ, nK)

    # curves of this file, only added to `acc` once the whole file has been read
    flips_curves = fill(NaN, max_sweeps, nγ, nK)
    frus_curves  = fill(NaN, max_sweeps, nγ, nK)

    jldopen(file, "r") do f
        f["meta/n_sims"] == n_sims         || error("n_sims = $(f["meta/n_sims"]), expected $n_sims")
        f["meta/max_sweeps"] == max_sweeps || error("max_sweeps = $(f["meta/max_sweeps"]), expected $max_sweeps")

        W    = f["meta/W"]
        nsum      = zeros(Int, size(W, 1), lat.n_cells)
        cell_frus = zeros(lat.n_cells)
        lu = zeros(Int, lat.n_cells)                 # intracellular:  unsatisfied interactions per cell
        gu = zeros(Int, lat.n_cells)                 # intercellular: unsatisfied couplings per cell
        gn = zeros(Int, lat.n_cells)                 # intercellular: nonzero couplings per cell
        γmap = Dict(zip(f["meta/gamma_names"], f["meta/gamma_values"]))
        Kmap = f["meta/K"]

        for (k, Kn) in enumerate(grns), (g, γn) in enumerate(gamma_names)
            haskey(γmap, γn) || error("γ $γn not in file")
            haskey(Kmap, Kn) || error("GRN $Kn not in file")
            grp = f["$(γn)/$(Kn)"]

            su = grp["sweeps_used"]
            E  = grp["E_selfish"]
            σf = grp["sigma_finals_selfish"]
            sW = sign.(γmap[γn] .* W)
            Kp = grn_pairs(Kmap[Kn])
            nE = length(Kp)

            for sim in 1:n_sims
                final_E[sim, g, k]      = E[su[sim], sim]
                sweeps_used[sim, g, k]  = su[sim]
                # intercellular: per-cell values, then the tissue mean (bit-identical to fast_tissue_frustration)
                intercellular_cell_unsat!(gu, gn, σf[sim], sW, lat, nsum)
                for c in 1:lat.n_cells
                    cell_frus[c] = gn[c] == 0 ? 0.0 : gu[c] / gn[c]
                    v = gn[c] == 0 ? 0 // 1 : gu[c] // gn[c]
                    intercellular_counts[g, k][v] = get(intercellular_counts[g, k], v, 0) + 1
                end
                final_frus[sim, g, k] = mean(cell_frus)

                # intracellular: per-cell values, then the tissue mean
                intracellular_cell_unsat!(lu, σf[sim], Kp)
                if nE > 0
                    for c in 1:lat.n_cells
                        cell_frus[c] = lu[c] / nE
                        v = lu[c] // nE
                        intracellular_counts[g, k][v] = get(intracellular_counts[g, k], v, 0) + 1
                    end
                    intracellular_frus[sim, g, k] = mean(cell_frus)
                end
                n_profiles[sim, g, k]   = length(unique(eachrow(σf[sim])))
            end
            equilibrated[:, g, k] .= grp["equilibrated"]

            fl = grp["avg_flip_counts"]
            fr = grp["avg_tissue_frustration"]
            last = maximum(su)
            last_flips[g, k] = fl[last]
            last_frus[g, k]  = fr[last]
            flips_curves[:, g, k] .= fl
            frus_curves[:, g, k]  .= fr
        end
    end

    # W average of the curves: running sum / count, ignoring NaN (= original average_over_Ws)
    for i in eachindex(flips_curves)
        v = flips_curves[i]
        isnan(v) || (acc.flips_sum[i] += v; acc.flips_cnt[i] += 1)
        v = frus_curves[i]
        isnan(v) || (acc.frus_sum[i] += v; acc.frus_cnt[i] += 1)
    end
    for i in eachindex(intracellular_counts)
        mergewith!(+, acc.intracellular_counts[i], intracellular_counts[i])
        mergewith!(+, acc.intercellular_counts[i], intercellular_counts[i])
    end
    acc.n_lattices .+= n_sims

    return (; final_E, sweeps_used, equilibrated, final_frus, n_profiles, intracellular_frus, last_flips, last_frus)
end

new_acc(max_sweeps, nγ, nK) = (flips_sum = zeros(max_sweeps, nγ, nK), flips_cnt = zeros(Int, max_sweeps, nγ, nK),
                               frus_sum  = zeros(max_sweeps, nγ, nK), frus_cnt  = zeros(Int, max_sweeps, nγ, nK),
                               intracellular_counts = new_counts(nγ, nK), intercellular_counts = new_counts(nγ, nK),
                               n_lattices = zeros(Int, nγ, nK))

"""
summarize_W_bank(files; gamma_names, grns, n_cells, n_sims, max_sweeps, cache_file = nothing)

Summarise many per-W files. Memory use does not grow with the number of files beyond
the small summary arrays (~100 bytes per sim per (γ, GRN)).

- Uses all Julia threads (each thread reads its own files).
- Files that are missing a requested γ / GRN, or have other settings, are skipped with a warning.
- With `cache_file`, the summary is saved and later calls only read files that are not
  in the cache yet, so re-running after more W files arrive is fast.

Returns a NamedTuple; arrays are indexed [sim, W, γ, GRN] or [W, γ, GRN] or [sweep, γ, GRN],
in the order of `S.W_files`, `gamma_names` and `grns`.
"""
function summarize_W_bank(files; gamma_names, grns, n_cells, n_sims, max_sweeps, cache_file = nothing)
    nγ, nK = length(gamma_names), length(grns)
    settings = Dict("gamma_names" => collect(gamma_names), "grns" => collect(grns),
                    "n_cells" => n_cells, "n_sims" => n_sims, "max_sweeps" => max_sweeps,
                    "summary_version" => "V4")   # V4 adds renamed frustration metrics + cell counts

    # ---- previous summary (only reused if it was made with the same settings) ----
    S_old = nothing
    if cache_file !== nothing && isfile(cache_file)
        d = load(cache_file)
        if d["settings"] == settings
            S_old = (; (Symbol(k) => v for (k, v) in d)...)
        else
            @warn "Cache $cache_file was made with different settings; rebuilding it"
        end
    end
    done_files = S_old === nothing ? Set{String}() : Set(S_old.W_files)
    todo = [f for f in files if !(basename(f) in done_files)]
    println("$(length(files)) files requested: $(length(files) - length(todo)) from cache, ",
            "$(length(todo)) to read on $(Threads.nthreads()) thread(s)")

    # ---- read new files, one chunk per thread ----
    lat     = Lattice(n_cells; n_neighbors = 8)
    results = Vector{Any}(nothing, length(todo))
    n_read  = Threads.Atomic{Int}(0)
    t0      = time()
    chunks  = collect(Iterators.partition(eachindex(todo), max(1, cld(length(todo), Threads.nthreads()))))

    accs = fetch.([Threads.@spawn begin
        acc = new_acc(max_sweeps, nγ, nK)
        for i in chunk
            try
                results[i] = summarize_W_file(todo[i], gamma_names, grns, lat, acc;
                                              n_sims = n_sims, max_sweeps = max_sweeps)
            catch e
                @warn "Skipping $(basename(todo[i])): $(sprint(showerror, e))"
            end
            k = Threads.atomic_add!(n_read, 1) + 1
            k % 100 == 0 && println("  read $k / $(length(todo)) files ($(round(time() - t0, digits = 1)) s)")
        end
        acc
    end for chunk in chunks])

    ok      = findall(!isnothing, results)
    new     = results[ok]
    W_files = vcat(S_old === nothing ? String[] : S_old.W_files, basename.(todo[ok]))

    # ---- assemble ----
    stack_sim(field) = cat((S_old === nothing ? [] : [getfield(S_old, field)])...,
                           [reshape(r[field], n_sims, 1, nγ, nK) for r in new]...; dims = 2)
    stack_W(field)   = cat((S_old === nothing ? [] : [getfield(S_old, field)])...,
                           [reshape(r[field], 1, nγ, nK) for r in new]...; dims = 1)
    acc_total(field) = reduce(+, (getfield(a, field) for a in accs);
                              init = S_old === nothing ? zero(getfield(new_acc(max_sweeps, nγ, nK), field)) :
                                                         getfield(S_old, field))

    if isempty(W_files)
        error("no usable files")
    end

    function counts_total(field)
        tot = S_old === nothing ? new_counts(nγ, nK) : deepcopy(getfield(S_old, field))
        for a in accs, i in eachindex(tot)
            mergewith!(+, tot[i], getfield(a, field)[i])
        end
        return tot
    end

    S = (
        settings     = settings,
        W_files      = W_files,
        W_names      = [first(split(f, "_")) for f in W_files],   # "W17", "-W17", ...
        final_E      = stack_sim(:final_E),
        sweeps_used  = stack_sim(:sweeps_used),
        equilibrated = stack_sim(:equilibrated),
        final_frus   = stack_sim(:final_frus),
        n_profiles   = stack_sim(:n_profiles),
        intracellular_frus   = stack_sim(:intracellular_frus),
        last_flips   = stack_W(:last_flips),
        last_frus    = stack_W(:last_frus),
        flips_sum    = acc_total(:flips_sum),
        flips_cnt    = acc_total(:flips_cnt),
        frus_sum     = acc_total(:frus_sum),
        frus_cnt     = acc_total(:frus_cnt),
        intracellular_counts  = counts_total(:intracellular_counts),
        intercellular_counts = counts_total(:intercellular_counts),
        n_lattices    = acc_total(:n_lattices),
    )

    if cache_file !== nothing && !isempty(todo)
        mkpath(dirname(abspath(cache_file)))
        jldsave(cache_file; pairs(S)...)
        println("Saved summary of $(length(W_files)) files to $cache_file")
    end
    println("Done in $(round(time() - t0, digits = 1)) s")
    return S
end

"""Per-sweep W average of a curve (same as the original average_over_Ws): returns (sweeps, values)."""
function W_average_curve(S, which::Symbol, g, k)
    s, c = which == :flips ? (S.flips_sum, S.flips_cnt) : (S.frus_sum, S.frus_cnt)
    sweeps = findall(>(0), view(c, :, g, k))
    return sweeps, s[sweeps, g, k] ./ c[sweeps, g, k]
end

"""Keep only the W files whose name satisfies `keep`, e.g. W_subset(S, n -> !startswith(n, "-"))."""
function W_subset(S, keep)
    w = findall(keep, S.W_names)
    @warn "W_subset: the per-sweep curves and the cell counts cannot be split and still cover all Ws" maxlog = 1
    return merge(S, (W_files = S.W_files[w], W_names = S.W_names[w],
                     final_E = S.final_E[:, w, :, :], sweeps_used = S.sweeps_used[:, w, :, :],
                     equilibrated = S.equilibrated[:, w, :, :], final_frus = S.final_frus[:, w, :, :],
                     n_profiles = S.n_profiles[:, w, :, :], intracellular_frus = S.intracellular_frus[:, w, :, :],
                     last_flips = S.last_flips[w, :, :], last_frus = S.last_frus[w, :, :]))
end

"""
frustration_distribution(S, which, g, k) -> (values, cells)

The pooled distribution of cell frustration at γ index g and GRN k, over all sims and Ws:
every frustration value that occurs (`which` = :intracellular or :intercellular) and the AVERAGE number of
cells per tissue with that value. `sum(cells) == n_cells`. Because every tissue has the same
number of cells, this average histogram is exactly the distribution of all cells pooled.
"""
function frustration_distribution(S, which::Symbol, g, k)
    d = which == :intracellular ? S.intracellular_counts[g, k] : which == :intercellular ? S.intercellular_counts[g, k] :
        error("which = :intracellular or :intercellular")
    v = sort(collect(keys(d)))
    return Float64.(v), [d[x] for x in v] ./ S.n_lattices[g, k]
end

"""
violin_counts!(p, x, values, weights; bandwidth, width = 0.35, side = :both, kw...)

Draw a violin at `x` for a distribution given as (value, weight) pairs, e.g. from
frustration_distribution. Same construction as violin!: a Gaussian kernel density, clipped to
the range of the data, mirrored, and scaled so its widest point is `width`. Weighting by the
cell counts gives the violin of all pooled cells without storing every cell. `side` = :left or
:right draws half a violin (for intracellular / intercellular side by side). Other keywords go to plot!.
"""
function violin_counts!(p, x, values, weights; bandwidth, width = 0.35, side = :both, npoints = 400, kw...)
    keep = weights .> 0
    values, weights = values[keep], weights[keep]
    isempty(values) && return p
    lo, hi = extrema(values)
    y = lo < hi ? range(lo, hi; length = npoints) : range(lo - bandwidth, hi + bandwidth; length = npoints)
    dens = [sum(w * exp(-0.5 * ((yi - v) / bandwidth)^2) for (v, w) in zip(values, weights)) for yi in y]
    d = width .* dens ./ maximum(dens)
    xl = side == :right ? fill(float(x), npoints) : x .- d
    xr = side == :left  ? fill(float(x), npoints) : x .+ d
    plot!(p, Shape(vcat(xl, reverse(xr)), vcat(collect(y), reverse(collect(y)))); label = "", kw...)
    return p
end

"""Re-bin a (values, cells) distribution into bins of width `bw` starting at 0 (for histograms)."""
function rebin(values, cells, bw)
    edges = 0:bw:(1 + bw)
    out = zeros(length(edges) - 1)
    for (v, c) in zip(values, cells)
        out[min(length(out), floor(Int, v / bw) + 1)] += c
    end
    return collect(edges[1:end-1]) .+ bw / 2, out
end