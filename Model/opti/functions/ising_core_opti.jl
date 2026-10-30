####################################################################
# ising_core_opti.jl  (was ising_core_V3.jl)
#
# Optimised core for the nonreciprocal (selfish) multicellular Ising
# model + intercellular frustration index.
#
# V3 change (only difference from ising_core.jl): a local energy change with
# |ΔE| < ZERO_TOL is treated as exactly 0, so the ΔE = 0 tie rule applies.
# Without this, when a spin's neighbours cancel exactly (Σ_j σ[j,B] = 0 for every
# gene B) the intercellular term comes out as floating-point noise (~1e-19)
# instead of 0, and the sign of that noise decided the flip. Genuine nonzero ΔE
# are many orders of magnitude larger than ZERO_TOL.
#
# Otherwise a replacement for
#   functions_ising_model_finalsweeps_gamma_flipequ.jl  (ising2d_local_selfish, run_nsims_final)
#   functions_frustration_index_opti.jl                 (state_dept_frustration_idx_faster + intercellular_frustration_density)
# with the same summation order and RNG consumption, but:
#   * local energy only visits the lattice neighbours of a cell (sparse J_cell)
#     instead of all n_cells  -> O(G * 8G) per spin instead of O(N * G * G)
#   * tissue frustration is computed directly from the 8-neighbour pairs,
#     never building the dense (N*G)x(N*G) J_frus matrix, and only once per sweep
#   * no σ history is stored; site flips are counted on the fly
#   * no untyped globals, no abstract containers, inner loops allocation-free
####################################################################

using Random
using Statistics
using LinearAlgebra

const ZERO_TOL = 1e-12

####################################################################
############################ Lattice ###############################
####################################################################
"""
create_2d_4nn_J()

2D lattice adjacency matrix with 4-nearest neighbors
(von Neumann), optionally periodic.
"""
function create_2d_4nn_J(n_cells::Int; periodic::Bool=true, weight::Float64=1.0)
    L = Int(round(sqrt(n_cells)))
    @assert L^2 == n_cells "n_cells must be a perfect square"

    J_cell = zeros(Float64, n_cells, n_cells)

    for i in 1:n_cells
        x = (i-1) % L + 1
        y = (i-1) ÷ L + 1

        neighbors = [
            (periodic ? (x == 1 ? L : x-1) : x-1, y),
            (periodic ? (x == L ? 1 : x+1) : x+1, y),
            (x, periodic ? (y == 1 ? L : y-1) : y-1),
            (x, periodic ? (y == L ? 1 : y+1) : y+1)
        ]

        for (nx, ny) in neighbors
            if 1 <= nx <= L && 1 <= ny <= L
                j = (ny-1)*L + nx
                J_cell[i, j] = weight
                J_cell[j, i] = weight
            end
        end
    end

    return J_cell
end

"""
create_2d_8nn_J()

2D lattice adjacency matrix with 8-nearest neighbors
(Moore neighborhood), optionally periodic.
"""
function create_2d_8nn_J(n_cells::Int; periodic::Bool=true, weight::Float64=1.0)
    L = Int(round(sqrt(n_cells)))
    @assert L^2 == n_cells "n_cells must be a perfect square"

    J_cell = zeros(Float64, n_cells, n_cells)

    for i in 1:n_cells
        x = (i-1) % L + 1
        y = (i-1) ÷ L + 1

        neighbors = [
            (periodic ? (x == 1 ? L : x-1) : x-1, y),
            (periodic ? (x == L ? 1 : x+1) : x+1, y),
            (x, periodic ? (y == 1 ? L : y-1) : y-1),
            (x, periodic ? (y == L ? 1 : y+1) : y+1),
            (periodic ? (x == 1 ? L : x-1) : x-1,
             periodic ? (y == 1 ? L : y-1) : y-1),
            (periodic ? (x == L ? 1 : x+1) : x+1,
             periodic ? (y == 1 ? L : y-1) : y-1),
            (periodic ? (x == 1 ? L : x-1) : x-1,
             periodic ? (y == L ? 1 : y+1) : y+1),
            (periodic ? (x == L ? 1 : x+1) : x+1,
             periodic ? (y == L ? 1 : y+1) : y+1)
        ]

        for (nx, ny) in neighbors
            if 1 <= nx <= L && 1 <= ny <= L
                j = (ny-1)*L + nx
                J_cell[i, j] = weight
                J_cell[j, i] = weight
            end
        end
    end

    return J_cell
end

"""
eight_neighbors_indices()

The 8 nearest neighbouring cells (N, NE, E, SE, S, SW, W, NW) of `cell`
on a periodic square lattice. Same as the original (duplicates kept for L <= 2).
"""
function eight_neighbors_indices(cell::Int, n_cells::Int; periodic::Bool=true)
    L = Int(round(sqrt(n_cells)))
    @assert L^2 == n_cells "n_cells must be a perfect square"

    x = (cell-1) % L + 1
    y = (cell-1) ÷ L + 1

    neighbors = [
        (periodic ? (x == 1 ? L : x-1) : x-1, y),
        (periodic ? (x == L ? 1 : x+1) : x+1, y),
        (x, periodic ? (y == 1 ? L : y-1) : y-1),
        (x, periodic ? (y == L ? 1 : y+1) : y+1),
        (periodic ? (x == 1 ? L : x-1) : x-1,
         periodic ? (y == 1 ? L : y-1) : y-1),
        (periodic ? (x == L ? 1 : x+1) : x+1,
         periodic ? (y == 1 ? L : y-1) : y-1),
        (periodic ? (x == 1 ? L : x-1) : x-1,
         periodic ? (y == L ? 1 : y+1) : y+1),
        (periodic ? (x == L ? 1 : x+1) : x+1,
         periodic ? (y == L ? 1 : y+1) : y+1)
    ]

    neighbors_indices = Int[]
    for (nx, ny) in neighbors
        if 1 <= nx && nx <= L && 1 <= ny && ny <= L
            push!(neighbors_indices, (ny-1)*L + nx)
        end
    end

    return neighbors_indices
end

"""
Lattice

Sparse neighbour structure of J_cell, built once and shared (read-only)
by every simulation / thread.

- `nbr[c]`, `nbr_J[c]`   : nonzero columns of J_cell[c, :] (ascending) and their weights.
                           Used by the local energy (replaces the loop over all cells).
- `frus_nbr[c]`, `frus_J[c]` : eight_neighbors_indices(c) and J_cell[c, c2] for each,
                           exactly the pairs intercellular_frustration_density() visits.
"""
struct Lattice
    n_cells::Int
    nbr::Vector{Vector{Int}}
    nbr_J::Vector{Vector{Float64}}
    frus_nbr::Vector{Vector{Int}}
    frus_J::Vector{Vector{Float64}}
end

function Lattice(n_cells::Int; n_neighbors::Int = 8, periodic::Bool = true)
    J_cell = if n_neighbors == 4
        create_2d_4nn_J(n_cells; periodic = periodic)
    elseif n_neighbors == 8
        create_2d_8nn_J(n_cells; periodic = periodic)
    else
        error("n_neighbors must be 4 or 8")
    end

    nbr   = [findall(!iszero, view(J_cell, c, :)) for c in 1:n_cells]
    nbr_J = [J_cell[c, nbr[c]] for c in 1:n_cells]

    frus_nbr = [eight_neighbors_indices(c, n_cells) for c in 1:n_cells]
    # within-cell blocks are excluded from the frustration index (i == j skipped originally)
    frus_J = [[c2 == c ? 0.0 : J_cell[c, c2] for c2 in frus_nbr[c]] for c in 1:n_cells]

    return Lattice(n_cells, nbr, nbr_J, frus_nbr, frus_J)
end

"""
Model

One (W, γ, K) condition on a lattice. Matrices are stored transposed
(Wt[B, A] = γ W[A, B], Kt[B, A] = K[A, B]) so inner loops run down columns.

Equivalent to J_system = I ⊗ K_gene + J_cell ⊗ (γ W).
"""
struct Model
    lat::Lattice
    n_genes::Int
    Wt::Matrix{Float64}
    Kt::Matrix{Float64}
end

function Model(lat::Lattice, W_scaled::AbstractMatrix, K_gene::AbstractMatrix)
    n_genes = size(K_gene, 1)
    size(W_scaled) == (n_genes, n_genes) || error("W and K must both be n_genes x n_genes")
    return Model(lat, n_genes,
                 Matrix{Float64}(permutedims(W_scaled)),
                 Matrix{Float64}(permutedims(K_gene)))
end

####################################################################
######################### Local energy #############################
####################################################################
"""
local_energy()

Local directed energy of spin σ[cell, gene] (σt is stored as σt[gene, cell]).
Same quantity and summation order as the original delta_local_energy().
"""
@inline function local_energy(σt::Matrix{Float64}, m::Model, cell::Int, gene::Int)
    G  = m.n_genes
    Wt = m.Wt
    Kt = m.Kt
    s  = σt[gene, cell]

    nb = m.lat.nbr[cell]
    nJ = m.lat.nbr_J[cell]

    ΔE_cell = 0.0
    @inbounds for k in eachindex(nb)
        j  = nb[k]
        Jw = nJ[k]
        for B in 1:G
            ΔE_cell += Jw * Wt[B, gene] * s * σt[B, j]
        end
    end

    ΔE_gene = 0.0
    @inbounds for B in 1:G
        ΔE_gene += Kt[B, gene] * s * σt[B, cell]
    end

    return ΔE_cell + ΔE_gene
end

####################################################################
###################### Tissue frustration ##########################
####################################################################
"""
tissue_frustration!()

Mean over cells of the intercellular frustration density
(= mean(intercellular_frustration_density(i, state_dept_frustration_idx_faster(J_system, σ))))
computed directly from the 8-neighbour pairs. `cell_frus` is a work buffer.
"""
function tissue_frustration!(cell_frus::Vector{Float64}, σt::Matrix{Float64}, m::Model)
    G   = m.n_genes
    Wt  = m.Wt
    lat = m.lat
    ϵ   = 1e-12

    @inbounds for c1 in 1:lat.n_cells
        n_unsat = 0
        n_sat   = 0
        nb = lat.frus_nbr[c1]
        nJ = lat.frus_J[c1]

        for k in eachindex(nb)
            c2 = nb[k]
            Jw = nJ[k]
            for A in 1:G
                σA = σt[A, c1]
                for B in 1:G
                    f = Jw * Wt[B, A] * σA * σt[B, c2]
                    if f < -ϵ
                        n_unsat += 1
                    elseif f > ϵ
                        n_sat += 1
                    end
                end
            end
        end

        n_nonzero = n_unsat + n_sat
        cell_frus[c1] = n_nonzero == 0 ? 0.0 : n_unsat / n_nonzero
    end

    return mean(cell_frus)
end

"""Convenience wrapper: tissue frustration of a σ[n_cells, n_genes] matrix."""
function tissue_frustration(σ::AbstractMatrix, m::Model)
    σt = Matrix{Float64}(permutedims(σ))
    return tissue_frustration!(zeros(Float64, m.lat.n_cells), σt, m)
end

####################################################################
###################### Equilibrium check ###########################
####################################################################
"""
Termination conditions (OR logic):
  1. Frozen:       all flip counts in last `equilibrium_window` sweeps < `flip_tol`
  2. Quasi-stable: range of flip counts over the current `oscillation_window` sweeps
                   differs from the range over the previous window by <= `flip_range_tol`
"""
function check_equilibrium(fh::Vector{Int},
                           equilibrium_window::Int, flip_tol::Int,
                           oscillation_window::Int, flip_range_tol::Int)
    n = length(fh)

    frozen = n >= equilibrium_window &&
             all(<(flip_tol), view(fh, n-equilibrium_window+1:n))

    quasi_stable = false
    if n >= 2 * oscillation_window
        cur  = view(fh, n-oscillation_window+1:n)
        prev = view(fh, n-2*oscillation_window+1:n-oscillation_window)
        quasi_stable = abs((maximum(cur) - minimum(cur)) -
                           (maximum(prev) - minimum(prev))) <= flip_range_tol
    end

    return frozen || quasi_stable
end

####################################################################
########################## Simulation ##############################
####################################################################
"""
run_single_sim()

Stochastic nonreciprocal (directed) Ising dynamics using local energies
(Glauber rule), identical update rule to ising2d_local_selfish().

Returns a NamedTuple:
  σ_final, E_hist, flip_hist, frus_hist, site_flips, equilibrated, n_sweeps_used

`site_flips` counts flips in sweeps 2..end (matches the original, which compared
consecutive σ snapshots); `frus_hist` is the tissue frustration after every sweep.
"""
function run_single_sim(
    σ0::AbstractMatrix,
    m::Model,
    β::Float64,
    n_sweeps::Int;
    rng::AbstractRNG         = Random.default_rng(),
    equilibrium_window::Int  = 10,
    flip_tol::Int            = 5,
    oscillation_window::Int  = 100,
    flip_range_tol::Int      = 5
)
    n_cells = m.lat.n_cells
    n_genes = m.n_genes
    size(σ0) == (n_cells, n_genes) || error("σ0 must be n_cells x n_genes")

    σt = Matrix{Float64}(permutedims(σ0))

    E = 0.0
    for cell in 1:n_cells
        for gene in 1:n_genes
            E += local_energy(σt, m, cell, gene)
        end
    end

    E_hist    = Float64[]
    flip_hist = Int[]
    frus_hist = Float64[]
    sizehint!(E_hist, n_sweeps)
    sizehint!(flip_hist, n_sweeps)
    sizehint!(frus_hist, n_sweeps)

    site_flips = zeros(Int, n_cells, n_genes)
    cell_frus  = zeros(Float64, n_cells)

    indices = [(cell, gene) for cell in 1:n_cells for gene in 1:n_genes]
    equilibrated = false

    for sweep in 1:n_sweeps
        shuffle!(rng, indices)
        n_flips = 0

        @inbounds for (cell, gene) in indices
            ΔE = 2 * local_energy(σt, m, cell, gene)
            abs(ΔE) < ZERO_TOL && (ΔE = 0.0)   # rounding noise is a tie, not a decision

            _x = iszero(ΔE) && isinf(β) ? 0.0 : β * ΔE
            _p = iszero(ΔE) && isinf(β) ? 0.5 : 1 / (1 + exp(_x))

            if rand(rng) < _p
                if !(iszero(ΔE) && isone(σt[gene, cell]))
                    σt[gene, cell] *= -1
                    E += ΔE
                    n_flips += 1
                    if sweep > 1
                        site_flips[cell, gene] += 1
                    end
                end
            end
        end

        push!(E_hist, E)
        push!(flip_hist, n_flips)
        push!(frus_hist, tissue_frustration!(cell_frus, σt, m))

        equilibrated = check_equilibrium(flip_hist, equilibrium_window, flip_tol,
                                         oscillation_window, flip_range_tol)
        equilibrated && break
    end

    σ_final = Matrix{eltype(σ0)}(permutedims(σt))

    return (σ_final       = σ_final,
            E_hist        = E_hist,
            flip_hist     = flip_hist,
            frus_hist     = frus_hist,
            site_flips    = site_flips,
            equilibrated  = equilibrated,
            n_sweeps_used = length(E_hist))
end

####################################################################
######################### Aggregation ##############################
####################################################################
"""
aggregate_sims()

Combine the per-sim results of one (W, γ, K) condition into exactly the
quantities run_nsims_final() returned / the old script saved.
"""
function aggregate_sims(sims::AbstractVector, n_sweeps::Int)
    n_sims = length(sims)

    σ_finals_selfish = [s.σ_final for s in sims]

    E_selfish          = fill(NaN, n_sweeps, n_sims)
    flip_counts        = fill(NaN, n_sweeps, n_sims)
    tissue_frustration = fill(NaN, n_sweeps, n_sims)

    sweeps_used  = zeros(Int, n_sims)
    equilibrated = falses(n_sims)

    for (sim, s) in enumerate(sims)
        n = s.n_sweeps_used
        sweeps_used[sim]  = n
        equilibrated[sim] = s.equilibrated
        E_selfish[1:n, sim]          = s.E_hist
        flip_counts[1:n, sim]        = s.flip_hist
        tissue_frustration[1:n, sim] = s.frus_hist
    end

    nanmean_row(M, sweep) = begin
        values = filter(!isnan, M[sweep, :])
        isempty(values) ? NaN : mean(values)
    end

    avg_flip_counts        = [nanmean_row(flip_counts, sweep) for sweep in 1:n_sweeps]
    avg_tissue_frustration = [nanmean_row(tissue_frustration, sweep) for sweep in 1:n_sweeps]
    avg_site_flip_counts   = sum(s.site_flips for s in sims) ./ n_sims
    n_active               = [count(!isnan, view(E_selfish, sweep, :)) for sweep in 1:n_sweeps]

    return (
        sigma_finals_selfish   = σ_finals_selfish,
        E_selfish              = E_selfish,
        avg_flip_counts        = avg_flip_counts,
        avg_site_flip_counts   = avg_site_flip_counts,
        avg_tissue_frustration = avg_tissue_frustration,
        equilibrated           = equilibrated,
        sweeps_used            = sweeps_used,
        n_active               = n_active
    )
end
