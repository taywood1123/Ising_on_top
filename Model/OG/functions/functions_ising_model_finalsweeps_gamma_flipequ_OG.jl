using Random 
using LinearAlgebra  
using Statistics
using DelimitedFiles
using StatsPlots
using Plots 
using Measures
using Distributions
using JLD2
using Pkg

# |ΔE| below ZERO_TOL is treated as exactly 0, so the ΔE = 0 tie rule applies.
# Without this, floating-point noise (~1e-16) from terms that cancel exactly
# could decide a flip. Same rule as ising_core_opti.jl.
const ZERO_TOL = 1e-12

##############################################
########### Ising Model Simulation ###########
##############################################
"""
delta_local_energy()

Compute the local directed energy contribution of a single spin
`σ[cell, gene]` in the nonreciprocal (selfish) model.

Includes:
- intercellular coupling via `J_cell ⊗ W`
- intracellular GRN coupling via `K_gene`

This is a local (non-global) energy; no symmetry or Lyapunov
assumption is made.
"""
function delta_local_energy(
    J_cell::Matrix, 
    W::Matrix,
    K_gene::Matrix, 
    σ::Matrix,
    cell::Int,
    gene::Int
)
    n_cells, n_genes = size(σ)
    L = Int(sqrt(n_cells))
    s = σ[cell, gene]

    ΔE_cell = 0.0
    for j in 1:n_cells
        for B in 1:n_genes
            ΔE_cell += J_cell[cell, j] * W[gene, B] * s * σ[j, B]
        end
    end

    ΔE_gene = 0.0 #change to ground state
    for B in 1:n_genes
        ΔE_gene += K_gene[gene, B] * s * σ[cell, B]
    end
    return ΔE_cell + ΔE_gene
end

"""
update_sign!()

Deterministic synchronous update rule for reciprocal dynamics:

σ(t+1) = sign(J_system * σ(t))

Zero-temperature Hopfield-like update.
"""
function update_sign!(σ::Matrix, J_system::Matrix)
    sgn(x::Number) = x >= 0 ? 1 : -1 
    x = vec(σ') #NOTE: need to take the transpose
    h = reshape(J_system * x, size(σ'))
    x_new = sgn.(h)  
    copyto!(σ, x_new')

end
"""
ising2d_local_selfish()

Stochastic nonreciprocal (directed) Ising dynamics using local energies.

Spins update via Glauber rule using `delta_local_energy()`.

No global energy minimization is assumed.

Termination conditions (OR logic — either triggers a break):
  1. Frozen:       all flip counts in last `equilibrium_window` sweeps < `flip_tol`
  2. Quasi-stable: the range of flip counts over the current `oscillation_window` sweeps
                   differs from the range over the previous `oscillation_window` sweeps
                   by less than `flip_range_tol`, indicating a consistent oscillation
                   band has been maintained across 2 × `oscillation_window` sweeps

Arguments:
  equilibrium_window  -- number of sweeps to check for frozen condition (default: 10)
  flip_tol            -- max flips per sweep to count as frozen (default: 5)
  oscillation_window  -- number of sweeps in each comparison window (default: 100)
  flip_range_tol      -- max allowed difference between current and previous window
                         flip count ranges to count as quasi-stable (default: 5)
"""
function ising2d_local_selfish(
    σ::AbstractMatrix,
    J_system::AbstractMatrix,
    J_cell::AbstractMatrix,
    W::AbstractMatrix,
    K_gene::AbstractMatrix,
    β::Float64,
    n_cells::Int,
    n_genes::Int,
    n_sweeps::Int;
    rng::AbstractRNG         = Random.GLOBAL_RNG,
    equilibrium_window::Int  = 10,
    flip_tol::Int            = 5,
    oscillation_window::Int  = 100,
    flip_range_tol::Int      = 5
)

    ### compute initial energy ###
    E = 0.0
    for cell in 1:n_cells
        for gene in 1:n_genes
            E += delta_local_energy(
                J_cell,
                W,
                K_gene,
                σ,
                cell,
                gene
            )
        end
    end

    ### storage ###
    E_history    = Vector{Float64}()
    σ_history    = Vector{Matrix}()
    flip_history = Vector{Int}()
    frus_history = Vector{Float64}()

    sizehint!(E_history,    n_sweeps)
    sizehint!(σ_history,    n_sweeps)
    sizehint!(flip_history, n_sweeps)

    ### spin update indices ###
    indices = [(cell, gene) for cell in 1:n_cells for gene in 1:n_genes]

    ### simulation ###
    for sweep in 1:n_sweeps
        shuffle!(rng, indices)
        n_flips = 0

        for (cell, gene) in indices
            ΔE = 2 * delta_local_energy(
                J_cell,
                W,
                K_gene,
                σ,
                cell,
                gene
            )

            abs(ΔE) < ZERO_TOL && (ΔE = 0.0)   # rounding noise is a tie, not a decision
            _x = iszero(ΔE) && isinf(β) ? 0.0 : β * ΔE
            _p = iszero(ΔE) && isinf(β) ? 0.5 : 1 / (1 + exp(_x))

            if rand(rng) < _p
                if !(iszero(ΔE) && isone(σ[cell, gene]))
                    σ[cell, gene] *= -1
                    E += ΔE
                    n_flips += 1
                end
            end
        end

        push!(σ_history,    copy(σ))
        push!(E_history,    E)
        push!(flip_history, n_flips)

        # ------------------------------------------------------------
        # Compute frustration at every sweep and store
        # ------------------------------------------------------------
        J_frus_mtx = state_dept_frustration_idx_faster(J_system, σ)
        cell_frus  = zeros(Float64, n_cells)
        for i in 1:n_cells
            cell_frus[i] = intercellular_frustration_density(i, J_frus_mtx)
        end
        push!(frus_history, mean(cell_frus))

        # ------------------------------------------------------------
        # Condition 1 — Frozen: all flips in last equilibrium_window sweeps < flip_tol
        # ------------------------------------------------------------
        frozen = (length(flip_history) >= equilibrium_window) &&
                 all(flip_history[end-equilibrium_window+1:end] .< flip_tol)

        # ------------------------------------------------------------
        # Condition 2 — Quasi-stable: range of flip counts over current
        # oscillation_window differs from previous by less than flip_range_tol
        # ------------------------------------------------------------
        quasi_stable = (length(flip_history) >= 2 * oscillation_window) &&
                       abs(
                           (maximum(flip_history[end-oscillation_window+1:end]) -
                            minimum(flip_history[end-oscillation_window+1:end]))
                           -
                           (maximum(flip_history[end-2*oscillation_window+1:end-oscillation_window]) -
                            minimum(flip_history[end-2*oscillation_window+1:end-oscillation_window]))
                       ) <= flip_range_tol

        if frozen || quasi_stable
            break
        end
    end

    return σ_history, E_history, flip_history, frus_history

end

##############################################
############# Compute Variables #############
##############################################
"""
compute_interaction()

Construct interaction matrices:

J_system = I ⊗ K_gene + J_cell ⊗ W

Returns:
- J_cell (spatial coupling)
- K_gene_eff (GRN)
- K_gene_smart (projected GRN)
- J_system (full system coupling)
"""
function compute_interaction(
    W::Matrix,
    n_cells::Int,
    K_gene::Matrix;
    n_neighbors::Int = 8,
    periodic::Bool = true
)
    J_cell = if n_neighbors == 4
        create_2d_4nn_J( n_cells; periodic = periodic )

    elseif n_neighbors == 8
        create_2d_8nn_J(n_cells; periodic = periodic )

    else
        error("n_neighbors must be 4 or 8")
    end

    I_M = Matrix(I, n_cells, n_cells)

    J_system =
        kron(I_M, K_gene) +
        kron(J_cell, W)
    return J_cell, J_system
end

"""
compute_W()

Generate symmetric intercellular gene-gene coupling matrix W
with entries in [-1, 1].
"""
function compute_W(n_genes::Int)
    W_rand = 2 .* rand(n_genes, n_genes) .- 1
    W_sym = Matrix(Symmetric(W_rand))
    return W_sym
end 


"""
noise_matrix()

Generate symmetric noisy coupling matrix with optional local
asymmetry perturbations.
"""
function noise_matrix(; n=n_genes,
                            mean=0.0,
                            stdev=0.01,
                            asym_sites=Tuple[])

    dist = Normal(mean, stdev)
    R = rand(dist, n, n)
    R = (R + R') / 2

    for (i, j) in asym_sites
        R[i, j] = rand(dist)
        R[j, i] = rand(dist)
    end

    return R
end

##############################################
############### Helper Functions #############
##############################################
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

        # Compute neighbor positions with periodic boundary conditions
        neighbors = [
            (periodic ? (x == 1 ? L : x-1) : x-1, y),
            (periodic ? (x == L ? 1 : x+1) : x+1, y),
            (x, periodic ? (y == 1 ? L : y-1) : y-1),
            (x, periodic ? (y == L ? 1 : y+1) : y+1)
        ]

        for (nx, ny) in neighbors
            if 1 <= nx <= L && 1 <= ny <= L  # only valid indices
                j = (ny-1)*L + nx
                J_cell[i,j] = weight
                J_cell[j,i] = weight  # symmetric
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
                J_cell[j, i] = weight  # symmetric
            end
        end
    end

    return J_cell
end



#################################################
########### RUN SIMULATION + STOARGE ############
#################################################
function run_nsims_final(
    σ_init::Matrix,
    W_sym::Matrix,
    γ::Float64;
    K_gene::Matrix,
    β = Inf,
    n_sims = n_sims,
    n_sweeps = n_sweeps,
    equilibrium_window::Int  = 10,
    flip_tol::Int            = 5,
    oscillation_window::Int  = 100,
    flip_range_tol::Int      = 5
)

    W_scaled = γ .* copy(W_sym)

    J_cell, J_system =
        compute_interaction(
            W_scaled,
            n_cells,
            K_gene;
            n_neighbors = 8
        )

    σ_sim = copy(σ_init)

    # ------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------
    σ_finals_selfish = Vector{Matrix}(undef, n_sims)

    E_selfish   = fill(NaN, n_sweeps, n_sims)
    flip_counts = fill(NaN, n_sweeps, n_sims)

    site_flip_counts = [
        zeros(Int, n_cells, n_genes)
        for _ in 1:n_sims
    ]

    tissue_frustration = fill(NaN, n_sweeps, n_sims)

    sweeps_used  = zeros(Int, n_sims)
    equilibrated = falses(n_sims)

    master_rng = MersenneTwister()

    # ------------------------------------------------------------
    # Run simulations
    # ------------------------------------------------------------
    for sim in 1:n_sims
        rng_selfish = MersenneTwister(rand(master_rng, UInt64))

        σ_sf, E_sf, flips_sf = ising2d_local_selfish(
            copy(σ_sim),
            J_system,
            J_cell,
            W_scaled,
            K_gene,
            β,
            n_cells,
            n_genes,
            n_sweeps;
            rng                = rng_selfish,
            equilibrium_window = equilibrium_window,
            flip_tol           = flip_tol,
            oscillation_window = oscillation_window,
            flip_range_tol     = flip_range_tol
        )

        n_actual_sweeps = length(E_sf)

        sweeps_used[sim]      = n_actual_sweeps
        σ_finals_selfish[sim] = σ_sf[end]

        E_selfish[1:n_actual_sweeps, sim]   = E_sf
        flip_counts[1:n_actual_sweeps, sim] = flips_sf

        # ============================================================
        # Equilibrium status — true if either condition was met
        # ============================================================
        frozen = (n_actual_sweeps >= equilibrium_window) &&
                 all(flips_sf[end-equilibrium_window+1:end] .< flip_tol)

        quasi_stable = (n_actual_sweeps >= 2 * oscillation_window) &&
                       abs(
                           (maximum(flips_sf[end-oscillation_window+1:end]) -
                            minimum(flips_sf[end-oscillation_window+1:end]))
                           -
                           (maximum(flips_sf[end-2*oscillation_window+1:end-oscillation_window]) -
                            minimum(flips_sf[end-2*oscillation_window+1:end-oscillation_window]))
                       ) <= flip_range_tol

        equilibrated[sim] = frozen || quasi_stable

        # ============================================================
        # Site flip counts (from σ history)
        # ============================================================
        if n_actual_sweeps > 1
            for sweep in 1:(n_actual_sweeps - 1)
                site_flip_counts[sim] .+=
                    (σ_sf[sweep] .!= σ_sf[sweep + 1])
            end
        end

        # ============================================================
        # Tissue frustration
        # ============================================================
        for sweep in 1:n_actual_sweeps
            J_frus_mtx = state_dept_frustration_idx_faster(
                J_system,
                σ_sf[sweep]
            )
            full_cell_frus_idx = zeros(Float64, n_cells)
            for i in 1:n_cells
                full_cell_frus_idx[i] = intercellular_frustration_density(i, J_frus_mtx)
            end
            tissue_frustration[sweep, sim] = mean(full_cell_frus_idx)
        end

        println("Completed simulation $sim")
    end

    # ================================================================
    # Average flip counts across simulations
    # ================================================================
    avg_flip_counts = [
        begin
            values = filter(!isnan, flip_counts[sweep, :])
            isempty(values) ? NaN : mean(values)
        end
        for sweep in 1:n_sweeps
    ]

    # ================================================================
    # Average site flip counts across simulations
    # ================================================================
    avg_site_flip_counts = sum(site_flip_counts) ./ n_sims

    # ================================================================
    # Average tissue frustration across simulations
    # ================================================================
    avg_tissue_frustration = [
        begin
            values = filter(!isnan, tissue_frustration[sweep, :])
            isempty(values) ? NaN : mean(values)
        end
        for sweep in 1:n_sweeps
    ]

    # ================================================================
    # Number of active simulations at each sweep
    # ================================================================
    n_active = [
        count(!isnan, E_selfish[sweep, :])
        for sweep in 1:n_sweeps
    ]

    return (
        σ_finals_selfish       = σ_finals_selfish,
        E_selfish              = E_selfish,

        avg_flip_counts        = avg_flip_counts,
        avg_site_flip_counts   = avg_site_flip_counts,
        avg_tissue_frustration = avg_tissue_frustration,

        sweeps_used            = sweeps_used,
        equilibrated           = equilibrated,
        n_active               = n_active
    )
end