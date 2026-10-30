using Random 
using LinearAlgebra  
using Statistics
using DelimitedFiles
using StatsPlots
using Plots 
using Measures
using Distributions

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
- intracellular GRN coupling via `K_gene`single_configuration_plot

This is a local (non-global) energy; no symmetry or Lyapunov
assumption is made.
"""
function delta_local_energySC(
    K_gene::Matrix, 
    σ::Matrix,
    cell::Int,
    gene::Int
)
    n_cells, n_genes = size(σ)
    L = Int(sqrt(n_cells))
    s = σ[cell, gene]

    #ΔE_cell = 0.0
    #for j in 1:n_cells
    #    for B in 1:n_genes
    #        ΔE_cell += J_cell[cell, j] * W[gene, B] * s * σ[j, B]
    #    end
    #end

    ΔE_gene = 0.0 #change to ground state
    for B in 1:n_genes
        ΔE_gene += K_gene[gene, B] * s * σ[cell, B]
    end
    return ΔE_gene
end

"""
update_sign!()

Deterministic synchronous update rule for reciprocal dynamics:

σ(t+1) = sign(J_system * σ(t))

Zero-temperature Hopfield-like update.
"""
function update_signSC!(σ::Matrix, K_gene::Matrix)
    sgn(x::Number) = x >= 0 ? 1 : -1

    for cell in axes(σ,1)
        σ[cell,:] .= sgn.(K_gene * view(σ, cell, :))
    end
end

"""
isingSC_local_selfish()

Stochastic nonreciprocal (directed) Ising dynamics using local energies.

Spins update via Glauber rule using `delta_local_energy()`.

No global energy minimization is assumed.
"""
function isingSC_local_selfish(
    σ::Matrix,
    K_gene::Matrix,
    β::Float64,
    n_cells::Int,
    n_genes::Int,
    max_sweeps::Int;
    rng::AbstractRNG = Random.GLOBAL_RNG,
    equilibrium_window::Int = 50,
    flip_tol::Int = 5
)
    ### storage ###
    σ_history = Vector{Matrix{Int8}}(undef, max_sweeps + 1)
    E_history = Vector{Float64}(undef, max_sweeps + 1)
    flip_history = Vector{Int}(undef, max_sweeps)

    σ_history[1] = copy(σ)

    ### compute initial energy ###
    E = 0.0
    for cell in 1:n_cells
        for gene in 1:n_genes
            E += delta_local_energySC(
                K_gene,
                σ,
                cell,
                gene
            )
        end
    end

    E_history[1] = -E

    ### spin update ###
    indices = [
        (cell, gene)
        for cell in 1:n_cells
        for gene in 1:n_genes
    ]

    for step in 1:max_sweeps
        shuffle!(rng, indices)
        n_flips = 0

        for (cell, gene) in indices

            ΔE = 2 * delta_local_energySC(
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

        σ_history[step + 1] = copy(σ)
        E_history[step + 1] = E
        flip_history[step] = n_flips

        # ------------------------------------------------------------
        # Check for equilibrium:
        # all flips in the last `equilibrium_window` sweeps
        # are below `flip_tol`
        # ------------------------------------------------------------
        if step >= equilibrium_window
            if all(
                flip_history[step-equilibrium_window+1:step] .< flip_tol
            )
                # Trim unused preallocated storage
                resize!(σ_history, step + 1)
                resize!(E_history, step + 1)
                resize!(flip_history, step)

                break
            end
        end
    end

    return σ_history, E_history, flip_history
end


"""
smart_deterministicSC()

Deterministic reciprocal dynamics using synchronous updates:

    σ(t+1) = sign(K_gene * σ(t))

Zero-temperature Hopfield limit.
"""
function smart_deterministicSC(
    σ::Matrix,
    K_gene::Matrix,
    max_sweeps::Int
)
    σ_history = Vector{Matrix{Int8}}(undef, max_sweeps + 1)

    σ_history[1] = copy(σ)

    for step in 1:max_sweeps
        update_signSC!(σ, K_gene)
        σ_history[step + 1] = copy(σ)
    end

    return σ_history
end

###############################################
########### HELPER FUNCTIONS ############
###############################################
"""
antiferromagnetic_spin_matrix()

Generate alternating ±1 spin configuration across cells and genes.
"""
function antiferromagnetic_spin_matrix(n_cells::Int, n_genes::Int)
    spins = Array{Int}(undef, n_cells, n_genes)

    for i in 1:n_cells
        for j in 1:n_genes
            spins[i, j] = (-1)^(i + j)
        end
    end

    return spins
end


function scramble_GRN(K_GRN , n_genes)
    nonzero_coords = Tuple.(findall(!=(0), K_GRN))

    new_coords = [(i, rand(1:n_genes)) for (i, _) in nonzero_coords]

    K_GRN_random = zeros(n_genes,n_genes)
    for (A, B) in new_coords
        K_GRN_random[A, B] = rand([-1, 1])
    end
    return K_GRN_random
end

###############################################
########### SIMULATIONS + STOARGE ############
###############################################
"""
run_nsims()

Run multiple simulations comparing:
- deterministic reciprocal dynamics
- stochastic reciprocal dynamics
- stochastic nonreciprocal dynamics

"""
function run_nsims(
    σ_init::Matrix;
    K_gene::Union{Nothing, Matrix} = nothing,
    β = Inf,
    n_sims = n_sims,
    max_sweeps = n_sweeps,
    equilibrium_window::Int = 50,
    flip_tol::Int = 5
)
    σ_sim = copy(σ_init)

    σ_histories_smart_det = Vector{Vector{Matrix{Int8}}}(undef, n_sims)
    σ_finals_smart_det    = Vector{Matrix{Int8}}(undef, n_sims)

    σ_histories_smart      = Vector{Vector{Matrix{Int8}}}(undef, n_sims)
    E_histories_smart      = Vector{Vector{Float64}}(undef, n_sims)
    σ_finals_smart         = Vector{Matrix{Int8}}(undef, n_sims)

    σ_histories_selfish    = Vector{Vector{Matrix{Int8}}}(undef, n_sims)
    E_histories_selfish    = Vector{Vector{Float64}}(undef, n_sims)
    flip_histories_selfish = Vector{Vector{Int}}(undef, n_sims)
    σ_finals_selfish       = Vector{Matrix{Int8}}(undef, n_sims)

    master_rng = MersenneTwister()

    for sim in 1:n_sims
        println("Simulation $(sim) / $(n_sims)")

        rng_selfish = MersenneTwister(rand(master_rng, UInt64))

        # ------------------------------------------------------------
        # SMART DETERMINISTIC
        # ------------------------------------------------------------

        σ_det = smart_deterministicSC(
            copy(σ_sim),
            K_gene,
            max_sweeps
        )

        σ_histories_smart_det[sim] = σ_det
        σ_finals_smart_det[sim]    = σ_det[end]

        # ------------------------------------------------------------
        # SELFISH
        # ------------------------------------------------------------

        σ_sf, E_sf, flips_sf = isingSC_local_selfish(
            copy(σ_sim),
            K_gene,
            β,
            n_cells,
            n_genes,
            max_sweeps;
            rng = rng_selfish,
            equilibrium_window = equilibrium_window,
            flip_tol = flip_tol
        )

        σ_histories_selfish[sim]    = σ_sf
        E_histories_selfish[sim]    = E_sf
        flip_histories_selfish[sim] = flips_sf
        σ_finals_selfish[sim]       = σ_sf[end]
    end

    ####################################################
    # RETURN
    ####################################################

    return (
        σ_histories_smart_det  = σ_histories_smart_det,
        σ_finals_smart_det     = σ_finals_smart_det,

        σ_histories_selfish    = σ_histories_selfish,
        E_histories_selfish    = E_histories_selfish,
        flip_histories_selfish = flip_histories_selfish,
        σ_finals_selfish       = σ_finals_selfish
    )
end
