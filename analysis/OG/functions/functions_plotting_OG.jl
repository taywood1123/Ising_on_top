using Plots
using LinearAlgebra  
using Measures
using Statistics

"""
single_configuration_plot()
Given a simulation has been run and we have a series of spin matrices following this shape σ_histories[n_sims][n_sweep][n_cells x n_genes] this function defines a unique color to each possible set configeration of the n_gene cell vector ( σ_histories[n_sims][n_sweep][i, :]). Once each vector is defined a color. the matrix is reshaped inot a 2D tissue pattern to visualize teh effects of encoded PBCs.

Each binary configuration is assigned a deterministic ID according to its binary/lexicographic ordering across the `2^n_genes` possible configurations. For 9 binary genes, this gives 512 possible configurations, from `(0,0,0,0,0,0,0,0,0)` to `(1,1,1,1,1,1,1,1,1)`. This ordering is intentionally neutral, as no assumptions are currently made about functional similarity between configurations. The resulting IDs will be used as a stable key for assigning colours consistently across runs.
"""
function single_configuration_plot(
    σ_test;
    sim=test_sim,
    sweep=test_sweep,
    color_key=create_configuration_color_key(9)
)
    σ = σ_test[sim][sweep]
    n_cells, n_genes = size(σ)

    L = Int(sqrt(n_cells))
    @assert L * L == n_cells "n_cells must be a perfect square"

    configs = [Tuple(σ[i, :]) for i in 1:n_cells]

    types = [configuration_id(cfg) for cfg in configs]

    grid = reshape(types, L, L)

    p = heatmap(
        grid;
        c = color_key,
        clims = (0.5, 2^n_genes + 0.5),
        aspect_ratio = :equal,
        title = "sweep=$sweep"
    )

    return p
end

"""

"""
function count_cell_types(σ_test; sim=test_sim, test_sweep=test_sweep)
    σ = σ_test[sim][sweep]
    n_sims = size(σ, 1)

    for sim in 1:n_sims 
        num_cell_types = size(unique(σ, dims=1), 1) 
    end
    return num_cell_types
end 

"""
overlap_with_patterns()
Given we have a defined set of cell type petterns (ξ_1,...,ξ_μ) map each unique configeration in σ_test to a cell type pattern in ξ (vector of the ξ  mtx). 

Note 2 self: this follows the very loosly guided methods of Mathew Mart's Multicellular hamiltonian paper but cell type mapping in a very complex and indepth feild on its own - i do not believe this mapping method is up-to-snuff for a complex bioloigical system. 
"""
function overlap_with_patterns(σ_test, ξ::Matrix; sim=test_sim, sweep=test_sweep)
    σ = σ_test[test_sim][test_sweep]
    n_cells = size(σ, 1)
    
    # each pattern is stores as a vetcor in matrix ξ
    n_patterns = size(ξ, 2)
    overlaps = zeros(n_cells, n_patterns)
    for i in 1:n_cells
        for μ in 1:n_patterns
            overlaps[i, μ] = dot(σ[i,:], ξ[:,μ]) / size(σ,2)
        end
    end
    return overlaps
end


"""
"""
function plot_cell_energies(σ_test; sim=test_sim, test_sweep=test_sweep)
    σ = σ_test[test_sim][test_sweep]
    row_sums = sum(σ, dims=2)
    row_sums = vec(row_sums)

    n = length(row_sums)
    side = Int(sqrt(n))
    if side * side != n
        error("Number of rows must be a perfect square to reshape into a square matrix.")
    end

    matrix = reshape(row_sums, side, side)'

    plt = heatmap(
        matrix,
        c=:viridis,
        title="Energy per cell",
        colorbar_title="Energy"
    )

    return plt
end

"""
helper Functions:
Sets an external defined color key for all cell types in order for us to compare across runs.
"""
function configuration_id(config)
    id = 0

    for x in config
        id = 2id + (x == 1 ? 1 : 0)
    end

    return id + 1
end

function create_configuration_color_key(n_genes)
    n_configs = 2^n_genes

    return palette(:tab20, n_configs)
end


"""
Counts the number of times a cell and a gene flips it states during a simulation (counts for each sweep.)
Averages this over all simulations. 
"""
function average_flip_count(σ_histories; 
        n_cells=n_cells, 
        n_genes=n_genes)
    n_sims = length(σ_histories)
    n_sweeps = length(σ_histories[1])

    flip_counts = zeros(Int, n_sweeps - 1, n_sims)

    site_flip_counts = [
        zeros(Int, n_cells, n_genes)
        for sim in 1:n_sims
    ]

    for sim in 1:n_sims
        for sweep in 1:(n_sweeps - 1)
            σ_test1 = σ_histories[sim][sweep]
            σ_test2 = σ_histories[sim][sweep + 1]

            hold = σ_test1 - σ_test2
            flip_counts[sweep, sim] = count(!iszero, hold)
            site_flip_counts[sim] .+= (σ_test1 .!= σ_test2)
        end
    end

    avg_flip_count = vec(mean(flip_counts, dims=2))
    avg_site_flip_counts = sum(site_flip_counts) ./ n_sims

    return avg_flip_count, avg_site_flip_counts
end