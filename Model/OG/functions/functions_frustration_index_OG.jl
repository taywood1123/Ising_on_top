using Plots
using LinearAlgebra
using Random

####################################################
############### Generate random data ###############
####################################################
"""
random_K_matrix():
Generates a random gene interaction matrix
Opt: symmetric or asymmetric
"""
function random_K_matrix(n_genes::Int; symmetric=true)
    K = zeros(Int, n_genes, n_genes)

    for i in 1:n_genes
        for j in 1:n_genes
            if i != j  # keep diagonal zero
                if symmetric && j < i
                    # mirror upper triangle for symmetric case
                    K[i, j] = K[j, i]
                else
                    K[i, j] = rand((-1, 0, 1))
                end
            end
        end
    end

    return K
end

"""
scramble_GRN()
Generates a random gene interaction matrix with the same number od edges as the defined K_GRN
** is asymmetic
"""
function scramble_GRN(K_GRN , n_genes)
    nonzero_coords = Tuple.(findall(!=(0), K_GRN))

    new_coords = [(i, rand(1:n_genes)) for (i, _) in nonzero_coords]

    K_GRN_random = zeros(n_genes,n_genes)
    for (A, B) in new_coords
        K_GRN_random[A, B] = rand([-1, 1])
    end
    return K_GRN_random
end

"""
make_K_pool():

Makes a random interaction matrix
opt: generated over a range of n_edges (edge_range)
"""
function make_K_pool(
    n_genes;
    edge_range = 0:n_genes
)
    K_pool = Matrix{Int}[]
    positions = [(i,j) for i in 1:n_genes, j in 1:n_genes if i != j]
    max_possible = length(positions)

    for n_edges in edge_range
        if n_edges > max_possible
            error("Requested $n_edges edges, but maximum possible is $max_possible")
        end

        K = zeros(Int, n_genes, n_genes)
        chosen = randperm(max_possible)[1:n_edges]

        for idx in chosen
            i, j = positions[idx]
            K[i,j] = rand([-1, 1])

        end
        push!(K_pool, K)
    end
    return K_pool
end


####################################################
################## Evaluate Data ###################
####################################################
"""
eval_KGRN_pool():

Computes the edge counts and frustration index 
of a single gene interaction network matrix (K_GRN)
"""

function eval_KGRN(K_test, σ_mtx, n_cells)
    n_edges = count(!iszero, K_test)
    frus_density = Vector{Float64}(undef, n_cells)

    for i in 1:n_cells
        _, frus_density[i] =
            intracellular_frustration_density(K_test, σ_mtx[i, :])
    end

    return n_edges, frus_density
end


"""
eval_Krand_pool():

Computes the edge counts and frustration index 
of all random Matrices in K_pool
"""
function eval_Krand_pool(K_pool, σ_mtx)
    edge_counts = Int[]
    all_frus_vals = Float64[]
    
    n_cells = size(σ_mtx, 1)
    sq_cell = Int(sqrt(n_cells))

    for K_test in K_pool
        n_edges = count(x -> x != 0, K_test) #- size(K_test,1)
        frus_idx = Vector{Float64}(undef, n_cells)
        
        for i in 1:n_cells
            _, frus_idx[i] = intracellular_frustration_density(K_test, σ_mtx[i,:])

        end

        frus_mtx = reshape(frus_idx, sq_cell, sq_cell)
        unique_frus = vec(frus_mtx) # unique(vec(frus_mtx))

        for f in unique_frus

            push!(edge_counts, n_edges)
            push!(all_frus_vals, f)

        end
    end
    return edge_counts, all_frus_vals
end

"""
count_triangle_loops():
Computes the number of triangulat loops in a matrix
"""
function count_triangle_loops(A::Matrix)
    n = size(A, 1)
    count = 0

    for i in 1:n, j in 1:n, k in 1:n
        if A[i,j] != 0 && A[j,k] != 0 && A[k,i] != 0
            count += 1
        end
    end

    return count/3
end

"""
count_square_loops():
Computes the number of square loops in a matrix
"""
function count_square_loops(A::Matrix)
    n = size(A, 1)
    count = 0

    for i in 1:n, j in 1:n, k in 1:n, l in 1:n
        if A[i,j] != 0 && A[j,k] != 0 && A[k,l] != 0 && A[l,i] != 0
            count += 1
        end
    end

    return count/4
end

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


####################################################
############ Frustration Metrics ###################
####################################################

### Single Cell ###
"""
Computes the frustration index for a GRN (K_test), given a single cell (sigma_vec). Then computes the frustration density of a cell by counding the amout of frustrated gene-gene interactions in the given cell.
"""
function intracellular_frustration_density(K_test, σ_vec)  
    n_genes = size(K_test, 1)
    frus_idx_mtx = zeros(Float64, n_genes, n_genes)

    ϵ = 1e-12

    for i in 1:n_genes
        for j in 1:n_genes
            if i != j
                frus_idx_mtx[i,j] = K_test[i,j] * σ_vec[i] * σ_vec[j]
            end
        end
    end

    n_frustrated = count(x -> x < -ϵ, frus_idx_mtx)
    n_satisfied  = count(x -> x >  ϵ, frus_idx_mtx)

    frus_density = n_frustrated / (n_frustrated + n_satisfied)

    return frus_idx_mtx, frus_density
end

####################################################

### Multi-Cell ###
"""
Computes the frustration indices for all cobinations of gene and cell interactions
by use of a system wide interaction matrix (J_sys) of shape [n_cells*n_genes x n_cells*n_genes]
and a spin matric sigma of shape [n_cells x n_genes]
"""
function state_dept_frustration_idx(J_sys, σ_all;
        n_genes::Int = n_genes,
        n_cells::Int = n_cells,
        sweep::Int = n_sweeps,
        sim::Int = 10)

    NM = n_cells * n_genes
    J_frus_mtx = zeros(eltype(J_sys), NM, NM)

    for i in 1:n_cells, j in 1:n_cells
        if i == j && A == B
            continue
        end

        for A in 1:n_genes, B in 1:n_genes

            i_sys = (i - 1) * n_genes + A
            j_sys = (j - 1) * n_genes + B

            J_frus_mtx[i_sys, j_sys] += J_sys[i_sys, j_sys] * σ_all[sim][sweep][i, A] * σ_all[sim][sweep][j, B]
        end
    end
    return J_frus_mtx
end

"""
Computes the intercellular frustration index of one cell with respect to the system interaction matrix (J_sys).
Given the intercellular frustration matrix (J_frus_mtx), containing the frustration index for each combination of gene and cell interactions, this returns the density over the cell's neighboring genes.
"""
function intercellular_frustration_density(cell1, J_frus_mtx;
        n_cells::Int = n_cells,
        n_genes::Int = n_genes)

    NN_idx = eight_neighbors_indices(cell1, n_cells)

    n_unsat = 0
    n_sat = 0

    ϵ = 1e-12

    for cell2 in NN_idx
        for A in 1:n_genes, B in 1:n_genes

            f = J_idx_helper(J_frus_mtx, A, cell1, B, cell2)

            if f < -ϵ
                n_unsat += 1
            elseif f > ϵ
                n_sat += 1
            end
        end
    end

    n_nonzero = n_unsat + n_sat

    if n_nonzero == 0
        return 0.0  
    end

    return n_unsat / n_nonzero
end
####################################################
################ Helper Functions #################
####################################################

"""
given a two cells and two genes, thsi function returns the corrosponding 
idices in the J_system matrix which is of shape [n_cells*n_genes x n_cells*n_genes]
"""
function J_idx_helper(J, gene1::Int, cell1::Int,
                         gene2::Int, cell2::Int;
                         n_genes::Int=n_genes)
    i = (cell1 - 1) * n_genes + gene1
    j = (cell2 - 1) * n_genes + gene2

    return J[i, j]
end


"""
eight_neighbors_indices()
Given a cell thsi function restunf the 8 nearest neighboring cells in a spatial lattice:
N, NE, E, SE, S, SW, W, NW - applies periodi boundary conditions
"""
function eight_neighbors_indices(cell::Int, n_cells::Int; periodic::Bool=true)
    L = Int(round(sqrt(n_cells)))
    @assert L^2 == n_cells "n_cells must be a perfect square"

    # convert 1D index to 2D coordinates (x=col, y=row)
    x = (cell-1) % L + 1
    y = (cell-1) ÷ L + 1

    # define neighbors
    neighbors = [
        (periodic ? (x == 1 ? L : x-1) : x-1, y),                  # W
        (periodic ? (x == L ? 1 : x+1) : x+1, y),                  # E
        (x, periodic ? (y == 1 ? L : y-1) : y-1),                  # N
        (x, periodic ? (y == L ? 1 : y+1) : y+1),                  # S
        (periodic ? (x == 1 ? L : x-1) : x-1,                      # NW
         periodic ? (y == 1 ? L : y-1) : y-1),
        (periodic ? (x == L ? 1 : x+1) : x+1,                      # NE
         periodic ? (y == 1 ? L : y-1) : y-1),
        (periodic ? (x == 1 ? L : x-1) : x-1,                      # SW
         periodic ? (y == L ? 1 : y+1) : y+1),
        (periodic ? (x == L ? 1 : x+1) : x+1,                      # SE
         periodic ? (y == L ? 1 : y+1) : y+1)
    ]

    # convert neighbors back to 1D indices
    neighbors_indices = Int[]
    for (nx, ny) in neighbors
        if 1 <= nx && nx <= L && 1 <= ny && ny <= L
            push!(neighbors_indices, (ny-1)*L + nx)
        end
    end

    return neighbors_indices
end


############# faster versions for big sims #############

function state_dept_frustration_idx_faster(
    J_sys::Matrix,
    σ::Matrix;
    n_genes::Int = n_genes,
    n_cells::Int = n_cells
)

    NM = n_cells * n_genes

    J_frus_mtx =
        zeros(eltype(J_sys), NM, NM)

    for i in 1:n_cells
        i_start = (i - 1) * n_genes + 1
        i_end   = i * n_genes

        for j in 1:n_cells
            j_start = (j - 1) * n_genes + 1
            j_end   = j * n_genes

            for A in 1:n_genes
                i_sys = i_start + A - 1
                σ_i = σ[i, A]

                for B in 1:n_genes
                    # Skip only gene self-interactions
                    if i == j && A == B
                        continue
                    end

                    j_sys = j_start + B - 1

                    J_frus_mtx[i_sys, j_sys] =
                        J_sys[i_sys, j_sys] *
                        σ_i *
                        σ[j, B]
                end
            end
        end
    end

    return J_frus_mtx
end
