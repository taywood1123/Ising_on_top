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
    
    for i in 1:n_genes
        for j in 1:n_genes
            if i != j
                frus_idx_mtx[i,j] = K_test[i,j] * σ_vec[i] * σ_vec[j]
            end
        end
    end

    n_frustrated = count(x -> x < 0, frus_idx_mtx)
    n_satisfied = count(x -> x > 0, frus_idx_mtx)

    frus_density = n_frustrated / (n_frustrated+n_satisfied) #length(frus_idx_mtx)

    return frus_idx_mtx, frus_density
end
