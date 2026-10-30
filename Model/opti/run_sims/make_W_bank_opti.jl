####################################################################
# make_W_bank.jl
#
# Generate a bank of random symmetric intercellular coupling matrices W
# and save them in ONE file, read by run_ising_Wbank.jl (W_BANK=<file>.jld2).
#
# Each W is built exactly like compute_W(): entries uniform in [-1, 1],
# then symmetrised with Symmetric() (upper triangle mirrored, diagonal kept).
#
# Usage:
#   julia make_W_bank.jl n_W out_file [seed=2024] [n_genes=9] [existing1.csv,existing2.csv,...]
#
# Examples:
#   julia make_W_bank.jl 2000 W_bank.jld2
#   julia make_W_bank.jl 2000 W_bank.jld2 2024 9 \
#       /home/twwoodw/my_folder/compute/data/W_sym9x9_1.csv,/home/twwoodw/my_folder/compute/data/W_sym9x9_2.csv,/home/twwoodw/my_folder/compute/data/W_sym9x9_3.csv
#
# With existing CSVs, they become W1, W2, ... and the random ones fill the rest
# (n_W is the total). Re-running with the same arguments gives the same bank.
#
# File contents:
#   "W"       : n_genes x n_genes x n_W  Float64 array, W[:, :, i] = W<i>
#   "seed", "n_genes", "n_W", "n_existing", "existing_files", "created"
####################################################################

using Random
using LinearAlgebra
using DelimitedFiles
using Dates
using JLD2

length(ARGS) >= 2 || error("usage: julia make_W_bank.jl n_W out_file [seed] [n_genes] [existing1.csv,...]")

n_W      = parse(Int, ARGS[1])
out_file = ARGS[2]
seed     = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 2024
n_genes  = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 9
existing = length(ARGS) >= 5 ? String.(strip.(split(ARGS[5], ","))) : String[]

n_W >= length(existing) || error("n_W ($n_W) is smaller than the number of existing W files ($(length(existing)))")
isfile(out_file) && error("$out_file already exists; delete it or choose another name")

W_bank = Array{Float64}(undef, n_genes, n_genes, n_W)

# existing matrices first
for (i, file) in enumerate(existing)
    W = Matrix{Float64}(readdlm(file, ','))
    size(W) == (n_genes, n_genes) || error("$file is $(size(W)), expected $(n_genes)x$(n_genes)")
    issymmetric(W) || @warn "$file is not symmetric"
    W_bank[:, :, i] = W
end

# random matrices, one independent stream per index so W<i> does not depend on n_W
for i in (length(existing) + 1):n_W
    rng = Xoshiro(UInt64[seed, i])
    W_rand = 2 .* rand(rng, n_genes, n_genes) .- 1
    W_bank[:, :, i] = Matrix(Symmetric(W_rand))
end

# sanity checks
@assert all(issymmetric(W_bank[:, :, i]) for i in (length(existing) + 1):n_W)
@assert all(-1 .<= W_bank .<= 1) || !isempty(existing)
dups = n_W - length(unique(eachslice(W_bank; dims = 3)))
dups == 0 || @warn "$dups duplicate W matrices in the bank"

mkpath(dirname(abspath(out_file)))
jldsave(out_file;
        W              = W_bank,
        seed           = seed,
        n_genes        = n_genes,
        n_W            = n_W,
        n_existing     = length(existing),
        existing_files = existing,
        created        = string(now()))

println("Saved $n_W W matrices ($(n_genes)x$(n_genes)) to $out_file")
println("  W1..W$(length(existing)) from existing CSVs, W$(length(existing)+1)..W$n_W random (seed=$seed)")
println("  size on disk: $(round(filesize(out_file) / 1024^2, digits = 2)) MB")
