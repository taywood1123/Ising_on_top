# Ising on top: a nonreciprocal multicellular Ising model of GRN frustration

Code for *"A method for analyzing the effects of competing gene interactions on emergent
tissue structure"* (T. E. Woodward, in preparation).

A tissue is modelled as a 20×20 lattice of cells. Each cell carries the same small,
**directed** (nonreciprocal) gene regulatory network (GRN). Gene states are Ising spins
(`+1` expressed, `-1` repressed). Genes interact inside a cell through the GRN matrix `K`,
and between neighbouring cells through a symmetric gene-gene kernel `W` scaled by `γ`.
We measure **frustration**, the fraction of interactions a cell cannot satisfy, to ask how
loop motifs in the GRN and cell-cell communication shape tissue patterning.

All code is Julia: simulation functions in `.jl` files, runs and analysis in Jupyter notebooks.

---

## Repository layout

```
Ising_on_top_V3/
├── README.md
├── data/
│   └── W_bank.jld2                      2000 intercellular kernels W (the paper uses W1–W1000)
├── Model/                               simulation code
│   ├── OG/                              readable version
│   │   ├── functions/
│   │   │   ├── functions_ising_model_finalsweeps_gamma_flipequ_OG.jl   multicellular dynamics
│   │   │   ├── functions_frustration_index_OG.jl                       frustration index
│   │   │   └── functions_SCising_model_ALLsweeps_OG.jl                 single-cell dynamics
│   │   └── run_sims/
│   │       ├── run_sim_multicell_OG.ipynb      multicellular runs, one W at a time
│   │       └── run_SCsim_ALLsweeps_OG.ipynb    single-cell runs
│   └── opti/                            optimised version, for the full W bank
│       ├── functions/
│       │   └── ising_core_opti.jl              multicellular dynamics + frustration
│       └── run_sims/
│           ├── run_ising_Wbank_opti.jl         command-line driver (multithreaded)
│           ├── run_ising_Wbank_opti.slurm      example SLURM job array
│           └── make_W_bank_opti.jl             builds data/W_bank.jld2
└── analysis/
    ├── OG/
    │   ├── functions/   functions_frustration_indexSC_OG.jl, functions_plotting_OG.jl
    │   └── notebooks/   Gammarange_frustration_analysis_OG.ipynb, SCfrustration_index_OG.ipynb
    └── opti/
        ├── functions/   function_Gammarange_analysis_opti.jl, load_results_opti.jl
        └── notebooks/   Gammarange_frustration_analysis_opti.ipynb, Gammafixed_frustration_analysis_opti.ipynb
```

Files ending in `_OG` belong to the readable version and files ending in `_opti` to the
optimised version.

---

## Two versions: OG and opti

| | **OG** (readable) | **opti** (optimised) |
|---|---|---|
| What it is | The original implementation, written to be followed line by line | A rewrite of the same model for speed |
| How it runs | Jupyter notebooks on a laptop | A command-line script, multithreaded; on a cluster via SLURM, or locally for a few Ws |
| Scale | One `W` at a time | The full W bank (the paper: 1000 Ws × 20 γ × 6 GRNs × 10 sims) |
| Output | One `.jld2` file per variable | One `.jld2` file per `W`, holding every γ and GRN |

**Why there are two.** Running the W bank with the readable code would take far too long.
At every spin update it sums over all cells, and every sweep it builds the full
`(N·M)×(N·M)` frustration matrix. The opti core only visits a cell's 8 lattice neighbours,
computes tissue frustration directly from the neighbour pairs, keeps no per-sweep state
history, and runs every `(W, γ, GRN, sim)` as a separate task across threads.

**Both versions use the same model.** Same energy, same update rule, same `|ΔE| < 1e-12`
tie rule, same stopping rule and same frustration definitions. They draw random numbers
differently, so they agree statistically, not bit for bit. Also:
- OG starts all `n_sims` simulations from one random tissue.
- opti gives each simulation its own seeded random start. That start is shared across every
  γ and GRN for the same `W`, so comparisons between conditions are paired.

**Single cell has only an OG version.** Isolated cells have no cell-cell terms, so the
single-cell runs are fast and do not need optimising.

---

## Getting started

1. Install [Julia](https://julialang.org/downloads/) 1.11 or later.
2. From the repository root, install the packages:
   ```julia
   julia> using Pkg
   julia> Pkg.activate(".")
   julia> Pkg.instantiate()
   ```
3. To run the notebooks, add a Jupyter kernel with `Pkg.add("IJulia")` and open the
   notebooks from Jupyter or VS Code. Start Julia with several threads
   (e.g. `julia -t 4`) for the opti notebooks.

Every notebook includes its functions with paths relative to the notebook itself, so open
it from its own folder. **Data paths** (`base_path`, `cache_file`) at the top of the
analysis notebooks and the single-cell run notebook point to where the results were stored
during the project. Change them to your own output folder before running.

---

## Running simulations

### Single cell (OG)

`Model/OG/run_sims/run_SCsim_ALLsweeps_OG.ipynb` runs each GRN, and its symmetrised
version, as 400 independent cells (no cell-cell coupling) for 20 simulations, each from its
own random start. It keeps the state of every sweep.

### Multicellular, one W (OG)

`Model/OG/run_sims/run_sim_multicell_OG.ipynb` loads one `W` from `data/W_bank.jld2`
(choose it with `W_idx`), or draws a new random one, and runs every GRN at every γ. Results
go to `results_OG/` next to the notebook. Shorten `gammas` or `GRNs` to run a subset.

### Multicellular, the W bank (opti)

From the repository root:

```bash
ISING_OUT_DIR=results_opti julia -t 4 Model/opti/run_sims/run_ising_Wbank_opti.jl \
    n_cells n_genes n_sims max_sweeps β β_name σ_name \
    "γ1,γ2,..." "γname1,γname2,..." \
    W_bank W_start W_end [W_signs="1"] [seed=1234]
```

A quick local test (2 Ws, 2 γ, a 10×10 tissue, 200 sweeps; a few seconds):

```bash
ISING_OUT_DIR=results_opti julia -t 4 Model/opti/run_sims/run_ising_Wbank_opti.jl \
    100 9 2 200 Inf betaInf σrand "0.1,0.0" "g01,g0" data/W_bank.jld2 1 2 1 1234
```

The paper settings are in `run_ising_Wbank_opti.slurm`: 400 cells, 9 genes, 10 sims,
at most 10 000 sweeps, `β = Inf`, random starts, the 20 γ values below, seed 1234, and
W1–W1000 split over 100 array tasks of 10 Ws each. The slurm file runs both `W` and `-W`
(`W_SIGNS="1,-1"`). The paper analysis uses `W` only. To run it on your own cluster:
- set `REPO_DIR` to your clone of this repository;
- set `ISING_OUT_DIR` and `W_BANK`;
- set `--partition` and `module load Julia` to match your system;
- create the log folder named in `--output` before submitting.

Then submit with `sbatch run_ising_Wbank_opti.slurm`.

The driver is restart-safe. If a job is killed, resubmit it: groups already in a file are
skipped. Re-running with new γ values adds them to the existing files. It first checks that
every other setting matches and stops with an error if anything differs.

### Rebuilding the W bank

`data/W_bank.jld2` was made with

```bash
julia Model/opti/run_sims/make_W_bank_opti.jl 2000 W_bank.jld2 2024 9 W_sym9x9_1.csv,W_sym9x9_2.csv,W_sym9x9_3.csv
```

W1–W3 are three hand-picked kernels from earlier work. W4–W2000 are drawn uniformly on
[-1, 1] and symmetrised. Each one has its own random stream (`Xoshiro(2024, i)`), so `W<i>`
does not depend on how many Ws are generated. The arguments are also stored inside the
file (`seed`, `n_existing`, `existing_files`, `created`).

---

## Analysis

| Notebook | What it does | Reads |
|---|---|---|
| `analysis/opti/notebooks/Gammarange_frustration_analysis_opti.ipynb` | Main γ-range analysis over the W bank: energy, flips, tissue frustration, unique expression profiles, intracellular and intercellular frustration vs γ, histograms and violins | opti output |
| `analysis/opti/notebooks/Gammafixed_frustration_analysis_opti.ipynb` | One γ in detail: averages over all Ws, per-cell frustration for a set of Ws, tissue patterns and heatmaps for one simulation | opti output |
| `analysis/OG/notebooks/Gammarange_frustration_analysis_OG.ipynb` | Readable γ-range analysis that loads results into dictionaries and averages over Ws | opti output (a subset of Ws) |
| `analysis/OG/notebooks/SCfrustration_index_OG.ipynb` | Single-cell analysis: intracellular frustration, final tissues, flips, energy and unique profiles over sweeps | single-cell output |

The opti γ-range notebook first summarises every W file once (`summarize_W_bank`) into a
cache file and then plots from that. Later runs only read new files. A cache made with a
different list of γ values, GRNs or settings is rebuilt automatically.

---

## Model

| Symbol | Code name | Meaning |
|---|---|---|
| `N` | `n_cells` | cells on an `L×L` periodic lattice (`N = 400`, `L = 20`) |
| `M` | `n_genes` | genes per cell (`M = 9`) |
| `σ[i, A]` | `σ` | state of gene `A` in cell `i`, an `N×M` matrix of ±1 |
| `K` | `K_gene`, `K_*` | intracellular GRN (`M×M`, not symmetric) |
| `J` | `J_cell` | intercellular adjacency (`N×N`, 8 neighbours, periodic) |
| `W` | `W` | intercellular gene-gene kernel (`M×M`, symmetric) |
| `γ` | `γ` | strength of cell-cell coupling relative to the GRN |
| `J_sys` | `J_system` | `I_N ⊗ K + J ⊗ (γW)`, shape `NM×NM` |
| `β` | `β` | inverse temperature (`Inf` throughout) |

**Dynamics.** Because `K` is not symmetric there is no global Hamiltonian. Each spin is
updated using its own ("selfish") local energy:

```
E_local(i, A) = Σ_j Σ_B J[i,j] (γW)[A,B] σ[i,A] σ[j,B]  +  Σ_B K[A,B] σ[i,A] σ[i,B]
ΔE            = 2 · E_local(i, A)
```

Each sweep visits every spin once in random order. A spin flips with probability
`1 / (1 + exp(β ΔE))`, which at `β = Inf` means "flip if `ΔE < 0`".

**Ties.** When `ΔE = 0` (taken as `|ΔE| < 1e-12`, so floating-point rounding is not mistaken
for a real energy change), a `-1` spin flips with probability ½ and a `+1` spin stays. Genes
with no inputs therefore drift to `+1`.

**Stopping rule** (multicellular). A simulation stops at `max_sweeps`, or earlier when
either of these holds:
- **frozen:** fewer than `flip_tol = 5` flips per sweep for `equilibrium_window = 30` sweeps
  in a row;
- **quasi-stable:** the range of flips per sweep over the last `oscillation_window = 100`
  sweeps is within `flip_range_tol = 5` of the previous window's range. This catches GRNs
  that oscillate and never freeze.

The single-cell runs use only the frozen rule, with `equilibrium_window = 50`.

**Frustration.** An interaction is *frustrated* when `coupling · σ_a · σ_b < 0`. Densities
divide by the number of nonzero interactions.

| Metric | Interactions counted |
|---|---|
| Intracellular | the cell's own `K` edges; at `γ = 0` cells are isolated, while at `γ > 0` states arise in the tissue |
| Intercellular | the `J ⊗ γW` couplings between a cell and its 8 neighbours |

The isolated-cell case is the intracellular metric evaluated at `γ = 0`.

*Tissue frustration* is the mean intercellular frustration density over all cells. At `γ = 0`
there are no cell-cell couplings, so intercellular frustration is 0 by definition.

---

## GRNs and γ values

All six GRNs have 9 genes. Genes 4–9 are wired the same way in every GRN:
`2→4 (+)`, `3→6 (+)`, then the read-out layer `4→7 (−)`, `5→8 (−)`, `6→9 (−)`.
They differ only in the 1–2–3 module:

| Name | 1–2–3 module | Motif |
|---|---|---|
| `tree` | `1→2 +`, `1→3 +`, `3→5 +` | no loop |
| `CFF` | `1→2 +`, `2→3 +`, `1→3 +` | coherent feed-forward loop |
| `ICFF` | `1→2 +`, `2→3 +`, `1→3 −` | incoherent feed-forward loop |
| `negtri` | `1→2 +`, `2→3 +`, `3→1 −` | feedback loop, 1 inhibitory edge (negative) |
| `negtritwin` | `1→2 −`, `2→3 +`, `3→1 −` | feedback loop, 2 inhibitory edges (positive) |
| `negtritri` | `1→2 −`, `2→3 −`, `3→1 −` | feedback loop, 3 inhibitory edges (negative) |

Edges are written `A → B` for `K[A,B] ≠ 0`, matching how the matrices are typed. In the
dynamics, row `A` of `K` lists the genes that gene `A` responds to (`h_A = Σ_B K[A,B] σ_B`).
Reversing every edge does not change a motif's class or loop sign.

The single-cell study also runs a symmetrised copy of each GRN, `K + Kᵀ`, in which every edge acts both ways (suffix `sym`).

**γ values** (20): `1.0, 0.8, 0.5, 0.3, 0.25, 0.22, 0.2, 0.18, 0.16, 0.14, 0.125, 0.12,
0.1, 0.08, 0.07, 0.06, 0.05, 0.04, 0.001, 0.0`. They are named `g1, g08, …, g0001, g0` in
files. The extra values between 0.3 and 0.1 resolve the transition in that range.
For `0 < γ < 1/72 ≈ 0.014` the dynamics do not depend on γ. The cell-cell part of `ΔE` is
at most `8 × 9 × γ < 1`, so it can never overturn a nonzero GRN part (an integer). It only
decides ties, and the sign it gives does not depend on γ. At `β = Inf` only the sign of `ΔE`
matters, so 0.001 stands in for that whole range.

---

## Output files

**OG multicellular**, one file per variable:
```
<variable>_<W>_<GRN>_<β>_<γ>_<σ>_<n_sims>_max<max_sweeps>sweeps_flipequ.jld2
```

**opti**, one file per `W` (and sign):
```
<W>_<β>_<σ>_<n_sims>_max<max_sweeps>sweeps_flipequ.jld2     e.g. W17_betaInf_σrand_10_max10000sweeps_flipequ.jld2

meta/…                                  W, initial states, GRN matrices, γ values, seed and all settings
<γ_name>/<GRN>/sigma_finals_selfish     final state of each sim (n_cells × n_genes)
<γ_name>/<GRN>/E_selfish                energy per sweep × sim (NaN after a sim stops)
<γ_name>/<GRN>/avg_flip_counts          flips per sweep, mean over running sims
<γ_name>/<GRN>/avg_site_flip_counts     mean flips per (cell, gene)
<γ_name>/<GRN>/avg_tissue_frustration   tissue frustration per sweep, mean over running sims
<γ_name>/<GRN>/equilibrated             frozen or quasi-stable before max_sweeps, per sim
<γ_name>/<GRN>/sweeps_used              sweeps run, per sim
<γ_name>/<GRN>/n_active                 sims still running at each sweep
```

`analysis/opti/functions/load_results_opti.jl` has small helpers for reading these files.

**Single cell**, one file per variable, keeping every sweep:
```
<variable>1D_<GRN>_<β>_<σ>_ALL<max_sweeps>sweeps.jld2
```
(`1D` in these file names is the older name for the single-cell study.)

---

## References

- M. Smart & A. Zilman, *Emergent properties of collective gene-expression patterns in
  multicellular systems*, Cell Rep. Phys. Sci. 4 (2023). Source of the
  `J_sys = I ⊗ K + J ⊗ γW` construction.
- S. Tripathi, D. A. Kessler & H. Levine, *Biological networks regulating cell fate choice
  are minimally frustrated*, PRL 125 (2020). Source of the frustration measure.
- Y. Avni et al., *Nonreciprocal Ising model*, PRL 134 (2025).
- R. Thomas, *Logical analysis of systems comprising feedback loops*, J. Theor. Biol. 73
  (1978). Source of the loop-sign rule.

## Contact

Taylor E. Woodward, University of Melbourne.
