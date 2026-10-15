# QuantumSimulator.jl

Julia package for building and simulating closed quantum systems with exact diagonalization and sparse/dense matrices. An emphasis is placed on doing dynamics and studying quantum chaos diagnostics (including through Krylov space expansions) on spin-1/2 chains. This package is meant to be high-level with a wide range of techniques for doing research. For more specialized and efficient versions of the simulation methods contained within, one should check out [PauliStrings.jl](https://paulistrings.org/dev/) for Pauli string binary encoding, [QuantumKitHub](https://github.com/QuantumKitHub) for Krylov and tensor network methods, [ITensors.jl](https://docs.itensor.org/Overview/) for tensor network methods, and [Xdiag.jl](https://github.com/awietek/XDiag.jl) for more efficient exact-diagonalization.

Claude Code has been used to write and edit part of this package, as well as to create examples and validations.

## Installation

Not registered, so install it straight from the repository:
```julia
using Pkg; Pkg.add(url="https://github.com/cpeacockc/QuantumSimulator.jl")
```
To edit the source as well, clone the repo and link the folder instead:
```julia
using Pkg; Pkg.develop(path="path/to/QuantumSimulator")
```
Either way, from anywhere:
```julia
using QuantumSimulator
```

The package is Julia standard libraries only (`LinearAlgebra`, `SparseArrays`, `Statistics`, `Random`), so loading it is instant.

## Contents

| File | Content |
|---|---|
| `src/Pauli_generator.jl` | Pauli/spin operators, product states, density matrices |
| `src/Models.jl` | Hamiltonians: XYZ, Ising, Anderson, Aubry-André, ultrametric |
| `src/Time_evolution.jl` | Schrödinger and Heisenberg time evolution |
| `src/Lanczos_functions.jl` | Operator-growth Lanczos, Krylov complexity, moments |
| `src/Measurements.jl` | Level statistics, spectral functions, correlators, spectral form factor |
| `src/Lanczos_statistics.jl` | Disorder statistics of the `b_n`, Krylov-space zero modes (LIOMs) |
| `src/QuantumSimulator.jl` | Loads all of the above |
| `test/runtests.jl` | Test suite (`julia --project=. test/runtests.jl`) |

## Operators and states — `Pauli_generator.jl`

### Create a Pauli Z operator on site 5
Operator options are `"X"`, `"Y"`, `"Z"`, `"+"`, `"-"` (the final two are $\sigma^+ = X+iY$ and $\sigma^- = X-iY$)
```julia
N = 10 #Number of sites
Op = "Z"
site = 5 # site on lattice
spin_flag = "pauli" # options are "pauli" or "spin" for \sigma^a and S^a = 0.5 \sigma^a respectively

Z_5 = Op_create(N, Op, site, spin_flag)
```

Multi-site strings are also available; unlisted sites stay as the identity:
```julia
XZ = op_create(N, [("X",1),("Z",3)])           # X_1 Z_3
```

### Product states
Options are `"Neel"`, `"Up"`, `"Dn"` (or `"Down"`), `"Rand"`, for alternating up and down, all up, all down, and a random product state of up/down spins. `"RandVec"` gives a Haar-random state in the full Hilbert space.
```julia
psi = Product_state(N,"Neel")
rho = Product_densm(N,"Neel")   # the same state as a density matrix
```

Site indexing is consistent between operators and states, so `Op_create(N,"Z",s,…)` measures site `s` of `Product_state`:
```julia
psi = Product_state(4,"Neel")
[real(psi' * Op_create(4,"Z",s,"pauli") * psi) for s in 1:4]   # [1.0, -1.0, 1.0, -1.0]
```

## Built-in Hamiltonians — `Models.jl`
### Isotropic Heisenberg model
```julia
Jx,Jy,Jz = 1,1,1 # Define tunneling probabilities in each direction
X_fields, Y_fields, Z_fields = zeros(N),zeros(N),zeros(N) # Set potentials in each direction
H_XXX = H_XYZ(N,[Jx,Jy,Jz],X_fields, Y_fields, Z_fields);
H_XXX_spin_open_BCs = H_XYZ(N,[Jx,Jy,Jz],X_fields, Y_fields, Z_fields;periodic=false,spinflag="spin"); # extra options to convert to spins and open BC's
```
### Chaotic Ising model (chaotic for nonzero hx)
```julia
hx = 1.0;                            # X-field
H_Ising = H_Quantum_Ising(N,hx);     # Z-field defaults to 0
H_Ising = H_Quantum_Ising(N,hx,0.5); # ...or set it explicitly
```
### Other spin chains
```julia
H = H_XXZ(N,1.0,2.0)                        # XXZ chain, J and Δ
H = XXZnnn_H(N)                             # XXZ with next-nearest-neighbour terms
H = H_Disordered_Heisenberg(N,1.0,5.0)      # disordered Heisenberg (MBL)
H = H_mb_Anderson1D(N,1.0,randn(N))         # XX chain with Z disorder
H = Rand_Heisenberg(N)                      # random nearest-neighbour couplings
```
### Anderson Model (1d–4d)
```julia
d=3; # Spatial dimension
W=20; # Disorder
t=1; # Tunneling probability
H = H_Anderson(d,N,W,t)              # Anderson localization model
H = H_Anderson(d,N,W,t;h=my_h)       # ...or supply the disorder yourself
H = H_Random_Hop1D(N,W,t)            # off-diagonal (hopping) disorder instead
```
### Aubry-André quasiperiodic chain
`L` is rounded to the nearest Fibonacci number so the potential is properly quasiperiodic; the size actually used is returned alongside `H`.
```julia
H_AA, L_used = Aubry_Andre_H(N,2.0)
H_AA, L_used = Aubry_Andre_H(N,2.0;t2=0.5)   # with next-nearest-neighbour hopping
```
### Random matrices
```julia
H = H_GOE(N)                  # 2^N x 2^N GOE matrix
H = Random_XYZ(N)             # random all-to-all 2-body Paulis, GOE-normalized bandwidth
H, dot = UltraMetric_H(2,2,0.5,1.0)   # hierarchical GOE-block model
```

## Time evolution — `Time_evolution.jl`
### Evolve a state by diagonalizing H
Sparse and dense `H` both work.
```julia
N = 10
t_array = collect(0:10)          # array of times to calculate
psi = Product_state(N,"Neel")
psi_t = Diag_TE(psi,H_XXX,t_array)      # rows are times, columns are amplitudes
psi_t = state_evolve(H_XXX,psi,t_array) # same thing, as a Vector of state vectors
```

Or do the same plus measuring Z on each site:
```julia
Ops = [Op_create(N,"Z",site,"pauli") for site in 1:N];
Ops_t = Diag_TE_measure(psi,H_Quantum_Ising(N,1.0),t_array,Ops);
Ops_t = exp_TE_measure(psi,H,t_array,Ops);   # matrix-exponential route instead
```

Heisenberg-picture operator evolution, reusing one diagonalization:
```julia
vals,vecs = eigen(Hermitian(Matrix(H)))
O_t = Diag_TE(Matrix(Ops[1]),vecs,vals,t_array)   # Vector of O(t)
```

## Chaos and spectral diagnostics — `Measurements.jl`
### Level statistics
Mean level-spacing ratio is ≈ 0.386 for a localized (Poisson) spectrum and ≈ 0.536 for a chaotic (GOE) one. Resolve symmetry sectors first — a uniform periodic chain mixes them and gives a misleadingly low value.
```julia
r = Level_ratios(H,"mid")     # "mid" windows the middle of the band, "all" uses everything
s = Level_spacings(H,"all")
mean(r)
```
### Spectral functions and correlators
```julia
freqs, peaks = Spectral_fct_ED(H,O)                       # all (ω_nm, |O_nm|²)
freqs, peaks = Spectral_fct_ED(H,O;e_start=100,e_end=900) # restrict to a spectral window
centers, heights = Spectral_function_binned(H,O,0.1)      # binned S(ω)

C_t = Correlation_fct_ED(H,O,t_array)                     # tr(O(t)O)/D
K_tau = SFF(tau_array,H)                                  # filtered spectral form factor
```
### One diagonalization, all observables
`analyze_ED` runs a single `eigen!` and returns the correlation function, binned spectral function, level ratios and Lanczos `b_n` together, rather than diagonalizing once per quantity.
```julia
r = analyze_ED(Matrix(H),Matrix(O),t_array; Nsteps=30, bin_width=0.005)
r.C_t; r.sf_heights; r.ratios; r.bn
r.tr_P                      # C(t) as t -> infinity, the diagonal-ensemble weight
```
### Disorder averaging
```julia
mean_r, median_r, var_r = disorder_average(() -> Level_ratios(H_Anderson(3,8,W,1),"mid"), 20)
```
See `Example_localization.jl` for the Anderson localization transition, and `Example_analyze_ED.jl` for the full battery.

## Operator growth / Krylov — `Lanczos_functions.jl`
```julia
H = H_Quantum_Ising(8,1.05,0.5)
O = Matrix(Op_create(8,"Z",1,"pauli"))

b = Lanczos(copy(O),H,30)      # Lanczos coefficients b_n, direct
b = Lanczos_ED(O,H,30)         # ...via the energy eigenbasis (faster for many steps)

vals,vecs = eigen(Hermitian(Matrix(H)))    # reuse one diagonalization across probes
b = Lanczos_from_spectrum(vecs'*O*vecs,vals,30)

O_ed = Matrix{Float64}(vecs'*O*vecs)       # same b_n at roughly half the memory,
b = lanczos_bn_inplace!(O_ed,vals,30)      # but it consumes O_ed
```
### The `b_n` storage convention
`Lanczos` and `Lanczos_ED` put a placeholder in front — `b[1]=1`, or `0` from `Lanczos_diag` — so that `b[k+1]` is the k-th coupling. `Lanczos_monic` and `Lanczos_subtract` return the bare couplings with nothing in front. Everything downstream routes through one pair of helpers rather than each re-deciding which it was handed:
```julia
has_placeholder(b)                          # guessed from the value of b[1]
bn_couplings(b)                             # bare couplings, placeholder dropped
bn_with_placeholder(b)                      # the padded form
bn_couplings(b; placeholder=false)          # say so by hand if the guess would be wrong
```
Every function that consumes a `b_n` array takes the same `placeholder` keyword, so bare couplings that happen to start at exactly `1` or `0` can still be passed — pre-stripping them yourself would not work, since the guess would just run again:
```julia
K_complexity(collect(1.0:15.0), t_vec; placeholder=false)   # 15 couplings, not 14
```
The placeholder is **not** a coupling, so it is dropped before any Krylov chain is built. Keeping it adds a spurious site to the front of the chain and shifts every result — `Autocorr_fct_from_bn` was doing exactly that, and now reproduces `Correlation_fct_ED` to machine precision.

From the `b_n` you can get the spectral function, autocorrelation, and Krylov complexity:
```julia
vals, weights = Spectral_fct_from_bn(b)
centers, density = bin_spectral_function(vals,weights,100)
tvec, C_t = Autocorr_fct_from_bn(b,collect(0:0.1:10))
K_t = K_complexity(b,collect(0:0.1:10))
```
Krylov-space zero modes (LIOMs) and the moment map:
```julia
phi = LIOM_from_b(b)             # zero-mode amplitudes (= ZeroMode_from_b)
ipr, nrm = LIOM_avg_IPR(b_tot)   # log-averaged IPR over disorder samples
m = moments_from_b(b)            # moments of the Liouvillian
b2 = b_from_moments(real.(m))    # ...and back again
```
Variants of the recursion, all returning the same `b_n` plus extra diagnostics:
```julia
b, d      = Lanczos_diag(copy(O),H,30)          # ...plus the trace of each Krylov operator
b, q, d, Ofull = Lanczos_full(copy(O),H,30)     # ...plus the whole Krylov basis
b, r      = Lanczos_level_Ratios_ED(copy(O),H,30)  # ...plus the level ratios of the same spectrum
Δ         = Lanczos_monic(copy(O),H,30)         # monic recursion, returns Δ_n = b_n²
b1,b2,bsb = Lanczos_subtract(copy(O),H_1,H_2,30)   # two Hamiltonians and their difference
b, cn, ZM_real_space, vecs, overlaps, errors = Lanczos_cn_calc(copy(O),H,30)
```

### Localization in Krylov space
Lanczos coefficients for the Anderson model, and the integrals of motion which are the zero modes of the Liouvillian. Here the *Hamiltonians and operators are written in the single-particle basis*, so `H_Anderson(d,L,W,t)` is `L^d × L^d` rather than `2^N`.
```julia
d=3; L=5; t=1; W=30
H = H_Anderson(d,L,W,t)

#probe: the density on site L/2 of each dimension, the exact origin of the torus
site=Int(round(L/2)); Diag = zeros(L); Diag[site] = 1
Probe = spdiagm(Diag)
for j in 2:d
  Probe = kron(Probe,spdiagm(Diag))
end

b = Lanczos(Matrix(Probe),H,100)
```
Storing the Hamiltonian as sparse and the probe as dense is generally recommended. In general the Krylov space dimension is `K_dim = V^2 - V + 1` where `V = L^d`, which sets how many steps are available.

After collecting samples in `b_tot[Nsteps,Nsamples]`, the average zero mode can be taken two ways — average the coefficients first, or take the zero mode of each sample and log-average:
```julia
ZeroMode_b_avg = ZeroMode_from_b(mean(b_tot,dims=2)[:])
ZeroMode_avg   = ZeroMode_log_avg_from_b(b_tot)
```
In the localized regime the zero mode follows the predicted scaling `ZeroMode(n) ~ exp(-n^(1/2d))`. Note that the zero mode is exactly zero on every other site.

`Example_Krylov_localization.jl` collects the samples and `Example_Krylov_zero_modes.jl` reloads them and plots both zero modes against that scaling (plotting is written for PyPlot.jl).

## Disorder statistics and LIOMs — `Lanczos_statistics.jl`
`bn_tot` holds one disorder realization per column.
```julia
xs = x_generate(bn_tot,K)              # x_n = -log(b_2n+1/b_2n)
st = bn_statistics(bn_tot)             # xs, means, medians, variances, covariances, cumsums
means, vars = kappa_cumsum_stats(bn_tot,K)
phi, eta = phi_eta_calc(bn_tot,K)

ZeroMode_log_avg_from_b(bn_tot)        # log-averaged zero mode
LIOM_vars_from_b(bn_tot)
LIOM_avg_TunnelRate(bn_tot)            # G = b_n |phi_n|²
LIOM_from_xs(bn_tot)
avg_dimerization_from_b_tot(bn_tot)    # staggering of the b_n sequence
```
Zero modes by diagonalizing the Krylov Liouvillian directly, sweeping the chain length kept:
```julia
vals, vecs = LIOM_avg_construct(bn_tot,[20,40,60],1)
vals, vecs = LIOM_log_avg_construct(bn_tot,[20,40,60],1)
```

## Examples
| File | Technique |
|---|---|
| `Example.jl` | Operators, states, time evolution, level statistics, Lanczos |
| `Example_localization.jl` | Disorder averaging and the Anderson localization transition |
| `Example_analyze_ED.jl` | One diagonalization, all observables |
| `Example_Krylov_localization.jl` | Lanczos coefficients of the Anderson model, sampled over disorder |
| `Example_Krylov_zero_modes.jl` | Reload those samples and plot the Krylov-space zero modes |

The Krylov examples write to and read from `examples/Outs/`, which ships with 16 sample realizations for d=3, L=10, W=40 so the zero-mode script runs without collecting anything first.

The plotting in `Example_Krylov_zero_modes.jl` needs `PyPlot`, which is not a core dependency. The examples have their own environment in `examples/`. Set it up once:
```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```
then run any example with:
```bash
julia --project=examples examples/Example.jl
```

## Tests
```bash
julia --project=. test/runtests.jl
```
