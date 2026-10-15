#One diagonalization, all observables. analyze_ED returns the correlation function,
#binned spectral function, level ratios and Lanczos b_n from a single eigen! of H.
using QuantumSimulator
using Statistics

N = 10
hx = 0.5
hz = 0.5

H = H_Quantum_Ising(N,hx,hz)
O = Op_create(N,"Z",1,"pauli")           #pauli-Z on site 1
times = vcat(0.0, exp.(range(log(1e-1), log(1e3), length = 40)))

r = analyze_ED(Matrix(H),Matrix(O),times; Nsteps=30, bin_width=0.005)

@show mean(r.ratios)                    #~0.53 in the chaotic phase
@show r.C_t[1], r.C_t[end]              #autocorrelation decay
@show r.tr_P                            #C(infinity), the diagonal-ensemble weight
@show r.bn[1:5]
@show length(r.sf_heights)

#the same b_n, the memory-lean way, when the probe is already in the eigenbasis
vals,vecs = eigen(Hermitian(Matrix(H)))
O_ed = Matrix{Float64}(vecs' * Matrix(O) * vecs)
@show lanczos_bn_inplace!(O_ed,vals,30)[1:5]
