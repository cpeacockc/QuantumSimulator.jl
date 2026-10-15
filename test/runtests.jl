#Test suite for QuantumSimulator.  Run with:  julia --project=. test/runtests.jl
using Test, LinearAlgebra, SparseArrays, Statistics, Random

using QuantumSimulator

Random.seed!(1234)
herm(H) = norm(Matrix(H) - Matrix(H)') < 1e-10

@testset "QuantumSimulator" begin

@testset "Pauli generator" begin
    N = 4
    @test size(Op_create(N,"Z",2,"pauli")) == (2^N,2^N)
    @test Matrix(Op_create(N,"Z",2,"spin")) ≈ 0.5*Matrix(Op_create(N,"Z",2,"pauli"))

    #Pauli algebra
    X = Op_create(N,"X",1,"pauli"); Y = Op_create(N,"Y",1,"pauli"); Z = Op_create(N,"Z",1,"pauli")
    @test Matrix(X*Y) ≈ im*Matrix(Z)
    @test Matrix(X*X) ≈ Matrix(I,2^N,2^N)
    #operators on different sites commute
    @test norm(Matrix(comm(Op_create(N,"X",1,"pauli"),Op_create(N,"Z",3,"pauli")))) < 1e-12
    #and on the same site they do not
    @test norm(Matrix(comm(X,Z))) > 1e-6

    @test_throws ErrorException Op_create(N,"Q",1,"pauli")
    @test_throws ErrorException pauli_expand(zeros(N),"paulli")

    #multi-site strings
    @test Matrix(op_create(N,[("X",1),("Z",3)])) ≈ Matrix(Op_create(N,"X",1,"pauli")*Op_create(N,"Z",3,"pauli"))
    @test_throws ErrorException op_create(N,[("X",99)])

    #site_ops must return exactly the operators state_measure builds from its op string -- that
    #equivalence is the whole reason the Ops keyword can replace the measure=true arrays, so it is
    #checked for every op and both conventions rather than spot-checked on one.
    Random.seed!(4); psi = normalize(randn(ComplexF64,2^N))
    for flag in ("spin","pauli"), o in ("X","Y","Z","J")
        A = site_ops(N,o;flag=flag)
        @test length(A) == (o=="J" ? N-1 : N)
        #the Ops path returns real expectations only, so each operator has to be Hermitian
        @test maximum(norm(M-M') for M in A) < 1e-14
        @test [real(dot(psi,M*psi)) for M in A] ≈ state_measure(psi,o,(o=="J" ? (1:1:N-1) : (1:1:N)),flag)
    end
    #the bond/site distinction is the off-by-one site_ops exists to absorb, so "J" must reject
    #an index list running to L while "Z" accepts it
    @test_throws ErrorException site_ops(N,"J",1:1:N)
    @test_throws ErrorException site_ops(N,"Q")
    @test length(site_ops(N,"Z",1:1:N)) == N
    #an explicit index list is honoured, and lands on the site it names
    @test Matrix(site_ops(N,"Z",[3])[1]) ≈ Matrix(Op_create(N,"Z",3,"pauli"))

end

@testset "Product states and site ordering" begin
    N = 4
    for flag in ("Neel","Up","Dn","Down","Rand","RandVec")
        @test norm(Product_state(N,flag)) ≈ 1
    end
    @test Product_state(N,"Dn") == Product_state(N,"Down")
    @test_throws ErrorException Product_state(N,"Downn")

    #Op_create and Product_state must agree on which site is which.
    #Neel is up,dn,up,dn so <Z_s> alternates +1,-1,...
    psi = Product_state(N,"Neel")
    for s in 1:N
        expected = isodd(s) ? 1.0 : -1.0
        @test real(psi' * Op_create(N,"Z",s,"pauli") * psi) ≈ expected
    end

    #all-up must give <Z_s> = +1 everywhere
    psi_up = Product_state(N,"Up")
    @test all(real(psi_up' * Op_create(N,"Z",s,"pauli") * psi_up) ≈ 1.0 for s in 1:N)

    #"Rand" is a genuine product state: every <Z_s> is exactly +-1
    psi_r = Product_state(6,"Rand")
    @test all(abs(real(psi_r' * Op_create(6,"Z",s,"pauli") * psi_r)) ≈ 1.0 for s in 1:6)

    #density matrices
    rho = Product_densm(N,"Neel")
    @test tr(rho) ≈ 1
    @test rho ≈ rho'
    @test tr(Product_densm(N,"RandMixed")) ≈ 1

    #infinite-temperature (maximally mixed) and fully-excited states
    rho_inf = Product_densm(N,"InfTemp")
    @test rho_inf ≈ Matrix{ComplexF64}(I,2^N,2^N)/2^N
    @test tr(rho_inf) ≈ 1
    rho_up = Product_densm(N,"Up")
    @test rho_up ≈ psi_up*psi_up'
    @test all(real(tr(rho_up*Op_create(N,"Z",s,"pauli"))) ≈ 1.0 for s in 1:N)
end

@testset "Models: hermiticity and shape" begin
    N = 4
    models = Dict(
        "H_XYZ"                   => H_XYZ(N,[1,1,1],rand(N),rand(N),rand(N)),
        "H_XYZ open/spin"         => H_XYZ(N,[1,0.5,2],zeros(N),zeros(N),zeros(N);periodic=false,spinflag="spin"),
        "H_Quantum_Ising"         => H_Quantum_Ising(N,1.0),
        "H_Quantum_Ising 3-arg"   => H_Quantum_Ising(N,1.0,0.5),
        "H_Disordered_Heisenberg" => H_Disordered_Heisenberg(N,1.0,5.0),
        "H_GOE"                   => H_GOE(N),
        "Random_XYZ"              => Random_XYZ(N),
        "Rand_XYZ"                => Rand_XYZ(N),
        "Rand_Heisenberg"         => Rand_Heisenberg(N),
        "H_XXZ"                   => H_XXZ(N,1.0,2.0),
        "XXZnnn_H"                => XXZnnn_H(N),
        "H_mb_Anderson1D"         => H_mb_Anderson1D(N,1.0,randn(N)),
    )
    for (name,H) in models
        @test size(H) == (2^N,2^N)
        @test herm(H)
    end
end

@testset "Models: XYZ couplings and boundary conditions" begin
    N = 3
    Jx,Jy,Jz = 1.0,10.0,100.0
    Hp = H_XYZ(N,[Jx,Jy,Jz],zeros(N),zeros(N),zeros(N);periodic=true)
    Ho = H_XYZ(N,[Jx,Jy,Jz],zeros(N),zeros(N),zeros(N);periodic=false)

    #the wrap bond must use Jy and Jz, not Jx three times
    XX = Op_create(N,"X",1,"pauli")*Op_create(N,"X",N,"pauli")
    YY = Op_create(N,"Y",1,"pauli")*Op_create(N,"Y",N,"pauli")
    ZZ = Op_create(N,"Z",1,"pauli")*Op_create(N,"Z",N,"pauli")
    @test Matrix(Hp - Ho) ≈ Matrix(Jx*XX + Jy*YY + Jz*ZZ)

    #the fields land on the right sites
    Zf = zeros(N); Zf[2] = 3.0
    @test Matrix(H_XYZ(N,[0,0,0],zeros(N),zeros(N),Zf)) ≈ 3.0*Matrix(Op_create(N,"Z",2,"pauli"))

    #the Ising wrap bond is ZZ, matching the bulk chain
    Ip = H_Quantum_Ising(N,0.0,0.0;periodic=true)
    Io = H_Quantum_Ising(N,0.0,0.0;periodic=false)
    @test Matrix(Ip - Io) ≈ Matrix(ZZ)

    #Z_field defaults to zero
    @test Matrix(H_Quantum_Ising(N,1.0)) ≈ Matrix(H_Quantum_Ising(N,1.0,0.0))
end

@testset "Anderson models" begin
    for d in 1:4
        H = H_Anderson(d,3,20,1)
        @test size(H) == (3^d,3^d)
        @test herm(H)
    end
    @test_throws ErrorException H_Anderson(5,3,20,1)

    #supplying the disorder explicitly must reproduce itself on the diagonal
    h = randn(6)
    H = H_Anderson1D(6,0,1.0;h=h,periodic=false)
    @test diag(Matrix(H)) ≈ h

    #open vs periodic differ only in the corners
    Hp = H_Anderson1D(6,0,1.0;h=zeros(6),periodic=true)
    @test Hp[1,6] == -1.0 && Hp[6,1] == -1.0

    @test herm(H_Random_Hop1D(8,2.0,1.0))
    AA,f = Aubry_Andre_H(8,2.0)
    @test size(AA) == (f,f) && herm(AA)
    AA2,f2 = Aubry_Andre_nnn_H(8,2.0;t2=0.5)
    @test size(AA2) == (f2,f2) && herm(AA2)
end

@testset "Quantum Sun family" begin
    N,L = 3,3
    for f in (QuantumSunH, QuantumSunH_norand, QuantumSunH_fixedpower, QuantumSunH_randXYZ)
        H = f(N,L,0.75;seed=7)
        @test size(H) == (2^(N+L),2^(N+L))
        @test herm(H)
    end
    #the seed must make it reproducible
    @test QuantumSunH(N,L,0.75;seed=42) ≈ QuantumSunH(N,L,0.75;seed=42)
    #optional extra returns
    H,core = QuantumSunH(N,L,0.75;seed=7,core_return=true)
    @test size(core) == size(H)
    H,spin = QuantumSunH(N,L,0.75;seed=7,furthest_spin_return=true)
    @test size(spin) == size(H)

    UM,dot = UltraMetric_H(2,2,0.5,1.0)
    @test size(UM) == (2^4,2^4) && herm(UM)
end

@testset "Time evolution" begin
    N = 4
    psi = Product_state(N,"Neel")
    Hs = H_Quantum_Ising(N,1.0,0.5)     #sparse
    Hd = Matrix(Hs)                     #dense
    ts = collect(0.0:0.5:3.0)

    #sparse and dense H must agree, and both must work
    @test Diag_TE(psi,Hs,ts) ≈ Diag_TE(psi,Hd,ts)
    #integer time arrays and ranges must work too
    @test size(Diag_TE(psi,Hs,collect(0:3))) == (4,2^N)
    @test size(Diag_TE(psi,Hs,0.0:0.5:1.0)) == (3,2^N)

    #evolution is unitary
    psi_t = Diag_TE(psi,Hs,ts)
    @test all(norm(psi_t[i,:]) ≈ 1 for i in 1:length(ts))

    #t=0 returns the initial state
    @test Diag_TE(psi,Hs,[0.0])[1,:] ≈ psi

    #measurement path agrees with evolving then measuring by hand
    Ops = [Op_create(N,"Z",s,"pauli") for s in 1:N]
    M = Diag_TE_measure(psi,Hs,ts,Ops)
    @test size(M) == (length(ts),N)
    @test real(M[1,:]) ≈ [real(psi'*O*psi) for O in Ops]

    #the exponential path agrees with the diagonalization path
    @test exp_TE_measure(psi,Hd,ts,Ops) ≈ M

    #state_evolve returns the same trajectory
    sv = state_evolve(Hs,psi,ts)
    @test length(sv) == length(ts)
    @test all(abs(abs(dot(sv[i],psi_t[i,:])) - 1) < 1e-8 for i in 1:length(ts))

    #Heisenberg picture: <psi|O(t)|psi> must match <psi(t)|O|psi(t)>
    vals,vecs = eigen(Hermitian(Hd))
    O_t = Diag_TE(Matrix(Ops[1]),vecs,vals,ts)
    @test all(real(psi'*O_t[i]*psi) ≈ real(psi_t[i,:]'*Ops[1]*psi_t[i,:]) for i in 1:length(ts))

    #conserved energy
    @test all(abs(real(psi_t[i,:]'*Hd*psi_t[i,:]) - real(psi'*Hd*psi)) < 1e-8 for i in 1:length(ts))
end

@testset "Measurements" begin
    N = 6
    H = Matrix(H_Quantum_Ising(N,1.05,0.5))
    O = Matrix(Op_create(N,"Z",1,"pauli"))

    @test 0 < sparsity(H_Quantum_Ising(N,1.0)) <= 1
    @test length(gap_find(eigvals(H))) == 3

    #A GOE matrix must give <r> ~ 0.536, an uncorrelated spectrum ~ 0.386.
    #NB: a *uniform periodic* chain is not a good test here -- translation and
    #spin-flip symmetry mix independent sectors and push <r> well below Poisson.
    @test 0.48 < mean(Level_ratios(H_GOE(8),"all")) < 0.60
    @test 0.30 < mean(Level_ratios(diagm(randn(256)),"all")) < 0.45
    #a symmetry-free chaotic many-body model also lands near GOE
    @test 0.45 < mean(Level_ratios(Matrix(QuantumSunH(3,4,0.9;seed=3)),"all")) < 0.62

    @test length(Level_spacings(H,"all")) == 2^N - 1
    @test length(Level_ratios(H,"all")) == 2^N - 2
    @test_throws ErrorException Level_spacings(H,"middle")

    #spectral function: total weight is the Frobenius norm of O in the eigenbasis
    freqs,peaks = Spectral_fct_ED(H,O)
    @test length(freqs) == (2^N)^2
    @test sum(peaks) ≈ sum(abs2,O)

    #windowing must slice the operator too, not just the eigenvalues
    fw,pw = Spectral_fct_ED(H,O;e_start=5,e_end=12)
    vals,vecs = eigen(Hermitian(H)); Oe = vecs'*O*vecs
    @test sort(pw) ≈ sort([abs2(Oe[n,m]) for n in 5:12 for m in 5:12])

    #multi-operator method agrees with the single-operator one
    f2,p2 = Spectral_fct_ED(H,[O,O])
    @test p2[1] ≈ peaks && p2[2] ≈ peaks

    c,h = Spectral_function_binned(H,O,0.5;x_tot_max=10.0)
    @test length(c) == length(h)
    c2,h2 = spectral_function_from_spectrum(Oe,vals,0.5)
    @test length(c2) == length(h2)

    #binning conserves total weight
    edges = collect(0.0:0.5:5.0)
    x = rand(200).*5; y = rand(200)
    @test sum(Bin_heights(x,y,edges)) ≈ sum(y)

    #correlation functions: C(0) = tr(O'O)/D, and the two routines agree
    ts = collect(0.0:0.25:2.0)
    C1 = Correlation_fct_ED(H,O,ts)
    C2 = Correlator_ED_t(ts,vals,vecs,O)
    @test C1 ≈ C2
    @test C1[1] ≈ real(tr(O'*O))

    #SFF is real, positive, and starts at 1
    @test SFF_iter(0.0,H) ≈ 1.0
    @test all(SFF(collect(0.0:1.0:5.0),H) .>= 0)
    @test filter_dist(5.0,0.0,1.0) < 1e-6   #the filter decays, it does not blow up
end

@testset "Lanczos / Krylov" begin
    N = 5
    H = H_Quantum_Ising(N,1.05,0.5)
    O = Matrix(Op_create(N,"Z",1,"pauli"))
    Nsteps = 15

    b = Lanczos(copy(O),H,Nsteps)
    @test length(b) == Nsteps
    @test b[1] == 1
    @test all(b[2:end] .> 0)

    #BigFloat in, BigFloat chain out. A diagonal H makes the eigenbasis exact, so the matrix recursion and the Stieltjes
    #procedure on nodes E_a - E_b, weights |O_ab|^2 agree far below Float64 rounding only if no b_n is demoted
    b_256 = setprecision(() -> Lanczos(BigFloat.(O),BigFloat.(H),Nsteps),BigFloat,256)
    @test eltype(b_256) == BigFloat
    @test maximum(abs.(b_256 .- b)) < 1e-12
    E = randn(16); O_rand = randn(16,16); O_rand = O_rand + O_rand'
    gap_exact = setprecision(BigFloat,256) do
        Eb = BigFloat.(E); Ob = BigFloat.(O_rand)
        maximum(abs.(Lanczos(Ob,Matrix(Diagonal(Eb)),Nsteps) .- jacobi_from_measure(vec(Eb .- Eb'),vec(abs2.(Ob)),Nsteps)))
    end
    @test gap_exact < 1e-60

    #the eigenbasis route must reproduce the direct route
    b_ed = Lanczos_ED(O,H,Nsteps)
    @test maximum(abs.(b .- b_ed)) < 1e-8

    #and so must the precomputed-spectrum route
    vals,vecs = eigen(Hermitian(Matrix(H)))
    @test Lanczos_from_spectrum(vecs'*O*vecs,vals,Nsteps) ≈ b_ed

    #Op_Norm is the normalized Frobenius norm, dense and sparse agreeing
    M = randn(8,8)
    @test Op_Norm(M) ≈ sqrt(tr(M'*M)/8)
    @test Op_Norm(sparse(M)) ≈ Op_Norm(M)

    #b_1 is the norm of [H,O], the first Krylov step
    @test b[2] ≈ Op_Norm(Matrix(comm(Matrix(H),O)))/Op_Norm(O)

    #the b[1] placeholder is not a coupling, so the Krylov chain has length(b)-1 bonds
    @test bn_couplings(b) == b[2:end]
    @test bn_with_placeholder(b) == b
    @test bn_with_placeholder(b[2:end]) == b
    @test bn_couplings(b[2:end]; placeholder=false) == b[2:end]

    #spectral function from b_n carries the right number of poles
    v,w = Spectral_fct_from_bn(b)
    @test length(v) == length(b)
    cen,den = bin_spectral_function(v,w,20)
    @test length(cen) == 20

    #autocorrelation starts at 1 and stays bounded
    tv,Ct = Autocorr_fct_from_bn(b,collect(0.0:0.5:2.0))
    @test abs(Ct[1] - 1) < 1e-10
    @test all(abs.(Ct) .<= 1 + 1e-8)

    #and it must actually reproduce the ED autocorrelation. Keeping the placeholder as a
    #coupling still satisfies the two tests above, but is wrong by O(0.5) here.
    let times = collect(0.0:0.25:1.5), b_long = Lanczos(copy(O),H,60)
        C_ed = real.(Correlation_fct_ED(H,O,times)); C_ed ./= C_ed[1]
        @test maximum(abs.(real.(Autocorr_fct_from_bn(b_long,times)[2]) .- C_ed)) < 1e-6
    end

    #Krylov complexity grows from zero. These are bare couplings that happen to start at
    #1.0, which is exactly the case the b[1] guess gets wrong, so say so explicitly.
    K = K_complexity(collect(1.0:15.0),collect(0.0:0.25:2.0); placeholder=false)
    @test K[1] < 1e-8
    @test issorted(K)
    @test length(K_phi_t(collect(1.0:15.0),collect(0.0:0.5:2.0); placeholder=false)) == 5
    #and with the guess left on, the leading 1.0 is taken for a placeholder and dropped
    @test length(bn_couplings(collect(1.0:15.0))) == 14

    #moments <-> b_n round trip
    bn = [1.0,2.0,2.5,2.8,3.0,3.1]
    m = real.(moments_from_b(bn))
    @test b_from_moments(m)[1:4] ≈ bn[2:5] atol=1e-6

    #zero mode / LIOM helpers
    @test LIOM_from_b(b) == ZeroMode_from_b(b)
    @test length(LIOM_from_zero_mode(b)) == length(b)
    b_tot = hcat([Lanczos_ED(O,H_Quantum_Ising(N,1.05,0.5),Nsteps) for _ in 1:3]...)
    @test length(ZeroMode_log_avg_from_b(b_tot)) == Nsteps
    ipr,nrm = LIOM_avg_IPR(b_tot)
    @test ipr > 0 && nrm > 0

    #dimerization diagnostics
    @test length(dimerization_from_b(b)) == length(dimerization_ratio_from_b(b))

    #Krylov Liouvillian shapes
    @test size(L_matrix(b)) == (length(b),length(b))
    @test size(L_sparse(b)) == (length(b),length(b))
    @test size(buildL(b)) == (length(b)-1,length(b)-1)
end

@testset "jacobi_from_measure (Stieltjes procedure)" begin
    #Chebyshev T: the M-node Chebyshev-Gauss rule x_j = cos((2j-1)pi/(2M)), w_j = pi/M is exact for b_1..b_(M-1), and
    #T_(n+1) = 2x T_n - T_(n-1) gives b_1 = 1/sqrt(2), b_n = 1/2 (en.wikipedia.org/wiki/Chebyshev-Gauss_quadrature)
    M = 64
    b = jacobi_from_measure([cos((2*j-1)*pi/(2*M)) for j in 1:1:M],fill(pi/M,M),40)
    @test length(b) == 40 && b[1] == 1
    @test abs(b[2] - 1/sqrt(2)) < 1e-14
    @test maximum(abs.(b[3:end] .- 0.5)) < 1e-14

    #Legendre: Gauss-Legendre nodes and weights from the Jacobi matrix of b_n = n/sqrt(4n^2-1) (Bonnet's recursion) by
    #Golub-Welsch (en.wikipedia.org/wiki/Gaussian_quadrature, "The Golub-Welsch algorithm"); the round trip returns the b_n
    bL = [n/sqrt(4*n^2-1) for n in 1:1:M-1]
    F = eigen(SymTridiagonal(zeros(M),bL))
    b = jacobi_from_measure(F.values,2 .* F.vectors[1,:].^2,40)
    @test maximum(abs.(b[2:end] .- bL[1:39])) < 1e-14

    #Hermite: a Gaussian of variance sigma^2 on a wide uniform grid (the trapezoid rule converges exponentially), b_n = sigma sqrt(n)
    sigma = 1.7; omega = collect(-14*sigma:sigma/20:14*sigma)
    b = jacobi_from_measure(omega,exp.(-omega.^2 ./ (2*sigma^2)),21)
    @test maximum(abs.(b[2:end] ./ (sigma .* sqrt.(1:1:20)) .- 1)) < 1e-14

    #the Liouvillian measure (nodes E_a - E_b, weights |O_ab|^2) of a 4-spin quantum sun is Lanczos_from_spectrum exactly,
    #and the matrix Lanczos to the accuracy of the Float64 eigendecomposition (measured 1.1e-13 at n = 14)
    H,O = QuantumSunH(2,2,0.7;seed=3,furthest_spin_return=true); H = Matrix(H); O = Matrix(O)
    vals,vecs = eigen(Hermitian(H)); O_ed = vecs'*O*vecs
    nodes = vec(vals .- vals'); weights = vec(abs2.(O_ed))
    b = jacobi_from_measure(nodes,weights,15)
    @test maximum(abs.(b .- Lanczos_from_spectrum(O_ed,vals,15))) < 1e-14
    @test maximum(abs.(b .- Lanczos(copy(O),H,15))) < 1e-12

    #BigFloat in, BigFloat chain out: no Float64 literal demotes it
    b_big = setprecision(BigFloat,256) do
        jacobi_from_measure(BigFloat.(nodes),BigFloat.(weights),15)
    end
    @test eltype(b_big) == BigFloat
    @test maximum(abs.(b_big .- b)) < 1e-14
end

@testset "analyze_ED / disorder_average / Z_wave_op" begin
    N = 8
    H = Matrix(H_Quantum_Ising(N,1.05,0.5))
    O = Matrix(Op_create(N,"Z",1,"pauli"))
    times = collect(0.0:0.5:3.0)
    vals,vecs = eigen(Hermitian(H))

    #the in-place Lanczos is the same algorithm as Lanczos_from_spectrum, just fused. Bitwise
    #equality holds for a fixed BLAS but not across platforms, so compare to machine precision.
    b_ref = Lanczos_from_spectrum(vecs'*O*vecs, vals, 20)
    b_ip  = lanczos_bn_inplace!(Matrix{Float64}(vecs'*O*vecs), Vector{Float64}(vals), 20)
    @test b_ip ≈ b_ref rtol=1e-12

    r = analyze_ED(H,O,times; Nsteps=20, bin_width=0.05)
    @test r.C_t ≈ Correlation_fct_ED(H,O,times)
    @test r.bn ≈ b_ref
    @test r.ratios == Level_ratios(vals,"mid")
    @test r.vals ≈ vals
    #tr_P is the diagonal-ensemble weight, i.e. C(t) as t -> infinity
    @test r.tr_P ≈ sum(abs2, diag(vecs'*O*vecs))
    @test r.C_t[1] ≈ sum(abs2, vecs'*O*vecs)

    #analyze_ED must not clobber the caller's H
    H2 = copy(H); analyze_ED(H2,O,[0.0]); @test H2 == H
    #Nsteps=0 skips the Lanczos
    @test isempty(analyze_ED(H,O,[0.0]).bn)

    #disorder_average pools K realizations
    m,md,v = disorder_average(() -> Level_ratios(H_GOE(7),"all"), 4)
    @test 0.45 < m < 0.62
    @test v > 0
    @test disorder_average(() -> [1.0,3.0], 3) == (2.0, 2.0, var([1.0,3.0,1.0,3.0,1.0,3.0]))

    #Z_wave_op is a real diagonal wave of Z's
    Zw = Z_wave_op(6,2)
    @test size(Zw) == (2^6,2^6)
    @test herm(Zw)
    @test Matrix(Zw) ≈ sum(cos(2pi*2*i/6) .* Matrix(Op_create(6,"Z",i,"pauli")) for i in 1:6)
end

@testset "Lanczos variants" begin
    N = 6
    H = H_Quantum_Ising(N,1.05,0.5); Hd = Matrix(H)
    O = Matrix(Op_create(N,"Z",1,"pauli"))
    b_ref = Lanczos(copy(O),Hd,16)

    #every variant must reproduce the same b_n as the plain recursion
    bd,d = Lanczos_diag(copy(O),Hd,16)
    @test bd[2:end] ≈ b_ref[2:end]
    @test length(d) == length(bd)

    bf,q,dfull,Ofull = Lanczos_full(copy(O),Hd,15)
    @test bf ≈ b_ref[2:end]
    @test size(Ofull) == (16,2^N,2^N)
    #the stored Krylov basis is orthonormal under the infinite-temperature inner product
    @test Op_Norm(Ofull[1,:,:]) ≈ 1
    @test abs(Op_Inner(Ofull[1,:,:],Ofull[3,:,:])) < 1e-8

    bl,r = Lanczos_level_Ratios_ED(copy(O),Hd,20)
    @test bl ≈ Lanczos_ED(O,Hd,20)
    @test r ≈ Level_ratios(eigvals(Hermitian(Hd)),"mid")

    #a traceless probe keeps every Krylov operator traceless
    @test maximum(abs.(d)) < 1e-8

    #same H twice must give identical chains
    b1,b2,bsb = Lanczos_subtract(copy(O),Hd,Hd,12)
    @test b1 ≈ b2

    #monic recursion: Delta_n = b_n^2
    Δ = Lanczos_monic(copy(O),Hd,12)
    @test Δ[1] ≈ b_ref[2]^2

    bc,cn,zm,ov,mo,er = Lanczos_cn_calc(copy(O),Hd,10)
    @test bc ≈ Lanczos(copy(O),Hd,11)
    @test size(cn,2) == 11
    @test size(zm) == (2^N,2^N)
    bce,cne,zme,ove,moe,ere = Lanczos_cn_calc_ED(copy(O),Hd,10)
    @test bce ≈ bc

    @test length(cn_calc([randn(2^N) for _ in 1:3], O)) == 3
    @test length(vec_to_krylov(randn(2^N),Ofull)) == size(Ofull,1)

    #the two moment inversions agree
    bn = [1.0,2.0,2.5,2.8,3.0,3.1]
    m = real.(moments_from_b(bn))
    @test b_from_moments_old(m) ≈ b_from_moments(m) atol=1e-6
end

@testset "Lanczos statistics and LIOMs" begin
    N = 6
    b_tot = hcat([Lanczos_ED(Matrix(Op_create(N,"Z",s,"pauli")),
                             H_Disordered_Heisenberg(N,1.0,5.0),20) for s in 1:6]...)
    b = b_tot[:,1]

    @test length(x_from_b(b)) == length(b[1:2:end])-1
    xs = x_generate(copy(b_tot),6)
    @test size(xs,2) == 6
    m,v = kappa_cumsum_stats(copy(b_tot),6)
    @test length(m) == size(xs,1)

    st = bn_statistics(copy(b_tot))
    @test st.xs == x_generate(copy(b_tot),6)
    @test st.xs_mean ≈ mean(st.xs,dims=2)[:]
    @test size(st.xs_cov) == (size(xs,1),size(xs,1))
    @test st.xs_mean_cumsum ≈ mean(cumsum(st.xs,dims=1),dims=2)[:]

    phi,eta = phi_eta_calc(copy(b_tot),6)
    @test length(phi) == length(eta)
    @test length(var_of_sum(cov(randn(5,10),dims=2),5)) == 5

    #dimerization diagnostics
    @test length(avg_dimerization_from_b_tot(copy(b_tot))) == length(dimerization_from_b(copy(b)))
    @test length(avg_dimerization_ratio_from_b_tot(copy(b_tot))) ==
          length(log_avg_dimerization_ratio_from_b_tot(copy(b_tot)))
    bdp = dimerized_powerlaw_decay_bn(20,1.0,0.5,1.0)
    @test length(bdp) == 19
    @test all(bdp[1:2:end] .≈ 1.0)          #odd entries stay at a
    @test all(bdp[2:2:end] .> 1.0)          #even entries are lifted

    #LIOM_from_b normalizes the b[1] convention, so these must agree
    @test LIOM_from_b(copy(b)) ≈ ZeroMode_from_b(b)
    @test LIOM_from_b(vcat(0.0,b[2:end])) ≈ LIOM_from_b(copy(b))
    @test length(ZeroMode_log_avg_from_b(copy(b_tot))) == length(LIOM_from_b(copy(b)))
    @test length(LIOM_vars_from_b(copy(b_tot))) == length(LIOM_from_b(copy(b)))
    @test LIOM_avg_TunnelRate(copy(b_tot)) > 0
    @test LIOM_avg_final_val(copy(b_tot)) > 0
    @test length(LIOM_from_xs(copy(b_tot))) == size(x_generate(copy(b_tot),6),1)
    @test LIOM_from_xs_cumsum([0.0,1.0]) ≈ [1.0, exp(-1)]

    O = Matrix(Op_create(N,"Z",1,"pauli"))
    bb,q,dd,Of = Lanczos_full(copy(O),Matrix(H_Quantum_Ising(N,1.05,0.5)),10)
    @test size(LIOM_real_space(Of,bb)) == (2^N,2^N)

    #Krylov-Liouvillian constructions
    for f in (LIOM_avg_construct, LIOM_log_avg_construct, LIOM_log_avg_construct_L)
        vals,vecs = f(copy(b_tot),[8,10],1)
        @test length(vals) == 2 && length(vecs) == 2
    end
    @test length(LIOM_log_avg_construct_EA(copy(b_tot),[8,10],1)) == 2
    mv,mvec = LIOM_construct(copy(b),[8,10],1)
    @test length(mv) == 2
    @test norm(mvec[1]) ≈ 1     #LIOM_construct returns normalized vectors

    #log staggering, pair geometric mean, gamma fit (Krylov_sun, arXiv:2510.26920v4 eq. (10)); b WITHOUT the placeholder
    g=0.3;bst=[1.7*(1+(-1)^n*g) for n in 1:1:121];ks=collect(1:1:60)
    lam=log_staggering_from_b(bst)
    @test length(lam)==div(length(bst)-1,2)
    @test all(abs.(lam.-log((1+g)/(1-g))).<1e-12)                #pure staggering: ln((1+g)/(1-g)) at every k
    bsm=[n^0.5 for n in 1:1:121]
    @test all(abs.(log_staggering_from_b(bsm)).<=0.1 ./ ks.^2)    #a smooth b_n leaves only O(1/k^2)
    Lm=log_staggering_from_b_tot(hcat(bst,bsm))
    @test size(Lm)==(60,2) && Lm[:,2]==log_staggering_from_b(bsm)
    delta=-0.05;bdg=[n^delta*(1+(-1)^n*g) for n in 1:1:121]
    Bk=pair_geomean_from_b(bdg);envelope=[(2*k*(2*k-1))^(delta/2) for k in 1:1:60]
    @test length(Bk)==60
    @test all(abs.(Bk./envelope.-sqrt(1-g^2)).<1e-12)              #B_k = (2k(2k-1))^(delta/2) sqrt(1-g^2) exactly
    kf=collect(50.0:1.0:195.0);A0=0.01;g0=0.7;lam0=A0 .* kf.^(g0-1);sig=fill(1e-4,length(kf))
    fit=gamma_powerlaw_fit(kf,lam0,sig)
    @test isapprox(fit.gamma,g0;atol=1e-9) && isapprox(fit.A,A0;atol=1e-12) && fit.chi2<1e-12
    fitneg=gamma_powerlaw_fit(kf,-lam0,sig)
    @test isapprox(fitneg.gamma,g0;atol=1e-9) && isapprox(fitneg.A,-A0;atol=1e-12)   #no censoring on sign
    rng=MersenneTwister(7);fitn=gamma_powerlaw_fit(kf,lam0.+sig.*randn(rng,length(kf)),sig)
    @test abs(fitn.gamma-g0)<0.03 && abs(fitn.A-A0)<0.15*A0        #400 draws give sigma_gamma = 0.009, sigma_A/A = 0.04: thresholds at 3.4 and 3.7 sigma
    @test LIOM_from_b([2.0,4.0,3.0,6.0];placeholder=false)≈[1,0,0.5,0,0.25]   #|Q|^2 = LIOM_from_b(b;placeholder=false).^2
end

@testset "README examples" begin
    N = 10
    Z_5 = Op_create(N,"Z",5,"pauli")
    psi = Product_state(N,"Neel")
    @test real(psi'*Z_5*psi) ≈ 1.0

    Jx,Jy,Jz = 1,1,1
    X_fields, Y_fields, Z_fields = zeros(N),zeros(N),zeros(N)
    H_XXX = H_XYZ(N,[Jx,Jy,Jz],X_fields,Y_fields,Z_fields)
    @test herm(H_XXX)
    @test herm(H_XYZ(N,[Jx,Jy,Jz],X_fields,Y_fields,Z_fields;periodic=false,spinflag="spin"))

    hx = 1.0
    @test herm(H_Quantum_Ising(N,hx))
    @test herm(H_Anderson(3,4,20,1))
    @test herm(H_GOE(6))

    #time evolution straight off the sparse H, as written in the README
    t_array = collect(0:10)
    Nsmall = 8
    psi_s = Product_state(Nsmall,"Neel")
    H_s = H_XYZ(Nsmall,[1,1,1],zeros(Nsmall),zeros(Nsmall),zeros(Nsmall))
    @test size(Diag_TE(psi_s,H_s,t_array)) == (11,2^Nsmall)

    Ops = [Op_create(Nsmall,"Z",site,"pauli") for site in 1:Nsmall]
    @test size(Diag_TE_measure(psi_s,H_Quantum_Ising(Nsmall,hx),t_array,Ops)) == (11,Nsmall)
end


end
