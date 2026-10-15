using LinearAlgebra, SparseArrays, Statistics


#Liouvillian function for sparse matrices
function L_n(O::SparseMatrixCSC,H::SparseMatrixCSC,n::Integer)
    return H*O-O*H
end
#Liouvillian function for dense matrices
function L_n(O::AbstractMatrix,H::AbstractMatrix,n::Integer)
    prod = H*O
    return prod + ((-1)^(n+1))*adjoint(prod)
end
#Liouvillian function for dense O, sparse H (recommended)
function L_n(O::Matrix,H::SparseMatrixCSC,n::Integer) 
    prod = O*H
    prod += ((-1)^(n+1))*adjoint(prod)
    return -(prod)
end

#norm function for dense matrices
#(this is tr(O'O)/n, but summing |O_ij|^2 directly avoids forming the n x n product)
function Op_Norm(O::Matrix)
    return sqrt(sum(abs2,O)/size(O,1))
end
#norm function for sparse matrices
function Op_Norm(O::SparseMatrixCSC)
    return sqrt(sum(abs2,nonzeros(O))/size(O,1))
end
#Infinite temp inner product
function Op_Inner(A::AbstractArray,B::AbstractArray)
    return tr(A'*B)/size(A)[1]
end

#Unvalidated
function Lanczos(Probe::AbstractMatrix, H::AbstractMatrix, Nsteps::Integer)


    #Base Krylov vector 
    O = []

    #b1 is set to 1, in the precision of the inputs (Float64[1] demoted every b_n of a BigFloat chain)
    b = float(real(promote_type(eltype(Probe),eltype(H))))[1]

    #If you want, track orthogonalization error
    #error_0 = Float64[]
    #error_1 = Float64[]
    
    #Define O0, normalize
    
    Probe/=Op_Norm(Probe)
    push!(O,Probe)
    LO_0 = L_n(O[1],H,2)

    #Define b2
    push!(b,Op_Norm(LO_0))
    
    #Define O1
    push!(O,LO_0/b[2])
    
    for n in 3:Nsteps
        #Construct new Krylov basis vector
        A_n = L_n(O[2],H,n) - b[n-1]*O[1]
        #Calculate Lanczos coefficient
        b_n = Op_Norm(A_n)
        push!(b,b_n)
        #Reset vectors, don't store entire basis in memory
        O[1]=O[2]
        O[2] = (A_n/b_n)

        #Optional during loop
        #push!(error_0,Op_Inner(Probe,O[2]))
        #push!(error_1,Op_Inner(LO_0/b[2],O[2]))
        #@show n,b,error_0, error_1
    end
    return b
end


#The following ZeroMode functions give the probability amplitudes of the zero modes in Krylov space
function ZeroMode_from_b(b)
    #This function assumes starting with b[1]=1 as in the previous code

    phi_n = zeros(length(b))
    phi_n[1]=1 #phi_0
    for n in 3:2:length(b)
        phi=1
        for i in 1:Int((n-1)/2)
            phi *=(b[2*i]/b[2*i+1]) 
        end
        phi_n[n]=phi
    end
    return phi_n
end

function ZeroMode_log_avg_from_b(b_tot)
    Nsamples = size(b_tot)[2]

    ZeroModes=zeros(length(ZeroMode_from_b(b_tot[:,1])),Nsamples)

    for j in 1:Nsamples
        ZeroModes[:,j]=ZeroMode_from_b(b_tot[:,j])
    end

    #We take the geometric mean
    ZeroMode_log_avgs=exp.(mean(log.(abs.(ZeroModes)),dims=2))
    return ZeroMode_log_avgs
end

########## The b_n storage convention ##########
#b_n arrays come in two shapes. Lanczos and Lanczos_ED put a placeholder in front -- b[1]=1,
#or 0 from Lanczos_diag -- so that b[k+1] is the k-th coupling. Lanczos_monic and
#Lanczos_subtract return the bare couplings with nothing in front. The functions below route
#through these two rather than each re-deciding which they were handed. The guess is made on
#the value of b[1], so pass placeholder=true/false by hand in the rare case that you have
#bare couplings starting at exactly 1 or 0.
has_placeholder(b) = b[1] == 0 || b[1] == 1
bn_couplings(b; placeholder::Bool=has_placeholder(b)) = placeholder ? collect(b[2:end]) : collect(b)
bn_with_placeholder(b; placeholder::Bool=has_placeholder(b)) = vcat(1, bn_couplings(b; placeholder))

#Construct Liouvillian in Krylov space. The placeholder is not a coupling, so it is dropped
#before the chain is built -- keeping it adds a spurious site and shifts every result.
L_matrix(b; placeholder::Bool=has_placeholder(b)) =
    (c = bn_couplings(b; placeholder); Tridiagonal(c,zeros(length(c)+1),c))
L_sparse(b; placeholder::Bool=has_placeholder(b)) =
    (c = bn_couplings(b; placeholder); spdiagm(-1 => c, 1 => c))

#Antisymmetric generator used for real-time Krylov dynamics
buildL(b; open=false, placeholder::Bool=has_placeholder(b)) = begin
    c = bn_couplings(b; placeholder)
    L = Tridiagonal(c[1:end-1], zeros(length(c)), -c[1:end-1])
    if open
        L[end, end] = -2 * c[end]
        L[end, end-1] += c[end]
    end
    L
end

########## Lanczos variants ##########

#Also returns the trace of each Krylov operator, which vanishes for a traceless probe
function Lanczos_diag(Probe::AbstractMatrix,H::AbstractMatrix,Nsteps::Int64)

    #Base vector
    O = []
    b = Float64[0]
    d=Float64[0]
    #Define O0
    Probe/=Op_Norm(Probe)
    push!(O,Probe)
    LO_0 = L_n(O[1],H,2)
    #Define b1, b0 is set to 0
    push!(b,Op_Norm(LO_0))
    push!(d,sum(diag(LO_0)))

    #Define O1
    push!(O,LO_0/b[2])

    for n in 3:Nsteps
        A_n = L_n(O[2],H,n) - b[n-1]*O[1]
        b_n = Op_Norm(A_n)
        push!(b,b_n)
        push!(d,sum(diag(A_n)))
        O[1]=O[2]
        O[2] = (A_n/b_n)
    end
    return b,d
end

#Runs the recursion under two Hamiltonians at once and on their difference
function Lanczos_subtract(Probe::AbstractMatrix,H_1::AbstractMatrix,H_2::AbstractMatrix,Nsteps::Int64)

    #Base vector
    O_1 = [];O_2 = [];O_sb = []
    b_1 = Float64[0];b_2 = Float64[0];b_sb = Float64[0]
    #Define O0
    Probe/=Op_Norm(Probe)
    push!(O_1,Probe),push!(O_2,Probe),push!(O_sb,Probe)
    LO_1_0 = L_n(O_1[1],H_1,2)
    LO_2_0 = L_n(O_2[1],H_2,2)
    LO_sb_0 = LO_1_0-LO_2_0
    #Define b1, b0 is set to 0
    push!(b_1,Op_Norm(LO_1_0))
    push!(b_2,Op_Norm(LO_2_0))
    push!(b_sb,Op_Norm(LO_sb_0))

    #Define O1
    push!(O_1,LO_1_0/b_1[2])
    push!(O_2,LO_2_0/b_2[2])
    push!(O_sb,LO_sb_0/b_sb[2])

    for n in 3:Nsteps
        A_1_n = L_n(O_1[2],H_1,n) - b_1[n-1]*O_1[1]
        A_2_n = L_n(O_2[2],H_2,n) - b_2[n-1]*O_2[1]
        A_sb_n = A_1_n - A_2_n

        b1 = Op_Norm(A_1_n)
        b2 = Op_Norm(A_2_n)
        bsb = Op_Norm(A_sb_n)
        push!(b_1,b1);push!(b_2,b2);push!(b_sb,bsb)
        O_1[1]=O_1[2];O_2[1]=O_2[2];O_sb[1]=O_sb[2]
        O_1[2] = (A_1_n/b1);O_2[2] = (A_2_n/b2);O_sb[2] = (A_sb_n/bsb);
    end
    return b_1[2:end],b_2[2:end],b_sb[2:end]
end

#Monic (unnormalised) recursion, returning the Delta_n rather than the b_n
function Lanczos_monic(Probe::AbstractMatrix,H::AbstractMatrix,Nsteps::Int64)

    #Base vector
    O = []
    Δ = Float64[0]
    #Define O0
    Probe/=Op_Norm(Probe)
    push!(O,Probe)
    LO_0 = L_n(O[1],H,2)
    #Define b1, b0 is set to 0
    push!(Δ,Op_Norm(LO_0)^2)

    #Define O1
    push!(O,LO_0)

    for n in 3:Nsteps
        A_n = L_n(O[2],H,n) - Δ[n-1]*O[1]
        Δ_n = (Op_Norm(A_n)^2)/(Op_Norm(O[2])^2)
        push!(Δ,Δ_n)
        O[1]=O[2]
        O[2] = (A_n)
    end
    return Δ[2:end]
end

#Keeps the whole Krylov basis in memory and also returns the off-diagonal norms q_n
#and traces d_n at each step
function Lanczos_full(Probe::AbstractMatrix,H::AbstractMatrix,Nsteps::Int64)
    L = size(H)[1]
   #Base vector
   O = zeros(Nsteps+1,L,L)
   b = Float64[0]
   q = Float64[0]
   d = Float64[0]
   #Define O0
   Probe/=Op_Norm(Probe)
   O[1,:,:] = Probe
   LO_0 = L_n(O[1,:,:],H,2)
   #Define b1, b0 is set to 0
    push!(b,Op_Norm(LO_0))
    push!(q,Op_Norm(LO_0-diagm(diag(LO_0))))
    push!(d,sum(diag(LO_0)))

   #Define O1
   O[2,:,:]=LO_0/b[2]

   for n in 3:Nsteps+1
       A_n = L_n(O[n-1,:,:],H,n) - b[n-1]*O[n-2,:,:]
       b_n = Op_Norm(A_n)
       q_n = Op_Norm(A_n-diagm(diag(A_n)))
       d_n = sum(diag(A_n))
       push!(b,b_n)
       push!(q,q_n)
       push!(d,d_n)
       O[n,:,:] = (A_n/b_n)
   end
   return b[2:end],q[2:end],d[2:end],O
end

########## Lanczos in the energy eigenbasis ##########
#Once H is diagonal, the Liouvillian is just an elementwise multiply by the energy
#differences, so each Krylov step costs O(D^2) instead of a matrix product.

L_n_ED(Op::AbstractMatrix,Vals::AbstractVector) = (Vals .- Vals') .* Op

function Lanczos_ED(Probe::AbstractMatrix, H::AbstractMatrix, Nsteps::Integer)
    vals,vecs = eigen(Hermitian(Matrix(H)))
    return Lanczos_from_spectrum(vecs' * Matrix(Probe) * vecs, vals, Nsteps)
end

#Same as Lanczos_ED, but for when the spectrum is already in hand.
#Probe_ed must already be in the eigenbasis: vecs' * Probe * vecs.
function Lanczos_from_spectrum(Probe_ed::AbstractMatrix, vals::AbstractVector, Nsteps::Integer)
    O = []
    b = Float64[1] #b[1]=1 convention, matching Lanczos()

    Probe_ed = Probe_ed ./ Op_Norm(Matrix(Probe_ed))
    push!(O, Probe_ed)

    LO_0 = L_n_ED(O[1], vals)
    push!(b, Op_Norm(Matrix(LO_0)))
    push!(O, LO_0 / b[2])

    for n in 3:Nsteps
        A_n = L_n_ED(O[2], vals) - b[n-1] * O[1]
        b_n = Op_Norm(Matrix(A_n))
        push!(b, b_n)
        O[1] = O[2]
        O[2] = A_n / b_n
    end
    return b
end

# ED-spectrum operator-Lanczos b-coefficients for a probe already in the H eigenbasis.
# Reuses vals (no re-diagonalisation). Fully in-place: two dim×dim scratch buffers beyond
# the probe, and the frequency weight (vals[n]-vals[m]) is applied via a fused broadcast so
# no dim×dim frequency matrix is ever materialised — this keeps N=14 inside the same memory
# envelope as the rest of the job (the shared Lanczos_from_spectrum allocates per step and
# would need ~2x the RAM). Returns b with b[1]=1, numerically identical to
# Lanczos_from_spectrum. NOTE: normalises the probe in place, so call it as the LAST use of
# the eigenbasis probe.
function lanczos_bn_inplace!(O1::Matrix{Float64}, vals::Vector{Float64}, Nsteps::Int)
    dim = size(O1, 1)
    vT  = reshape(vals, 1, :)                    # row view; (vals .- vT)[n,m] = vals[n]-vals[m]
    opn(M) = sqrt(sum(abs2, M) / dim)
    b = Vector{Float64}(undef, Nsteps)
    b[1] = 1.0
    O1 ./= opn(O1)                               # |O_1)
    O2 = similar(O1)
    @. O2 = (vals - vT) * O1                     # L|O_1)
    b[2] = opn(O2)
    O2 ./= b[2]                                  # |O_2)
    A = similar(O1)
    for n in 3:Nsteps
        @. A = (vals - vT) * O2 - b[n-1] * O1    # A_n (fused, no temporaries)
        b[n] = opn(A)
        A ./= b[n]                               # |O_n)
        O1, O2, A = O2, A, O1                    # rotate buffers, no allocation
    end
    return b
end

#Jacobi coefficients of a discrete measure mu = sum_j w_j delta(omega - omega_j), symmetric about omega = 0, by the
#Stieltjes procedure: b_n p_n = omega p_(n-1) - b_(n-1) p_(n-2), b_n = ||omega p_(n-1) - b_(n-1) p_(n-2)||, ||f||^2 = sum_j w_j f_j^2,
#p_n the orthonormal polynomials stored by their values on the nodes. References: en.wikipedia.org/wiki/Gaussian_quadrature
#"Recurrence relation" (monic form); en.wikipedia.org/wiki/Lanczos_algorithm "The algorithm" (the same recursion on
#diag(omega_j) from the vector sqrt(w_j)); W. Gautschi, Orthogonal Polynomials: Computation and Approximation (OUP 2004) §2.2.3.
#Nodes E_a - E_b and weights |O_ab|^2 give Lanczos_from_spectrum, since O_n[a,b] = p_n(E_a - E_b) O_ab. The diagonal
#<omega p_n, p_n> vanishes for a symmetric measure and is not computed. The element type follows the inputs (BigFloat in,
#BigFloat chain). Returns b with b[1]=1 and length Nsteps, as Lanczos().
#Unvalidated
function jacobi_from_measure(nodes::AbstractVector,weights::AbstractVector,Nsteps::Integer)
    T=float(promote_type(eltype(nodes),eltype(weights)))
    omega=Vector{T}(nodes);w=Vector{T}(weights)
    b=ones(T,Nsteps)
    #p[1] = p_(n-2), p[2] = p_(n-1); start from p_(-1) = 0, p_0 = 1/||1||
    p=[zeros(T,length(omega)),fill(1/sqrt(sum(w)),length(omega))]
    for n in 2:1:Nsteps
        A_n=omega.*p[2] .- b[n-1].*p[1]
        b[n]=sqrt(sum(w.*A_n.^2))
        p[1]=p[2]
        p[2]=A_n./b[n]
    end
    return b
end

#Lanczos plus the level ratios of the same spectrum, from one diagonalization
function Lanczos_level_Ratios_ED(Probe::AbstractMatrix,H::AbstractMatrix,Nsteps::Int64;spectrum_flag="mid")

    H = Matrix(H)
    vals,vecs=eigen(Hermitian(H))
    Probe=vecs' * Matrix(Probe) * vecs

    #Base vector
    O = []
    b = Float64[1]
    #Define O0
    Probe/=Op_Norm(Probe)
    push!(O,Probe)
    LO_0 = L_n_ED(O[1],vals)
    #Define b1, b0 is set to 0
    push!(b,Op_Norm(LO_0))

    #Define O1
    push!(O,LO_0/b[2])

    for n in 3:Nsteps
        A_n = L_n_ED(O[2],vals) - b[n-1]*O[1]
        b_n = Op_Norm(A_n)
        push!(b,b_n)
        O[1]=O[2]
        O[2] = (A_n/b_n)
    end

    # Now compute level ratios
    ratios = Level_ratios(vals,spectrum_flag)

    return b,ratios
end

#Overlap of each Krylov operator with a set of eigenstate projectors |v><v|
function cn_calc(vecs,On)
    cns =ComplexF64[]
    for vec in vecs
        P_vec = vec*transpose(vec)
        push!(cns,Op_Inner(On,P_vec))
    end
    return cns
end

function vec_to_krylov(vec::Union{Array{Float64,1},Vector},On)

    P_vec = vec*transpose(vec)
    cn=Float64[]
    for n in 1:size(On)[1]
        push!(cn,tr(P_vec*On[n,:,:]) / length(vec))
    end
    return cn
end

#Lanczos which also tracks the overlap c_n of each Krylov operator with the eigenstate
#projectors it has the largest overlap with, and accumulates the real-space zero mode.
function Lanczos_cn_calc(Probe::AbstractMatrix,H::AbstractMatrix,Nsteps::Int64)

    Probe = Matrix(Probe)
    Probe/=Op_Norm(Probe)
    H = Matrix(H)
    N=size(Probe)[1]
    N_2 = Int(round(N/2))
    vecs=eigvecs(H)

    #Find eigenvalue that has the maximum overlap with the initial probe
    N_overlaps = N_2  # how many top overlaps you want to keep

    all_overlaps = Float64[]
    all_indices = Int[]
    all_vecs = Vector{Vector{Float64}}()

    # Step 1: collect everything
    for i in 1:N
        vec = vecs[:, i]
        P_vec = vec * transpose(vec)
        overlap = Op_Inner(P_vec, Probe)
        push!(all_overlaps, overlap)
        push!(all_indices, i)
        push!(all_vecs, vec)
    end

    # Step 2: get indices of top N_overlaps overlaps
    top_idx = partialsortperm(all_overlaps, 1:N_overlaps,rev=true)
    # Step 3: extract top N_overlaps results
    max_overlap   = all_overlaps[top_idx]
    max_overlap_i = all_indices[top_idx]
    overlap_vecs  = all_vecs[top_idx]

    vecs=nothing

    cn=complex(zeros(length(overlap_vecs),Nsteps+1))
    cn_test=complex(zeros(2,Nsteps+1))
    errors = zeros(2,Nsteps+1)

    test_vec1 = zeros(N);test_vec1[1]=1
    test_vec2 = ones(N);
    test_vecs=[test_vec1,test_vec2]
    #Base vector
    O = Vector{Matrix{Float64}}(undef, 2)
    O[1] = Probe
    b = Float64[1]
    ZM_real_space = zeros(N,N)

    #Define O0
    cn[:,1]=cn_calc(overlap_vecs,O[1])
    cn_test[:,1]=cn_calc(test_vecs,O[1])
    ZM_real_space+=LIOM_from_b(copy(b))[1]*O[1]

    LO_0 = L_n(O[1],H,2)
    #Define b1, b0 is set to 0
    push!(b,Op_Norm(LO_0))

    #Define O1
    O[2] = LO_0 / b[2]
    cn[:,2]=cn_calc(overlap_vecs,O[2])
    cn_test[:,2]=cn_calc(test_vecs,O[2])
    ZM_real_space+=LIOM_from_b(copy(b))[2]*O[2]

    for n in 3:Nsteps+1
        A_n = L_n(O[2],H,n) - b[n-1]*O[1]
        b_n = Op_Norm(A_n)
        O[1], O[2] = O[2], A_n/b_n
        push!(b,b_n)
        cn_test[:,n] = cn_calc(test_vecs,O[2])
        cn[:,n]=cn_calc(overlap_vecs,O[2])

        ZM_real_space+=LIOM_from_b(copy(b))[n]*O[2]

        error_0 = abs(Op_Inner(Probe,O[2]))
        error_1 = abs(Op_Inner(LO_0/b[2],O[2]))
        errors[:,n] = [error_0,error_1]
        if error_0 > 1E-4 || error_1 > 1E-4
            @show n,error_0, error_1
        end
    end

    return b,cn,ZM_real_space,overlap_vecs,max_overlap,errors
end

#As Lanczos_cn_calc, but running the recursion in the energy eigenbasis
function Lanczos_cn_calc_ED(Probe::AbstractMatrix,H::AbstractMatrix,Nsteps::Int64)

    H = Matrix(H)
    vals,e_vecs=eigen(Hermitian(H))
    Probe=e_vecs' * Matrix(Probe) * e_vecs

    Probe/=Op_Norm(Probe)
    N=size(Probe)[1]
    N_2 = Int(round(N/2))

    #Find eigenvalue that has the maximum overlap with the initial probe
    N_overlaps = N_2  # how many top overlaps you want to keep

    all_overlaps = Float64[]
    all_indices = Int[]
    all_vecs = Vector{Vector{Float64}}()

    # Step 1: collect everything
    for i in 1:N
        vec = e_vecs[:, i]
        P_vec = vec * transpose(vec)
        overlap = Op_Inner(P_vec, Probe)
        push!(all_overlaps, overlap)
        push!(all_indices, i)
        push!(all_vecs, vec)
    end

    # Step 2: get indices of top N_overlaps overlaps
    top_idx = partialsortperm(all_overlaps, 1:N_overlaps,rev=true)
    # Step 3: extract top N_overlaps results
    max_overlap   = all_overlaps[top_idx]
    max_overlap_i = all_indices[top_idx]
    overlap_vecs  = all_vecs[top_idx]

    cn=complex(zeros(length(overlap_vecs),Nsteps+1))
    cn_test=complex(zeros(2,Nsteps+1))
    errors = zeros(2,Nsteps+1)

    test_vec1 = zeros(N);test_vec1[1]=1
    test_vec2 = ones(N);
    test_vecs=[test_vec1,test_vec2]
    #Base vector
    O = Vector{Matrix{Float64}}(undef, 2)
    O[1] = Probe
    b = Float64[1]
    ZM_real_space = zeros(N,N)

    #Define O0
    cn[:,1]=cn_calc(overlap_vecs,O[1])
    cn_test[:,1]=cn_calc(test_vecs,O[1])
    ZM_real_space+=LIOM_from_b(copy(b))[1]*(e_vecs *O[1]* e_vecs')

    LO_0 = L_n_ED(O[1],vals)
    #Define b1, b0 is set to 0
    push!(b,Op_Norm(LO_0))

    #Define O1
    O[2] = LO_0 / b[2]
    cn[:,2]=cn_calc(overlap_vecs,O[2])
    cn_test[:,2]=cn_calc(test_vecs,O[2])
    ZM_real_space+=LIOM_from_b(copy(b))[2]*(e_vecs *O[2]* e_vecs')

    for n in 3:Nsteps+1
        A_n = L_n_ED(O[2],vals) - b[n-1]*O[1]
        b_n = Op_Norm(A_n)
        O[1], O[2] = O[2], A_n/b_n
        push!(b,b_n)

        cn_test[:,n] = cn_calc(test_vecs,O[2])
        cn[:,n]=cn_calc(overlap_vecs,O[2])

        ZM_real_space+=LIOM_from_b(copy(b))[n]*(e_vecs *O[2]* e_vecs')

        error_0 = abs(Op_Inner(Probe,O[2]))
        error_1 = abs(Op_Inner(LO_0/b[2],O[2]))
        errors[:,n] = [error_0,error_1]
        if error_0 > 1E-4 || error_1 > 1E-4
            @show n,error_0, error_1
        end
    end

    return b,cn,ZM_real_space,overlap_vecs,max_overlap,errors
end

########## Quantities derived from the Lanczos coefficients ##########

#Spectral function of the probe, as (frequencies, weights)
function Spectral_fct_from_bn(bn::AbstractVector; placeholder::Bool=has_placeholder(bn))
    L = L_matrix(bn; placeholder)
    vals,vecs = eigen(L)
    return (vals, 2pi .* vecs[1,:] .^2)
end

#Autocorrelation function C(t) = <O(t)|O(0)> reconstructed from the b_n
function Autocorr_fct_from_bn(bn::AbstractVector,tvec::AbstractVector; placeholder::Bool=has_placeholder(bn))
    L = L_matrix(bn; placeholder)
    C_t = [exp(im*Matrix(L)*t)[1,1] for t in tvec]
    return (tvec, C_t)
end

#Histogram a spectral function onto a uniform frequency grid
function bin_spectral_function(vals::AbstractVector, weights::AbstractVector, nbins::Int=100; range=nothing)
    lo, hi = range === nothing ? (minimum(vals), maximum(vals)) : range
    edges = collect(LinRange(lo, hi, nbins+1))
    dω = edges[2] - edges[1]

    density = zeros(nbins)
    for (v, w) in zip(vals, weights)
        idx = clamp(searchsortedlast(edges, v), 1, nbins)
        density[idx] += w
    end
    density ./= dω

    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    return centers, density
end

#Krylov-space wavefunction amplitude phi_i(t)
function K_phi_t(bn::AbstractVector, t_vec::AbstractVector, i::Int=1; placeholder::Bool=has_placeholder(bn))
    bn = bn_couplings(bn; placeholder)
    x0 = zeros(length(bn)+1); x0[1] = 1
    L = Tridiagonal(bn,zeros(length(bn)+1),(-1 .* bn))
    F = eigen(L)
    V = F.vectors
    y0 = V \ x0
    return [real((V * (exp.(F.values .* t) .* y0)))[i] for t in t_vec]
end

#Krylov complexity K(t) = sum_n n |phi_n(t)|^2
function K_complexity(bn::AbstractVector,t_vec::AbstractVector; placeholder::Bool=has_placeholder(bn))
    bn = bn_couplings(bn; placeholder)
    x0 = zeros(length(bn)+1); x0[1] = 1
    L = Tridiagonal(bn,zeros(length(bn)+1),(-1 .* bn))
    F = eigen(L)
    V = F.vectors
    y0 = V \ x0
    ns = collect(0:length(bn))

    K_t = Float64[]
    for t in t_vec
        phi = real.(V * (exp.(F.values .* t) .* y0))
        push!(K_t, sum(ns .* abs2.(phi)) / sum(abs2.(phi)))
    end
    return K_t
end

#Staggering of the b_n sequence, a diagnostic for a Krylov-space zero mode
function dimerization_from_b(b::AbstractVector; placeholder::Bool=has_placeholder(b))
    b = bn_with_placeholder(b; placeholder)
    b_even = b[2:2:end]; b_odd = b[3:2:end]
    m = min(length(b_even), length(b_odd))
    return b_odd[1:m] .- b_even[1:m]
end

function dimerization_ratio_from_b(b::AbstractVector; placeholder::Bool=has_placeholder(b))
    b = bn_with_placeholder(b; placeholder)
    b_even = b[2:2:end]; b_odd = b[3:2:end]
    m = min(length(b_even), length(b_odd))
    return b_odd[1:m] ./ b_even[1:m]
end

#Zero-mode amplitudes in Krylov space. Same quantity as ZeroMode_from_b, but this one
#also normalises the b[1] storage convention first.
function LIOM_from_b(bn; placeholder::Bool=has_placeholder(bn))
    #Taking care of different storage conventions (this no longer mutates its argument)
    bn = bn_with_placeholder(bn; placeholder)

    phi_n = zeros(length(bn))
    phi_n[1]=1 #phi_0
    for n in 3:2:length(bn)
        phi=1
        for i in 1:Int((n-1)/2)
            phi *=(bn[2*i]/bn[2*i+1]) #Because the bn array starts at b2, this ratio is flipped from analytical expression
        end
        phi_n[n]=phi #n+1 because of 0 start
    end
    return phi_n
end

#Zero mode obtained by diagonalizing the Krylov Liouvillian directly
function LIOM_from_zero_mode(bn::AbstractVector; placeholder::Bool=has_placeholder(bn))
    L = L_matrix(bn; placeholder)
    vals,vecs = eigen(L)
    return vecs[:,argmin(abs.(vals))]
end

#Log-averaged inverse participation ratio of the zero mode across disorder samples
function LIOM_avg_IPR(bn_tot::AbstractMatrix)
    Nsamples = size(bn_tot,2)
    IPRs = zeros(Nsamples); Norms = zeros(Nsamples)
    for j in 1:Nsamples
        LIOM = LIOM_from_b(bn_tot[:,j])
        IPRs[j] = sum(abs.(LIOM).^4)/sum(abs.(LIOM).^2)^2
        Norms[j] = sum(abs.(LIOM).^2)
    end
    return exp(mean(log.(IPRs))), exp(mean(log.(Norms)))
end

########## Moments <-> Lanczos coefficients ##########

#Even moments m_2n = <O|L^2n|O> of the Liouvillian
function moments_from_b(bn::AbstractVector; placeholder::Bool=has_placeholder(bn))
    bn = bn_couplings(bn; placeholder)
    L = Tridiagonal(bn,zeros(length(bn)+1), bn)
    return [(Matrix(L)^n)[1,1] for n in 1:2*length(bn)]
end

function M_mom_calc(bs,moments,two_k,m)

    #bs should start with index 1
    #Indexing starts at -1
    #moments should start with index 1 and include odd terms, even if set to zero

    if m==-1
        M_m_2k = 0
    elseif m==0
        M_m_2k = moments[two_k]
    else
        M_m_2k = (M_mom_calc(bs,moments,two_k,m-1)/bs[m-1+2]^2)-(M_mom_calc(bs,moments,two_k-2,m-2)/bs[m-2+2]^2)
    end
    return M_m_2k
end

#Direct (unmemoised) inverse map, kept for reference; b_from_moments below is the same
#recursion with caching and a positivity check.
function b_from_moments_old(m_vec,K=length(m_vec[2:2:end]))
    bs = Float64[1,1]
    for n in 1:K
        M = M_mom_calc(bs,m_vec,2n,n)
        if M<0
            error("M=$M <0 at n=$n")
        end
        push!(bs,sqrt(M))
    end

    return bs[3:end]
end

#Inverse map, by the standard Hankel recursion. Memoized, and it errors out
#rather than returning nonsense once floating-point noise breaks positivity.
function b_from_moments(m_vec::AbstractVector{T}, K::Int=length(m_vec)÷2; m0::T=one(T)) where {T}
    bs = T[one(T), sqrt(m0)]
    cache = Dict{Tuple{Int,Int}, T}()

    function M(two_k::Int, m::Int)
        m == -1 && return zero(T)
        m ==  0 && return m_vec[two_k]
        haskey(cache, (two_k, m)) && return cache[(two_k, m)]
        val = M(two_k, m-1)/bs[m+1]^2 - M(two_k-2, m-2)/bs[m]^2
        cache[(two_k, m)] = val
        return val
    end

    for n in 1:K
        Mnn = M(2n, n)
        Mnn < 0 && error("M=$Mnn < 0 at n=$n (Hankel positivity lost -- precision issue)")
        push!(bs, sqrt(Mnn))
    end
    return bs[3:end]
end
