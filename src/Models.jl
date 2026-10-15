using SparseArrays, LinearAlgebra, Random

#Models.jl is built on top of pauli_expand, so pull it in if it isn't already loaded
isdefined(@__MODULE__, :pauli_expand) || include(joinpath(@__DIR__,"Pauli_generator.jl"))


######## Random GOE matrix ##########
function H_GOE(N::Int64)
    R_ = randn(2^N,2^N)
    R = (1/sqrt(2)) .* (R_ + transpose(R_))
    return R
end

######## Random matrix of all 2 body pauli operators ######
#The same operator as Rand_XYZ further down, term for term; the factor is the one that
#makes the bandwidth scaling equal to GOE.
Random_XYZ(N::Int64) = Rand_XYZ(N) .* (2^N / ((N/2)*(3N+1)))


########## General Heisenberg/ XYZ-spin chain ###########
function H_XYZ(N::Int64,J_vec::AbstractVector,X_vec::AbstractVector,Y_vec::AbstractVector,Z_vec::AbstractVector;periodic::Bool=true,spinflag::String="pauli")
    #Hopping strengths are J_vec=[Jx,Jy,Jz]
    #X,Y,Z Field strengths are X_vec,Y_vec,Z_vec respectively
    
    Jx=J_vec[1]
    Jy=J_vec[2]
    Jz=J_vec[3]
    Big_Op=spzeros(2^N,2^N)
    
    for i in 1:N-1
        str = zeros(N)
        str[i]=1;str[i+1]=1 #X_i X_i+1
        Big_Op += Jx .* pauli_expand(str,spinflag)


        str = zeros(N)
        str[i]=2;str[i+1]=2 #Y_i Y_i+1
        Big_Op += Jy .* pauli_expand(str,spinflag)

        str = zeros(N)
        str[i]=3;str[i+1]=3 #Z_i Z_i+1
        Big_Op += Jz .* pauli_expand(str,spinflag)
    end

    if periodic==true
        str = zeros(N)
        str[N]=1;str[1]=1 #X_i X_i+1
        Big_Op += Jx .* pauli_expand(str,spinflag)

        str = zeros(N)
        str[N]=2;str[1]=2 #Y_i Y_i+1
        Big_Op += Jy .* pauli_expand(str,spinflag)

        str = zeros(N)
        str[N]=3;str[1]=3 #Z_i Z_i+1
        Big_Op += Jz .* pauli_expand(str,spinflag)
    end
    
    
    for i in 1:N
        str = zeros(N)
        str[i]=1;#X_i
        Big_Op += X_vec[i] .* pauli_expand(str,spinflag)
    end

    for i in 1:N
    
        str = zeros(N)
        str[i]=2;#Y_i
        Big_Op += Y_vec[i] .* pauli_expand(str,spinflag)
    end
    
    for i in 1:N
        str = zeros(N)
        str[i]=3;#Z_i
        Big_Op += Z_vec[i] .* pauli_expand(str,spinflag)
    end
    
    return Big_Op
end

######### Disordered Heisenberg (model for MBL) #########
H_Disordered_Heisenberg(N::Int64,J::Real,W::Real) = H_XYZ(N,J*ones(3),zeros(N),zeros(N),W .* (rand(N).*2 .-1))

######### A chaotic Ising chain #########
function H_Quantum_Ising(N::Int64,X_field::Real,Z_field::Real=0.0;periodic::Bool=true,spinflag::String="pauli")

    Big_Op = spzeros(2^N,2^N)

    for i in 1:N-1
        str=zeros(N)
        str[i]=3 #Z_i
        str[i+1]=3 #Z_i+1
        Big_Op += pauli_expand(str,spinflag)
    end

    #add periodic BCs -- same ZZ bond as the bulk, wrapping site N back to site 1
    if periodic==true
    str=zeros(N)
    str[1]=3
    str[N]=3
    Big_Op += pauli_expand(str,spinflag)
    end

    for i in 1:N
        str=zeros(N)
        str[i] = 1 #X_i
        Big_Op += X_field * pauli_expand(str,spinflag)

        str=zeros(N)
        str[i] = 3 #Z_i
        Big_Op += Z_field * pauli_expand(str,spinflag)
    end

    return Big_Op
end

######## Various Anderson Models ###########
function H_Anderson1D(N::Integer,W::Real,t::Real;h::Union{Nothing, AbstractVector}=nothing,periodic::Bool=true)
    
    #If an array of disorder is not given, one will be provided
    if isnothing(h)
        h = (W/2) .* (rand(N).*2 .-1) #random potentials
    end

    #Default is sparse format
    H_Anderson = spdiagm(-1 => (-t)*ones(N-1), 1 => (-t)*ones(N-1), 0=>h)

    #If periodic boundary conditions (default), connect the boundaries
    if periodic
    H_Anderson[1,N]=-t
    H_Anderson[N,1]=-t
    end
    return H_Anderson
end

function H_Anderson2D(N::Integer,W::Real,t::Real;h::Union{Nothing, AbstractVector}=nothing,periodic::Bool=true)

    H = spzeros(N^2,N^2)
    Id = spdiagm(ones(N))
        
        Big_Op = H_Anderson1D(N,0,t;periodic)
        Big_Op = kron(Big_Op,Id)
        H += Big_Op

        Big_Op = Id
        Big_Op = kron(Big_Op,H_Anderson1D(N,0,t;periodic))
        H += Big_Op
        
    if isnothing(h)
        h = (W/2) .* (rand(N^2).*2 .-1) #random potentials
    end
    H += spdiagm(h)
    return H
end

function H_Anderson3D(N::Integer,W::Real,t::Real;h::Union{Nothing, AbstractVector}=nothing,periodic::Bool=true)

    H = spzeros(N^3,N^3)
    Id = spdiagm(ones(N))
        Big_Op = H_Anderson1D(N,0,t;periodic)
        Big_Op = kron(Big_Op,Id)
        Big_Op = kron(Big_Op,Id)
        H += Big_Op

        Big_Op = Id
        Big_Op = kron(Big_Op,H_Anderson1D(N,0,t;periodic))
        Big_Op = kron(Big_Op,Id)
        H += Big_Op

        Big_Op = Id
        Big_Op = kron(Big_Op,Id)
        Big_Op = kron(Big_Op,H_Anderson1D(N,0,t;periodic))
        H += Big_Op

    if isnothing(h)
        h = (W/2) .* (rand(N^3).*2 .-1) #random potentials
    end
    H += spdiagm(h)
    return H
end

function H_Anderson4D(N::Integer,W::Real,t::Real;h::Union{Nothing, AbstractVector}=nothing,periodic::Bool=true)

    H = spzeros(N^4,N^4)
    Id = spdiagm(ones(N))
    
        Big_Op = H_Anderson1D(N,0,t;periodic)
        Big_Op = kron(Big_Op,Id)
        Big_Op = kron(Big_Op,Id)
        Big_Op = kron(Big_Op,Id)
        H += Big_Op

        Big_Op = Id
        Big_Op = kron(Big_Op,H_Anderson1D(N,0,t;periodic))
        Big_Op = kron(Big_Op,Id)
        Big_Op = kron(Big_Op,Id)
        H += Big_Op

        Big_Op = Id
        Big_Op = kron(Big_Op,Id)
        Big_Op = kron(Big_Op,H_Anderson1D(N,0,t;periodic))
        Big_Op = kron(Big_Op,Id)
        H += Big_Op

        Big_Op = Id
        Big_Op = kron(Big_Op,Id)
        Big_Op = kron(Big_Op,Id)
        Big_Op = kron(Big_Op,H_Anderson1D(N,0,t;periodic))
        H += Big_Op

    if isnothing(h)
        h = (W/2) .* (rand(N^4).*2 .-1) #random potentials
    end
    H += spdiagm(h)
    return H
end

function H_Anderson(d::Integer,N::Integer,W::Real,t::Real;h::Union{Nothing, AbstractVector}=nothing,periodic::Bool=true)

    if d==1
       return H_Anderson1D(N,W,t;h,periodic)
    elseif d==2
        return H_Anderson2D(N,W,t;h,periodic)
    elseif d==3
        return H_Anderson3D(N,W,t;h,periodic)
    elseif d==4
        return H_Anderson4D(N,W,t;h,periodic)
    else
        error("Incorrect dimension (only d=1-4 are currently included)")
    end
end

function Z_wave_op(N::Int64,k::Real)
    O = spzeros(2^N,2^N)
    for i in 1:N
        O += cos(2pi*k*i/N) .* op_create(N,[("Z",i)])
    end
    return O
end

######## Random hopping chain (off-diagonal disorder) ########
function H_Random_Hop1D(L::Integer,W::Real,t::Real;periodic::Bool=true)
    h1 = (W/2) .* (rand(L-1).*2 .-1) #random hoppings
    H = spdiagm(-1 => (-t)*h1, 1 => (-t)*h1, 0=>zeros(L))
    if periodic
        H[1,L] = -(W/2) * (rand()*2 -1)
        H[L,1] = H[1,L]
    end
    return H
end

######## Aubry-Andre quasiperiodic chain ########
#L is rounded to the nearest Fibonacci number f_n so that alpha = f_{n+1}/f_n is a
#rational approximant to the golden ratio; the actual size used is returned alongside H.
function Aubry_Andre_H(L::Integer,W::Real;t::Real=1,t2::Union{Nothing,Real}=nothing)
    a, b = 0, 1
    while b < L
        a, b = b, a + b
    end
    f_n,f_n_p_1 = abs(L - a) <= abs(L - b) ? (a, b) : (b, a + b)

    alpha = f_n_p_1/f_n
    h = W .* cos.((2pi* alpha) .* collect(1:f_n)) #quasiperiodic potential

    H_AA = spdiagm(-1 => (-t)*ones(f_n-1), 1 => (-t)*ones(f_n-1), 0=>h)
    H_AA[1,f_n]=-t
    H_AA[f_n,1]=-t

    #optional next-nearest-neighbour hopping
    if !isnothing(t2)
        H_AA += spdiagm(-2 => (-t2)*ones(f_n-2), 2 => (-t2)*ones(f_n-2))
        H_AA[2,f_n]=-t2
        H_AA[f_n,2]=-t2
    end
    return H_AA,f_n
end

Aubry_Andre_nnn_H(L::Integer,W::Real;t::Real=1,t2::Real=1) = Aubry_Andre_H(L,W;t=t,t2=t2)

######## Many-body Anderson / XX chain with Z disorder ########
function H_mb_Anderson1D(N::Integer,t::Real,h::AbstractVector;spinflag::String="pauli")
    H = spzeros(2^N,2^N)
    for i in 0:N-1
        str=zeros(N); str[i+1]=1 ; str[(i+1)%N+1]=1 #X_i X_i+1
        H += -t .* pauli_expand(str,spinflag)

        str=zeros(N); str[i+1]=2 ; str[(i+1)%N+1]=2 #Y_i Y_i+1
        H += -t .* pauli_expand(str,spinflag)

        str=zeros(N); str[i+1]=3 #Z_i
        H += h[i+1] .* pauli_expand(str,spinflag)
    end
    return H
end

######## XXZ and next-nearest-neighbour XXZ ########
H_XXZ(N::Integer,J::Real,Δ::Real;spinflag::String="pauli") =
    H_XYZ(N,[J,J,Δ],zeros(N),zeros(N),zeros(N);periodic=true,spinflag=spinflag)

function XXZnnn_H(N::Integer;Δ::Real=2,γ::Real=1/2,spinflag::String="pauli")
    H = spzeros(2^N,2^N)
    for i in 0:N-1
        for (a,J1,J2) in ((1,1.0,γ),(2,1.0,γ),(3,Δ,Δ*γ))
            str=zeros(N); str[i+1]=a ; str[(i+1)%N+1]=a #nearest neighbour
            H += J1 .* pauli_expand(str,spinflag)

            str=zeros(N); str[i+1]=a ; str[(i+2)%N+1]=a #next-nearest neighbour
            H += J2 .* pauli_expand(str,spinflag)
        end
    end
    return H
end

######## Random-coupling Heisenberg chain ########
function Rand_Heisenberg(N::Integer;spinflag::String="spin")
    Big_Op=spzeros(2^N,2^N)
    for i in 0:N-1
        for a in 1:3
            str = zeros(N); str[i+1]=a; str[(i+1)%N+1]=a
            Big_Op += randn() .* pauli_expand(str,spinflag)
        end
        #X and Z fields only: a random Y field would make H complex
        for a in (1,3)
            str = zeros(N); str[i+1]=a
            Big_Op += randn() .* pauli_expand(str,spinflag)
        end
    end
    return real(Big_Op)
end

#=============== Quantum Sun (avalanche) model family ===============#
#A chaotic "dot" of N spins coupled to L outer spins, with coupling strength
#decaying as alpha^u_i along the outer chain. Used as a minimal model for the
#avalanche instability of many-body localization.

GOE(N::Integer) = H_GOE(N) #alias: the dot is just a GOE matrix

#Shared engine. core is the already-normalized 2^N dot; h_i and u_i are the
#per-outer-spin Z-field and coupling exponent.
function _quantum_sun_build(N::Integer,L::Integer,alpha::Real,core::AbstractMatrix,
                            h_i::AbstractVector,u_i::AbstractVector,
                            connections::AbstractVector,spinflag::String)
    Big_Op = core
    Id = spdiagm([1,1])
    for i in 1:L
        Big_Op = kron(Id,Big_Op) #lift the dot into the full 2^(N+L) space
    end
    core_full = deepcopy(Big_Op)

    for i in 1:L
        n_i = connections[i] #dot site this outer spin attaches to

        str = zeros(N+L)
        str[n_i] = 1 #X on dot site n_i
        str[i+N] = 1 #X on outer site i
        X_iX_n_i = pauli_expand(str,spinflag)

        str = zeros(N+L)
        str[i+N] = 3 #Z on outer site i
        Z_i = pauli_expand(str,spinflag)

        Big_Op += real((alpha^u_i[i]) .* X_iX_n_i + h_i[i] .* Z_i)
    end

    str = zeros(N+L)
    str[L+N] = 3 #Z on the outermost spin, the usual probe operator
    furthest_spin = pauli_expand(str,spinflag)

    return Big_Op, core_full, furthest_spin
end

function _quantum_sun_return(H,core,furthest_spin,core_return,furthest_spin_return)
    if core_return
        return H,core
    elseif furthest_spin_return
        return H,furthest_spin
    else
        return H
    end
end

function QuantumSunH(N::Integer,L::Integer,alpha::Real;core_function=GOE,spinflag::String="spin",
                     h_L::Union{AbstractVector,Nothing}=nothing,connections::Union{Vector{Int},Nothing}=nothing,
                     u_L::Union{AbstractVector,Nothing}=nothing,seed::Integer=rand(Int),
                     core_return=false,furthest_spin_return=false,no_random_final_coupling=false)
    Random.seed!(seed)
    h_L = h_L === nothing ? rand(L) : h_L
    connections = connections === nothing ? [rand(1:N) for _ in 1:L] : connections
    u_L = u_L === nothing ? rand(L) : u_L

    h_i = (1.5-0.5) .* h_L .+ 0.5 #random fields iid in [0.5,1.5]
    xi_i = 0.2                    #jitter of the coupling exponent about u_i = i

    u_i = [((i+xi_i)-(i-xi_i))*u_L[i] + (i-xi_i) for i in 1:L]
    if no_random_final_coupling
        u_i[L] = L
    end

    core = core_function(N) .* (1/sqrt(2^N + 1))
    H,core_full,furthest = _quantum_sun_build(N,L,alpha,core,h_i,u_i,connections,spinflag)
    return _quantum_sun_return(H,core_full,furthest,core_return,furthest_spin_return)
end

#Outer-chain Z-field disorder removed (h_i -> h_fixed) and the coupling exponent
#pinned to u_i = i. The dot and the random dot-site connections are unchanged.
function QuantumSunH_norand(N::Integer,L::Integer,alpha::Real;core_function=GOE,spinflag::String="spin",
                            h_fixed::Real=1.0,connections::Union{Vector{Int},Nothing}=nothing,
                            seed::Integer=rand(Int),core_return=false,furthest_spin_return=false)
    Random.seed!(seed)
    connections = connections === nothing ? [rand(1:N) for _ in 1:L] : connections

    core = core_function(N) .* (1/sqrt(2^N + 1))
    H,core_full,furthest = _quantum_sun_build(N,L,alpha,core,fill(h_fixed,L),collect(1:L),connections,spinflag)
    return _quantum_sun_return(H,core_full,furthest,core_return,furthest_spin_return)
end

#Coupling exponent pinned to u_i = i, but the Z-field disorder h_i is kept.
function QuantumSunH_fixedpower(N::Integer,L::Integer,alpha::Real;core_function=GOE,spinflag::String="spin",
                                h_L::Union{AbstractVector,Nothing}=nothing,connections::Union{Vector{Int},Nothing}=nothing,
                                seed::Integer=rand(Int),core_return=false,furthest_spin_return=false)
    Random.seed!(seed)
    h_L = h_L === nothing ? rand(L) : h_L
    connections = connections === nothing ? [rand(1:N) for _ in 1:L] : connections

    h_i = (1.5-0.5) .* h_L .+ 0.5
    core = core_function(N) .* (1/sqrt(2^N + 1))
    H,core_full,furthest = _quantum_sun_build(N,L,alpha,core,h_i,collect(1:L),connections,spinflag)
    return _quantum_sun_return(H,core_full,furthest,core_return,furthest_spin_return)
end

#Dot replaced by a random all-to-all 1- and 2-body XYZ model instead of a GOE matrix.
#The 1/sqrt(N(3N+1)/2) normalization matches its Frobenius spread to the GOE dot.
function QuantumSunH_randXYZ(N::Integer,L::Integer,alpha::Real;spinflag::String="spin",
                             h_L::Union{AbstractVector,Nothing}=nothing,connections::Union{Vector{Int},Nothing}=nothing,
                             u_L::Union{AbstractVector,Nothing}=nothing,seed::Integer=rand(Int),
                             core_return=false,furthest_spin_return=false)
    Random.seed!(seed)
    h_L = h_L === nothing ? rand(L) : h_L
    connections = connections === nothing ? [rand(1:N) for _ in 1:L] : connections
    u_L = u_L === nothing ? rand(L) : u_L

    h_i = (1.5-0.5) .* h_L .+ 0.5
    xi_i = 0.2
    u_i = [((i+xi_i)-(i-xi_i))*u_L[i] + (i-xi_i) for i in 1:L]

    core = Rand_XYZ(N) ./ sqrt(N*(3N+1)/2)
    H,core_full,furthest = _quantum_sun_build(N,L,alpha,core,h_i,u_i,connections,spinflag)
    return _quantum_sun_return(H,core_full,furthest,core_return,furthest_spin_return)
end

#Random all-to-all 1- and 2-body XYZ model, unnormalized (used as a Quantum Sun dot)
function Rand_XYZ(N::Integer;spinflag::String="pauli")
    Big_Op=spzeros(2^N,2^N)
    for i in 1:N
        for j in i+1:N
            for a in 1:3
                str = zeros(N); str[i]=a; str[j]=a
                Big_Op += randn() .* pauli_expand(str,spinflag)
            end
        end
        #X and Z fields only: a random Y field would make H complex
        for a in (1,3)
            str = zeros(N); str[i]=a
            Big_Op += randn() .* pauli_expand(str,spinflag)
        end
    end
    return real(Big_Op)
end

######## Ultrametric model ########
#Hierarchy of GOE blocks with strengths decaying as alpha^k. Returns (H, dot).
function UltraMetric_H(N::Integer,L::Integer,alpha::Real,J::Real)
    gamma=1
    R = H_GOE(N)
    Big_Op = (gamma .* R) ./ (sqrt((2^N) + 1))
    Id = diagm([1,1])

    for i in 1:L
        Big_Op = kron(Id,Big_Op) #lift the dot into the full 2^(N+L) space
    end
    #NB: this local must not be named GOE, or it would shadow the GOE() function below
    core = copy(Big_Op)

    for k in 1:L
        Hk = zeros(2^(N+L),2^(N+L))
        for i in 1:2^(L-k)
            P = zeros(2^(L-k),2^(L-k))
            P[i,i]=1 #projector selecting the i-th block on this level
            Hk += kron(P,H_GOE(N+k) ./ (sqrt((2^(N+k)) + 1)))
        end
        Big_Op += J*(alpha^k) .* Hk
    end

    return Big_Op, core
end

######## All-to-all raising/lowering coupling ########
#Unvalidated
function Two_Level_H(N::Integer,J::AbstractMatrix;spinflag::String="pauli")
    Big_Op = spzeros(ComplexF64,2^N,2^N)
    for n in 1:N, m in 1:N
        Big_Op += J[n,m] .* (op_create(N,[("+",n)];flag=spinflag) * op_create(N,[("-",m)];flag=spinflag))
    end
    return Big_Op
end
