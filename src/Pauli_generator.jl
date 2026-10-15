#Code to take in strings and generate pauli-based Hamiltonians
using SparseArrays, LinearAlgebra

#Unvalidated
function pauli_expand(paulistr::Union{Vector{Int64},Vector{Float64}},flag::String)
    I=spdiagm([1,1])
    local X
    local Y
    local Z
    local P
    local M
    if flag=="spin"
        X=Complex.(sparse([2,1],[1,2],[1/2,1/2]))
        Y=Complex.(sparse([2,1],[1,2],[im/2,-im/2]))
        Z=Complex.(spdiagm([1/2,-1/2]))
        P = X+im*Y
        M = X-im*Y
    elseif flag=="pauli"
        X=Complex.(sparse([2,1],[1,2],[1,1]))
        Y=Complex.(sparse([2,1],[1,2],[im,-im]))
        Z=Complex.(spdiagm([1,-1]))
        #sigma+/sigma- kept normalized (PM = diag(1,0)) even though X,Y,Z are not:
        #standard ladder-operator convention, {P,M}=I, regardless of the X,Y,Z scale.
        P = (X+im*Y)/2
        M = (X-im*Y)/2
    else
        error("invalid spinflag \"$flag\" (options: \"spin\", \"pauli\")")
    end
    N = length(paulistr)
    BigOp=0
    for i in 1:1:N
        n=paulistr[i]
        if n==0
            Op=I
        elseif n==1
            Op=X
        elseif n==2
            Op=Y
        elseif n==3
            Op=Z
        elseif n==4
            Op=P
        elseif n==5
            Op=M
        end

        if i==1
            BigOp=copy(Op)
        else
            BigOp = kron(Op,BigOp)
        end
    end

    #if real(BigOp)==BigOp
    #    BigOp = real(BigOp)
    #end

    return BigOp
end

function Op_create(N::Int64,Op::String,site::Int64,flag::String)
    str = zeros(N)
    if Op == "X"
        str[site]=1
    elseif Op == "Y"
        str[site]=2
    elseif Op == "Z"
        str[site]=3
    elseif Op =="+"
        str[site]=4
    elseif Op == "-"
        str[site]=5
    else
        error("invalid operator \"$Op\" (options: \"X\", \"Y\", \"Z\", \"+\", \"-\")")
    end
    return pauli_expand(str,flag)
end



#Site 1 is the *innermost* kron factor here, matching the ordering built by
#pauli_expand. Keeping the two in step is what makes <Product_state|Op_create(N,"Z",s,..)|Product_state>
#actually report site s rather than site N+1-s.
function Product_state(N::Int64,state_flag::String)
    up=[1, 0]
    dn=[0, 1]
    local state
    if state_flag=="Neel"
        state=copy(up)
        for i in 2:N
            if iseven(i)==true
                state=kron(dn,state)
            else
                state=kron(up,state)
            end
        end
    elseif state_flag=="Up"
        state=copy(up)
        for i in 2:N
            state=kron(up,state)
        end
    elseif state_flag=="Dn" || state_flag=="Down"
        state=copy(dn)
        for i in 2:N
            state=kron(dn,state)
        end
    elseif state_flag=="Rand"
        #A random *product* state: each site independently up or down
        state = rand(Bool) ? copy(up) : copy(dn)
        for i in 2:N
            state = kron(rand(Bool) ? up : dn, state)
        end
    elseif state_flag=="RandVec"
        #A Haar-random state in the full Hilbert space (not a product state)
        state = randn(ComplexF64,2^N)
    else
        error("invalid state_flag \"$state_flag\" (options: \"Neel\", \"Up\", \"Dn\"/\"Down\", \"Rand\", \"RandVec\")")
    end
    return state./norm(state) #normalize
end

#Multi-site operator strings, e.g. op_create(6,[("X",1),("Z",3),("+",5)])
#Sites not listed are left as the identity.
function op_create(N::Int64,Ops::Vector{<:Tuple{String,Integer}};flag::String="pauli")
    str = zeros(N)
    for (Op,site) in Ops
        (1 <= site <= N) || error("site $site out of range 1:$N")
        if Op == "X"
            str[site]=1
        elseif Op == "Y"
            str[site]=2
        elseif Op == "Z"
            str[site]=3
        elseif Op == "+"
            str[site]=4
        elseif Op == "-"
            str[site]=5
        else
            error("invalid operator \"$Op\" (options: \"X\", \"Y\", \"Z\", \"+\", \"-\")")
        end
    end
    return pauli_expand(str,flag)
end

#Density matrix versions of Product_state
#Unvalidated
function Product_densm(N::Int64,state_flag::String)
    if state_flag=="RandMixed"
        Q = randn(ComplexF64,2^N,2^N)
        return (Q*Q')/tr(Q*Q')
    elseif state_flag=="InfTemp"
        return Matrix{ComplexF64}(I,2^N,2^N)/2^N
    end
    psi = Product_state(N,state_flag)
    return psi*psi'
end

#Interpret a bit array as an integer (site 1 = least significant, matching pauli_expand)
bitarr_to_int(arr::AbstractVector) = sum(b << (i-1) for (i,b) in enumerate(arr))

#Site-resolved expectation values. op is "X", "Y", "Z" or "J" (the current between site
#i and i+1), or any matrix operator.
function state_measure(state::AbstractVector,op,sites,spinflag::String)
    N = Int(log2(size(state)[1]))
    meas_arr = []
    if typeof(op)==String
        for i in sites
            str = zeros(N)
            if op == "X"
                str[i]=1
                A = pauli_expand(str,spinflag)
                push!(meas_arr,state'*A*state)
            elseif op == "Y"
                str[i]=2
                A = pauli_expand(str,spinflag)
                push!(meas_arr,state'*A*state)
            elseif op == "Z"
                str[i]=3
                A = pauli_expand(str,spinflag)
                push!(meas_arr,state'*A*state)
            elseif op == "J"
                str[i]=1
                X1 = pauli_expand(str,spinflag)
                str = zeros(N)
                str[i+1]=1
                X2 = pauli_expand(str,spinflag)
                str = zeros(N)
                str[i]=2
                Y1 = pauli_expand(str,spinflag)
                str = zeros(N)
                str[i+1]=2
                Y2 = pauli_expand(str,spinflag)
                J = -2 .*(Y1*X2-X1*Y2)
                push!(meas_arr,state'*J*state)
            else
                error("Incorrect Op string. Try inputting matrix operator")
            end
        end
        return real.(meas_arr)
    else
        return real(state'*op*state)
    end

end

#The operators state_measure builds internally, returned as a list so they are built ONCE rather
#than rebuilt from the op string at every call. op is "X", "Y" or "Z" (one operator per site) or
#"J" (the current between site i and i+1, so one per BOND -- L-1 of them, not L, which is why the
#index range defaults per op instead of being left to the caller). Everything here is Hermitian,
#so the list drops straight into the Ops keyword of Jump_TE/Liouvillian_TE_obs; for ladder
#operators or multi-site strings, build the list out of op_create directly.
#Unvalidated
function site_ops(L::Integer,op::String,sites=(op=="J" ? (1:1:L-1) : (1:1:L));flag::String="pauli")
    op in ("X","Y","Z","J") || error("invalid op \"$op\" (options: \"X\", \"Y\", \"Z\", \"J\")")
    N = Int(L)
    #"J" indexes BONDS, not sites: bond i joins sites i and i+1, so i=L has no partner. Caught
    #here rather than left to op_create, which would report it as an out-of-range SITE L+1 and
    #so hide the very off-by-one this helper exists to remove from the caller.
    op=="J" && !all(1 .<= sites .< N) && error("bonds $sites out of range 1:$(N-1) for op \"J\"")
    #Assembled exactly as state_measure assembles it -- same -2 prefactor, same operand order,
    #same flag -- so the two agree term by term and not merely up to sign or normalization.
    bond(i) = -2 .*(op_create(N,[("Y",i)];flag=flag)*op_create(N,[("X",i+1)];flag=flag) -
                    op_create(N,[("X",i)];flag=flag)*op_create(N,[("Y",i+1)];flag=flag))
    return op=="J" ? [bond(i) for i in sites] : [op_create(N,[(op,i)];flag=flag) for i in sites]
end


function DM_measure(rho::AbstractMatrix,flag,sites,spinflag::String)
    N = Int(log2(size(rho)[1]))
    meas_arr = []
    for i in sites
        str = zeros(N)
        if flag == "X"
            str[i]=1
            A = pauli_expand(str,spinflag)
            push!(meas_arr,tr(rho*A))
        elseif flag == "Y"
            str[i]=2
            A = pauli_expand(str,spinflag)
            push!(meas_arr,tr(rho*A))
        elseif flag == "Z"
            str[i]=3
            A = pauli_expand(str,spinflag)
            push!(meas_arr,tr(rho*A))
        elseif flag == "J"
            str[i]=1
            X1 = pauli_expand(str,spinflag)
            str = zeros(N)
            str[i+1]=1
            X2 = pauli_expand(str,spinflag)
            str = zeros(N)
            str[i]=2
            Y1 = pauli_expand(str,spinflag)
            str = zeros(N)
            str[i+1]=2
            Y2 = pauli_expand(str,spinflag)
            J = -2 .* (Y1*X2-X1*Y2)
            push!(meas_arr,tr(rho*J))
        else
            error("Incorrect Op string (options: \"X\", \"Y\", \"Z\", \"J\")")
        end
    end
    return meas_arr
end

#Generate simple unitary
U_t(t::Real,H::AbstractMatrix) = exp(-im .* Matrix(H) .* t)

# Simple commutator and anti-commutator functions
comm(A::AbstractArray,B::AbstractArray) = A*B - B*A
acomm(A::AbstractArray,B::AbstractArray) = A*B + B*A



