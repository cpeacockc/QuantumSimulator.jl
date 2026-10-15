using LinearAlgebra, SparseArrays

#eigen() has no sparse method, so densify first. Hermitian() then guarantees real
#eigenvalues and picks up LAPACK's symmetric solver (roughly 2x faster).
_dense_herm(H::AbstractMatrix) = Hermitian(Matrix(H))

# Function for diagonalizing H and time evolving a state, measuring ops
function Diag_TE_measure(psi::AbstractVector, H::AbstractMatrix, t_array::AbstractVector{<:Real}, Ops::AbstractArray)
    eigH=eigen(_dense_herm(H))
    Measure_t_array=Matrix{ComplexF64}(undef, length(t_array), length(Ops))

    c_n = eigH.vectors' * psi  # Project psi onto eigenbasis

    for (ti,t) in (enumerate(t_array))
        phase_factors = exp.(-im * t .* eigH.values)
        psi_t = eigH.vectors * (phase_factors .* c_n)

        Measure_t_array[ti,:] = ([(psi_t'*Ops[i]*psi_t) for i in eachindex(Ops)])

    end
    return Measure_t_array
end

#Function for time evolving a wave function from diagonalizing H
function Diag_TE(psi::AbstractVector, H::AbstractMatrix, t_array::AbstractVector{<:Real})
    eigH = eigen(_dense_herm(H))
    c_n = eigH.vectors' * psi  # Project psi onto eigenbasis
    psi_t_array = Matrix{ComplexF64}(undef, length(t_array), length(psi))

    for (ti, t) in (enumerate(t_array))
        phase_factors = exp.(-im * t .* eigH.values)
        psi_t = eigH.vectors * (phase_factors .* c_n)
        psi_t_array[ti, :] = psi_t
    end

    return psi_t_array
end

#As Diag_TE, but returns a Vector of state vectors rather than a matrix
function state_evolve(H::AbstractMatrix, psi0::AbstractVector, t_vec::AbstractVector{<:Real})
    E,V = eigen(_dense_herm(H))
    c0 = V' * psi0
    return [V * (exp.(-im * E * t) .* c0) for t in t_vec]
end

#Heisenberg-picture evolution of an operator, reusing an existing diagonalization.
#Returns a Vector of O(t) = U'(t) O U(t).
function Diag_TE(O::AbstractMatrix, H_vecs::AbstractMatrix, H_vals::AbstractVector, t_array::AbstractVector{<:Real})
    O_eig = H_vecs' * Matrix(O) * H_vecs
    O_t = Vector{Matrix{ComplexF64}}(undef,length(t_array))
    for (ti,t) in enumerate(t_array)
        phase = exp.(im * t .* H_vals)
        #(U'OU)_nm = e^{+iE_n t} O_nm e^{-iE_m t}; phase' supplies the conjugated row factor
        O_t[ti] = H_vecs * (phase .* O_eig .* phase') * H_vecs'
    end
    return O_t
end

#Time evolution by direct matrix exponential rather than diagonalization.
#Cheaper when only a handful of times are needed, or when H is not Hermitian.
function exp_TE_measure(psi::AbstractVector, H::AbstractMatrix, t_array::AbstractVector{<:Real}, Ops::AbstractArray)
    Hd = Matrix(H)
    Measure_t_array = Matrix{ComplexF64}(undef, length(t_array), length(Ops))
    for (ti,t) in enumerate(t_array)
        psi_t = exp(-im .* Hd .* t) * psi
        Measure_t_array[ti,:] = [(psi_t'*Ops[i]*psi_t) for i in eachindex(Ops)]
    end
    return Measure_t_array
end
