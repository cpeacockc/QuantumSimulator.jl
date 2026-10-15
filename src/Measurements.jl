#Diagnostics for quantum chaos and thermalization: level statistics, spectral
#functions, correlation functions, and the spectral form factor.
using LinearAlgebra, SparseArrays, Statistics

#analyze_ED calls into the Lanczos routines
isdefined(@__MODULE__, :lanczos_bn_inplace!) || include(joinpath(@__DIR__,"Lanczos_functions.jl"))

######################## Basic utilities ########################

#Fraction of stored entries that are structurally nonzero
sparsity(M::SparseMatrixCSC) = nnz(M)/length(M)
sparsity(M::AbstractMatrix) = count(!iszero,M)/length(M)

#Infinite-temperature two-point function tr(AB)/2^N
auto_corr(A::AbstractMatrix,B::AbstractMatrix,N::Integer) = tr(A*B)/2^N

n_choose_k(n::Int,k::Int) = binomial(BigInt(n), BigInt(k))

#Pool a quantity over K disorder realizations. f() should return a scalar or a vector.
function disorder_average(f::Function,K::Integer)
    vals = Float64[]
    for i in 1:K
        vals = vcat(vals,f())
    end
    return mean(vals), median(vals), var(vals)
end

######################## Level statistics ########################

#The three eigenvalues bracketing zero energy
function gap_find(e_vals::AbstractVector)
    sorted_vals = sort(e_vals)
    idx = argmin(abs.(sorted_vals))
    lower = idx > 1 ? sorted_vals[idx - 1] : nothing
    center = sorted_vals[idx]
    upper = idx < length(sorted_vals) ? sorted_vals[idx + 1] : nothing
    return (lower, center, upper)
end

#spectrum_flag "all" uses the whole spectrum; "mid" keeps a window about the
#middle of the band (min(dim/10,500) levels), where the density of states is flattest.
function Level_spacings(H::AbstractMatrix,spectrum_flag::String="mid")
    vals = eigvals(Hermitian(Matrix(H)))
    return Level_spacings(vals,spectrum_flag)
end

function Level_spacings(vals::AbstractVector,spectrum_flag::String="mid")
    spacings = diff(sort(vals))

    dimH = length(vals)
    Half_dimH = round(Int,dimH/2)
    mid_N = round(Int,minimum([dimH/10,500])/2)

    if spectrum_flag == "all"
        return spacings
    elseif spectrum_flag == "mid"
        return spacings[Half_dimH-mid_N:Half_dimH+mid_N]
    else
        error("invalid spectrum_flag \"$spectrum_flag\" (options: \"all\", \"mid\")")
    end
end

#Consecutive level-spacing ratios r_n = min(s_n,s_{n+1})/max(s_n,s_{n+1}).
#Mean r ~ 0.386 for Poisson (localized), ~ 0.536 for GOE (chaotic).
function Level_ratios(H::AbstractMatrix,spectrum_flag::String="mid")
    return Level_ratios(Level_spacings(H,spectrum_flag))
end

function Level_ratios(vals::AbstractVector,spectrum_flag::String)
    return Level_ratios(Level_spacings(vals,spectrum_flag))
end

function Level_ratios(spacings::AbstractVector)
    return [min(spacings[i],spacings[i+1])/max(spacings[i],spacings[i+1]) for i in 1:length(spacings)-1]
end

######################## Spectral form factor ########################

#Gaussian filter selecting the middle of the band; Gamma^2 is the spectral variance
function filter_dist(e_n::Real,e_avg::Real,Gamma::Real)
    eta = 0.5 #effective fraction of eigenstates included in the SFF
    return exp(-(e_n-e_avg)^2 / (2*eta*Gamma)^2)
end

#Filtered spectral form factor K(tau) at a single time
function SFF_iter(tau::Real,H::AbstractMatrix)
    vals = eigvals(Hermitian(Matrix(H)))
    e_avg = mean(vals)
    Gamma = std(vals)

    rho = filter_dist.(vals,e_avg,Gamma)
    Z = sum(rho)
    return abs2(sum(rho .* exp.(-im*tau .* vals)))/Z^2
end

SFF(tau_vec::AbstractVector,H::AbstractMatrix) = [SFF_iter(tau,H) for tau in tau_vec]

######################## Correlation functions ########################

#Infinite-temperature autocorrelation C(t) = tr(O(t) O)/D from a precomputed spectrum
function Correlator_ED_t(t_vec::AbstractVector,vals::AbstractVector,vecs::AbstractMatrix,O::AbstractMatrix)
    C_t_vec = Vector{Float64}(undef, length(t_vec))

    Op_eigen = vecs' * O * vecs
    Op_eigen_sq = abs2.(Op_eigen)
    e_diff = (vals .- vals')

    for (i,t) in enumerate(t_vec)
        phase_t = exp.(im .* e_diff .* t)
        C_t_vec[i] = real(tr(phase_t * Op_eigen_sq))
    end
    return C_t_vec
end

#Same quantity straight from H. Uses one matrix-vector product per time instead of
#a full D x D product, so it is much cheaper than Correlator_ED_t for long t_vec.
function Correlation_fct_ED(H::AbstractMatrix, O::AbstractMatrix, times::AbstractVector{<:Real})
    vals, vecs = eigen(Hermitian(Matrix(H)))
    P = abs2.(vecs' * O * vecs)

    C_t = Vector{Float64}(undef, length(times))
    for (i, t) in enumerate(times)
        v = exp.(im .* vals .* t)
        C_t[i] = real(dot(conj(v), P * conj(v)))
    end
    return C_t
end

######################## Spectral functions ########################

#All (omega_nm, |O_nm|^2) pairs. e_start:e_end restricts to a window of the spectrum.
function Spectral_fct_ED(H::AbstractMatrix,O::AbstractMatrix;e_start::Int64=1,e_end::Int64=size(H,1))
    vals,vecs = eigen(Hermitian(Matrix(H)))
    O_eig = vecs' * Matrix(O) * vecs

    #the window must be applied to O_eig as well as to vals, or the matrix
    #elements below are read from the wrong indices
    O_eig = O_eig[e_start:e_end, e_start:e_end]
    vals = vals[e_start:e_end]

    freqs = Float64[]
    peaks = Float64[]
    for n in eachindex(vals), m in eachindex(vals)
        push!(freqs,real(vals[n]-vals[m]))
        push!(peaks,abs2(O_eig[n,m]))
    end
    return freqs, peaks
end

#Several operators sharing one diagonalization
function Spectral_fct_ED(H::AbstractMatrix,Os::AbstractVector{<:AbstractMatrix};e_start::Int64=1,e_end::Int64=size(H,1))
    vals,vecs = eigen(Hermitian(Matrix(H)))
    Os_eig = [vecs' * Matrix(O) * vecs for O in Os]

    Os_eig = [Oeig[e_start:e_end, e_start:e_end] for Oeig in Os_eig]
    vals = vals[e_start:e_end]

    freqs = Float64[]
    peaks = [Float64[] for _ in 1:length(Os)]
    for n in eachindex(vals), m in eachindex(vals)
        push!(freqs,real(vals[n]-vals[m]))
        for (i, Oeig) in enumerate(Os_eig)
            push!(peaks[i], abs2(Oeig[n, m]))
        end
    end
    return freqs, peaks
end

#Which bin of `edges` each x falls into
function bin_indices(x::AbstractVector{<:Real}, edges::AbstractVector{<:Real})
    nbins = length(edges) - 1
    bins = zeros(Int,length(x))
    for i in eachindex(x)
        xi = x[i]
        b = searchsortedlast(edges, xi)
        if b < 1 || b > nbins
            #the last edge is a closed right endpoint
            if xi == edges[end]
                b = nbins
            else
                error("value ($xi) outside of edges ($(edges[1]) to $(edges[end]))")
            end
        end
        bins[i] = b
    end
    return bins
end

function Bin_heights(x::AbstractVector{<:Real},y::AbstractVector{<:Real},edges::AbstractVector{<:Real})
    bin_inds = bin_indices(x, edges)
    binned_y = zeros(length(edges) - 1)
    for (i, bin_idx) in enumerate(bin_inds)
        binned_y[bin_idx] += y[i]
    end
    return binned_y
end

function Bin_heights(x::AbstractVector{<:Real},y::AbstractVector{<:AbstractVector{<:Real}},edges::AbstractVector{<:Real})
    bin_inds = bin_indices(x, edges)
    binned_y = [zeros(length(edges) - 1) for _ in 1:length(y)]
    for (i, bin_idx) in enumerate(bin_inds)
        for (j, yj) in enumerate(y)
            binned_y[j][bin_idx] += yj[i]
        end
    end
    return binned_y
end

#Binned spectral function S(omega), returning (bin centers, heights)
function Spectral_function_binned(H::AbstractMatrix,O::AbstractMatrix,bin_width::Real;
                                  e_start::Int64=1,e_end::Int64=size(H,1),
                                  x_tot_max::Real=15.,x_tot_min::Real=0.)
    freqs,peaks = Spectral_fct_ED(Matrix(H),Matrix(O);e_start=e_start,e_end=e_end)
    mask = x_tot_max .> freqs .> x_tot_min

    edges_tot = collect(x_tot_min:bin_width:x_tot_max)
    if x_tot_max > edges_tot[end]
        edges_tot = vcat(edges_tot,edges_tot[end]+bin_width)
    end

    centers = edges_tot[1:end-1] .+ bin_width/2
    heights = Bin_heights(freqs[mask],peaks[mask],edges_tot)
    return centers, heights
end

#Positive-frequency spectral function straight from a precomputed spectrum.
#S is symmetric in omega, so storing omega > 0 loses no information.
function spectral_function_from_spectrum(O_eig::AbstractMatrix, vals::AbstractVector, bin_width::Real)
    x_max = maximum(vals) - minimum(vals)
    nbins = ceil(Int, x_max / bin_width)
    centers = collect(((0:nbins-1) .+ 0.5) .* bin_width)
    heights = zeros(nbins)
    dim = length(vals)
    for n in 1:dim, m in 1:dim
        freq = vals[n] - vals[m]
        if freq > 0.0
            bin_idx = floor(Int, freq / bin_width) + 1
            if bin_idx <= nbins
                @inbounds heights[bin_idx] += abs2(O_eig[n, m])
            end
        end
    end
    return centers, heights
end

######################## One diagonalization, all observables ########################

#Correlation function, spectral function, level ratios and Lanczos b_n from a single
#eigen! of H. Set Nsteps>0 to include the b_n. Returns a NamedTuple.
function analyze_ED(H::AbstractMatrix,O::AbstractMatrix,times::AbstractVector{<:Real};
                    Nsteps::Int=0,bin_width::Real=0.05,spectrum_flag::String="mid")

    vals,vecs = eigen!(Hermitian(Matrix(H))) #eigen! overwrites the copy, not the caller's H
    O_ed = vecs' * Matrix(O) * vecs
    vecs = nothing

    P = abs2.(O_ed)
    C_t = zeros(Float64, length(times))
    for (i,t) in enumerate(times)
        v = exp.(im .* vals .* t)
        C_t[i] = real(dot(conj(v), P * conj(v)))
    end
    tr_P = tr(P)
    P = nothing

    sf_centers,sf_heights = spectral_function_from_spectrum(O_ed,vals,bin_width)
    ratios = Level_ratios(vals,spectrum_flag)

    #lanczos_bn_inplace! mutates O_ed, so it goes last
    bn = Float64[]
    if Nsteps > 0
        bn = O_ed isa Matrix{Float64} ? lanczos_bn_inplace!(O_ed,vals,Nsteps) :
                                        Lanczos_from_spectrum(O_ed,vals,Nsteps)
    end

    return (vals=vals, C_t=C_t, tr_P=tr_P, sf_centers=sf_centers, sf_heights=sf_heights,
            ratios=ratios, bn=bn)
end
