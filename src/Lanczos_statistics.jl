#Disorder statistics of the Lanczos coefficients, and the Krylov-space zero modes (LIOMs)
#built from them. Throughout, bn_tot is a matrix with one disorder realization per column.
using LinearAlgebra, SparseArrays, Statistics

isdefined(@__MODULE__, :LIOM_from_b) || include(joinpath(@__DIR__,"Lanczos_functions.jl"))

######################## Dimerization statistics ########################

function avg_dimerization_from_b_tot(bn_tot::AbstractArray)
    Nsamples = size(bn_tot)[2]
    dimerizations=zeros(length(dimerization_from_b(bn_tot[:,1])),Nsamples)
    for i in 1:Nsamples
        dimerizations[:,i] = dimerization_from_b(bn_tot[:,i])
    end
    b_dimerization_avg = mean(dimerizations,dims=2)
    return b_dimerization_avg[:]
end

function log_avg_dimerization_ratio_from_b_tot(bn_tot::AbstractArray)
    Nsamples = size(bn_tot)[2]
    dimerizations=zeros(length(dimerization_ratio_from_b(bn_tot[:,1])),Nsamples)
    for i in 1:Nsamples
        dimerizations[:,i] = dimerization_ratio_from_b(bn_tot[:,i])
    end
    b_dimerization_avg = exp.(mean(log.(dimerizations),dims=2))
    return b_dimerization_avg[:]
end

function avg_dimerization_ratio_from_b_tot(bn_tot::AbstractArray)
    Nsamples = size(bn_tot)[2]
    dimerizations=zeros(length(dimerization_ratio_from_b(bn_tot[:,1])),Nsamples)
    for i in 1:Nsamples
        dimerizations[:,i] = dimerization_ratio_from_b(bn_tot[:,i])
    end
    b_dimerization_avg = mean((dimerizations),dims=2)
    return b_dimerization_avg[:]
end

#Synthetic b_n with a dimerized power-law decay, for testing the analysis above
function dimerized_powerlaw_decay_bn(K_dim::Int,a::Real,b::Real,alpha::Real)
    bn = ones(K_dim-1) .* a
    for n in collect(2:2:K_dim-1)
        bn[n] += (b)*(n^(-alpha))
    end
    return bn
end

######################## x_n = -log(b_2n/b_2n+1) statistics ########################

function x_from_b(bn)
    x_is = Float64[]
    for i in 1:1:length(bn[1:2:end])-1
        x_i = -log(bn[2i-1]/bn[2i])
        push!(x_is,x_i)
    end
    return x_is
end

var_of_sum(cov_M, N::Int64) = [sum(cov_M[1:n, 1:n]) for n in 1:N]

function x_generate(bn_tot,K)
    arr_length = length((bn_tot[:,1])[1:2:end])-1
    xs = zeros(arr_length,K)
    if K>size(bn_tot)[2]
        println("K larger than bn_tot[2]")
    else
        for k in 1:1:K
            bn = bn_tot[1:end,k]
            if bn[1]==0
                bn[1]=1
            elseif bn[1]==1
            else
                bn = vcat(1,bn)
            end
            x_is = Float64[]
            for i in 1:1:arr_length
                x_i = -log(bn[2i+1]/bn[2i])
                push!(x_is,x_i)
            end
                xs[:,k] = x_is
        end
    end
    return xs
end

function kappa_cumsum_stats(bn_tot,K)
    xs = x_generate(bn_tot,K)
    cumsum_xs = zeros(size(xs))
    for k in 1:K
        x = xs[:,k]
        cumsum_xs[:,k]=cumsum(x)
    end
    cumsum_vars=var(cumsum_xs,dims=2)
    cumsum_means=mean(cumsum_xs,dims=2)

    return cumsum_means,cumsum_vars
end

#The statistical core of the bn_statistics family in General_Quantum, taking bn_tot
#directly rather than loading it from an HDF5 export.
function bn_statistics(bn_tot,K=size(bn_tot)[2])
    xs = x_generate(bn_tot,K)
    xs_cumsum = cumsum(xs,dims=1)

    xs_var = var(xs,dims=2)[:]
    xs_var_cumsum = var(xs_cumsum,dims=2)[:]
    xs_mean = mean(xs,dims=2)[:]
    xs_med = median(xs,dims=2)[:]
    xs_cov = cov(xs,dims=2)
    gn_cov = cov(xs_cumsum,dims=2)

    return (xs=xs, xs_var=xs_var, xs_var_cumsum=xs_var_cumsum,
            xs_mean=xs_mean, xs_median=xs_med, xs_cov=xs_cov, gn_cov=gn_cov,
            xs_mean_cumsum=mean(xs_cumsum,dims=2)[:],
            xs_median_cumsum=median(xs_cumsum,dims=2)[:])
end

function phi_eta_calc(bn_tot,K)
    arr_length = length((bn_tot[:,1])[1:2:end])-1
    xs = zeros(arr_length,K)
    ys = zeros(arr_length,K)
    if K>size(bn_tot)[2]
        println("K larger than bn_tot[2]")
    else
        for k in 1:1:K
            bn = bn_tot[1:end,k]
            x_is = Float64[]
            y_is = Float64[]
            for i in 1:1:arr_length
                x_i = -log(bn[2i-1]/bn[2i])
                y_i = -log(bn[2i]/bn[2i+1])
                push!(x_is,x_i)
                push!(y_is,y_i)
            end
                xs[:,k] = x_is
                ys[:,k]= y_is
        end
    end
    xs_cumsum = cumsum(xs,dims=1)
    ys_cumsum = cumsum(ys,dims=1)
    xs_mean_cumsum=mean(xs_cumsum,dims=2)[:]
    ys_mean_cumsum=mean(ys_cumsum,dims=2)[:]

    phi=exp.(-1 .*(xs_mean_cumsum))
    eta = exp.(-1 .*(ys_mean_cumsum))
    return phi,eta
end

######################## Zero modes / LIOMs from the b_n ########################

function LIOM_vars_from_b(bn_tot)
    Nsamples = size(bn_tot)[2]
    LIOMs=zeros(length(LIOM_from_b(bn_tot[:,1])),Nsamples)

    for j in 1:Nsamples
        LIOMs[:,j]=LIOM_from_b(bn_tot[:,j])
    end
    LIOM_vars=var(log.(abs.(LIOMs)),dims=2)
    return LIOM_vars
end

#Tunnelling rate out of the end of the Krylov chain, G = b_n |phi_n|^2
function LIOM_avg_TunnelRate(bn_tot)
    Nsamples = size(bn_tot)[2]
    Gs=zeros(Nsamples)
    for j in 1:Nsamples

        LIOM = LIOM_from_b(bn_tot[1:end-1,j])
        LIOM /= sqrt(sum(abs.(LIOM).^2)) #normalize

        if LIOM[end]==0.
            LIOM = LIOM_from_b(bn_tot[1:end-2,j])
            LIOM /= sqrt(sum(abs.(LIOM).^2)) #normalize
            b_n_out =bn_tot[end-1,j]
        else
            b_n_out =bn_tot[end,j]
        end

        Gs[j]=b_n_out*(LIOM[end])^2
    end
    G_avg=exp.(mean(log.(Gs)))
    return G_avg
end

function LIOM_avg_final_val(bn_tot)
    Nsamples = size(bn_tot)[2]
    Gs=zeros(Nsamples)
    for j in 1:Nsamples
        LIOM = LIOM_from_b(bn_tot[:,j])
        LIOM /= sqrt(sum(abs.(LIOM).^2)) #normalize
        Gs[j]=maximum((LIOM[end],LIOM[end-1]))
    end
    G_avg=exp.(mean(log.(Gs)))
    return G_avg
end

function LIOM_from_xs(bn_tot)
    K=size(bn_tot)[2]
    xs=x_generate(bn_tot,K)
    xs_cumsum = cumsum(xs,dims=1)
    xs_mean_cumsum=mean(xs_cumsum,dims=2)[:]
    LIOM=exp.(-1 .*(xs_mean_cumsum))
    return LIOM
end

function LIOM_from_xs_cumsum(xs_mean_cumsum)
    LIOM=exp.(-1 .*(xs_mean_cumsum))
    return LIOM
end

#Real-space zero mode from a full Krylov basis (as returned by Lanczos_full)
function LIOM_real_space(O::Array{Float64,3},bn::Vector{Float64})
    LIOM = O[1,:,:] #O_0

    for n in 2:2:length(bn)
        phi_n=1
        for i in 1:Int(n/2)
            phi_n *=(bn[2*i-1]/bn[2*i])
        end
        LIOM += phi_n*O[n+1,:,:] #O_2n but i+1 because of 0 shift
    end
    return LIOM
end

######################## LIOMs by diagonalizing the Krylov Liouvillian ########################
#x_arr sweeps the Krylov chain length kept; seig picks which eigenvector to call the LIOM.

function LIOM_avg_construct(bn_tot,x_arr,seig)
    avg_vals=Float64[]
    avg_vecs=[]
    for x in x_arr
        min_eigvals=Float64[]
        LIOM_arr = zeros(Int(x/2),size(bn_tot)[2])

        for i in 1:1:size(bn_tot)[2]

            bn_i= bn_tot[:,i][2:end]

            L = Tridiagonal(bn_i,zeros(length(bn_i)+1),bn_i)

            M=(transpose(L)*L)[1:x,1:x]
            eigM=eigen(M)

            eig_vals=eigM.values
            eig_vecs=eigM.vectors

            min_eig_val=eig_vals[seig]
            min_eig_vec=eig_vecs[:,seig]

            LIOM_arr[:,i]=min_eig_vec[1:2:end]
            push!(min_eigvals,min_eig_val)

            end
        avg_min_val=median(min_eigvals)[1]
        avg_LIOM=mean(LIOM_arr,dims=2)[1:end]

        push!(avg_vals,avg_min_val)
        push!(avg_vecs,avg_LIOM)
        end
    return (avg_vals,avg_vecs)
end

function LIOM_log_avg_construct(bn_tot,x_arr,seig)
    avg_vals=Float64[]
    avg_vecs=[]
    for x in x_arr
        min_eigvals=Float64[]
        LIOM_arr = zeros(x,size(bn_tot)[2])

        for i in 1:1:size(bn_tot)[2]

            bn_i= bn_tot[:,i][2:end]

            L = Tridiagonal(bn_i,zeros(length(bn_i)+1),bn_i)[1:x,1:x]

            M=(transpose(L)*L)
            eigM=eigen(M)

            eig_vals=eigM.values
            eig_vecs=eigM.vectors

            min_eig_val=eig_vals[seig]
            min_eig_vec=eig_vecs[:,seig]

            LIOM_arr[:,i]=log.(abs.(min_eig_vec))
            push!(min_eigvals,min_eig_val)

            end
        avg_min_val=median(min_eigvals)[1]
        avg_LIOM=mean(LIOM_arr,dims=2)[1:end]

        push!(avg_vals,avg_min_val)
        push!(avg_vecs,avg_LIOM)
        end
    return (avg_vals,avg_vecs)
end

function LIOM_log_avg_construct_L(bn_tot,x_arr,seig)
    avg_vals=Float64[]
    avg_vecs=[]
    for x in x_arr
        min_eigvals=Float64[]
        LIOM_arr = zeros(x,size(bn_tot)[2])

        for i in 1:1:size(bn_tot)[2]

            bn_i= bn_tot[:,i][2:end]

            L = Tridiagonal(bn_i,zeros(length(bn_i)+1),bn_i)

            eigM=eigen(L[1:x,1:x])

            eig_vals=eigM.values
            eig_vecs=eigM.vectors

            min_eig_val=eig_vals[seig]
            min_eig_vec=eig_vecs[:,seig]

            LIOM_arr[:,i]=log.(abs.(min_eig_vec))
            push!(min_eigvals,min_eig_val)

            end
        avg_min_val=median(min_eigvals)[1]
        avg_LIOM=mean(LIOM_arr,dims=2)[1:end]

        push!(avg_vals,avg_min_val)
        push!(avg_vecs,avg_LIOM)
        end
    return (avg_vals,avg_vecs)
end

function LIOM_log_avg_construct_EA(bn_tot,x_arr,seig)
    avg_vals=Float64[]
    avg_vecs=[]
    for x in x_arr
        min_eigvals=Float64[]
        LIOM_arr = zeros(x,size(bn_tot)[2])

        for i in 1:1:size(bn_tot)[2]

            bn_i= bn_tot[:,i]

            L = Tridiagonal(bn_i,zeros(length(bn_i)+1),bn_i)

            M=(transpose(L)*L)[1:x,1:x]
            eigM=eigen(M)

            eig_vals=eigM.values
            eig_vecs=eigM.vectors

            min_eig_val=eig_vals[seig]
            min_eig_vec=eig_vecs[:,seig]

            LIOM_arr[:,i]=log.(abs.(min_eig_vec))
            push!(min_eigvals,min_eig_val)

            end
        avg_min_val=median(min_eigvals)[1]
        avg_LIOM=mean(LIOM_arr,dims=2)[1:end]

        push!(avg_vals,avg_min_val)
        push!(avg_vecs,avg_LIOM)
        end
    return avg_vecs
end

function LIOM_construct(bn_avg,x_arr,seig)
    min_vals=Float64[]
    min_vecs=[]
    for x in x_arr

            bn_i= bn_avg[2:end]

            L = Tridiagonal(bn_i,zeros(length(bn_i)+1),bn_i)

            M=(transpose(L)*L)[1:x,1:x]
            eigM=eigen(M)

            eig_vals=eigM.values
            eig_vecs=eigM.vectors

            min_eig_val=eig_vals[seig]
            min_eig_vec=eig_vecs[:,seig]

        push!(min_vals,min_eig_val)
        push!(min_vecs,min_eig_vec[1:2:end]./norm(min_eig_vec[1:2:end]))
        end
    return (min_vals,min_vecs)
end

#--- Log staggering, pair geometric mean and the power-law fit of the dimerization (Krylov_sun, arXiv:2510.26920v4)
#All four take b WITHOUT the stored placeholder: b[1] is the first true coefficient b_1. The Krylov_sun
#HDF5 files store bn[1] = 1 (the norm b_0), so read them as bn[2:end,:] before calling; passing the
#placeholder shifts every k by one and turns b_1 = alpha^L/2 into a fake b_2. Map: docs/KRYLOV_SUN_PAPER_MAP.md.

#Log staggering lambda_k = ln b_2k - (ln b_(2k-1) + ln b_(2k+1))/2, k = 1..div(length(b)-1,2): the log
#form of the dimerization b_(2n+1)/b_(2n) of eq. (10) (our b_k is the paper's b_(k+1)). Symmetric in the
#two odd neighbours, so a smooth envelope b_n ~ n^delta leaves only O(delta/k^2); the sample mean of
#lambda_k has no Jensen bias, the mean of ratios does.
#Unvalidated
function log_staggering_from_b(b::AbstractVector)
    K=div(length(b)-1,2)
    return [log(b[2*k])-(log(b[2*k-1])+log(b[2*k+1]))/2 for k in 1:1:K]
end

#lambda_k for every disorder sample: bn_tot is coefficients x samples without the placeholder, the result
#is k x samples. Averaging is left to the caller because the window mean must be taken inside each sample
#first and then over samples; any other order reintroduces the Jensen bias.
#Unvalidated
function log_staggering_from_b_tot(bn_tot::AbstractMatrix)
    K=div(size(bn_tot,1)-1,2);Nsamples=size(bn_tot,2)
    lam=zeros(float(eltype(bn_tot)),K,Nsamples)
    for j in 1:1:Nsamples
        lam[:,j]=log_staggering_from_b(bn_tot[:,j])
    end
    return lam
end

#Pair geometric mean B_k = sqrt(b_(2k-1) b_2k), k = 1..div(length(b),2): the staggering-free envelope.
#For b_n = n^delta (1 + (-1)^n g) it is exactly (2k(2k-1))^(delta/2) sqrt(1-g^2), so the slope of ln B_k
#against ln k measures the envelope exponent delta with no dimerization leaking in.
#Unvalidated
function pair_geomean_from_b(b::AbstractVector)
    K=div(length(b),2)
    return [sqrt(b[2*k-1]*b[2*k]) for k in 1:1:K]
end

#Weighted power-law fit lam_k = A k^(gamma-1): eq. (10) with eps = 0, the -delta/(2k) term already removed
#by the caller. gamma is scanned over `gammas`; at each gamma the amplitude has the closed form
#A = sum_k w_k lam_k s_k / sum_k w_k s_k^2 with the shape s_k = k^(gamma-1) and weights w_k = 1/sigma_k^2,
#and chi2 = sum_k w_k lam_k^2 - (sum_k w_k lam_k s_k)^2 / sum_k w_k s_k^2. Every k enters with its sign:
#no censoring on lam_k > 0, which a log-log fit needs and which biases gamma upward where lam_k is small.
#Returns (gamma, A, chi2) at the minimum over the grid.
#Unvalidated
function gamma_powerlaw_fit(k::AbstractVector,lam::AbstractVector,sigma::AbstractVector;gammas=-1.0:0.002:2.0)
    w=1.0 ./ sigma.^2
    lam_norm=sum(w.*lam.^2)
    chi2=zeros(length(gammas));A=zeros(length(gammas))
    for i in 1:1:length(gammas)
        shape=k.^(gammas[i]-1)
        shape_norm=sum(w.*shape.^2);overlap=sum(w.*lam.*shape)
        A[i]=overlap/shape_norm
        chi2[i]=lam_norm-overlap^2/shape_norm
    end
    i=argmin(chi2)
    return (gamma=gammas[i],A=A[i],chi2=chi2[i])
end
