using LinearAlgebra, Statistics,SparseArrays
using QuantumSimulator; cd(@__DIR__)

#Define parameters
L = 10 # length of lattice
d=3 #spatial dimension
W=40 # disorder
t=1 # tunneling probability
Nsteps=1000 #must match the Nsteps used in Anderson_Lanczos_run.jl

#Load in some data
#path_out="C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//Anderson_krylov//Outs//Lanczos//"*H_flag
#cd(path_out)
cd("Outs//Lanczos_data//b-d$d-L$L-W$W-t$t")
Nsamples = length(readdir()) #Take all samples in the folder
b_tot = zeros(Nsteps,Nsamples)
for k in 1:Nsamples
    file="b$k.txt"
    #take the values straight from the parsed "b = [...]" line. Assigning a global with eval and
    #reading it back in the same loop stopped working in Julia 1.12 (world age of new bindings)
    b_tot[:,k]=eval(Meta.parse(split(readline(file)," = ")[2]))
end

#Calculate the average
b_avg=mean(b_tot,dims=2)[:]

#Construct the zero modes
#First from each sample, averaging after
ZeroMode_avg = ZeroMode_log_avg_from_b(b_tot)
#Next from average b
ZeroMode_b_avg = ZeroMode_from_b(b_avg)

#plot the results, comparing the two zero modes and checking
#
using PyPlot
fig,axs=subplots()

ns=collect(1:length(ZeroMode_avg)) .^(1/(2*d)) #Rescale as predicted
axs.plot(ns[1:2:end] ,ZeroMode_b_avg[1:2:end].^2,label="ZM from average b")
axs.plot(ns[1:2:end],ZeroMode_avg[1:2:end].^2,label="ZM log-averaged over samples")
axs.set_xlabel("Krylov space n^(1/$(2d))")
axs.set_ylabel("Zero mode probability density")
axs.set_yscale("log")
fig
fig.savefig("..//..//Plots//Zero-modes.pdf")
