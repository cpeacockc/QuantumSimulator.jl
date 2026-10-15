module QuantumSimulator

using LinearAlgebra, SparseArrays, Statistics, Random

#Load the whole library in dependency order.
#
#   using QuantumSimulator
#
#The individual files can still be included on their own -- each pulls in what it
#needs -- but this is the convenient entry point.
#
#Lanczos_functions comes before Measurements because Measurements uses
#lanczos_bn_inplace!; loading it the other way round includes Lanczos_functions twice.

include("Pauli_generator.jl")    #operators, product states
include("Models.jl")             #Hamiltonians
include("Time_evolution.jl")     #Schrodinger / Heisenberg evolution
include("Lanczos_functions.jl")  #Krylov / operator growth
include("Measurements.jl")       #level statistics, spectral functions
include("Lanczos_statistics.jl") #disorder statistics of the b_n, LIOMs

########## Exports ##########
#Everything defined above is exported, so `using QuantumSimulator` behaves the same way
#`include("Quantum_simulator.jl")` used to. Names starting with _ stay internal.

#include used to drop LinearAlgebra, SparseArrays, Statistics and Random straight into
#Main, and the examples and the README lean on that -- eigen, spdiagm, mean and friends
#are all used unqualified. Passing them straight back out keeps those scripts working.
for m in (LinearAlgebra, SparseArrays, Statistics, Random)
    for n in names(m)
        isdefined(m,n) || continue        #ask the source module, not this one
        Base.isexported(m,n) || continue  #mirror `using`: public-but-unexported names stay put
        getfield(m,n) isa Module && continue
        @eval export $n
    end
end

for n in names(@__MODULE__; all=true)
    s = String(n)
    startswith(s,"#") && continue                       #compiler-generated
    startswith(s,"_") && continue                       #internal helpers
    n in (:QuantumSimulator,:eval,:include) && continue
    isdefined(@__MODULE__,n) || continue
    getfield(@__MODULE__,n) isa Module && continue      #LinearAlgebra, SparseArrays, ...
    @eval export $n
end

end #module
