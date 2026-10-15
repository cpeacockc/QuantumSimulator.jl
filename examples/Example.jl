#Example code for using Quantum Simulator, a Julia code for simulating quantum mechanics
using QuantumSimulator
using LinearAlgebra, Statistics

N = 10 #Number of lattice sites

Z_5  = Op_create(N,"Z",5,"pauli") #Pauli Z operator on site 5
Sz_5 = Op_create(N,"Z",5,"spin")  #spin Z operator on site 5 (Sz = 0.5 Z)

#Other options are "X","Y","Z","+","-" (final two are \sigma^+ = X+iY and \sigma^- = X-iY)
#Multi-site strings work too:
X1Z3 = op_create(N,[("X",1),("Z",3)])

#Now let's create a simple product state of up and down spins
#Options: "Neel" (alternating up & down), "Dn"/"Down" (all down), "Up" (all up),
#         "Rand" (random up/down product state), "RandVec" (Haar-random state)
psi = Product_state(N,"Neel")

#Site indexing is consistent between operators and states:
@show [round(real(psi' * Op_create(N,"Z",s,"pauli") * psi),digits=6) for s in 1:N]

#Build a Hamiltonian and evolve
H = H_Quantum_Ising(N,1.05,0.5)
t_array = collect(0:0.5:10)
Ops = [Op_create(N,"Z",site,"pauli") for site in 1:N]
Ops_t = Diag_TE_measure(psi,H,t_array,Ops)
@show size(Ops_t)

#Level statistics: is this model chaotic? (~0.386 Poisson, ~0.536 GOE)
#NB: H above is a *uniform periodic* chain, so translation and spin-flip symmetry mix
#independent sectors and the ratio comes out far below Poisson. Break the symmetries
#(open BCs + disordered fields) before reading anything into the number.
H_nosym = H_XYZ(N,[0,0,1.0],1.05*ones(N),zeros(N),0.5 .+ 0.3*randn(N);periodic=false)
@show mean(Level_ratios(H_nosym,"mid"))    # chaotic, lands near 0.53
@show mean(Level_ratios(H_GOE(8),"all"))   # a true GOE matrix, for reference

#Operator growth: Lanczos coefficients for Z_1 under H
b = Lanczos_ED(Matrix(Op_create(N,"Z",1,"pauli")),H,30)
@show b[1:5]
#Krylov complexity. Only trust times short enough that the operator has not yet
#reached the end of the truncated Krylov chain, or it reflects back off the wall.
@show K_complexity(b,collect(0:0.2:1.0))
