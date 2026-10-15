#Localization transition: disorder-averaged level ratios vs disorder strength.
#<r> ~ 0.536 in the delocalized (GOE) phase, ~ 0.386 in the localized (Poisson) phase.
#1d and 2d Anderson are localized at any W; 3d has a transition near W = 16.5.
using QuantumSimulator
using Statistics

K = 20        #disorder realizations per point
Ws = collect(1:5:41)
t = 1

for (d,L) in ((1,500),(2,22),(3,8))
    means = Float64[]
    for W in Ws
        m,md,v = disorder_average(() -> Level_ratios(H_Anderson(d,L,W,t),"mid"), K)
        push!(means,m)
    end
    println("d=$d (L=$L, dim=$(L^d))")
    for (W,m) in zip(Ws,means)
        println("  W=$(lpad(W,2))  <r>=$(round(m,digits=3))")
    end
end
