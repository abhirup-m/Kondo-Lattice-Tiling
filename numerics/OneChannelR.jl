using Fermions, PyPlot

sites = 60
#=KE = repeat((10).^range(-0, stop=-5, length=div(sites, 4)), inner=2)=#
#=KE .*= (-1).^(1:div(sites, 2))=#
#=KE = repeat(KE, inner=2)=#
#=println(KE)=#
J = 10 .^ range(-0.5, stop=-5, length=sites)
J[1] = 0.5
J[2:end] .= 0
t = 1.

H = Tuple{String, Vector{Int64}, Float64}[]
push!(H, ("nn", [1, 2], 10.))
push!(H, ("n", [1], -5.))
push!(H, ("n", [2], -5.))
for i in 1:sites
    up = 2 * i + 1
    push!(H, ("nn", [1, up], J[i]/4))
    push!(H, ("nn", [1, up + 1], -J[i]/4))
    push!(H, ("nn", [2, up], -J[i]/4))
    push!(H, ("nn", [2, up + 1], J[i]/4))
    push!(H, ("+-+-", [1, 2, up + 1, up], J[i]/2))
    push!(H, ("+-+-", [2, 1, up, up + 1], J[i]/2))

    if i < (sites - 1)
        push!(H, ("+-", [up, up + 2], -t))
        push!(H, ("+-", [up + 2 , up], -t))
        push!(H, ("+-", [up + 1, up + 3], -t))
        push!(H, ("+-", [up + 3, up + 1], -t))
    end
end

specFunc = Dict("Ad" => Dict("create" => [("+-+", [2, 1, 3], 1.)], "destroy" => [("+--", [1, 2, 3], 1.)]), 
                "Af" => Dict("create" => [("+-+", [2, 1, 5], 1.)], "destroy" => [("+--", [1, 2, 5], 1.)]),
                "Adf" => Dict("create" => [("+-+", [2, 1, 3], 1.), ("+-+", [2, 1, 5], 1.)], "destroy" => [("+--", [1, 2, 3], 1.), ("+--", [1, 2, 5], 1.)]),
               )
#=specFunc = Dict("Ad" => Dict("create" => [("+", [1], 1.), ], "destroy" => [("-", [1], 1.), ]))=#
family = MinceHamiltonian(H, 12:2:2*(1 + sites))
results = IterDiag(family, 3000; specFuncDefDict=specFunc)
ω = collect(-4:0.001:4)
Ad = SpecFunc(vcat(results["Ad"]...), ω, 0.1)
Af = SpecFunc(vcat(results["Af"]...), ω, 0.1)
Adf = SpecFunc(vcat(results["Adf"]...), ω, 0.1)
fig, ax = plt.subplots()
ax.plot(ω, Ad)
#=ax.plot(ω, Af)=#
#=ax.plot(ω, Adf)=#
fig.savefig("Ad_oneCK.pdf")
