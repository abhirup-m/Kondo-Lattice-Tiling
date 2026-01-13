using Fermions, PyPlot

sites = 20
KE = repeat((10).^range(0, stop=-5, length=div(sites, 2)), inner=2)
KE .*= (-1).^(1:sites)
J = 0.1

H = Tuple{String, Vector{Int64}, Float64}[]
#=push!(H, ("n", [1], 1e-8))=#
#=push!(H, ("n", [2], -1e-8))=#
for i in 1:sites
    up1 = 2 * i + 1
    for j in 1:sites
        up2 = 2 * j + 1
        push!(H, ("n+-", [1, up1, up2], J/4))
        push!(H, ("n+-", [1, up1 + 1, up2 + 1], -J/4))
        push!(H, ("n+-", [2, up1, up2], -J/4))
        push!(H, ("n+-", [2, up1 + 1, up2 + 1], J/4))
        push!(H, ("+-+-", [1, 2, up1 + 1, up2], J/2))
        push!(H, ("+-+-", [2, 1, up1, up2 + 1], J/2))
    end
    push!(H, ("n", [up1], KE[i]))
    push!(H, ("n", [up1 + 1], KE[i]))
end

specFunc = Dict("Ad" => Dict("create" => [("+", [1], 1.), ], "destroy" => [("-", [1], 1.), ]))
family = MinceHamiltonian(H, 4:2:2*(1 + sites))
results = IterDiag(family, 2000; specFuncDefDict=specFunc)
ω = collect(-10:0.01:10)
Ad = SpecFunc(vcat(results["Ad"]...), ω, 0.1)
fig, ax = plt.subplots()
ax.plot(ω, Ad)
fig.savefig("Ad_twoCK.pdf")
