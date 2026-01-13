using Fermions, PyPlot

sites = 20
KE = repeat((10).^range(-0, stop=-5, length=div(sites, 4)), inner=2)
KE .*= (-1).^(1:div(sites, 2))
KE = repeat(KE, inner=2)
println(KE)
J = 0.5

H = Tuple{String, Vector{Int64}, Float64}[]
push!(H, ("nn", [1, 2], 10.))
push!(H, ("n", [1], -5.))
push!(H, ("n", [2], -5.))
for i in 1:2:sites
    up1 = 2 * i + 1
    for j in 1:2:sites
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

for i in 2:2:sites
    up1 = 2 * i + 1
    for j in 2:2:sites
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

#=specFunc = Dict("Ad" => Dict("create" => [("+-+", [2, 1, 2 * i + 1], 1.) for i in 1:2:sites], "destroy" => [("+--", [1, 2, 2 * i + 1], 1.) for i in 1:2:sites]), =#
specFunc = Dict("Ad" => Dict("create" => [("+-+", [2, 1, 2 * sites - 1], 1.), ("+-+", [2, 1, 2 * sites + 1], 1.)], "destroy" => [("+--", [1, 2, 2 * sites - 1], 1.), ("+--", [1, 2, 2 * sites + 1], 1.)]),
                #="Af" => Dict("create" => [("+-+", [2, 1, 2 * sites + 1], 1.)], "destroy" => [("+--", [1, 2, 2 * sites + 1], 1.)]),=#
               )
#=specFunc = Dict("Ad" => Dict("create" => [("+", [1], 1.), ], "destroy" => [("-", [1], 1.), ]))=#
family = MinceHamiltonian(H, 8:4:2*(2 + sites))
results = IterDiag(family, 2000; specFuncDefDict=specFunc)
ω = collect(-10:0.01:10)
Ad = SpecFunc(vcat(results["Ad"]...), ω, 0.1)
#=Af = SpecFunc(vcat(results["Af"]...), ω, 0.1)=#
fig, ax = plt.subplots()
ax.plot(ω, Ad)
fig.savefig("Ad_twoCK.pdf")
