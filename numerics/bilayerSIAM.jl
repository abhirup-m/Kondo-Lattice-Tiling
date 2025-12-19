using Fermions, Plots

function BilayerSIAM(
        V::Float64,
        Δ::Float64,
        Ek::Vector,
        μ::Float64,
    )
    sites = length(Ek)
    #########
    # f1u  f1d  f2u  f2d  A1u  A1d  B1u  B1d ...
    # 1    2    3    4    5    6    7    8   ...
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]
    push!(hamiltonian, ("n", [1], -0.5))
    push!(hamiltonian, ("n", [2], -0.5))
    push!(hamiltonian, ("nn", [1, 2], 1.))
    push!(hamiltonian, ("n", [3], -0.5))
    push!(hamiltonian, ("n", [4], -0.5))
    push!(hamiltonian, ("nn", [3, 4], 1.))
    if Δ ≠ 0
        push!(hamiltonian, ("+-", [1, 3], -Δ))
        push!(hamiltonian, ("+-", [3, 1], -Δ))
        push!(hamiltonian, ("+-", [2, 4], -Δ))
        push!(hamiltonian, ("+-", [4, 2], -Δ))
    end
    rungA = 5:4:(4 + 4*sites)
    rungB = 7:4:(4 + 4*sites)
    for (i, (A, B)) in enumerate(zip(rungA, rungB))
        push!(hamiltonian, ("+-", [1, A], -V))
        push!(hamiltonian, ("+-", [A, 1], -V))
        push!(hamiltonian, ("+-", [2, A + 1], -V))
        push!(hamiltonian, ("+-", [A + 1, 2], -V))


        push!(hamiltonian, ("+-", [3, B], -V))
        push!(hamiltonian, ("+-", [B, 3], -V))
        push!(hamiltonian, ("+-", [4, B + 1], -V))
        push!(hamiltonian, ("+-", [B + 1, 4], -V))

        if Ek[i] ≠ 0
            push!(hamiltonian, ("n", [A], Ek[i] + Δ))
            push!(hamiltonian, ("n", [A + 1], Ek[i] + Δ))
            push!(hamiltonian, ("n", [B], Ek[i] - Δ))
            push!(hamiltonian, ("n", [B + 1], Ek[i] - Δ))
        end
        if μ ≠ 0
            push!(hamiltonian, ("n", [A], -μ))
            push!(hamiltonian, ("n", [B], -μ))
        end
    end
    return hamiltonian
end

sites = 20
@assert sites % 2 == 0
Ek = 10 .^range(0, stop=-6, length=div(sites, 2))
Ek = repeat(Ek, inner=2)
Ek[1:2:end] .*= -1
display(Ek)
ω = collect(-10:0.01:10)
η = 0.05
p = plot()
for Δ in [3.]
    hamiltonian = BilayerSIAM(1., Δ, Ek, 0.)
    family = MinceHamiltonian(hamiltonian, 8:2:(4 + 4*sites))
    specFunc = Dict("create" => [("+", [1], 1.0)], "destroy" => [("-", [1], 1.0)])
    results = IterDiag(family, 3000; specFuncDefDict=Dict("A1" => specFunc), maxMaxSize=3000)
    specCoeffs = vcat(results["A1"]...)
    specFunc = SpecFunc(specCoeffs, ω, η)
    specFunc = 0.5 * (specFunc .+ reverse(specFunc))
    plot!(p, ω, specFunc, label="\$\\Delta=$(Δ)\$")
end
savefig(p, "bisiam_specfunc.pdf")
