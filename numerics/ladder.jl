using Fermions, Plots

function LadderModel(
        hop_t::Float64,
        Δ::Float64,
        sites::Int64,
        μ::Float64,
    )
    #########
    # A1  B1  A2  B2  A3  B3 ...
    # 1   2   3   4   5   6  ...
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]
    rungA = 1:2:2*sites
    rungB = 2:2:2*sites
    for (A, B) in zip(rungA, rungB)
        if A + 2 ∈ rungA
            push!(hamiltonian, ("+-", [A, A + 2], -hop_t))
            push!(hamiltonian, ("+-", [A + 2, A], -hop_t))
            push!(hamiltonian, ("+-", [B, B + 2], -hop_t))
            push!(hamiltonian, ("+-", [B + 2, B], -hop_t))
        end
        if A == 1
            push!(hamiltonian, ("+-", [A, B], -Δ))
            push!(hamiltonian, ("+-", [B, A], -Δ))
        end
        push!(hamiltonian, ("n", [A], -μ))
        push!(hamiltonian, ("n", [B], -μ))
    end
    return hamiltonian
end

sites = 30
ω = collect(-10:0.01:10)
η = 0.05
p = plot()
for Δ in [3.]
    hamiltonian = LadderModel(1., Δ, sites, 0.)
    family = MinceHamiltonian(hamiltonian, 2:2:2*sites)
    specFunc = Dict("create" => [("+", [3], 1.0)], "destroy" => [("-", [3], 1.0)])
    results = IterDiag(family, 2000; specFuncDefDict=Dict("A1" => specFunc))
    specCoeffs = vcat(results["A1"]...)
    specFunc = SpecFunc(specCoeffs, ω, η)
    plot!(p, ω, specFunc, label="\$\\Delta=$(Δ)\$")
end
savefig(p, "ladder_specfunc.pdf")
