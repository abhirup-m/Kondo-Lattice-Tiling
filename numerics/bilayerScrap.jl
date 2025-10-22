using PyPlot, ProgressMeter

function getDenominators(params, ω)
    return Dict("Jf" => [ω(params["bw"]) - params["bw"] / 2 + params["Jf"] / 4 + params["Wf"] / 2 + 3 * params["J"] / 4, ω(params["bw"]) - params["bw"] / 2 + params["Jf"] / 4 + params["Wf"] / 2 - params["J"] / 4],
                "Jd" => [ω(params["bw"]) - params["bw"] / 2 + params["Jd"] / 4 + params["Wd"] / 2 + 3 * params["J"] / 4, ω(params["bw"]) - params["bw"] / 2 + params["Jd"] / 4 + params["Wd"] / 2 - params["J"] / 4],
                "J" => [ω(params["bw"]) - params["bw"] + params["Jf"] / 2 + params["Wf"] - params["J"] / 4, ω(params["bw"]) - params["bw"] + params["Jd"] / 2 - params["J"] / 4]
               )
end

function deltaJfd(ω, params, dos, initSign, key)
    @assert key ∈ "fd"
    denominators = getDenominators(params, ω)["J"*key]
    flags = [s == sign(d) ? 1 : 0 for (s,d) in zip(initSign, denominators)]
    return -(params["J"*key]^2 + 4 * params["J"*key] * params["W"*key]) * dos * sum([0.25, 0.75] .* flags ./ denominators)
end

function deltaJ(ω, params, dos, initSign)
    denominators = getDenominators(params, ω)["J"]
    flags = [s == sign(d) ? 1 : 0 for (s,d) in zip(initSign, denominators)]
    return 0.5 * params["Jf"] * params["Jd"] * dos * sum(flags ./ denominators)
end

function LatestDict(dict)
    return Dict(k => v[end] for (k, v) in dict)
end

function rgFlow(
        bareParams,
        ω,
        dos,
        ΔD
    )
    renormalisedParams = Dict(k => [v] for (k,v) in bareParams)
    D = bareParams["bw"]
    initDenominators = getDenominators(bareParams, ω)
    deltaSigns = Dict{String, Any}(g => nothing for g in ("Jf", "Jd", "J"))
    while renormalisedParams["bw"][end] > 0
        latestDict = LatestDict(renormalisedParams)
        delta = Dict()
        delta["Jf"] = (deltaSigns["Jf"] ≠ 0 && latestDict["Jf"] ≠ 0) ? abs(ΔD) * deltaJfd(ω, latestDict, dos, sign.(initDenominators["Jf"]), 'f') : 0
        delta["Jd"] = (deltaSigns["Jd"] ≠ 0 && latestDict["Jd"] ≠ 0) ? abs(ΔD) * deltaJfd(ω, latestDict, dos, sign.(initDenominators["Jd"]), 'd') : 0
        delta["J"] = (deltaSigns["J"] ≠ 0 && latestDict["J"] ≠ 0) ? abs(ΔD) * deltaJ(ω, latestDict, dos, sign.(initDenominators["J"])) : 0
        for (k, s) in deltaSigns
            if isnothing(s)
                deltaSigns[k] = sign(delta[k])
            elseif s * delta[k] < 0
                deltaSigns[k] = 0
                delta[k] = 0
            end
        end
        for (k, v) in delta
            push!(renormalisedParams[k], latestDict[k] + v)
            if renormalisedParams[k][end] * renormalisedParams[k][end-1] < 0
                renormalisedParams[k][end] = 0
            end
        end
        push!(renormalisedParams["bw"], latestDict["bw"] - abs(ΔD))

        if all(==(0), (renormalisedParams["Jf"][end], renormalisedParams["Jd"][end], renormalisedParams["J"][end]))
            break
        end
    end
    return renormalisedParams
end

function getFixedPointData(WfValues, JValues)
    FixedPoint = Dict(k => zeros(length(WfValues) * length(JValues)) for k in ["Jf", "Jd", "J"])
    Bools = Dict(k => zeros(length(WfValues) * length(JValues)) for k in ["Jf", "Jd", "J"])
    @showprogress for (index, (Wf, J)) in enumerate(Iterators.product(WfValues, JValues)) |> collect
        couplingsFlow = rgFlow(
               Dict("bw" => 1.0, "Jf" => 0.1, "Jd" => 0.1, "J" => J, "Wf" => Wf, "Wd" => 0),
               D -> -D,
               2.,
               0.001,
              )
        for k in ["Jf", "Jd", "J"]
            FixedPoint[k][index] = couplingsFlow[k][end]
            Bools[k][index] = ifelse(couplingsFlow[k][end] > couplingsFlow[k][1], 2, 
                                     ifelse(couplingsFlow[k][end] > 0.7 * couplingsFlow[k][1], 1, 0)
                                    )
        end
    end
    return FixedPoint, Bools
end

WfValues = -0:-0.002:-0.15
JValues = 0:0.002:0.15
FixedPoint, Bools = getFixedPointData(WfValues, JValues)
fig, axes = subplots(ncols=3, nrows=2, figsize=(8, 4))
fig.tight_layout()
for (i, key) in enumerate(["Jf", "Jd", "J"])
    title = Dict("Jf" => "\$J_f\$", "Jd" => "\$J_d\$", "J" => "\$J\$")[key]
    data = FixedPoint[key]
    bool = Bools[key]
    ax = axes[1, i]
    hm = ax.imshow(reshape(data, length(WfValues), length(JValues)), origin="lower", extent=(extrema(JValues)..., extrema(-WfValues)...), cmap="inferno", aspect="equal", vmax=minimum((0.4, maximum(data))))
    ax.set_xlabel("\$J\$")
    ax.set_ylabel("\$|Wf|\$")
    ax.set_title("RG flow of $(title)", pad=10)
    yPoints = repeat(-WfValues, outer=length(JValues))
    xPoints = repeat(JValues, inner=length(WfValues))
    fig.colorbar(hm)

    ax = axes[2, i]
    hm = ax.imshow(reshape(bool, length(WfValues), length(JValues)), origin="lower", extent=(extrema(JValues)..., extrema(-WfValues)...), cmap="inferno", aspect="equal", vmin=0, vmax=2)
    ax.set_xlabel("\$J\$")
    ax.set_ylabel("\$|Wf|\$")
    ax.set_title("Class of flow of $(title)", pad=10)
    yPoints = repeat(-WfValues, outer=length(JValues))
    xPoints = repeat(JValues, inner=length(WfValues))
    fig.colorbar(hm)
end
fig.tight_layout()
plt.rcParams["text.usetex"] = true
savefig("bilayerHubbard.pdf", bbox_inches="tight")
