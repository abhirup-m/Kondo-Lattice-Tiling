using PyPlot

function deltaJf(ω, params, dos, initSign)
    denominator = ω(params["bw"]) - params["bw"] / 2 + params["Jf"] / 4 + params["Wf"] / 2
    if sign(denominator) == initSign
        return -(params["Jf"]^2 + 4 * params["Jf"] * params["Wf"]) * dos / denominator
    else
        return 0
    end
end
function deltaJd(ω, params, dos, initSign)
    denominator = ω(params["bw"]) - params["bw"] / 2 + params["Jd"] / 4
    if sign(denominator) == initSign
        return -params["Jd"]^2 * dos / denominator
    else
        return 0
    end
end
function deltaJ(ω, params, dos, initSign)
    denominators =[ω(params["bw"]) - params["bw"] / 2 + params["Jf"] / 4 + params["Wf"] / 2 - params["J"] / 4, ω(params["bw"]) - params["bw"] / 2 + params["Jd"] / 4 - params["J"] / 4]
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
    initSigns = sign.([ω(bareParams["bw"]) - bareParams["bw"] / 2 + bareParams["Jf"] / 4 + bareParams["Wf"] / 2, 
                       ω(bareParams["bw"]) - bareParams["bw"] / 2 + bareParams["Jd"] / 4,
                       ω(bareParams["bw"]) - bareParams["bw"] / 2 + bareParams["Jf"] / 4 + bareParams["Wf"] / 2 - bareParams["J"] / 4,
                       ω(bareParams["bw"]) - bareParams["bw"] / 2 + bareParams["Jd"] / 4 - bareParams["J"] / 4
                      ])
    deltaSigns = Dict{String, Any}(g => nothing for g in ("Jf", "Jd", "J"))
    while renormalisedParams["bw"][end] > 0
        latestDict = LatestDict(renormalisedParams)
        delta = Dict()
        delta["Jf"] = (deltaSigns["Jf"] ≠ 0 && latestDict["Jf"] ≠ 0) ? abs(ΔD) * deltaJf(ω, latestDict, dos, initSigns[1]) : 0
        delta["Jd"] = (deltaSigns["Jd"] ≠ 0 && latestDict["Jd"] ≠ 0) ? abs(ΔD) * deltaJd(ω, latestDict, dos, initSigns[2]) : 0
        delta["J"] = (deltaSigns["J"] ≠ 0 && latestDict["J"] ≠ 0) ? abs(ΔD) * deltaJ(ω, latestDict, dos, initSigns[3:4]) : 0
        #=println(delta)=#
        for (k, s) in deltaSigns
            if isnothing(s)
                deltaSigns[k] = sign(delta[k])
            elseif s * delta[k] < 0
                deltaSigns[k] = 0
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

WfValues = -0:-0.005:-0.2
JValues = 0:0.005:0.2
JfFixedPoint = []
JFixedPoint = []
for J in JValues
    for Wf in WfValues
        couplingsFlow = rgFlow(
               Dict("bw" => 1.0, "Jf" => 0.1, "Jd" => 0.1, "J" => J, "Wf" => Wf),
               D -> -D,
               1.,
               0.001,
              )
        push!(JfFixedPoint, couplingsFlow["Jf"][end])
        push!(JFixedPoint, couplingsFlow["J"][end])
    end
end
fig, ax = subplots(figsize=(7, 4))
fig.tight_layout()
hm = ax.imshow(reshape(JfFixedPoint, length(WfValues), length(JValues)), origin="lower", extent=(extrema(JValues)..., extrema(-WfValues)...), cmap="inferno", aspect="equal")
ax.set_xlabel("J")
ax.set_ylabel("-Wf")
yPoints = repeat(-WfValues, outer=length(JValues))
xPoints = repeat(JValues, inner=length(WfValues))
#=println(JFixedPoint)=#
s = ax.scatter(xPoints, yPoints, c=JFixedPoint, s=5)
fig.colorbar(hm, location="left", label="HM: J_f", shrink=0.5)
fig.colorbar(s, label="SC: J", shrink=0.5)
savefig("bilayerHubbard.pdf", bbox_inches="tight")

#=plot!(p, WfValues, JfFixedPoint, label="Jf")=#
#=plot!(p, WfValues, JFixedPoint, label="J")=#
#=plot!(p, WfValues, JFixedPoint, label="J")=#
#=display(p)=#
