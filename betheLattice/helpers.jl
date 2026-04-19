function SpinCorr(i, j)
    spinCorr = Tuple{String, Vector{Int64}, Float64}[]
    push!(spinCorr, ("nn", [i, j], 0.25))
    push!(spinCorr, ("nn", [i, j+1], -0.25))
    push!(spinCorr, ("nn", [i+1, j], -0.25))
    push!(spinCorr, ("nn", [i+1, j+1], 0.25))
    push!(spinCorr, ("+-+-", [i, i+1, j+1, j], 0.5))
    push!(spinCorr, ("+-+-", [i+1, i, j, j+1], 0.5))
    return spinCorr
end

function Norm(A, ω)
    δω = ω[2] - ω[1]
    norm = sum(A) * δω
    return A / norm
end

function getDenominators(params, ω)
    denominators = Dict()
    for (k, kbar) in zip(["d", "f"], ["f", "d"])
        denominators["J"*k*"+"] = ω(params["bw"]) - params["bw"] / 2 + params["J$(k)"] / 4 + params["W$(k)"] / 2 + 3 * params["J⟂"] / 4
        denominators["J"*k*"-"] = denominators["J"*k*"+"] - 3 * params["J⟂"] / 4 - params["J⟂"] / 4
        denominators["K"*k*"+"] = ω(params["bw"]) - params["bw"] / 2 + params["K$(kbar)"] / 4 + params["W$(k)"] / 2 + 3 * params["J⟂"] / 4
        denominators["K"*k*"-"] = denominators["K"*k*"+"] - 3 * params["J⟂"] / 4 - params["J⟂"] / 4
        denominators["JK"*k] = ω(params["bw"]) - params["bw"] / 2 + params["J$(k)"] / 4 + params["K$(kbar)"] / 4 + params["W$(k)"] / 2 - params["J⟂"] / 4
        denominators["J"*k*k] = ω(params["bw"]) - params["bw"] + params["J$(k)"] / 2 + params["W$(k)"] - params["J⟂"] / 4
        denominators["K"*k*k] = ω(params["bw"]) - params["bw"] + params["K$(kbar)"] / 2 + params["W$(k)"] - params["J⟂"] / 4
    end
    return denominators
end

function deltaJfd(ω, params, initDenominators, key, flags)
    @assert key ∈ "fd"
    bar(k) = k == "d" ? "f" : "d"
    denominators = getDenominators(params, ω)
    for k in keys(denominators)
        flags[k] *= sign(initDenominators[k]) * sign(denominators[k]) ≤ 0 || abs(denominators[k] / initDenominators[k]) < RG_TOL ? 0 : 1
    end
    deltaJ = -(params["J"*key]^2 + 4 * params["J"*key] * params["W"*key]) * (0.25 * flags["J$(key)+"] / denominators["J$(key)+"] + 0.75 * flags["J$(key)-"] / denominators["J$(key)-"])
    deltaJ += 0.5 * (params["K"*bar(key)]^2 - 4 * params["K"*bar(key)] * params["W"*key]) * (flags["K$(key)+"] / denominators["K$(key)+"] + flags["K$(key)-"] / denominators["K$(key)-"])
    return deltaJ
end

function deltaKfd(ω, params, initSign, key, flags)
    @assert key ∈ "fd"
    bar(k) = k == "d" ? "f" : "d"
    denominators = getDenominators(params, ω)
    for k in keys(denominators)
        flags[k] *= initSign[k] == sign(denominators[k]) ? 1 : 0
    end
    deltaK = -(params["K"*key]^2 + 4 * params["K"*key] * params["W"*bar(key)]) * (0.25 * flags["K$(bar(key))+"] / denominators["K$(bar(key))+"] + 0.75 * flags["K$(bar(key))-"] / denominators["K$(bar(key))-"])
    deltaK += 0.5 * (params["J"*bar(key)]^2 - 4 * params["J"*bar(key)] * params["W"*bar(key)]) * (flags["J$(bar(key))+"] / denominators["J$(bar(key))+"] + flags["J$(bar(key))-"] / denominators["J$(bar(key))-"])
    return deltaK
end

function deltaJperp(ω, params, initSign, flags)
    denominators = getDenominators(params, ω)
    bar(k) = k == "d" ? "f" : "d"
    for k in keys(denominators)
        flags[k] *= initSign[k] == sign(denominators[k]) ? 1 : 0
    end
    deltaJperp = 0
    deltaJperp += -0.25 * sum(params["J$(k)"]^2 * (flags["J$(k)+"] / denominators["J$(k)+"] + flags["J$(k)-"] / denominators["J$(k)-"]) for k in ["d", "f"])
    deltaJperp += -0.25 * sum(params["K$(k)"]^2 * (flags["K$(bar(k))+"] / denominators["K$(bar(k))+"] + flags["K$(bar(k))-"] / denominators["K$(bar(k))-"]) for k in ["d", "f"])
    deltaJperp += sum(params["J$(k)"] * params["K$(bar(k))"] * flags["JK$(k)"] / denominators["JK$(k)"] for k in ["d", "f"])
    deltaJperp += 0.5 * params["Jf"] * params["Jd"] * sum(flags["J$(k)$(k)"] / denominators["J$(k)$(k)"] for k in ["d", "f"])
    deltaJperp += 0.5 * params["Kf"] * params["Kd"] * sum(flags["K$(k)$(k)"] / denominators["K$(k)$(k)"] for k in ["d", "f"])
    return deltaJperp
end

function rgFlow(
        bareParams,
        ω;
        loadData=false
    )
    savePath = "saveData/RG-BL-$(hash(bareParams))"
    if isfile(savePath) && loadData
        return deserialize(savePath)
    end
    scale = bareParams["scale"]
    renormalisedParams = Dict(k => [v] for (k,v) in bareParams)
    #=dos(E) = 1.0=#
    dos(E) = 2 * √(bareParams["bw"]^2 - E^2) / (π * bareParams["bw"]^2)
    initDenominators = getDenominators(bareParams, ω)
    flags = Dict(k => 1 for k in keys(initDenominators))
    initSigns = Dict(k => sign(v) for (k,v) in initDenominators)
    deltaSigns = Dict{String, Any}(g => nothing for g in ("Jf", "Jd", "Kf", "Kd", "J⟂"))
    totRenorm = nothing
    while (isnothing(totRenorm) || totRenorm / (bareParams["Jd"]^2 / bareParams["bw"]) > 1e-9) && renormalisedParams["bw"][end] > 0
        latestDict = Dict(k => v[end] for (k, v) in renormalisedParams)
        delta = Dict()
        ΔD = latestDict["bw"] * (1 - scale)
        #=ΔD = 1e-5=#
        delta["Jf"] = (deltaSigns["Jf"] ≠ 0 && latestDict["Jf"] ≠ 0) ? abs(ΔD) * dos(latestDict["bw"] - abs(ΔD)) * deltaJfd(ω, latestDict, initDenominators, 'f', flags) : 0
        delta["Jd"] = (deltaSigns["Jd"] ≠ 0 && latestDict["Jd"] ≠ 0) ? abs(ΔD) * dos(latestDict["bw"] - abs(ΔD)) * deltaJfd(ω, latestDict, initDenominators, 'd', flags) : 0
        delta["Kf"] = (deltaSigns["Kf"] ≠ 0 && latestDict["Kf"] ≠ 0) ? abs(ΔD) * dos(latestDict["bw"] - abs(ΔD)) * deltaKfd(ω, latestDict, initSigns, 'f', flags) : 0
        delta["Kd"] = (deltaSigns["Kd"] ≠ 0 && latestDict["Kd"] ≠ 0) ? abs(ΔD) * dos(latestDict["bw"] - abs(ΔD)) * deltaKfd(ω, latestDict, initSigns, 'd', flags) : 0
        delta["J⟂"] = (deltaSigns["J⟂"] ≠ 0 && latestDict["J⟂"] ≠ 0) ? abs(ΔD) * dos(latestDict["bw"] - abs(ΔD)) * deltaJperp(ω, latestDict, initSigns, flags) : 0
        for (k, s) in deltaSigns
            if isnothing(s)
                deltaSigns[k] = sign(delta[k])
            elseif s * delta[k] ≤ 0
                deltaSigns[k] = 0
                delta[k] = 0
            end
        end
        totRenorm = √(sum(values(delta).^2) / sum(values(bareParams).^2))
        #=println((totRenorm, ΔD))=#
        for (k, v) in delta
            push!(renormalisedParams[k], latestDict[k] + v)
            if renormalisedParams[k][end] * renormalisedParams[k][end-1] < 0
                renormalisedParams[k][end] = 0
            end
        end
        #=push!(renormalisedParams["bw"], latestDict["bw"] * scale)=#
        push!(renormalisedParams["bw"], latestDict["bw"] - abs(ΔD))

        if all(==(0), (renormalisedParams["Jf"][end], renormalisedParams["Jd"][end], renormalisedParams["J⟂"][end]))
            break
        end
    end
    serialize(savePath, renormalisedParams)
    return renormalisedParams
end

function DefineSpecFunc(fFactor, dFactor, perpFactor)
    specFunc = Dict{String, Dict{String, Vector{Tuple{String, Vector{Int64}, Float64}}}}()
    specFunc["Aff"] = Dict("create" => [("+", [1], 1.0), ("+", [2], 1.0)])
    specFunc["Add"] = Dict("create" => [("+", [3], 1.0), ("+", [4], 1.0)])
    specFunc["Af0"] = Dict("create" => [("+-+", [1, 2, 6], 0.5 * fFactor), ("+-+", [2, 1, 5], 0.5 * fFactor)])
    specFunc["Ad0"] = Dict("create" => [("+-+", [3, 4, 8], 0.5 * dFactor), ("+-+", [4, 3, 7], 0.5 * dFactor)])
    specFunc["Afd"] = Dict("create" => [("+-+", [1, 2, 4], 0.5 * perpFactor), ("+-+", [2, 1, 3], 0.5 * perpFactor)])
    for k in ["Aff", "Add", "Af0", "Ad0", "Afd"]
        specFunc[k]["destroy"] = Dagger(copy(specFunc[k]["create"]))
    end
    return specFunc
end
