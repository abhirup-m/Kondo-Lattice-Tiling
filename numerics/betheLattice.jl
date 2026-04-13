using PyPlot, ProgressMeter, Serialization, Fermions

RG_TOL = 10^(-5)
PARAMS = Dict(
              "bw" => 1.0,
              "Jf" => 0.1,
              "Jd" => 0.1,
              "Wd" => -0.0,
              "Vf" => 0.,
              "Vd" => 0.,
              "Uf" => 8.,
              "Ud" => 2.,
              "ηf" => 0.,
              "ηd" => 0.,
              "hop_t" => 0.,
             )
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
    )
    scale = bareParams["scale"]
    renormalisedParams = Dict(k => [v] for (k,v) in bareParams)
    dos(E) = 2 * √(bareParams["bw"]^2 - E^2) / (π * bareParams["bw"]^2)
    initDenominators = getDenominators(bareParams, ω)
    flags = Dict(k => 1 for k in keys(initDenominators))
    initSigns = Dict(k => sign(v) for (k,v) in initDenominators)
    deltaSigns = Dict{String, Any}(g => nothing for g in ("Jf", "Jd", "Kf", "Kd", "J⟂"))
    totRenorm = nothing
    while (isnothing(totRenorm) || totRenorm * bareParams["bw"] > 1e-9) && renormalisedParams["bw"][end] > 0
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
    return renormalisedParams
end

function getFixedPointData(WfValues, JValues, Kf, Kd, scale; loadData=true)
    quants = ["Jf", "Jd", "Kf", "Kd", "J⟂"]
    FixedPoint = Dict(k => zeros(length(WfValues) * length(JValues)) for k in quants)
    Bools = Dict(k => zeros(length(WfValues) * length(JValues)) for k in quants)
    @showprogress Threads.@threads for (index, (Wf, J)) in enumerate(Iterators.product(WfValues, JValues)) |> collect
        params = copy(PARAMS)
        params["Kf"] = Kf
        params["Kd"] = Kd
        params["Wf"] = Wf
        params["scale"] = scale
        savePath = "saveData/RG-BL-" * join(["$v" for v in values(params)], "-")
        if isfile(savePath) && loadData
            data = deserialize(savePath)
            for k in quants
                FixedPoint[k][index] = data[k]
            end
            continue
        end
        couplingsFlow = rgFlow(
                               params,
                               D -> -D/2,
                              )
        for k in quants
            FixedPoint[k][index] = couplingsFlow[k][end]
        end
        serialize(savePath, Dict(k => couplingsFlow[k][end] for k in quants))
    end
    return FixedPoint, Bools
end

function BilayerLEEReal(
        J::Matrix{Float64},
        Jp::Float64,
        hybrid::Vector{Float64},
        η::Dict{String, Float64},
        impCorr::Dict{String, Float64},
        hop_t::Union{Float64, Dict{String,Float64}},
        layerSpecs::Vector{String},
        hop_step::Dict;
        globalField::Union{Vector{Float64}, Float64}=0.,
        couplingTolerance::Number=1e-15,
        heisenberg::Dict{String, Vector{Float64}}=Dict{String, Vector{Float64}}(),
    )
    #=@assert layerSpecs[1] ≠ layerSpecs[end]=#
    @assert size(J)[1] == size(J)[2] == length(layerSpecs) == length(hybrid)

    if isa(hop_t, Float64)
        hop_t = Dict("f" => hop_t, "d" => hop_t)
    end
    if isa(globalField, Vector)
        @assert length(globalField) == length(layerSpecs) + 2
    else
        globalField = repeat([globalField], length(layerSpecs) + 2)
    end

    #### Indexing convention ####
    # Sf   Sd   γ1   γ2  ...
    # 1,2, 3,4, 5,6, 7,8 ...
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]
    if abs(Jp) > couplingTolerance
        push!(hamiltonian,
              ("nn",  [1, 3], Jp / 4)
             ) # n_{d up, n_{0 up}
        push!(hamiltonian,
              ("nn",  [1, 4], -Jp / 4)
             ) # n_{d up, n_{0 down}
        push!(hamiltonian,
              ("nn",  [2, 3], -Jp / 4)
             ) # n_{d down, n_{0 up}
        push!(hamiltonian,
              ("nn",  [2, 4], Jp / 4)
             ) # n_{d down, n_{0 down}
        push!(hamiltonian,
              ("+-+-",  [1, 2, 4, 3], Jp / 2)
             ) # S_d^+ S_0^-
        push!(hamiltonian,
              ("+-+-",  [2, 1, 3, 4], Jp / 2)
             ) # S_d^- S_0^+
    end

    # kondo terms
    for (i, J_i) in enumerate(eachrow(J))
        bath_i = 3 + 2 * i
        if layerSpecs[i] == "f"
            imp = 1
        else
            imp = 3
        end
        for (j, J_ij) in enumerate(J_i)
            if layerSpecs[i] ≠ layerSpecs[j]
                continue
            end
            #=if i ≠ j=#
            #=    @assert J_ij == 0=#
            #=end=#
            bath_j = 3 + 2 * j
            if abs(J_ij) > couplingTolerance
                push!(hamiltonian, ("n+-",  [imp, bath_i, bath_j], J_ij / 4))
                push!(hamiltonian, ("n+-",  [imp, bath_i + 1, bath_j + 1], -J_ij / 4))
                push!(hamiltonian, ("n+-",  [imp + 1, bath_i, bath_j], -J_ij / 4))
                push!(hamiltonian, ("n+-",  [imp + 1, bath_i + 1, bath_j + 1], J_ij / 4))
                push!(hamiltonian, ("+-+-",  [imp, imp + 1, bath_i + 1, bath_j], J_ij / 2))
                push!(hamiltonian, ("+-+-",  [imp + 1, imp, bath_i, bath_j + 1], J_ij / 2))
            end
        end
        if abs(hybrid[i]) > couplingTolerance
            push!(hamiltonian, ("+-",  [imp, bath_i], hybrid[i])) # n_{d up, n_{0 up}
            push!(hamiltonian, ("+-",  [imp + 1, bath_i + 1], hybrid[i])) # n_{d up, n_{0 up}
            push!(hamiltonian, ("+-",  [bath_i, imp], hybrid[i])) # n_{d up, n_{0 up}
            push!(hamiltonian, ("+-",  [bath_i + 1, imp + 1], hybrid[i])) # n_{d up, n_{0 up}
        end
    end

    for (i, t) in enumerate(layerSpecs)
        bath_i = 3 + 2 * i
        j = findnext(==(t), layerSpecs, i+1)
        if !isnothing(j)
            bath_j = 3 + 2 * j
            push!(hamiltonian, ("+-",  [bath_i, bath_j], -hop_t[t]))
            push!(hamiltonian, ("+-",  [bath_j, bath_i], -hop_t[t]))
            push!(hamiltonian, ("+-",  [bath_i + 1, bath_j + 1], -hop_t[t]))
            push!(hamiltonian, ("+-",  [bath_j + 1, bath_i + 1], -hop_t[t]))
        end
    end

    for (site, k) in zip([1, 3], ["f", "d"])
        if abs(η[k]) > couplingTolerance
            push!(hamiltonian, ("n",  [site], -η[k]))
            push!(hamiltonian, ("n",  [site + 1], -η[k]))
        end
    end

    if abs(impCorr["f"]) > couplingTolerance
        push!(hamiltonian, ("nn",  [1, 2], impCorr["f"]))
        push!(hamiltonian, ("n",  [1], -0.5 * impCorr["f"]))
        push!(hamiltonian, ("n",  [2], -0.5 * impCorr["f"]))
    end
    if abs(impCorr["d"]) > couplingTolerance
        push!(hamiltonian, ("nn",  [3, 4], impCorr["d"]))
        push!(hamiltonian, ("n",  [3], -0.5 * impCorr["d"]))
        push!(hamiltonian, ("n",  [4], -0.5 * impCorr["d"]))
    end

    # global magnetic field (to lift any trivial degeneracy)
    for site in 1:(2 + length(layerSpecs))
        if abs(globalField[site]) > couplingTolerance
            push!(hamiltonian, ("n",  [2 * site - 1], globalField[site]/2))
            push!(hamiltonian, ("n",  [2 * site], -globalField[site]/2))
        end
    end
    if length(heisenberg) > 0
        for (t, imp) in zip(["f", "d"], [1, 2])
            sites = [[imp]; 2 .+ findall(==(t), layerSpecs)[1:end-1]]
            for i in 1:(length(sites)-1)
                if abs(heisenberg[t][i]) < couplingTolerance
                    continue
                end
                j = 2 * sites[i] - 1
                k = 2 * sites[i+1] - 1
                push!(hamiltonian, ("nn",  [j, k], heisenberg[t][sites[i]]/4))
                push!(hamiltonian, ("nn",  [j, k + 1], -heisenberg[t][sites[i]]/4))
                push!(hamiltonian, ("nn",  [j + 1, k], -heisenberg[t][sites[i]]/4))
                push!(hamiltonian, ("nn",  [j + 1, k + 1], heisenberg[t][sites[i]]/4))
                push!(hamiltonian, ("+-+-",  [j, j + 1, k + 1, k], heisenberg[t][sites[i]]/2))
                push!(hamiltonian, ("+-+-",  [j + 1, j, k, k + 1], heisenberg[t][sites[i]]/2))
            end
        end
    end

    @assert !isempty(hamiltonian) "Hamiltonian is empty!"

    return hamiltonian
end


function RealCorr(
        Kf,
        Kd,
        Wf,
        Jp,
        scale,
        size,
        correlationDef,
        mutInfoDef,
        states;
        loadData=false,
    )

    params = copy(PARAMS)
    params["Kf"] = Kf
    params["Kd"] = Kd
    params["Wf"] = Wf
    params["J⟂"] = Jp
    params["scale"] = scale
    savePath = "saveData/RG-BL-$(size)-$(states)-" * join(["$(params[k])" for k in sort(keys(params))], "-")

    if loadData && isfile(savePath)
        return deserialize(savePath)
    end
    couplingsFlow = rgFlow(
                           params,
                           D -> -D/2,
                          )
    # ordering:
    # f  d  f1  d1  f2  d2 ...
    layerSpecs = repeat(["f", "d"], size)
    inplaneKondo = zeros(size * 2, size * 2)
    inplaneKondo[1, 1] = couplingsFlow["Jf"][end]
    inplaneKondo[2, 2] = couplingsFlow["Jd"][end]
    Jp = couplingsFlow["J⟂"][end]
    hybrid = zeros(size * 2)
    η = Dict("f" => params["ηf"], "d" => params["ηd"])
    impCorr = Dict("f" => params["Uf"], "d" => params["Ud"])
    hop_t = params["hop_t"]
    hop_step = Dict("f" => 1., "d" => 1.)

    hamiltonian = BilayerLEEReal(
                                 inplaneKondo,
                                 Jp,
                                 hybrid,
                                 η,
                                 impCorr,
                                 hop_t,
                                 layerSpecs,
                                 hop_step;
                                )
    hamiltonianFamily = MinceHamiltonian(hamiltonian, 8:1:2*(2 + length(layerSpecs)))
    results = IterDiag(
                      hamiltonianFamily,
                      states;
                      symmetries=Char['N', 'S'],
                      correlationDefDict=copy(correlationDef),
                      mutInfoDefDict=copy(mutInfoDef),
                      silent=false,
                      maxMaxSize=states,
                     )
    impResults = Dict(k => v for (k,v) in results if haskey(correlationDef, k) || haskey(mutInfoDef, k))
    serialize(savePath, impResults)
    return impResults
end


function RealCorrRunner()
    Kf = -1e-5
    Kd = -1e-5
    Wf = -0.1
    scale = 0.9999
    size = 30
    states = 1000
    Jp_values = 10 .^ (-2:0.1:-0.5)
    correlationDef = Dict("Sf.Sd" => SpinCorr(1, 3), "Sf.sf" => SpinCorr(1, 5), "Sd.sd" => SpinCorr(3, 7))
    mutInfoDef = Dict("fd" => ([1, 2], [3, 4]), "Ff" => ([1, 2], [5, 6]))
    corrResultsArr = Dict(k => 0 .* Jp_values for k in keys(correlationDef))
    mutInfoResultsArr = Dict(k => 0 .* Jp_values for k in keys(mutInfoDef))
    fig, ax = plt.subplots()
    @time Threads.@threads for (i, Jp) in collect(enumerate(Jp_values))
        results = RealCorr(
                     Kf,
                     Kd,
                     Wf,
                     Jp,
                     scale,
                     size,
                     correlationDef,
                     mutInfoDef,
                     states;
                     loadData=true
            )
        for k in keys(correlationDef)
            corrResultsArr[k][i] = -results[k]
        end
        for k in keys(mutInfoDef)
            mutInfoResultsArr[k][i] = results[k]
        end
    end
    ax.plot(Jp_values, corrResultsArr["Sf.Sd"], label="-<Sf.Sd>")
    ax.plot(Jp_values, corrResultsArr["Sf.sf"], label="-<Sf.sf>")
    ax.plot(Jp_values, corrResultsArr["Sd.sd"], label="-<Sd.sd>")
    ax.plot(Jp_values, mutInfoResultsArr["fd"], label="I2(f:d)")
    ax.plot(Jp_values, mutInfoResultsArr["Ff"], label="I2(f:f0)")
    ax.set_xlabel("J_perp")
    ax.set_ylabel("real-space correlation")
    ax.set_xscale("log")
    fig.savefig("BL-RC-$(size)-$(states).pdf")
end
RealCorrRunner()


function RGrunner()
    WfValues = -0.0:-0.02:-0.5
    JValues = 0.001:0.02:0.5
    titleDict = Dict("Jf" => "\$J_f\$", "Jd" => "\$J_d\$", "Kf" => "\$-K_f\$", "Kd" => "\$-K_d\$", "J⟂" => "\$J_\\perp\$")
    plottables = ["Jf", "Jd", "J⟂"]
    #=plottables = ["Kf", "Kd",]=#
    #=for (j, (Kf, Kd)) in enumerate([(-1e-4, -1e-4)])=#
    for (j, (Kf, Kd)) in enumerate([(0., 0.), (-1e-5, -1e-5)])
        fig, axes = subplots(ncols=length(plottables), figsize=(2.6 * length(plottables), 2))
        FixedPoint, Bools = getFixedPointData(WfValues, JValues, Kf, Kd, 0.99995; loadData=false)
        for (i, key) in enumerate(plottables)
            title = titleDict[key]
            #=data = FixedPoint[key]=#
            data = abs.(FixedPoint[key])
            #=data[sortperm(data)[end]] += 1e-6=#
            bool = Bools[key]
            ax = axes[i]
            hm = ax.imshow(reshape(data, length(WfValues), length(JValues)), origin="lower", extent=(extrema(JValues)..., extrema(WfValues)...), cmap="inferno", aspect="auto")#, norm=matplotlib.colors.LogNorm())
            ax.set_xlabel("\$J_\\perp\$")
            ax.set_ylabel("\$Wf\$")
            ax.set_title("Fixed-point $(title)", pad=10)
            yPoints = repeat(-WfValues, outer=length(JValues))
            xPoints = repeat(JValues, inner=length(WfValues))
            fig.colorbar(hm)
        end
        fig.tight_layout()
        plt.rcParams["text.usetex"] = true
        savefig("bilayerHubbard_$(j).pdf", bbox_inches="tight")
    end
end
