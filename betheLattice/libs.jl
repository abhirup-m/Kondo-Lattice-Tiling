@everywhere using ProgressMeter, Serialization, Fermions
@everywhere include("models.jl")
@everywhere include("helpers.jl")

function getFixedPointData(WfValues, JValues)
    quants = ["Jf", "Jd", "Kf", "Kd", "J⟂"]
    FixedPoint = Dict(k => zeros(length(WfValues) * length(JValues)) for k in quants)
    bareParams = Dict(k => zeros(length(WfValues) * length(JValues)) for k in quants)
    @showprogress Threads.@threads for (index, (Wf, J)) in enumerate(Iterators.product(WfValues, JValues)) |> collect
        bareParams["Jf"][index] = PARAMS["Jf"]
        bareParams["Jd"][index] = PARAMS["Jd"]
        bareParams["J⟂"][index] = maximum((J, 1e-5))
        params = copy(PARAMS)
        params["J⟂"] = J
        params["Wf"] = Wf
        params["Kf"] = -1e-4 * J
        params["Kd"] = -1e-4 * J
        for g in ("bw", "Jf", "Jd", "Kf", "Kd", "J⟂", "Wf", "Wd")
            params[g] *= params["upFactor"]
        end
        bareParams = filter(p -> p[1] ∈ ("Jf", "Jd", "Kf", "Kd", "Wf", "Wd", "J⟂", "bw", "scale"), params)
        couplingsFlow = rgFlow(
                               bareParams,
                               D -> - D/2,
                              )
        for k in quants
            FixedPoint[k][index] = couplingsFlow[k][end] / params["upFactor"]
        end
    end
    return FixedPoint, bareParams
end


@everywhere function RealCorr(
        params,
        correlationDef,
        mutInfoDef;
        loadData=false,
    )

    for (k, v) in filter(p -> !haskey(params, p[1]), PARAMS)
        params[k] = v
    end
    #=params = copy(PARAMS)=#
    #=params["Kf"] = Kf=#
    #=params["Kd"] = Kd=#
    #=params["Wf"] = Wf=#
    #=params["J⟂"] = Jp=#
    savePath = "saveData/RC-BL-$(hash(params))"

    if loadData && isfile(savePath)
        return deserialize(savePath)
    end
    # @assert false

    bareParams = filter(p -> p[1] ∈ ("Jf", "Jd", "Kf", "Kd", "J⟂",  "Wf", "Wd", "bw", "scale"), params)
    couplingsFlow = rgFlow(
                           bareParams,
                           D -> -D/2;
                           loadData=false,
                          )
    size = params["size"]
    states = params["states"]
    # ordering:
    # f  d  f1  d1  f2  d2 ...
    # if couplingsFlow["Jf"][end] ≥ couplingsFlow["Jf"][1]
    #     layerSpecs = repeat(["f", "d"], size)
    #     inplaneKondo = zeros(size * 2, size * 2)
    #     indirectKondo = zeros(size * 2, size * 2)
    #     hybrid = zeros(size * 2)
    # else
    #     layerSpecs = vcat(repeat(["f"], 3), repeat(["d"], size))
    #     inplaneKondo = zeros(size + 3, size + 3)
    #     indirectKondo = zeros(size + 3, size + 3)
    #     hybrid = zeros(size + 3)
    # end
    layerSpecs = repeat(["f", "d"], size)
    inplaneKondo = zeros(size * 2, size * 2)
    indirectKondo = zeros(size * 2, size * 2)
    hybrid = zeros(size * 2)
    inplaneKondo[1, 1] = couplingsFlow["Jf"][end]
    inplaneKondo[2, 2] = couplingsFlow["Jd"][end]
    indirectKondo[1, 1] = couplingsFlow["Kd"][end]
    indirectKondo[2, 2] = couplingsFlow["Kf"][end]
    Jp = couplingsFlow["J⟂"][end]
    hybrid[1] = (0.5 * couplingsFlow["Jf"][end] * params["Uf"])^0.5
    hybrid[2] = (0.5 * couplingsFlow["Jd"][end] * params["Ud"])^0.5
    η = Dict("f" => params["ηf"], "d" => params["ηd"])
    impCorr = Dict("f" => params["Uf"], "d" => params["Ud"])
    hop_t = Dict("f" => params["hop_t"], "d" => params["hop_t"])
    hop_step = Dict("f" => 1., "d" => 1.)
    heisenberg = zeros(size)

    if couplingsFlow["Jf"][end] < couplingsFlow["Jf"][1]
        hop_t["f"] *= abs(couplingsFlow["Jf"][end] / couplingsFlow["Jf"][1])
        hybrid[1] *= abs(couplingsFlow["Jf"][end] / couplingsFlow["Jf"][1])
        indirectKondo[1:2, 1:2] .= 0.
        heisenberg .= params["Jf"] * (1 - abs(couplingsFlow["Jf"][end] / couplingsFlow["Jf"][1]))
        # heisenberg[1] = 8 * params["Vf"]^2 / params["Uf"] * clamp((couplingsFlow["bw"][1] / couplingsFlow["bw"][end])^1.0, 0., 3.0)
    end
    # hybrid[1] = (couplingsFlow["Jf"][end] * params["Uf"])^0.5
    # hybrid[2] = (couplingsFlow["Jd"][end] * params["Ud"])^0.5
    # η = Dict("f" => params["ηf"], "d" => params["ηd"])
    # impCorr = Dict("f" => params["Uf"], "d" => params["Ud"])
    # hop_t = Dict("f" => params["hop_t"], "d" => params["hop_t"])
    # hop_step = Dict("f" => 1., "d" => 1.)
    # heisenberg = zeros(size)
    # if couplingsFlow["Jf"][end] < couplingsFlow["Jf"][1]
    #     hop_t["f"] *= abs(couplingsFlow["bw"][end] / couplingsFlow["bw"][1])
    #     # indirectKondo[1:2, 1:2] .= 0.
    #     heisenberg[1] = 8 * params["Vf"]^2 / params["Uf"] * clamp((couplingsFlow["bw"][1] / couplingsFlow["bw"][end])^1.0, 0., 3.0)
    #     inplaneKondo[1, 1] = 0.
    #     hybrid[1] = 0
    # end

    hamiltonian = BilayerLEEReal(
                                 inplaneKondo,
                                 indirectKondo,
                                 Jp,
                                 hybrid,
                                 η,
                                 impCorr,
                                 hop_t,
                                 layerSpecs,
                                 hop_step;
                                 heisenberg=heisenberg,
                                )
    hamiltonianFamily = MinceHamiltonian(hamiltonian, 8:2:2*(2 + length(layerSpecs)))
    sites = Dict("f" => 3 .+ 2 .* findall(==("f"), layerSpecs), "d" => 3 .+ 2 .* findall(==("d"), layerSpecs))
    if length(mutInfoDef) > 0
        mutInfoDef = Dict(k => v(sites) for (k,v) in mutInfoDef)
    else
        mutInfoDef = Dict{String, NTuple{2, Vector{Int64}}}()
    end
    if length(correlationDef) == 0
        correlationDefDict = Dict{String, Vector{Tuple{String, Vector{Int64}, Float64}}}()
    else
        correlationDefDict = copy(correlationDef)
    end
    results = IterDiag(
                      hamiltonianFamily,
                      states;
                      symmetries=Char['N', 'S'],
                      correlationDefDict=correlationDefDict,
                      mutInfoDefDict=copy(mutInfoDef),
                      silent=true,
                      maxMaxSize=states,
                     )
    impResults = Dict(k => v for (k,v) in results if haskey(correlationDef, k) || haskey(mutInfoDef, k))
    if isfile(savePath)
        impResults = merge(deserialize(savePath), impResults)
    end
    serialize(savePath, impResults)
    return impResults
end


function RealSpecFunc(
        params,
        ω,
        σ,
        height,
        heightTol;
        loadData=false,
    )
    size = Int(params["size"])
    states = Int(params["states"])

    #=println(couplingsFlow["Jd"][end-10:end])=#
    # ordering:
    # f  d  f1  d1  f2  d2 ...
    layerSpecs = repeat(["f", "d"], size)
    inplaneKondo = zeros(size * 2, size * 2)
    indirectKondo = zeros(size * 2, size * 2)
    specFunc = DefineSpecFunc()
    for (k, v) in filter(p -> !haskey(params, p[1]), PARAMS)
        params[k] = v
    end

    bareParams = filter(p -> p[1] ∈ ("Jf", "Jd", "Kf", "Kd", "Wf", "Wd", "J⟂", "bw", "scale"), params)
    couplingsFlow = rgFlow(
                           bareParams,
                           D -> -D/2,
                          )
    overallFactors = Dict(t => couplingsFlow[t][end]^2 / couplingsFlow["Jf"][1]^2 - 1 for t in ["Jf", "Jd", "J⟂"])

    SFresults = Dict(k => zeros(length(ω)) for k in keys(specFunc))
    accumCoeffs = @showprogress @distributed (d1, d2) -> merge(vcat, d1, d2) for factor in 10 .^ (1.3:-0.05:-1.3)
        factorParams = copy(params)
        for k in ["Jf", "Jd", "Kf", "Kd", "Wf", "Wd", "J⟂", "bw"]
            factorParams[k] *= factor
        end
        savePath = "saveData/SF-BL-$(hash(factorParams))"

        collectedResults = nothing
        iterCoeffs = Dict(k => Vector{Tuple{Float64, Float64}}[] for k in keys(specFunc))
        if loadData && isfile(savePath)
            iterCoeffs = deserialize(savePath)
        else
            # @assert false
            bareParams = filter(p -> p[1] ∈ ("Jf", "Jd", "Kf", "Kd", "Wf", "Wd", "J⟂", "bw", "scale"), factorParams)
            couplingsFlow = rgFlow(
                                   factorParams,
                                   D -> -D/2,
                                  )
            steps = unique(trunc.(Int, 2 .^ (0:1.0:log2(length(couplingsFlow["bw"])))))
            overallFactors = Dict(t => couplingsFlow[t][end]^2 / couplingsFlow["Jf"][1]^2 - 1 for t in ["Jf", "Jd", "J⟂"])
            if isnan(overallFactors["J⟂"])# || (perpFactor < fFactor && perpFactor < dFactor)
                overallFactors["J⟂"] = 0.
            end
            overallFactors = Dict(k => clamp(v, 0., 1.) for (k,v) in overallFactors)
            #=fFactor = clamp(fFactor, 0., 1.)=#
            #=dFactor = clamp(dFactor, 0., 1.)=#
            #=perpFactor = clamp(perpFactor, 0., 1.)=#
            specFunc = DefineSpecFunc(overallFactors)
            collectedResults = [Dict() for _ in steps]
            Threads.@threads for (i, step) in collect(enumerate(steps))
                inplaneKondo[1, 1] = couplingsFlow["Jf"][step]
                inplaneKondo[2, 2] = couplingsFlow["Jd"][step]
                indirectKondo[1, 1] = couplingsFlow["Kd"][step]
                indirectKondo[2, 2] = couplingsFlow["Kf"][step]
                hybrid = zeros(size * 2)
                hybrid[1] = overallFactors["Jf"] * √(factorParams["Vf"] * couplingsFlow["Jf"][step])
                hybrid[2] = overallFactors["Jd"] * √(factorParams["Vd"] * couplingsFlow["Jd"][step])
                η = Dict("f" => factorParams["ηf"], "d" => factorParams["ηd"])
                impCorr = Dict("f" => factorParams["Uf"], "d" => factorParams["Ud"])
                Jp = couplingsFlow["J⟂"][step] / factor + (couplingsFlow["J⟂"][step] * sum(values(impCorr)) / factor)^0.5
                hop_t = Dict("f" => couplingsFlow["bw"][step], "d" => couplingsFlow["bw"][step])
                hop_step = Dict("f" => 1., "d" => 1.)
                heisenberg = zeros(size)
                if overallFactors["Jf"] < 1.
                    hop_t["f"] *= abs(couplingsFlow["Jf"][end] / couplingsFlow["Jf"][1])
                    indirectKondo[1:2, 1:2] .= 0.
                    heisenberg .= params["Jf"] * (1 - abs(couplingsFlow["Jf"][end] / couplingsFlow["Jf"][1]))
                end
                hamiltonian = BilayerLEEReal(
                                             inplaneKondo,
                                             indirectKondo,
                                             Jp,
                                             hybrid,
                                             η,
                                             impCorr,
                                             hop_t,
                                             layerSpecs,
                                             hop_step;
                                             heisenberg=heisenberg
                                            )
                hamiltonianFamily = MinceHamiltonian(hamiltonian, 8:2:2*(2 + length(layerSpecs)))
                results = IterDiag(
                                  hamiltonianFamily,
                                  states;
                                  symmetries=Char['N', 'S'],
                                  specFuncDefDict=specFunc,
                                  silent=true,
                                  maxMaxSize=states,
                                 )
                for path in results["savePaths"]
                    rm(path; recursive=true, force=true)
                end
                collectedResults[i] = Dict(k => results[k] for k in keys(specFunc))
            end
            for results in collectedResults
                for k in keys(specFunc)
                    append!(iterCoeffs[k], results[k])
                end
            end
            serialize(savePath, iterCoeffs)
        end
        Dict(k => vcat(v...) for (k,v) in iterCoeffs)
    end
    for (k, v) in accumCoeffs
        # v = filter(p -> abs(p[1]) > 1e-3, v)
        if overallFactors["Jf"] ≥ 0 && k == "Af0"
            println((k, overallFactors["Jf"]))
            SFresults[k] = overallFactors["Jf"] * HeightFix(v, ω, height * (1 + overallFactors["Jf"]) / overallFactors["Jf"], heightTol, σ[k])
        else
            if isempty(v)
                SFresults[k] = zeros(length(ω))
            else
                SFresults[k] = Norm(sum(pmap(vi -> SpecFunc(collect(vi), ω, σ[k]; normalise=false), Iterators.partition(v, 100))), ω)
            end
        end
    end
    return SFresults
end
