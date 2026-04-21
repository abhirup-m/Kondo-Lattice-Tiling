@everywhere using ProgressMeter, Serialization, Fermions
@everywhere include("models.jl")
@everywhere include("helpers.jl")

function getFixedPointData(WfValues, JValues, Kf, Kd; loadData=true)
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
        params["Kf"] = Kf
        params["Kd"] = Kd
        for g in ("bw", "Jf", "Jd", "Kf", "Kd", "J⟂", "Wf", "Wd")
            params[g] *= params["upFactor"]
        end
        couplingsFlow = rgFlow(
                               params,
                               D -> - D/2,
                              )
        for k in quants
            FixedPoint[k][index] = couplingsFlow[k][end] / params["upFactor"]
        end
    end
    return FixedPoint, bareParams
end


function RealCorr(
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
    couplingsFlow = rgFlow(
                           params,
                           D -> -D/2;
                           loadData=true
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
    hybrid[1] = params["Vf"]
    hybrid[2] = params["Vd"]
    η = Dict("f" => params["ηf"], "d" => params["ηd"])
    impCorr = Dict("f" => params["Uf"], "d" => params["Ud"])
    hop_t = params["hop_t"]
    hop_step = Dict("f" => 1., "d" => 1.)
    heisenberg = Float64[]
    if couplingsFlow["Jf"][end] < couplingsFlow["Jf"][1]
        hop_t = Dict("f" => 0., "d" => params["hop_t"])
        inplaneKondo[1, 1] = 0.
        heisenberg = repeat([params["hop_t"]^2 / params["Uf"]], size)
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
                                 heisenberg=heisenberg,
                                )
    hamiltonianFamily = MinceHamiltonian(hamiltonian, 8:2:2*(2 + length(layerSpecs)))
    results = IterDiag(
                      hamiltonianFamily,
                      states;
                      symmetries=Char['N', 'S'],
                      correlationDefDict=copy(correlationDef),
                      #=mutInfoDefDict=copy(mutInfoDef),=#
                      silent=true,
                      maxMaxSize=states,
                     )
    impResults = Dict(k => v for (k,v) in results if haskey(correlationDef, k) || haskey(mutInfoDef, k))
    serialize(savePath, impResults)
    return impResults
end


function RealSpecFunc(
        params,
        ω,
        σ;
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
    return @distributed (d1, d2) -> merge(+, d1, d2) for factor in 10 .^ (1.0:-0.0125:-1.0)
        SFresults = Dict(k => zeros(length(ω)) for k in keys(specFunc))
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
            #=@assert false=#
            couplingsFlow = rgFlow(
                                   factorParams,
                                   D -> -D/2,
                                  )
            steps = length(couplingsFlow["bw"]) .- unique(trunc.(Int, 2 .^ (0:0.5:log2(length(couplingsFlow["bw"]))))) .+ 1
            overallFactors = Dict(t => couplingsFlow[t][end]^2 / couplingsFlow["Jf"][1]^2 - 1 for t in ["Jf", "Jd", "J⟂"])
            #=fFactor = =#
            #=dFactor = couplingsFlow["Jd"][end]^2 / couplingsFlow["Jd"][1]^2 - 1=#
            #=perpFactor = couplingsFlow["J⟂"][end]^2 / couplingsFlow["Jf"][1]^2 - 1=#
            if isnan(overallFactors["J⟂"])# || (perpFactor < fFactor && perpFactor < dFactor)
                overallFactors["J⟂"] = 0.
            end
            overallFactors = Dict(k => clamp(v, 0., 1.) for (k,v) in overallFactors)
            #=fFactor = clamp(fFactor, 0., 1.)=#
            #=dFactor = clamp(dFactor, 0., 1.)=#
            #=perpFactor = clamp(perpFactor, 0., 1.)=#
            println(overallFactors)
            #=return=#
            specFunc = DefineSpecFunc(overallFactors)
            collectedResults = [Dict() for _ in steps]
            @showprogress Threads.@threads for (i, step) in collect(enumerate(steps))
                inplaneKondo[1, 1] = couplingsFlow["Jf"][step]
                inplaneKondo[2, 2] = couplingsFlow["Jd"][step]
                indirectKondo[1, 1] = couplingsFlow["Kd"][step]
                indirectKondo[2, 2] = couplingsFlow["Kf"][step]
                hybrid = zeros(size * 2)
                hybrid[1] = √(factorParams["Vf"] * inplaneKondo[1, 1])
                hybrid[2] = √(factorParams["Vd"] * couplingsFlow["Jd"][step])
                η = Dict("f" => factorParams["ηf"], "d" => factorParams["ηd"])
                impCorr = Dict("f" => factorParams["Uf"], "d" => factorParams["Ud"])
                hop_t = Dict("f" => couplingsFlow["bw"][step], "d" => couplingsFlow["bw"][step])
                # hop_t = Dict(t => inplaneKondo[i, i] > 0 ? couplingsFlow["bw"][step] : 0. for (i, t) in enumerate(["f", "d"]))
                hop_step = Dict("f" => 1., "d" => 1.)
                hamiltonian = BilayerLEEReal(
                                             inplaneKondo,
                                             indirectKondo,
                                             couplingsFlow["J⟂"][end] / factor,
                                             hybrid,
                                             η,
                                             impCorr,
                                             hop_t,
                                             layerSpecs,
                                             hop_step;
                                             #=globalField=1e-8,=#
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
                collectedResults[i] = Dict(k => results[k] for k in keys(specFunc))
            end
            for results in collectedResults
                for k in keys(specFunc)
                    append!(iterCoeffs[k], results[k])
                end
            end
            serialize(savePath, iterCoeffs)
        end
        for (k, v) in copy(iterCoeffs)
            filteredWeights = filter(p -> true, vcat(v...))
            #=filteredWeights = filter(p -> abs(p[1]) > 1e-4, vcat(v...))=#
            SFresults[k] = SpecFunc(filteredWeights, ω, σ; normalise=false)#; normalise=true)
        end
        SFresults
    end
    #=return SFresults=#
end
