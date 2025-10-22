##### Functions for calculating various probes          #####
##### (correlation functions, Greens functions, etc)    #####

@everywhere using ProgressMeter, Combinatorics, Fermions, JSON3

"""
Function to calculate the total Kondo scattering probability Γ(k) = ∑_q J(k,q)^2
at the RG fixed point.
"""

@everywhere function ScattProb(
        size_BZ::Int64,
        kondoJArray::Array{Float64,3},
        dispersion::Vector{Float64},
    )

    # allocate zero arrays to store Γ at fixed point and for the bare Hamiltonian.
    results_scaled = zeros(size_BZ^2)

    # loop over all points k for which we want to calculate Γ(k).
    for point in 1:size_BZ^2
        targetStatesForPoint = collect(1:size_BZ^2)[abs.(dispersion).<=abs(dispersion[point])]

        # calculate the sum over q
        results_scaled[point] = sum(kondoJArray[point, targetStatesForPoint, end] .^ 2) ^ 0.5 / sum(kondoJArray[point, targetStatesForPoint, 1] .^ 2) ^ 0.5

    end

    # get a boolean representation of results for visualisation, using the mapping
    results_bool = ifelse.(abs.(results_scaled) .> 0, 1, 0)

    return results_scaled, results_bool
end

function SelfEnergyHelper(
        specFunc::Vector{Float64},
        freqValues::Vector{Float64},
        nonIntSpecFunc::Vector{Float64};
        pinBottom::Bool=true,
        normalise::Bool=true,
    )
    specFunc .+= 1e-8
    selfEnergy = SelfEnergy(nonIntSpecFunc, specFunc, freqValues; normalise=normalise)
    imagSelfEnergyCurrent = imag(selfEnergy)
    if pinBottom
        imagSelfEnergyCurrent .-= imagSelfEnergyCurrent[freqValues .≥ 0][1]
        imagSelfEnergyCurrent[imagSelfEnergyCurrent .≥ 0] .= 0
    end
    return real(selfEnergy) .+ 1im .* imagSelfEnergyCurrent
end

"""
Return the map of Kondo couplings M_k(q) = J^2_{k,q} given the state k.
Useful for visualising how a single state k interacts with all other states q.
"""
function KondoCoupMap(
        kx_ky::Tuple{Float64,Float64},
        size_BZ::Int64,
        kondoJArrayFull::Array{Float64,3};
        mapAmong::Union{Function, Nothing}=nothing
    )
    kspacePoint = map2DTo1D(kx_ky..., size_BZ)
    otherPoints = collect(1:size_BZ^2)
    if !isnothing(mapAmong)
        filter!(p -> mapAmong(map1DTo2D(p, size_BZ)...), otherPoints)
    end
    results = zeros(size_BZ^2) 
    results[otherPoints] .= kondoJArrayFull[kspacePoint, otherPoints, end]
    results_bare = kondoJArrayFull[kspacePoint, :, 1]
    reference = sum(abs.(results_bare)) / length(results_bare)
    @assert reference > RG_RELEVANCE_TOL
    results_bool = [abs(r / reference) ≤ RG_RELEVANCE_TOL ? -1 : 1 for (r, r_b) in zip(results, results_bare)]
    return results, results_bare, results_bool
end


@everywhere function IterDiagMomentumSpace(
        hamiltDetails::Dict,
        maxSize::Int64,
        momentumPoints::Vector{Int64},
        correlation::Dict;
        addPerStep::Int64=1,
    )
    allKeys = keys(correlation)
    corrResults = Dict{String, Vector{Float64}}(k => repeat([NaN], hamiltDetails["size_BZ"]^2) for k in allKeys)

    kondoGamma = zeros(2 * length(momentumPoints) + 1, 2 * length(momentumPoints) + 1)
    kondoGamma[1, 1] = hamiltDetails["kondoPerp"]
    hamiltonianIndices = Int64[0]
    for (i, point) in enumerate(momentumPoints)
        kondoGamma[1 + i, 2:2:end] .= hamiltDetails["kondoJArray"][point, momentumPoints] + hamiltDetails["kondoJArray"][ReflectY(point, hamiltDetails["size_BZ"]), momentumPoints]
        kondoGamma[2:2:end, i + 1] .= kondoGamma[i + 1, 2:2:end]
        kondoGamma[1 + 2 * i, 3:2:end] .= hamiltDetails["kondoJArray"][point, momentumPoints] - hamiltDetails["kondoJArray"][ReflectY(point, hamiltDetails["size_BZ"]), momentumPoints]
        kondoGamma[3:2:end, 1 + 2 * i] .= kondoGamma[1 + 2 * i, 3:2:end]
        append!(hamiltonianIndices, [point, -point])
    end
    #=priority = [sum(row.^2)^0.5 for row in eachrow(kondoGamma)]=#
    priority = abs.(diag(kondoGamma))
    sortedFiltered = sortperm(priority, rev=true)
    filter!(i -> priority[i] > 0, sortedFiltered)
    if 1 ∈ sortedFiltered
        diff = filter(≠(1), sortedFiltered)
        #=sortedFiltered = vcat(diff, [1])=#
        #=sortedFiltered = vcat([1], diff)=#
        #=sortedFiltered = vcat(diff[1:div(length(sortedFiltered)-1, 2)], [1], diff[div(length(sortedFiltered)-1, 2)+1:end])=#
    end
    hamiltonianIndices = hamiltonianIndices[sortedFiltered]
    kondoGamma = kondoGamma[sortedFiltered, sortedFiltered]
    
    hamiltonian = BilayerKondo(
                               kondoGamma;
                               epsilonF=hamiltDetails["epsilonF"],
                               impU=hamiltDetails["impU"],
                               bathChemPot=repeat([hamiltDetails["mu_c"]], length(sortedFiltered)),
                               globalField=1e-8,
                               couplingTolerance=1e-10,
                            )
    indexPartitions = [8]
    while indexPartitions[end] < 2 + 2 * length(sortedFiltered)
        push!(indexPartitions, indexPartitions[end] + 2 * addPerStep)
    end
    hamiltonianFamily = MinceHamiltonian(hamiltonian, indexPartitions)
    @assert all(!isempty, hamiltonianFamily)
    for hamiltonian in hamiltonianFamily
        @assert all(!isempty, hamiltonian)
    end

    corrOps = Dict{String, Vector{Tuple{String, Vector{Int64}, Float64}}}()

    corrResults = Dict()
    momentumPairs = vec(collect(Iterators.product(momentumPoints, momentumPoints)))
    for (name, (type, condition, corrFunc)) in correlation
        corrResults[name] = Dict()
        if type == "i"
            corrName = name
            corrOps[corrName] = corrFunc
            continue
        end
        if type == "s"
            if 0 ∈ hamiltonianIndices 
                upZero = 1 + 2 * findfirst(==(0), hamiltonianIndices)
                corrOps[name] = corrFunc(upZero, upZero)
            else
                corrResults[name] = 0.
            end
            continue
        end
        corrResults[name] = zeros(length(momentumPoints)^2) # Dict((k1, k2) => 0. for k1 in momentumPoints for k2 in momentumPoints)

        for (ki, kj) in Iterators.product(momentumPoints, momentumPoints)
            if !condition(ki, kj, momentumPoints)
                continue
            end
            #=corrName = name*"-"*string(findfirst(==(k_i), momentumPoints))*"-"*string(findfirst(==(k_j), momentumPoints))=#
            index = findfirst(==((ki, kj)), momentumPairs)
            corrName = "$(name)-$(index)"

            upLeft = [1 + 2 * findfirst(==(k), hamiltonianIndices) for k in [ki, -ki] if k ∈ hamiltonianIndices]
            upRight = [1 + 2 * findfirst(==(k), hamiltonianIndices) for k in [kj, -kj] if k ∈ hamiltonianIndices]
            if isempty(upLeft) || isempty(upRight)
                continue
            end
            corrOps[corrName] = vcat([corrFunc(iP, iM, ki, kj; factor=0.25) for (iP, iM) in Iterators.product(upLeft, upRight)]...)
        end
    end

    results = IterDiag(
                      hamiltonianFamily, 
                      maxSize;
                      symmetries=Char['N', 'S'],
                      magzReq=(m, N) -> -5 ≤ m ≤ 5,
                      occReq=(x, N) -> div(N, 2) - 5 ≤ x ≤ div(N, 2) + 5,
                      correlationDefDict=corrOps,
                      silent=true,
                      maxMaxSize=maxSize,
                     )
    @assert results["exitCode"] == 0
    for (name, (type, condition, _)) in correlation
        if type ≠ "f"
            if name ∈ keys(results)
                corrResults[name] = results[name]
            end
            continue
        end
        for (ki, kj) in Iterators.product(momentumPoints, momentumPoints)
            if !condition(ki, kj, momentumPoints)
                continue
            end
            #=inds = (findfirst(==(ki), momentumPoints), findfirst(==(kj), momentumPoints))=#
            index = findfirst(==((ki, kj)), momentumPairs)
            revIndex = findfirst(==((kj, ki)), momentumPairs)
            corrName = "$(name)-$(index)"
            if corrName ∈ keys(results)
                corrResults[name][index] = results[corrName]
            end
            if ki == kj
                continue
            end
            corrResults[name][revIndex] = corrResults[name][index]
        end
    end

    return corrResults
end


@everywhere function IterDiagRealSpace(
        hamiltDetails::Dict,
        realKondo1D::Vector{Dict{NTuple{2, Int64}, Float64}},
        maxSize::Int64,
        numBathSites::Int64,
        addPerStep::Int64,
    )
    numChannels = length(realKondo1D)

    hamiltDetails["imp_corr"] -= 2 * maximum(values(realKondo1D[1]))
    hamiltonian = KondoModel(numBathSites, HOP_T, realKondo1D; globalField=hamiltDetails["globalField"])
    append!(hamiltonian, [("n",  [1], -hamiltDetails["imp_corr"]/2), ("n",  [2], -hamiltDetails["imp_corr"]/2), ("nn",  [1, 2], hamiltDetails["imp_corr"])])

    mutInfoSites = 1:2:numBathSites # [1, div(numBathSites, 4), div(numBathSites, 2), numBathSites]
    I2_di = Dict("I2-d-$(i)" => ([1, 2], [3 + 2 * numChannels * (i-1), 4 + 2 * numChannels * (i-1)]) for i in mutInfoSites)
    I2_d0i = Dict("I2-d-0$(i)" => ([1, 2], [3 + 2 * numChannels * (i-1), 4 + 2 * numChannels * (i-1), 3 + 2 * numChannels * (mutInfoSites[end]-1), 4 + 2 * numChannels * (mutInfoSites[end]-1)]) for i in mutInfoSites[1:end-1])
    spinFlipCorrDefDict = Dict("SF-d-$(i)" => [("+-+-", [1, 2, 4 + 2 * numChannels * (i-1), 3 + 2 * numChannels * (i-1)], 0.5), ("+-+-", [2, 1, 3 + 2 * numChannels * (i-1), 4 + 2 * numChannels * (i-1)], 0.5)] for i in 1:numBathSites)
    isingCorrDefDict = Dict("ZZ-d-$(i)" => [("nn", [1, 3 + 2 * numChannels * (i-1)], 0.25), ("nn", [1, 4 + 2 * numChannels * (i-1)], -0.25), ("nn", [2, 3 + 2 * numChannels * (i-1)], -0.25), ("nn", [2, 4 + 2 * numChannels * (i-1)], 0.25)] for i in 1:numBathSites)
    imp_mag = Dict("Sdz" => [("n", [1], 0.5), ("n", [2], -0.5)], "Sdx" => [("+-", [1,2], 0.5), ("+-", [2,1], 0.5)], "Sdy" => [("+-", [1,2], 0.5), ("+-", [2,1], -0.5)])
    Sdz_sq = Dict("Sdz_sq" => [("n", [1], 0.25), ("n", [2], 0.25), ("nn", [1, 2], -0.5)])
    QFI = Dict("n_tot_sq" => Tuple{String, Vector{Int64}, Float64}[], "n_tot" => Tuple{String, Vector{Int64}, Float64}[])
    up(k) = 1 + 2 * k
    down(k) = up(k)
    count = length(0:2:numChannels*(numBathSites-1))
    for i in 0:2:numChannels*(numBathSites-1)
        for j in 0:numChannels:numChannels*(numBathSites-1)
            append!(QFI["n_tot_sq"], [("+-+-+-+-", [up(i), down(i), down(i+1), up(i+1), up(j), down(j), down(j+1), up(j+1)], 0.25 / count)])
            append!(QFI["n_tot_sq"], [("+-+-+-+-", [up(i+1), down(i+1), down(i), up(i), up(j), down(j), down(j+1), up(j+1)], 0.25 / count)])
            append!(QFI["n_tot_sq"], [("+-+-+-+-", [up(i+1), down(i+1), down(i), up(i), up(j+1), down(j+1), down(j), up(j)], 0.25 / count)])
            append!(QFI["n_tot_sq"], [("+-+-+-+-", [up(i), down(i), down(i+1), up(i+1), up(j+1), down(j+1), down(j), up(j)], 0.25 / count)])
        end
        append!(QFI["n_tot"], [("+-+-", [up(i), down(i), down(i+1), up(i+1)], 0.5 / count)])
        append!(QFI["n_tot"], [("+-+-", [up(i+1), down(i+1), down(i), up(i)], 0.5 / count)])
    end
    #=corrDefDict = imp_mag=#
    corrDefDict = merge(spinFlipCorrDefDict, isingCorrDefDict)
    indexPartitions = [4]
    while indexPartitions[end] < 2 + 2 * numChannels * numBathSites
        push!(indexPartitions, minimum((indexPartitions[end] + 2 * addPerStep, 2 + 2 * numChannels * numBathSites)))
    end

    hamiltonianFamily = MinceHamiltonian(hamiltonian, indexPartitions)
    @assert all(!isempty, hamiltonianFamily)
    for hamiltonian in hamiltonianFamily
        @assert all(!isempty, hamiltonian)
    end
    results = Dict()

    id = nothing
    exitCode = 0
    specDictSet = ImpurityExcitationOperators(1)
    while true
        @time output = IterDiag(
                          hamiltonianFamily, 
                          maxSize;
                          symmetries=Char['N', 'S'],
                          #=magzReq=(m, N) -> -3 ≤ m ≤ 3,=#
                          #=occReq=(x, N) -> div(N, 2) - 5 ≤ x ≤ div(N, 2) + 5,=#
                          #=mutInfoDefDict=deepcopy(I2_di),=#
                          #=correlationDefDict=deepcopy(QFI),=#
                          #=correlationDefDict=deepcopy(corrDefDict),=#
                          silent=false,
                          maxMaxSize=maxSize,
                          specFuncDefDict=specDictSet,
                         )
        #=savePaths, iterDiagResults, exitCode, specFuncOperators = output=#
        savePaths, iterDiagResults, specFuncOperators = output
        results["SFO"] = specFuncOperators
        #=savePaths, iterDiagResults, exitCode = output=#
        #=savePaths, iterDiagResults = output=#
        results["SP"] = savePaths


        if exitCode > 0
            id = rand()
            println("Error code $(exitCode). Retry id=$(id).")
        else
            if !isnothing(id)
                println("Passed $(id).")
            end
            for i in 1:numBathSites
                corrKeys = [("SF-di", "SF-d-$(i)"), ("ZZ-di", "ZZ-d-$(i)"), ("I2-di", "I2-d-$(i)"), ("I2-d0i", "I2-d-0$(i)")]
                for (k1, k2) in corrKeys
                    if k2 in keys(iterDiagResults)
                        if k1 ∉ keys(results)
                            results[k1] = Float64[]
                        end
                        push!(results[k1], iterDiagResults[k2])
                    end
                end
            end
            corrKeys = ["Sdx", "Sdy", "Sdz", "Sdz_sq", "n_tot_sq", "n_tot"]
            for k in corrKeys
                if k in keys(iterDiagResults)
                    results[k] = iterDiagResults[k]
                end
            end
            break
        end
    end
    return results, 1:numBathSites, mutInfoSites
end


@everywhere function IterDiagSpecFunc(
        hamiltDetails::Dict,
        maxSize::Int64,
        sortedPoints::Vector{Int64},
        specFuncDict::Dict{String, Vector{Tuple{String, Vector{Int64}, Float64}}},
        bathIntLegs::Int64,
        freqValues::Vector{Float64},
        standDev::Dict{String, Union{Float64, Vector{Float64}}};
        addPerStep::Int64=2,
        silent::Bool=false,
        broadFuncType::Union{String, Dict{String, String}}="gauss",
        normEveryStep::Bool=true,
        onlyCoeffs::Vector{String}=String[],
    )
    if typeof(broadFuncType) == String
        broadFuncType = Dict(name => broadFuncType for name in specFuncDict)
    end

    bathIntFunc = points -> hamiltDetails["bathIntForm"](0.,
                                                         # hamiltDetails["W_val"],
                                                         hamiltDetails["orbitals"][2],
                                                         hamiltDetails["size_BZ"],
                                                         points)
                            
    hamiltonian = KondoModel(
                             hamiltDetails["dispersion"][sortedPoints],
                             hamiltDetails["kondoJArray"][sortedPoints, sortedPoints] ./ length(sortedPoints),
                             sortedPoints, bathIntFunc;
                             bathIntLegs=bathIntLegs,
                             globalField=hamiltDetails["globalField"],
                             couplingTolerance=1e-10,
                            )

    kondoTemp = (sum(hamiltDetails["kondoJArray"][sortedPoints[1], sortedPoints] .|> abs) / sum(hamiltDetails["kondoJArray_bare"][sortedPoints[1], sortedPoints] .|> abs))^0.6

    # impurity local terms
    push!(hamiltonian, ("n",  [1], -hamiltDetails["imp_corr"]/2)) # Ed nup
    push!(hamiltonian, ("n",  [2], -hamiltDetails["imp_corr"]/2)) # Ed ndown
    push!(hamiltonian, ("nn",  [1, 2], hamiltDetails["imp_corr"])) # U nup ndown
    excludeRange = (1.0 * maximum(hamiltDetails["kondoJArray_bare"]), hamiltDetails["imp_corr"]/4)

    indexPartitions = [4]
    while indexPartitions[end] < 2 + 2 * length(sortedPoints)
        push!(indexPartitions, indexPartitions[end] + 2 * addPerStep)
    end
    hamiltonianFamily = MinceHamiltonian(hamiltonian, indexPartitions)
    
    savePaths, resultsDict, specFuncOperators = IterDiag(
                                                         hamiltonianFamily, 
                                                         maxSize;
                                                         symmetries=Char['N', 'S'],
                                                         magzReq=(m, N) -> -3 ≤ m ≤ 4,
                                                         occReq=(x, N) -> div(N, 2) - 4 ≤ x ≤ div(N, 2) + 4,
                                                         silent=silent,
                                                         maxMaxSize=maxSize,
                                                         specFuncDefDict=specFuncDict,
                                                        ) 

    specFuncResults = Dict()
    for (name, operator) in specFuncOperators
        if name ∈ onlyCoeffs
            specCoeffs = IterSpectralCoeffs(savePaths, operator;
                                            degenTol=1e-10, silent=silent,
                                           )
            scaledSpecCoeffs = NTuple{2, Float64}[(weight * kondoTemp, pole) for (weight, pole) in vcat(specCoeffs...)]
            specFuncResults[name] = scaledSpecCoeffs
        else
            specFunc = IterSpecFunc(savePaths, operator, freqValues, 
                                                    standDev[name]; normEveryStep=normEveryStep, 
                                                    degenTol=1e-10, silent=silent, 
                                                    broadFuncType=broadFuncType[name],
                                                    returnEach=false, normalise=false,
                                                   )
            specFuncResults[name] = specFunc
        end
    end
    return specFuncResults

end

@everywhere function AuxiliaryCorrelations(
        size_BZ::Int64,
        couplings::Dict{String, Float64},
        correlation::Dict,
        momentumPoints::Vector{Int64},
        maxSize::Int64;
        loadData::Bool=false,
    )

    function Average(results)
        averageResults = Dict{String, Float64}()
        for (name, (type, _, _)) in correlation
            if type == "f"
                averageResults[name] = sum(results[name])
            else
                averageResults[name] = results[name]
            end
        end
        return averageResults
    end

    _, dispersion = getDensityOfStates(tightBindDisp, size_BZ)

    savePath = joinpath(SAVEDIR, SavePath("mom-corr-", size_BZ, couplings, ".json"; maxSize=maxSize))
    availableResults = Dict()
    if isfile(savePath) && loadData
        loadedResults = JSON3.read(read(savePath, String), Dict{String, Any})
        if keys(correlation) ⊆ keys(loadedResults)
            return loadedResults
        else
            merge!(availableResults, loadedResults)
        end
    end

    kondoJArray, kondoPerpArray, dispersion = momentumSpaceRG(size_BZ, couplings; loadData=false, saveData=false)

    averageKondoScale = sum(abs.(kondoJArray[:, :, 1])) / length(kondoJArray[:, :, 1])
    @assert averageKondoScale > RG_RELEVANCE_TOL
    kondoJArray[:, :, end] .= ifelse.(abs.(kondoJArray[:, :, end]) ./ averageKondoScale .> RG_RELEVANCE_TOL, kondoJArray[:, :, end], 0)

    hamiltDetails = Dict()
    hamiltDetails["kondoJArray"] = kondoJArray[:, :, end]
    merge!(hamiltDetails, couplings)
    hamiltDetails["kondoPerp"] = kondoPerpArray[end]
    hamiltDetails["dispersion"] = dispersion
    hamiltDetails["size_BZ"] = size_BZ

    if abs(hamiltDetails["mu_c"]) < 4 * hamiltDetails["lightBandFactor"] * HOP_T
        hamiltDetails["mu_c"] = 0.
    elseif hamiltDetails["mu_c"] > 4 * hamiltDetails["lightBandFactor"] * HOP_T
        hamiltDetails["mu_c"] -= 4 * hamiltDetails["lightBandFactor"] * HOP_T
    else
        hamiltDetails["mu_c"] += 4 * hamiltDetails["lightBandFactor"] * HOP_T
    end

    results = IterDiagMomentumSpace(
                                    hamiltDetails, 
                                    maxSize, 
                                    momentumPoints,
                                    correlation;
                                    addPerStep=1,
                                   )

    merge!(availableResults, results)
    if !isnothing(savePath)
        mkpath(SAVEDIR)
        open(savePath, "w") do file JSON3.write(file, availableResults) end
    end

    return results
    #=return Average(results)=#

end


@everywhere function AuxiliaryRealSpaceEntanglement(
        hamiltDetails::Dict,
        numShells::Int64,
        maxSize::Int64;
        numChannels::Int64=1,
        savePath::Union{Nothing, String}=nothing,
        addPerStep::Int64=1,
        numProcs::Int64=nprocs(),
        loadData::Bool=false,
    )

    size_BZ = hamiltDetails["size_BZ"]

    cutoffEnergy = hamiltDetails["dispersion"][div(size_BZ - 1, 2) + 2 - numShells]
    cutoffWindow = filter(p -> abs(hamiltDetails["dispersion"][p]) ≤ cutoffEnergy, 1:size_BZ^2)

    # pick out k-states from the southwest quadrant that have positive energies 
    # (hole states can be reconstructed from them (p-h symmetry))
    shellPointsChannels = Vector{Int64}[]
    if numChannels == 1
        #=push!(shellPointsChannels, filter(p -> true, 1:size_BZ^2))=#
        push!(shellPointsChannels, filter(p -> abs(cutoffEnergy) ≥ abs(hamiltDetails["dispersion"][p]) && (hamiltDetails["kondoJArray"][p,:] |> maximum |> abs > 1e-4), cutoffWindow))
    else
        push!(shellPointsChannels, filter(p -> abs(cutoffEnergy) ≥ abs(hamiltDetails["dispersion"][p]) && prod(map1DTo2D(p, size_BZ)) ≥ 0 && map1DTo2D(p, size_BZ)[1] ≠ 0 && (hamiltDetails["kondoJArray"][p,:] |> maximum |> abs > 1e-4), cutoffWindow))
        push!(shellPointsChannels, filter(p -> abs(cutoffEnergy) ≥ abs(hamiltDetails["dispersion"][p]) && prod(map1DTo2D(p, size_BZ)) ≤ 0 && map1DTo2D(p, size_BZ)[2] ≠ 0 && (hamiltDetails["kondoJArray"][p,:] |> maximum |> abs > 1e-4), cutoffWindow))
    end

    if !isnothing(savePath) && loadData
        corrResults = Dict()
        xvals1, xvals2 = nothing, nothing
        if isfile(savePath)
            content = load(savePath)
            xvals1 = content["xvals1"]
            xvals2 = content["xvals2"]
            for (name, val) in content
                corrResults[name] = val
            end
            return corrResults, xvals1, xvals2
        end
    end

    fermSurfKondoChannels = [zeros(size_BZ^2, size_BZ^2) for _ in 1:numChannels]
    for i in 1:numChannels
        fermSurfKondoChannels[i][shellPointsChannels[i], shellPointsChannels[i]] .= hamiltDetails["kondoJArray"][shellPointsChannels[i], shellPointsChannels[i]]
    end
    distances = [trunc(sum((map1DTo2D(p, size_BZ) .* (size_BZ - 1)/ (2π)) .^ 2)^0.5, digits=6) for p in 1:size_BZ^2]
    sortedIndices = (1:size_BZ^2)[sortperm(distances)]
    impurity = Int((1 + size_BZ^2) / 2)

    filter!(p -> 0 ≤ map1DTo2D(p, size_BZ)[1] ≤ 5.5 && abs(map1DTo2D(p, size_BZ)[2]) < 1e-6, sortedIndices)
    if length(sortedIndices) % 2 != 0
        sortedIndices = sortedIndices[1:end-1]
    end

    realKondoChannels, kondoTemp = Fourier(fermSurfKondoChannels; shellPointsChannels=shellPointsChannels, calculateFor=sortedIndices)
    #=kondoTemp = maximum(realKondoChannels[1])=#

    kondoReal1D = [Dict{NTuple{2, Int64}, Float64}() for _ in 1:numChannels]

    for (k, kondoMatrix) in enumerate(realKondoChannels)
        for (i1, p1) in enumerate(sortedIndices)
            for (i2, p2) in enumerate(sortedIndices)
                if 1 ∈ (i1, i2)
                    continue
                end
                kondoReal1D[k][(i1-1, i2-1)] = kondoMatrix[p1, p2]
            end
        end
    end
    
    desc = "W=$(round(hamiltDetails["W_val"], digits=3))"
    @time corrResults, xvals1, xvals2 = IterDiagRealSpace(hamiltDetails,
                                kondoReal1D,
                                maxSize,
                                length(sortedIndices)-1,
                                addPerStep,
                               )
    corrResults["Tk"] = kondoTemp


    if !isnothing(savePath)
        mkpath(SAVEDIR)
        jldopen(savePath, "w"; compress = true) do file
            if "SFO" in keys(corrResults)
                for (i, path) in enumerate(corrResults["SP"])
                    newPath = joinpath(SAVEDIR, basename(path))
                    cp(path, newPath, force=true)
                    corrResults["SP"][i] = newPath
                end
            end
            for (name, val) in corrResults
                file[name] = val
            end
            file["xvals1"] = xvals1
            file["xvals2"] = xvals2
        end
    end

    return corrResults, xvals1, xvals2
end


function AuxiliaryLocalSpecfunc(
        hamiltDetails::Dict,
        numShells::Int64,
        specFuncDictFunc::Function,
        freqValues::Vector{Float64},
        standDevInner::Union{Vector{Float64}, Float64},
        standDevOuter::Union{Vector{Float64}, Float64},
        maxSize::Int64;
        targetHeight::Float64=0.,
        standDevGuess::Float64=0.1,
        heightTolerance::Float64=1e-4,
        bathIntLegs::Int64=2,
        addPerStep::Int64=1,
        maxIter::Int64=20,
        broadFuncType::String="gauss",
        nonIntSpecFunc::Union{Nothing,Vector{Number}}=nothing,
    )
    if targetHeight < 0
        targetHeight = 0
    end

    size_BZ = hamiltDetails["size_BZ"]
    cutoffEnergy = hamiltDetails["dispersion"][div(size_BZ - 1, 2) + 2 - numShells]

    # pick out k-states from the southwest quadrant that have positive energies 
    # (hole states can be reconstructed from them (p-h symmetry))
    SWIndicesAll = [p for p in 1:size_BZ^2 if 
                 map1DTo2D(p, size_BZ)[1] ≤ 0 &&
                 map1DTo2D(p, size_BZ)[2] ≤ 0 &&
                 #=map1DTo2D(p, size_BZ)[1] ≤ map1DTo2D(p, size_BZ)[2] &&=#
                 abs(cutoffEnergy) ≥ abs(hamiltDetails["dispersion"][p])
                ]

    SWIndices = filter(p -> maximum(abs.(hamiltDetails["kondoJArray"][p, SWIndicesAll])) > 0, SWIndicesAll)
    while length(SWIndices) < 2
        push!(SWIndices, SWIndicesAll[findfirst(∉(SWIndices), SWIndicesAll)])
    end
    distancesFromNode = [minimum([sum((map1DTo2D(p, size_BZ) .- node) .^ 2)^0.5 
                                  for node in NODAL_POINTS])
                        for p in SWIndices
                       ]
    distancesFromAntiNode = [minimum([sum((map1DTo2D(p, size_BZ) .- antinode) .^ 2)^0.5
                                     for antinode in ANTINODAL_POINTS])
                            for p in SWIndices
                           ]
    @assert distancesFromNode |> length == distancesFromAntiNode |> length

    sortedPointsNode = SWIndices[sortperm(distancesFromNode)]
    sortedPointsAntiNode = SWIndices[sortperm(distancesFromAntiNode)]
    
    onlyCoeffs = ["Sd+", "Sd-"]
    specDictSet = ImpurityExcitationOperators(length(SWIndices))
    standDev = Dict{String, Union{Float64, Vector{Float64}}}(name => ifelse(name ∈ ("Sd+", "Sd-"), standDevInner, standDevOuter) for name in keys(specDictSet))
    broadType = Dict{String, String}(name => ifelse(name ∈ ("Sd+", "Sd-"), "lorentz", "gauss")
                                     for name in keys(specDictSet)
                                    )
    sortedPointsSets = [sortedPointsNode, sortedPointsAntiNode]
    energySigns = [1, -1]
    argsPermutations = [(p, s) for p in sortedPointsSets for s in energySigns]
    specFuncResultsGathered = repeat(Any[nothing], length(argsPermutations))

    @sync for (threadCounter, (sortedPoints, energySign)) in enumerate(argsPermutations)
        Threads.@spawn begin
            hamiltDetailsModified = deepcopy(hamiltDetails)
            hamiltDetailsModified["imp_corr"] = hamiltDetails["imp_corr"] * energySign
            hamiltDetailsModified["W_val"] = hamiltDetails["W_val"] * energySign
            specFuncResultsGathered[threadCounter] = IterDiagSpecFunc(hamiltDetailsModified, maxSize, deepcopy(sortedPoints),
                                                  specDictSet, bathIntLegs, freqValues, 
                                                  standDev; addPerStep=addPerStep,
                                                  silent=false, broadFuncType=broadType,
                                                  normEveryStep=false, onlyCoeffs=onlyCoeffs,
                                              )
        end
    end
    fixedContrib = Vector{Float64}[]
    specCoeffs = Vector{Tuple{Float64, Float64}}[]
    for specFuncResults in specFuncResultsGathered
        for (key, val) in specFuncResults
            if key ∈ onlyCoeffs
                push!(specCoeffs, val)
            else
                push!(fixedContrib, val)
            end
        end
    end

    insideArea = sum(sum([SpecFunc(coeffs, freqValues, standDevInner; normalise=false) for coeffs in specCoeffs])) * (maximum(freqValues) - minimum(freqValues)) / (length(freqValues) - 1)
    centerSpecFuncArr, localSpecFunc, standDevInner = SpecFuncVariational(specCoeffs, freqValues, targetHeight, 1e-3;
                                 degenTol=1e-10, silent=false, 
                                 broadFuncType="lorentz", 
                                 fixedContrib=fixedContrib,
                                 standDevGuess=standDevGuess,
                                )
    outsideArea = 0
    for specFunc in fixedContrib
        outsideArea += sum(specFunc) * (maximum(freqValues) - minimum(freqValues)) / (length(freqValues) - 1)
    end
    quasipResidue = insideArea / (insideArea + outsideArea)

    return localSpecFunc, standDevInner, quasipResidue, centerSpecFuncArr
end


function LatticeKspaceDOS(
        hamiltDetails::Dict,
        numShells::Int64,
        specFuncDictFunc::Function,
        freqValues::Vector{Float64},
        standDevInner::Union{Vector{Float64}, Float64},
        standDevOuter::Union{Vector{Float64}, Float64},
        maxSize::Int64,
        savePath::String;
        loadData::Bool=false,
        onlyAt::Union{Nothing,NTuple{2, Float64}}=nothing,
        bathIntLegs::Int64=2,
        addPerStep::Int64=1,
        maxIter::Int64=20,
        broadFuncType::String="gauss",
        targetHeight::Float64=0.,
        standDevGuess::Float64=0.1,
        nonIntSpecBzone::Union{Vector{Vector{Float64}}, Nothing}=nothing,
        selfEnergyWindow::Float64=0.,
        singleThread::Bool=false,
    )
    size_BZ = hamiltDetails["size_BZ"]
    cutoffEnergy = hamiltDetails["dispersion"][div(size_BZ - 1, 2) + 2 - numShells]

    # pick out k-states from the southwest quadrant that have positive energies 
    # (hole states can be reconstructed from them (p-h symmetry))
    SWIndicesAll = [p for p in 1:size_BZ^2 if 
                 #=map1DTo2D(p, size_BZ)[1] ≤ 0 &&=#
                 #=map1DTo2D(p, size_BZ)[2] ≤ 0 &&=#
                 map1DTo2D(p, size_BZ)[1] ≤ -map1DTo2D(p, size_BZ)[2] &&
                 abs(cutoffEnergy) ≥ abs(hamiltDetails["dispersion"][p])
                ]
    calculatePoints = filter(p -> 0 ≥ map1DTo2D(p, size_BZ)[1] ≥ map1DTo2D(p, size_BZ)[2], SWIndicesAll)

    oppositePoints = Dict{Int64, Vector{Int64}}()
    for point in calculatePoints
        reflectDiagonal = map2DTo1D(reverse(map1DTo2D(point, size_BZ))..., size_BZ)
        reflectFS = map2DTo1D((-1 .* reverse(map1DTo2D(point, size_BZ)) .+ [-π, -π])..., size_BZ)
        reflectBoth = map2DTo1D((-1 .* map1DTo2D(point, size_BZ) .+ [-π, -π])..., size_BZ)
        oppositePoints[point] = [reflectDiagonal, reflectFS, reflectBoth]
    end
    rotationMatrix(rotateAngle) = [cos(rotateAngle) -sin(rotateAngle); sin(rotateAngle) cos(rotateAngle)]

    if ispath(savePath) && loadData
        centerSpecFuncArr=jldopen(savePath)["centerSpecFuncArr"]
        fixedContrib=jldopen(savePath)["fixedContrib"]
        localSpecFunc=jldopen(savePath)["localSpecFunc"]
        standDevFinal=jldopen(savePath)["standDevFinal"]
        println("Collected $(savePath) from saved data.")
    else
        specCoeffsBZone = [NTuple{2, Float64}[] for _ in calculatePoints]
        fixedContrib = [freqValues |> length |> zeros for _ in calculatePoints]

        onlyCoeffs = ["Sd+", "Sd-"]

        function SpectralCoeffsAtKpoint(
                kspacePoint::Int64,
                onlyCoeffs::Vector{String};
                invert::Bool=false,
            )
            SWIndices = filter(p -> maximum(abs.(hamiltDetails["kondoJArray"][p, SWIndicesAll])) > 0, SWIndicesAll)
            if kspacePoint ∉ SWIndices
                SWIndices = [kspacePoint]
                push!(SWIndices, SWIndicesAll[findfirst(∉(SWIndices), SWIndicesAll)])
            end
            while length(SWIndices) < 2
                push!(SWIndices, SWIndicesAll[findfirst(∉(SWIndices), SWIndicesAll)])
            end
            equivalentPoints = [rotationMatrix(rotateAngle) * map1DTo2D(kspacePoint, size_BZ) for rotateAngle in (0, π/2, π, 3π/2)]
            distancesFromPivot = [minimum([sum((map1DTo2D(p, size_BZ) .- pivot) .^ 2)^0.5 for pivot in equivalentPoints])
                                for p in SWIndices
                               ]
            sortedPoints = SWIndices[sortperm(distancesFromPivot)]
            
            specDictSet = specFuncDictFunc(length(SWIndices), (3, 4))
            if hamiltDetails["kondoJArray"][sortedPoints[1], sortedPoints] .|> abs |> maximum == 0 
                delete!(specDictSet, "Sd+")
                delete!(specDictSet, "Sd-")
            end

            standDev = Dict{String, Union{Float64, Vector{Float64}}}(name => ifelse(name ∈ ("Sd+", "Sd-"), standDevInner, standDevOuter) for name in keys(specDictSet))
            broadType = Dict{String, String}(name => ifelse(name ∈ ("Sd+", "Sd-"), "lorentz", "gauss")
                                             for name in keys(specDictSet)
                                            )

            hamiltDetailsModified = deepcopy(hamiltDetails)
            if invert
                hamiltDetailsModified["imp_corr"] = hamiltDetails["imp_corr"] * -1
                hamiltDetailsModified["W_val"] = hamiltDetails["W_val"] * -1
            end
            specFuncResultsPoint = IterDiagSpecFunc(hamiltDetailsModified, maxSize, sortedPoints,
                                               specDictSet, bathIntLegs, freqValues, 
                                               standDev; addPerStep=addPerStep,
                                               silent=false, broadFuncType=broadType,
                                               normEveryStep=false, onlyCoeffs=onlyCoeffs,
                                              )
            return specFuncResultsPoint
        end

        if !isnothing(onlyAt)
            pointIndex = map2DTo1D(onlyAt..., size_BZ)
            specFuncResults = SpectralCoeffsAtKpoint(pointIndex, String[])
            specFunc = Normalise(sum(values(specFuncResults)), freqValues, true)
            return specFunc
        end

        specFuncResults = Any[nothing, nothing]
        @sync begin
            if !singleThread
                @async specFuncResults[1] = fetch.([Threads.@spawn SpectralCoeffsAtKpoint(k, onlyCoeffs) for k in calculatePoints])
                @async specFuncResults[2] = fetch.([Threads.@spawn SpectralCoeffsAtKpoint(k, onlyCoeffs; invert=true) for k in calculatePoints])
            else
                for k in calculatePoints
                    push!(specFuncResults[1], SpectralCoeffsAtKpoint(k, onlyCoeffs))
                    push!(specFuncResults[2], SpectralCoeffsAtKpoint(k, onlyCoeffs; invert=true))
                end
            end
        end
        for results in specFuncResults
            for (index, specFuncResultsPoint) in enumerate(results)
                for (name, val) in specFuncResultsPoint 
                    if name ∉ onlyCoeffs
                        fixedContrib[index] .+= val
                    else
                        append!(specCoeffsBZone[index], val)
                    end
                end
            end
        end

        centerSpecFuncArr, localSpecFunc, standDevFinal = SpecFuncVariational(specCoeffsBZone, freqValues, targetHeight, 1e-3; 
                                       degenTol=1e-10, silent=false, 
                                       broadFuncType="lorentz", fixedContrib=fixedContrib,
                                       standDevGuess=standDevGuess,
                                      )
        jldsave(savePath; 
                centerSpecFuncArr=centerSpecFuncArr,
                fixedContrib=fixedContrib,
                localSpecFunc=localSpecFunc,
                standDevFinal=standDevFinal,
               )
    end

    specFuncKSpace = [freqValues |> length |> zeros for _ in calculatePoints]

    results = Dict("kspaceDOS" => zeros(size_BZ^2), "quasipRes" => zeros(size_BZ^2), "selfEnergyKspace" => zeros(size_BZ^2))

    results["kspaceDOS"] .= NaN
    results["quasipRes"] .= NaN
    results["selfEnergyKspace"] .= NaN

    for (index, centerSpecFunc) in enumerate(centerSpecFuncArr)
        centerArea = sum(centerSpecFunc) * (maximum(freqValues) - minimum(freqValues)) / (length(freqValues) - 1)
        sideArea = sum(fixedContrib[index]) * (maximum(freqValues) - minimum(freqValues)) / (length(freqValues) - 1)
        results["quasipRes"][calculatePoints[index]] = centerArea / (centerArea + sideArea)
        specFuncKSpace[index] = Normalise(centerSpecFunc .+ fixedContrib[index], freqValues, true)
        results["kspaceDOS"][calculatePoints[index]] = specFuncKSpace[index][freqValues .≥ 0][1]
    end

    if isnothing(nonIntSpecBzone)
        nonIntSpecBzone = [zeros(length(freqValues)) for _ in calculatePoints]
        for (index, centerSpecFunc) in enumerate(centerSpecFuncArr)
            nonIntSpecBzone[index] = centerSpecFunc
        end
    end

    for index in eachindex(calculatePoints)
        imagSelfEnergy = imag(SelfEnergyHelper(specFuncKSpace[index], freqValues, nonIntSpecBzone[index]; 
                                                                               normalise=true, 
                                                                               pinBottom=maximum(abs.(hamiltDetails["kondoJArray"][calculatePoints[index], SWIndicesAll])) > 0
                                                                              ))
        results["selfEnergyKspace"][calculatePoints[index]] = -(sum(imagSelfEnergy[abs.(freqValues) .≤ selfEnergyWindow]) * (maximum(freqValues) - minimum(freqValues)) ./ (length(freqValues) -1))
        results["selfEnergyKspace"][calculatePoints[index]] = minimum((1e2, results["selfEnergyKspace"][calculatePoints[index]]))
        results["selfEnergyKspace"][calculatePoints[index]] = maximum((1e-2, results["selfEnergyKspace"][calculatePoints[index]]))
    end

    results = PropagateIndices(calculatePoints, results, size_BZ, 
                                 oppositePoints)

    return specFuncKSpace, localSpecFunc, standDevFinal, results, nonIntSpecBzone
end
