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
        size_BZ::Int64,
        maxSize::Int64,
        momentumPoints::Dict{String, Vector{Int64}},
        typesOrder::Vector{String},
        specFunc::Dict;
        addPerStep::Int64=1,
        tolerance=-Inf,
    )

    t1, t2 = typesOrder
    # only momentum points with ky ≥ 0 need to be
    # solved for, the rest can be mapped exactly
    truncatedPoints = Dict()
    for k in ["f", "d"]
        #=truncatedPoints[k] = filter(p -> true, momentumPoints[k])=#
        truncatedPoints[k] = filter(p -> map1DTo2D(p, size_BZ)[2] ≥ 0, momentumPoints[k])
    end

    totalSize = length(truncatedPoints["f"]) + 1

    # define Kondo matrix just for the upper half, for both layers
    J = zeros(totalSize, totalSize)
    
    # this stores the information of which indices in J
    # store couplings from which layer. Since the first N
    # indices are f, we set them true, the last N is set to false.
    layerSpecs = repeat([t1], totalSize-1)
    push!(layerSpecs, t2)

    # the first NxN points store the d-Kondo couplings,
    # while the last NxN store the f-couplings. This means
    # that d-correlations must be calculated from the 
    # upper half, while f-correlations come from lower half.
    hybrid = hamiltDetails["V"][t1][truncatedPoints[t1]]
    push!(hybrid, (sum(hamiltDetails["V"][t2].^2)^0.5) / length(hamiltDetails["V"][t2]))
    indices = 1:length(truncatedPoints[t1])
    J[end, end] = (sum(diag(hamiltDetails[t2][truncatedPoints[t2], truncatedPoints[t2]]).^2)^0.5) / length(truncatedPoints[t2])

    momentumMapping = Dict()
    momentumMapping[t1] = Dict(zip(truncatedPoints[t1], indices))
    momentumMapping[t2] = Dict(map2DTo1D(π/2, π/2, size_BZ) => 1)
    J[indices, indices] = hamiltDetails[t1][truncatedPoints[t1], truncatedPoints[t1]]

    sortseq = sortperm([sum(Ji.^2) for Ji in eachcol(J[indices,indices])], rev=true)
    J[indices,indices] = J[indices,indices][sortseq, sortseq]
    layerSpecs[indices] = layerSpecs[indices][sortseq]
    hybrid[indices] = hybrid[indices][sortseq]
    momentumMapping[t1] = Dict(p => findfirst(==(i), sortseq) for (p, i) in momentumMapping[t1])

    #=channel1 = findall(p -> prod(map1DTo2D(p, size_BZ)) ≥ 0, truncatedPoints[t1])=#
    #=channel2 = findall(p -> prod(map1DTo2D(p, size_BZ)) < 0, truncatedPoints[t1])=#
    #=J_channel = 0 .* J[indices, indices]=#
    #=hybrid_channel = 0 .* hybrid[indices]=#
    #=sortseq = collect(1:length(indices))=#
    #=println(channel1, channel2)=#
    #=for channel in [channel1, channel2]=#
    #=    kxVals = first.(map1DTo2D.(truncatedPoints[t1][channel], size_BZ))=#
    #=    kyVals = last.(map1DTo2D.(truncatedPoints[t1][channel], size_BZ))=#
    #=    xVals = -div(length(channel), 2):div(length(channel)-1, 2)=#
    #=    fourierPhase = exp.(1im .* xVals .* kxVals')=#
    #=    J_channel[channel, channel] .= real.(fourierPhase * J[channel, channel] * fourierPhase') ./ length(channel)=#
    #=    hybrid_channel[channel] .= 0 * real.(fourierPhase * hybrid[channel])=#
    #=    sortseq[channel] = sortperm([Ji^2 for Ji in diag(J_channel[channel, channel])], rev=true)=#
    #=    J_channel[channel, channel] = J_channel[channel, channel][sortseq[channel], sortseq[channel]]=#
    #=    layerSpecs[channel] = layerSpecs[channel][sortseq[channel]]=#
    #=    hybrid_channel[channel] = hybrid[channel][sortseq[channel]]=#
    #=end=#
    #=J[indices, indices] .= J_channel=#
    #=hybrid[indices] .= hybrid_channel=#


    # obtain Hamiltonian with sorted Kondo matrix
    hamiltonian = BilayerLEE(
                             J,
                             hamiltDetails["⟂"],
                             hybrid,
                             hamiltDetails["η"],
                             hamiltDetails["impCorr"],
                             layerSpecs;
                             globalField=1e-8,
                             couplingTolerance=1e-10,
                            )
    # split hamiltonian into chunks for iterative diagonalisation.
    # The first chunk has 10 qubits, then we subsequently keep
    # adding 2 qubits (k↑, k↓) every iteration.
    indexPartitions = [10]
    while indexPartitions[end] < 2 * (2 + length(layerSpecs))
        push!(indexPartitions, indexPartitions[end] + 2)
    end
    hamiltonianFamily = MinceHamiltonian(hamiltonian, indexPartitions)
    @assert all(!isempty, hamiltonianFamily)
    for h_i in hamiltonianFamily
        @assert all(!isempty, h_i)
    end

    specFuncDefDict = Dict{String, Dict{String, Vector{Tuple{String, Vector{Int64}, Float64}}}}()
    fd_inds = Dict(k => findall(==(k), layerSpecs) for k in ["f", "d"])
    for (k, v) in specFunc
        if k == "ω" || k == "η"
            continue
        end
        type = v[1]
        @assert type ∈ ["if", "id", "f", "d", "+-"]
        if (type == "if" || type =="id") && type[2:2] == t1
            specFuncDefDict[k] = Dict()
            specFuncDefDict[k]["create"] = v[2]
        end
        if (type == "f" || type == "d") && type == t1
            specFuncDefDict[k] = Dict()
            func = v[2]
            specFuncDefDict[k]["create"] = vcat(unique([func(3 + 2 * i) for i in fd_inds[type]])...)
        end
        if type == "+-"
            specFuncDefDict[k] = Dict()
            condition, func = v[2:3]
            specFuncDefDict[k]["create"] = []
            mom = Dict(t2 => 3 + 2 * findfirst(==(t2), layerSpecs))
            kondoAvg = Dict()
            kondoAvg[t2] = maximum((1e-3, sum(abs.(hamiltDetails[t2][truncatedPoints[t2], truncatedPoints[t2]])) / length(truncatedPoints[t2])^2))
            for p in filter(condition, truncatedPoints[t1])
                mom[t1] = 3 + 2 * momentumMapping[t1][p]
                kondoAvg[t1] = maximum((1e-3, sum(abs.(hamiltDetails[t2][p, truncatedPoints[t1]])) / length(truncatedPoints[t2])))
                append!(specFuncDefDict[k]["create"], func(mom, kondoAvg))
            end
        end
        if haskey(specFuncDefDict, k)
            specFuncDefDict[k]["destroy"] = Dagger(copy(specFuncDefDict[k]["create"]))
        end
    end
    results = IterDiag(
                      hamiltonianFamily,
                      maxSize;
                      symmetries=Char['N', 'S'],
                      #=magzReq=(m, N) -> -5 ≤ m ≤ 5,=#
                      #=occReq=(x, N) -> div(N, 2) - 5 ≤ x ≤ div(N, 2) + 5,=#
                      specFuncDefDict=specFuncDefDict,
                      #=excludeLevels=E -> abs(E) > 1.0,=#
                      silent=false,
                      maxMaxSize=maxSize,
                     )

    @assert results["exitCode"] == 0

    corrResults = Dict(k => results[k] for k in keys(specFuncDefDict))
    return corrResults
end


@everywhere function IterDiagMomentumSpace(
        hamiltDetails::Dict,
        size_BZ::Int64,
        maxSize::Int64,
        momentumPoints::Dict{String, Vector{Int64}},
        correlation::Dict,
        entanglement::Dict;
        addPerStep::Int64=1,
        tolerance=-Inf,
    )

    # only momentum points with ky ≥ 0 need to be
    # solved for, the rest can be mapped exactly
    truncatedPoints = Dict()
    for k in ["f", "d"]
        truncatedPoints[k] = filter(p -> map1DTo2D(p, size_BZ)[2] ≥ 0, momentumPoints[k])
    end

    totalSize = sum(length.(values(truncatedPoints)))

    # define Kondo matrix just for the upper half, for both layers
    J = zeros(totalSize, totalSize)
    hybrid = zeros(totalSize)

    # this stores the information of which indices in J
    # store couplings from which layer. Since the first N
    # indices are f, we set them true, the last N is set to false.
    layerSpecs = String[]

    # the first NxN points store the d-Kondo couplings,
    # while the last NxN store the f-couplings. This means
    # that d-correlations must be calculated from the 
    # upper half, while f-correlations come from lower half.
    typesOrder = ["f", "d"]

    # go to diagonal basis of J, to make matrix sparse
    stargraph = 0 .* J
    unitary = 0 .* J

    for (i, k) in enumerate(typesOrder)
        if k == "f"
            indices = 1:length(truncatedPoints["f"])
        else
            indices = length(truncatedPoints["f"]) .+ (1:length(truncatedPoints["d"]))
        end
        J[indices, indices] = hamiltDetails[k][truncatedPoints[k], truncatedPoints[k]]
        #=if k == "f" && all(==(0), J[indices, indices])=#
        #=    J[1, 1] = HOP_T^2 / hamiltDetails["impCorr"]["f"]=#
        #=end=#
        append!(layerSpecs, repeat([k], length(truncatedPoints[k])))
        stargraph[diagind(stargraph)[indices]], unitary[indices, indices] = eigen(Hermitian(J[indices, indices]))
        if k == "d"
            hybrid[indices] = abs.(hamiltDetails["impCorr"][k] .* unitary[indices, indices] * diag(J[indices, indices])).^0.5
        end
    end

    sortseq = sortperm(diag(stargraph), rev=true)
    filter!(i -> abs(stargraph[i, i]) > tolerance, sortseq)
    for k in ["f", "d"]
        if k ∉ layerSpecs[sortseq]
            push!(sortseq, findfirst(==(k), layerSpecs))
        end
    end
    stargraph = stargraph[sortseq, sortseq]
    layerSpecs = layerSpecs[sortseq]
    hybrid = hybrid[sortseq]

    # obtain Hamiltonian with sorted Kondo matrix
    hamiltonian = BilayerLEE(
                             stargraph,
                             hamiltDetails["⟂"],
                             hybrid,
                             hamiltDetails["η"],
                             hamiltDetails["impCorr"],
                             layerSpecs;
                             globalField=1e-8,
                             couplingTolerance=1e-10,
                            )
    # split hamiltonian into chunks for iterative diagonalisation.
    # The first chunk has 10 qubits, then we subsequently keep
    # adding 2 qubits (k↑, k↓) every iteration.
    indexPartitions = [10]
    i = 3
    while i < length(layerSpecs)
        push!(indexPartitions, indexPartitions[end] + 2)
        i += 1
    end
    hamiltonianFamily = MinceHamiltonian(hamiltonian, indexPartitions)
    @assert all(!isempty, hamiltonianFamily)
    for h_i in hamiltonianFamily
        @assert all(!isempty, h_i)
    end

    corrOps = Dict{String, Vector{Tuple{String, Vector{Int64}, Float64}}}()
    corrResults = Dict()
    for (name, (type, corrFunc)) in correlation

        # if the type of the correlation is set
        # to purely impurity, no momentum indices
        # need to be passed to it.
        if type == "i"
            corrName = name
            corrOps[corrName] = corrFunc
            continue
        end

        # If the type is not purely impurity, we need a
        # matrix to store the k-dependence.
        corrResults[name] = Dict()

        for (i, j) in Iterators.product(1:length(layerSpecs), 1:length(layerSpecs))
            if j < i
                continue
            end
            # for any pair of points in the Kondo matrix,
            # check if they come from same layer. We are
            # not interested in inter-layer correlations.
            if layerSpecs[i] == layerSpecs[j] == type
                corrName = "$name-$i-$j"

                # the i_th index along the Kondo matrix will be
                # at the position 3 + 2 * i in the hilbert space
                # sequence, because 1-4 are the impurities, so
                # (i = 1) -> 5,6 and (i = 2) -> 7,8 and so on.
                corrOps[corrName] = corrFunc(3 + 2 * i, 3 + 2 * j)
            end
        end
    end
    vne = Dict{String, Vector{Int64}}()
    mutInfo = Dict{String, NTuple{2, Vector{Int64}}}()
    for (k, v) in entanglement
        if typeof(v) == Vector{Int64}
            vne[k] = v
        elseif typeof(v) == Tuple{Vector{Int64}, Vector{Int64}}
            mutInfo[k] = v
        else
            fmax = findfirst(==("f"), layerSpecs)
            dmax = findfirst(==("d"), layerSpecs)
            mutInfo[k] = v(3 + 2 * fmax, 3 + 2 * dmax)
        end
    end

    results = IterDiag(
                      hamiltonianFamily,
                      maxSize;
                      symmetries=Char['N', 'S'],
                      #=magzReq=(m, N) -> -5 ≤ m ≤ 5,=#
                      #=occReq=(x, N) -> div(N, 2) - 5 ≤ x ≤ div(N, 2) + 5,=#
                      correlationDefDict=corrOps,
                      vneDefDict=vne,
                      mutInfoDefDict=mutInfo,
                      #=excludeLevels=E -> abs(E) > 1.0,=#
                      silent=false,
                      maxMaxSize=maxSize,
                     )
    @assert results["exitCode"] == 0
    for k in keys(entanglement)
        corrResults[k] = results[k]
    end

    for (name, (type, _)) in correlation
        if type == "i"
            corrResults[name] = results[name]
            continue
        end
        rotatedCorrMatrix = zeros(totalSize, totalSize)
        for (i, j) in Iterators.product(eachindex(layerSpecs), eachindex(layerSpecs))
            if j < i || layerSpecs[i] ≠ type || layerSpecs[j] ≠ type
                continue
            end
            corrName = "$(name)-$(i)-$(j)"
            if abs(results[corrName]) < tolerance
                rotatedCorrMatrix[sortseq[i], sortseq[j]] = 0.
            else
                rotatedCorrMatrix[sortseq[i], sortseq[j]] = results[corrName]
            end
            rotatedCorrMatrix[sortseq[j], sortseq[i]] = rotatedCorrMatrix[sortseq[i], sortseq[j]]
        end
        corrMatrix = unitary * rotatedCorrMatrix * unitary'
        for ((i1, p1), (i2, p2)) in Iterators.product(enumerate(truncatedPoints[type]), enumerate(truncatedPoints[type]))
            if type == "f"
                corrResults[name][(p1, p2)] = corrMatrix[i1, i2]
            else
                corrResults[name][(p1, p2)] = corrMatrix[length(truncatedPoints["f"]) + i1, length(truncatedPoints["f"]) + i2]
            end
            p1_prime = Nest(p1, size_BZ)
            p2_prime = Nest(p2, size_BZ)
            corrResults[name][(p1_prime, p2)] = -corrResults[name][(p1, p2)]
            corrResults[name][(p1, p2_prime)] = -corrResults[name][(p1, p2)]
            corrResults[name][(p1_prime, p2_prime)] = corrResults[name][(p1, p2)]
        end
    end
    for (name, (type, _)) in correlation
        if type == "f" || type == "d"
            corrResults[name] = [corrResults[name][(k1, k2)] for (k1, k2) in Iterators.product(momentumPoints[type], momentumPoints[type])]
        end
    end

    return corrResults
end


@everywhere function IterDiagRealSpace(
        hamiltDetails::Dict,
        size_BZ::Int64,
        maxSize::Int64,
        momentumPoints::Dict{String, Vector{Int64}},
        typesOrder::Vector{String},
        specFunc::Dict;
        addPerStep::Int64=1,
        tolerance=-Inf,
    )

    t1, t2 = typesOrder
    # only momentum points with ky ≥ 0 need to be
    # solved for, the rest can be mapped exactly
    truncatedPoints = Dict()
    for k in ["f", "d"]
        #=truncatedPoints[k] = filter(p -> true, momentumPoints[k])=#
        truncatedPoints[k] = filter(p -> map1DTo2D(p, size_BZ)[2] ≥ 0, momentumPoints[k])
    end
    println(sort(diag(hamiltDetails["f"][truncatedPoints["f"], truncatedPoints["f"]])))

    totalSize = length(truncatedPoints["f"]) + 1

    # define Kondo matrix just for the upper half, for both layers
    J = zeros(totalSize)
    
    # this stores the information of which indices in J
    # store couplings from which layer. Since the first N
    # indices are f, we set them true, the last N is set to false.
    layerSpecs = repeat([t1], totalSize-1)
    push!(layerSpecs, t2)

    # the first NxN points store the d-Kondo couplings,
    # while the last NxN store the f-couplings. This means
    # that d-correlations must be calculated from the 
    # upper half, while f-correlations come from lower half.
    hybrid = hamiltDetails["V"][t1][truncatedPoints[t1]]
    push!(hybrid, (sum(hamiltDetails["V"][t2].^2)^0.5) / length(hamiltDetails["V"][t2]))
    indices = 1:(length(truncatedPoints[t1])-1)
    sortseq = collect(indices)
    @assert length(indices) % 2 == 0
    J[end] = (sum(diag(hamiltDetails[t2][truncatedPoints[t2], truncatedPoints[t2]]).^2)^0.5) / length(truncatedPoints[t2])

    hop_eff = HOP_T
    hop_step = 2
    if any(>(0), abs.(diag(hamiltDetails[t1]))) && any(==(0), abs.(diag(hamiltDetails[t1])))
        println(1)
        J[indices[1:2:end]] .= [sum([2 * abs(hamiltDetails[t1][k, q] * cos(r * (map1DTo2D(k, size_BZ)[1] - map1DTo2D(q, size_BZ)[1]))) for k in momentumPoints[t1] for q in momentumPoints[t1] if map1DTo2D(k, size_BZ)[1] ≥ 0 && map1DTo2D(q, size_BZ)[1] ≥ 0]) for r in 0:(div(indices[end], 2)-1)]
        sortseq[indices[1:2:end]] = sortperm(abs.(J[indices[1:2:end]]), rev=true)
        J[indices[2:2:end]] .= J[indices[1:2:end]]
        sortseq[indices[2:2:end]] = sortperm(abs.(J[indices[2:2:end]]), rev=true)
    else
        J[indices] .= [sum([2 * abs(hamiltDetails[t1][k, k] * cos(r * map1DTo2D(k, size_BZ)[1])) for k in momentumPoints[t1] if map1DTo2D(k, size_BZ)[1] ≥ 0]) for r in 0:(length(indices) - 1)]
        sortseq[indices] = sortperm(abs.(J[indices]), rev=true)
        hop_step = 1
    end

    J[indices] = J[indices][sortseq]
    layerSpecs[indices] = layerSpecs[indices][sortseq]
    hybrid[indices] = hybrid[indices][sortseq]

    # obtain Hamiltonian with sorted Kondo matrix
    
    hamiltonian = BilayerLEEReal(
                                 J,
                                 hamiltDetails["⟂"],
                                 1 * hybrid,
                                 hamiltDetails["η"],
                                 hamiltDetails["impCorr"],
                                 hop_eff,
                                 layerSpecs,
                                 hop_step;
                                 globalField=1e-8,
                                 couplingTolerance=1e-10,
                            )
    # split hamiltonian into chunks for iterative diagonalisation.
    # The first chunk has 10 qubits, then we subsequently keep
    # adding 2 qubits (k↑, k↓) every iteration.
    indexPartitions = [10]
    while indexPartitions[end] < 2 * (2 + length(layerSpecs))
        push!(indexPartitions, indexPartitions[end] + 2 * hop_step)
    end
    hamiltonianFamily = MinceHamiltonian(hamiltonian, indexPartitions)
    @assert all(!isempty, hamiltonianFamily)
    for h_i in hamiltonianFamily
        @assert all(!isempty, h_i)
    end

    specFuncDefDict = Dict{String, Dict{String, Vector{Tuple{String, Vector{Int64}, Float64}}}}()
    fd_inds = Dict(k => findall(==(k), layerSpecs) for k in ["f", "d"])
    for (k, v) in specFunc
        if k == "ω" || k == "η"
            continue
        end
        type = v[1]
        @assert type ∈ ["if", "id"]
        if type[2:2] == t1
            specFuncDefDict[k] = Dict()
            specFuncDefDict[k]["create"] = v[2]
        end
        if haskey(specFuncDefDict, k)
            specFuncDefDict[k]["destroy"] = Dagger(copy(specFuncDefDict[k]["create"]))
        end
    end
    results = IterDiag(
                      hamiltonianFamily,
                      maxSize;
                      symmetries=Char['N', 'S'],
                      #=magzReq=(m, N) -> -5 ≤ m ≤ 5,=#
                      #=occReq=(x, N) -> div(N, 2) - 5 ≤ x ≤ div(N, 2) + 5,=#
                      specFuncDefDict=specFuncDefDict,
                      #=excludeLevels=E -> abs(E) > 1.0,=#
                      silent=false,
                      maxMaxSize=maxSize,
                     )

    @assert results["exitCode"] == 0

    corrResults = Dict(k => results[k] for k in keys(specFuncDefDict))
    return corrResults
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
        bareCouplings::Dict{String, Float64},
        correlation::Dict,
        momentumPoints::Dict{String, Vector{Int64}},
        entanglement::Dict,
        specFunc::Dict,
        maxSize::Int64;
        loadData::Bool=false,
        saveData::Bool=true,
    )

    _, dispersion = getDensityOfStates(tightBindDisp, size_BZ)

    saveJLD = joinpath(SAVEDIR, SavePath("CORR2P-", size_BZ, bareCouplings, "jld2"; maxSize=maxSize))
    allReqKeys = []
    append!(allReqKeys, keys(correlation))
    append!(allReqKeys, filter(k -> k ≠ "ω" && k ≠ "η", keys(specFunc)))
    append!(allReqKeys, keys(entanglement))

    couplings = momentumSpaceRG(size_BZ, bareCouplings; loadData=false, saveData=false)
    for k in ["f", "d"]
        averageKondoScale = bareCouplings["J$k"] / 2
        @assert averageKondoScale > RG_RELEVANCE_TOL
        couplings[k] .= ifelse.(abs.(couplings[k]) ./ averageKondoScale .> RG_RELEVANCE_TOL, couplings[k], 0)
    end
    couplings["η"] = Dict(k => bareCouplings["η$k"] for k in ["f", "d"])
    couplings["impCorr"] = Dict(k => bareCouplings["U$k"] for k in ["f", "d"])

    couplings["V"] = Dict(k => abs.(bareCouplings["U$k"] .* diag(couplings[k])).^0.5 for k in ["d", "f"])
    couplings["Vbare"] = Dict(k => abs(bareCouplings["U$k"] * bareCouplings["J$k"]^0.5) for k in ["d", "f"])

    availableResults = Dict()
    if loadData
        if isfile(saveJLD)
            loadedResults = load(saveJLD)
            merge!(availableResults, loadedResults)
        end
        if allReqKeys ⊆ keys(availableResults)
            println("Data available. $(join(allReqKeys, ", "))")
            return availableResults, couplings
        else
            for k in filter(k -> haskey(correlation, k), keys(availableResults))
                delete!(correlation, k)
            end
        end
    end

    if !isempty(correlation) || !isempty(entanglement)
        results = IterDiagMomentumSpace(
                                        couplings,
                                        size_BZ,
                                        maxSize,
                                        momentumPoints,
                                        correlation,
                                        entanglement;
                                        addPerStep=1,
                                       )
        merge!(availableResults, results)
    end

    if !isempty(specFunc)
        for typesOrder in [["f", "d"], ["d", "f"]]
            if any(k -> contains(k, "_loc"), keys(specFunc))
                results = IterDiagRealSpace(
                                                couplings,
                                                size_BZ,
                                                maxSize,
                                                momentumPoints,
                                                typesOrder,
                                                specFunc;
                                                addPerStep=1,
                                               )
            else
                results = IterDiagMomentumSpace(
                                                couplings,
                                                size_BZ,
                                                maxSize,
                                                momentumPoints,
                                                typesOrder,
                                                specFunc;
                                                addPerStep=1,
                                               )
            end
            for (k, v) in results
                if typesOrder[1] == "f"
                    @assert !haskey(availableResults, k)
                end
                if haskey(availableResults, k)
                    availableResults[k] = vcat.(availableResults[k], v)
                else
                    availableResults[k] = v
                end
            end
        end
    end

    if saveData
        save(saveJLD, availableResults)
    end
    return availableResults, couplings
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
