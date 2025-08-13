using LinearAlgebra, Distributed, ProgressMeter

function getCutOffEnergy(size_BZ)
    kx_pos_arr = [kx for kx in range(K_MIN, K_MAX, length=size_BZ) if kx >= 0]
    return sort(-tightBindDisp(kx_pos_arr, 0 .* kx_pos_arr), rev=true)
end


function highLowSeparation(dispersionArray::Vector{Float64}, energyCutoff::Float64, proceed_flags::Matrix{Int64}, size_BZ::Int64)

    # get the k-points that will be decoupled at this step, by getting the isoenergetic contour at the cutoff energy.
    cutoffPoints = unique(getIsoEngCont(dispersionArray, energyCutoff))
    cutoffHolePoints = particleHoleTransf(cutoffPoints, size_BZ)

    # these cutoff points will no longer participate in the RG flow, so disable their flags
    proceed_flags[cutoffPoints, :] .= 0
    proceed_flags[:, cutoffPoints] .= 0
    proceed_flags[cutoffHolePoints, :] .= 0
    proceed_flags[:, cutoffHolePoints] .= 0

    # get the k-space points that need to be tracked for renormalisation, by getting the states 
    # below the cutoff energy. We only take points within the lower left quadrant, because the
    # other quadrant is obtained through symmetry relations.
    innerIndicesArr = [
        point for (point, energy) in enumerate(dispersionArray) if
        abs(energy) < (abs(energyCutoff) - TOLERANCE) &&
        map1DTo2D(point, size_BZ)[1] <= 0.5 * (K_MAX + K_MIN)
    ]
    excludedIndicesArr = [
        point for (point, energy) in enumerate(dispersionArray) if
        abs(energy) < (abs(energyCutoff) - TOLERANCE) &&
        map1DTo2D(point, size_BZ)[1] > 0.5 * (K_MAX + K_MIN)
    ]
    excludedVertexPairs = [
        (p1, p2) for p1 in sort(excludedIndicesArr) for
        p2 in sort(excludedIndicesArr)[sort(excludedIndicesArr).>=p1]
    ]
    mixedVertexPairs = [(p1, p2) for p1 in innerIndicesArr for p2 in excludedIndicesArr]
    return innerIndicesArr, excludedVertexPairs, mixedVertexPairs, cutoffPoints, cutoffHolePoints, proceed_flags
end


function initialiseKondoJ(
        size_BZ::Int64, 
        num_steps::Int64,
        kondo_f::Float64
    )
    # Kondo coupling must be stored in a 3D matrix. Two of the dimensions store the 
    # incoming and outgoing momentum indices, while the third dimension stores the 
    # behaviour along the RG flow. For example, J[i][j][k] reveals the value of J 
    # for the momentum pair (i,j) at the k^th Rg step.
    kondoJArray = Array{Float64}(undef, size_BZ^2, size_BZ^2, num_steps)
    k1x_vals, k1y_vals = map1DTo2D(collect(1:size_BZ^2), size_BZ)
    kondoJArray[:, :, 1] .= 0.5 * kondo_f .* (cos.(k1x_vals' .- k1x_vals) .+ cos.(k1y_vals' .- k1y_vals))
    return kondoJArray
end


function deltaJk1k2(
    denominators::Vector{Float64},
    proceed_flagk1k2::Int64,
    kondoJArrayPrev_k1k2::Float64,
    kondoJ_k2q_qk1::Vector{Float64},
    kondoJ_qqbar::Vector{Float64},
    deltaEnergy::Float64,
    bathIntArgs,
    densityOfStates_q::Vector{Float64},
    interLayerTerm::Float64,
)
    # if the flag is disabled for this momentum pair, don't bother to do the rest
    if proceed_flagk1k2 == 0 || length(denominators) == 0
        return 0, 0
    end

    # the renormalisation itself is given by the expression
    # ΔJ(k1, k2) = ∑_q [J(k2,q)J(q,k1) + 4 * J(q,qbar) * W(qbar, k2, k1, q)]/[ω - E/2 + J(q)/4 + W(q)/2]
    renormalisation =
        -deltaEnergy * sum(
            densityOfStates_q .*
            ((kondoJ_k2q_qk1 .+ 4 .* kondoJ_qqbar .* bathIntForm(bathIntArgs...)) ./ denominators .+ kondoJArrayPrev_k1k2 * interLayerTerm)
        )

    # if a non-zero coupling goes through a zero, we set it to zero, and disable its flag.
    if abs(kondoJArrayPrev_k1k2) > TOLERANCE && (kondoJArrayPrev_k1k2 + renormalisation) * kondoJArrayPrev_k1k2 < 0
        kondoJArrayNext_k1k2 = 0
        proceed_flagk1k2 = 0
    else
        kondoJArrayNext_k1k2 = kondoJArrayPrev_k1k2 + renormalisation
    end

    return kondoJArrayNext_k1k2, proceed_flagk1k2
end


function symmetriseRGFlow(innerIndicesArr, excludedVertexPairs, mixedVertexPairs, size_BZ, kondoJArrayNext, kondoJArrayPrev, proceed_flags)
    Threads.@threads for (innerIndex1, excludedIndex) in mixedVertexPairs
        innerIndex2 = particleHoleTransf(excludedIndex, size_BZ)
        @assert innerIndex1 in innerIndicesArr
        @assert innerIndex2 in innerIndicesArr
        kondoJArrayNext[innerIndex1, excludedIndex] = kondoJArrayNext[excludedIndex, innerIndex1] = -kondoJArrayNext[innerIndex1, innerIndex2]
        proceed_flags[innerIndex1, excludedIndex] = proceed_flags[innerIndex1, innerIndex2]
        proceed_flags[excludedIndex, innerIndex1] = proceed_flags[innerIndex1, innerIndex2]
    end
    Threads.@threads for (index1, index2) in excludedVertexPairs
        sourcePoint1, sourcePoint2 = particleHoleTransf([index1, index2], size_BZ)
        @assert sourcePoint1 in innerIndicesArr
        @assert sourcePoint2 in innerIndicesArr
        kondoJArrayNext[index1, index2] = kondoJArrayNext[index2, index1] = kondoJArrayNext[sourcePoint1, sourcePoint2]
        proceed_flags[index1, index2] = proceed_flags[sourcePoint1, sourcePoint2]
        proceed_flags[index2, index1] = proceed_flags[sourcePoint1, sourcePoint2]
    end
    return kondoJArrayNext, proceed_flags
end


function stepwiseRenormalisation(
    innerIndicesArr::Vector{Int64},
    omega_by_t::Float64,
    excludedVertexPairs::Vector{Tuple{Int64,Int64}},
    mixedVertexPairs::Vector{Tuple{Int64,Int64}},
    energyCutoff::Float64,
    cutoffPoints::Vector{Int64},
    cutoffHolePoints::Vector{Int64},
    proceed_flags::Matrix{Int64},
    kondoJArrayPrev::Array{Float64,2},
    kondoJArrayNext::Array{Float64,2},
    Wf::Float64,
    size_BZ::Int64,
    deltaEnergy::Float64,
    densityOfStates::Vector{Float64},
    interLayerTerm::Float64,
)

    # construct denominators for the RG equation, given by
    # d = ω - E/2 + J(q)/4 + W(q)/2

    OMEGA = omega_by_t * HOP_T
    denominators =
        OMEGA .- abs(energyCutoff) / 2 .+ diag(
            kondoJArrayPrev[cutoffPoints, cutoffPoints] / 4 .+
            bathIntForm(
                Wf,
                size_BZ,
                Tuple([cutoffPoints, cutoffPoints, cutoffPoints, cutoffPoints]),
            ) / 2,
           )

    # only consider those terms whose denominator haven't gone through zeros
    cutoffPoints = cutoffPoints[denominators.<0]
    cutoffHolePoints = cutoffHolePoints[denominators.<0]
    denominators = denominators[denominators.<0]

    if length(cutoffPoints) == 0
        proceed_flags[:, :] .= 0
        return kondoJArrayNext, proceed_flags
    end

    # loop over (k1, k2) pairs that represent the momentum states within the emergent window,
    # so that we can calculate the renormalisation of J(k1, k2), for all k1, k2.
    externalVertexPairs = [
        (p1, p2) for p1 in sort(innerIndicesArr) for
        p2 in sort(innerIndicesArr)[sort(innerIndicesArr).>=p1]
    ]
    kondoJ_qq_bar = diag(kondoJArrayPrev[cutoffPoints, cutoffHolePoints])
    dOfStates_cutoff = densityOfStates[cutoffPoints]
    Threads.@threads for (innerIndex1, innerIndex2) in externalVertexPairs
        kondoJ_k2_q_times_kondoJ_q_k1 = diag(kondoJArrayPrev[innerIndex2, cutoffPoints] * kondoJArrayPrev[innerIndex1, cutoffPoints]')
        kondoJArrayNext_k1k2, proceed_flag_k1k2 = deltaJk1k2(
            denominators,
            proceed_flags[innerIndex1, innerIndex2],
            kondoJArrayPrev[innerIndex1, innerIndex2],
            kondoJ_k2_q_times_kondoJ_q_k1,
            kondoJ_qq_bar,
            deltaEnergy,
            [
                Wf,
                size_BZ,
                Tuple([cutoffHolePoints, innerIndex2, innerIndex1, cutoffPoints]),
            ],
            dOfStates_cutoff,
            interLayerTerm,
        )
        kondoJArrayNext[innerIndex1, innerIndex2] = kondoJArrayNext_k1k2
        kondoJArrayNext[innerIndex2, innerIndex1] = kondoJArrayNext_k1k2
        proceed_flags[innerIndex1, innerIndex2] = proceed_flag_k1k2
        proceed_flags[innerIndex2, innerIndex1] = proceed_flag_k1k2
    end
    kondoJArrayNext, proceed_flags = symmetriseRGFlow(innerIndicesArr, excludedVertexPairs, mixedVertexPairs, size_BZ, kondoJArrayNext, kondoJArrayPrev, proceed_flags)
    return kondoJArrayNext, proceed_flags
end

@everywhere function momentumSpaceRG(
        size_BZ::Int64,
        couplings::Dict{String, Float64};
        progressbarEnabled=false,
        loadData::Bool=false,
        saveData::Bool=true,
    )
    omega_by_t, kondo_f, kondo_perp, Wf = [couplings[k] for k in ["omega_by_t", "kondo_f", "kondo_perp", "Wf"]]
    Wc, epsilon_f, mu_c, lightBandFactor = [couplings[k] for k in ["Wc", "epsilon_f", "mu_c", "lightBandFactor"]]

    #=savePath = joinpath(SAVEDIR, "rgflow-$(size_BZ)-$(omega_by_t)-$(kondo_f)-$(Wf)-$(kondo_perp)-$(Wc)-$(epsilon_f)-$(mu_c).jld2")=#
    savePath = joinpath(SAVEDIR, SavePath("rgflow", size_BZ, couplings, "jld2"))
    mkpath(SAVEDIR)
    if isfile(savePath) && loadData
        _, dispersionArray = getDensityOfStates(tightBindDisp, size_BZ)
        kondoJArray = zeros(size_BZ^2, size_BZ^2, 2)
        kondoJArray[:, :, 1] .= initialiseKondoJ(size_BZ, trunc(Int, (size_BZ + 1) / 2), kondo_f)[:, :, 1]
        loadedData = jldopen(savePath)
        kondoJArray[:, :, 2] = loadedData["kondoRenorm"]
        kondoPerpArray = loadedData["kondoPerpArray"]
        return kondoJArray, kondoPerpArray, dispersionArray
    end


    # ensure that [0, \pi] has odd number of states, so 
    # that the nodal point is well-defined.
    @assert (size_BZ - 5) % 4 == 0 "Size of Brillouin zone must be of the form N = 4n+5, n=0,1,2..., so that all the nodes and antinodes are well-defined."

    densityOfStates, dispersionArray = getDensityOfStates(tightBindDisp, size_BZ)

    cutOffEnergies = getCutOffEnergy(size_BZ)

    # Kondo coupling must be stored in a 3D matrix. Two of the dimensions store the 
    # incoming and outgoing momentum indices, while the third dimension stores the 
    # behaviour along the RG flow. For example, J[i][j][k] reveals the value of J 
    # for the momentum pair (i,j) at the k^th Rg step.
    kondoJArray = initialiseKondoJ(size_BZ, trunc(Int, (size_BZ + 1) / 2), kondo_f)

    kondoPerpArray = zeros(length(cutOffEnergies))
    kondoPerpArray[1] = kondo_perp

    # define flags to track whether the RG flow for a particular J_{k1, k2} needs to be stopped 
    # (perhaps because it has gone to zero, or its denominator has gone to zero). These flags are
    # initialised to one, which means that by default, the RG can proceed for all the momenta.
    proceed_flags = fill(1, size_BZ^2, size_BZ^2)

    # Run the RG flow starting from the maximum energy, down to the penultimate energy (ΔE), in steps of ΔE

    perpDeltaSign = nothing
    for (stepIndex, energyCutoff) in enumerate(cutOffEnergies[1:end-1])
        deltaEnergy = abs(cutOffEnergies[stepIndex+1] - cutOffEnergies[stepIndex])

        # set the Kondo coupling of all subsequent steps equal to that of the present step 
        # for now, so that we can just add the renormalisation to it later
        kondoJArray[:, :, stepIndex+1] = kondoJArray[:, :, stepIndex]

        # if there are no enabled flags (i.e., all are zero), stop the RG flow
        if all(==(0), proceed_flags)
            kondoJArray[:, :, stepIndex+2:end] .= kondoJArray[:, :, stepIndex]
            break
        end

        innerIndicesArr, excludedVertexPairs, mixedVertexPairs, cutoffPoints, cutoffHolePoints, proceed_flags = highLowSeparation(dispersionArray, energyCutoff, proceed_flags, size_BZ)

        lightDenominators = [omega_by_t * HOP_T * lightBandFactor - cutOffEnergies[stepIndex] * lightBandFactor / 2 + kondoPerpArray[stepIndex]/4 + Wc /2 + mu_c / 2,
                            omega_by_t * HOP_T * lightBandFactor - cutOffEnergies[stepIndex] * lightBandFactor / 2 + kondoPerpArray[stepIndex]/4 + Wc /2 - mu_c / 2
                           ]
        innerLoopMomenta = [
            point for (point, energy) in enumerate(dispersionArray) if
            abs(energy) < (abs(energyCutoff) - TOLERANCE)
        ]
        interLayerTerm = -0.5 * kondoPerpArray[stepIndex]^2 * sum(densityOfStates[innerLoopMomenta]) * deltaEnergy * sum(0.5 ./ lightDenominators)^2

        # calculate the renormalisation for this step and for all k1,k2 pairs
        kondoJArrayNext, proceed_flags_updated = stepwiseRenormalisation(
            innerIndicesArr,
            omega_by_t,
            excludedVertexPairs,
            mixedVertexPairs,
            energyCutoff,
            cutoffPoints,
            cutoffHolePoints,
            proceed_flags,
            kondoJArray[:, :, stepIndex],
            kondoJArray[:, :, stepIndex+1],
            Wf,
            size_BZ,
            deltaEnergy,
            densityOfStates,
            interLayerTerm,
        )
        kondoJArray[:, :, stepIndex+1] = round.(kondoJArrayNext, digits=trunc(Int, -log10(TOLERANCE)))
        proceed_flags = proceed_flags_updated

        kondoPerpArray[(stepIndex + 1):end] .= kondoPerpArray[stepIndex]

        if !isnothing(perpDeltaSign) && perpDeltaSign == 0
            continue
        end
        filter!(<(0), lightDenominators)
        if !isempty(lightDenominators)
            kondoPerpArray[stepIndex + 1] -= (kondoPerpArray[stepIndex]^2 + 4 * kondoPerpArray[stepIndex] * Wc) * deltaEnergy * sum(densityOfStates[cutoffPoints]) * 0.5 * sum(1 ./ lightDenominators)
        end

        for q in cutoffPoints
            heavyDenominator = omega_by_t * HOP_T - cutOffEnergies[stepIndex] / 2 + Wf / 2 + epsilon_f + kondoJArray[q, q, stepIndex] / 4
            if heavyDenominator ≥ 0
                continue
            end
            kondoPerpArray[stepIndex + 1] -= 0.5 * kondoPerpArray[stepIndex] * deltaEnergy * densityOfStates[q] * sum([densityOfStates[k] * deltaEnergy * kondoJArray[q, k, stepIndex].^2 for k in innerLoopMomenta]) / (heavyDenominator ^ 2)
        end
        if isnothing(perpDeltaSign)
            perpDeltaSign = sign(kondoPerpArray[stepIndex+1] - kondoPerpArray[stepIndex])
        elseif sign(kondoPerpArray[stepIndex+1] - kondoPerpArray[stepIndex]) * perpDeltaSign < 0
            kondoPerpArray[stepIndex+1] = kondoPerpArray[stepIndex]
            perpDeltaSign = 0
        end

    end
    if saveData
        jldsave(savePath, true; kondoRenorm=kondoJArray[:, :, end], kondoPerpArray=kondoPerpArray)
    end
    return kondoJArray, kondoPerpArray, dispersionArray
end
