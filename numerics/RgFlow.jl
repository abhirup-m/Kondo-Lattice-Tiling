using LinearAlgebra, JLD2

function initialiseKondoJ(
        size_BZ::Int64, 
        num_steps::Int64,
        kondoF::Float64
    )
    # Kondo coupling must be stored in a 3D matrix. Two of the dimensions store the 
    # incoming and outgoing momentum indices, while the third dimension stores the 
    # behaviour along the RG flow. For example, J[i][j][k] reveals the value of J 
    # for the momentum pair (i,j) at the k^th Rg step.
    kondoJArray = Array{Float64}(undef, size_BZ^2, size_BZ^2)
    k1x_vals, k1y_vals = map1DTo2D(collect(1:size_BZ^2), size_BZ)
    kondoJArray[:, :] .= 0.5 * kondoF .* (cos.(k1x_vals' .- k1x_vals) .+ cos.(k1y_vals' .- k1y_vals))
    return kondoJArray
end


function getCutOffEnergy(size_BZ)
    kx_pos_arr = [kx for kx in range(K_MIN, K_MAX, length=size_BZ) if kx >= 0]
    return sort(-tightBindDisp(kx_pos_arr, 0 .* kx_pos_arr), rev=true)
end


function highLowSeparation(
        dispersionArray,
        energyCutoff,
        proceedFlags,
        size_BZ
    )

    # get the k-points that will be decoupled at this step, by getting the isoenergetic contour at the cutoff energy.
    cutoffPoints = unique(getIsoEngCont(dispersionArray, energyCutoff))
    cutoffHolePoints = particleHoleTransf(cutoffPoints, size_BZ)

    # these cutoff points will no longer participate in the RG flow, so disable their flags
    for key in ["d", "f"]
        proceedFlags[key][[cutoffPoints; cutoffHolePoints], :] .= 0
        proceedFlags[key][:, [cutoffPoints; cutoffHolePoints]] .= 0
    end

    # get the k-space points that need to be tracked for renormalisation, by getting the states 
    # below the cutoff energy. We only take points within the lower left quadrant, because the
    # other quadrant is obtained through symmetry relations.
    innerIndices = Dict(key => [
                                point for (point, energy) in enumerate(dispersionArray) if
                                abs(energy) < (abs(energyCutoff) - TOLERANCE)
                                && any(proceedFlags[key][:, point])
                               ]
                        for key in ["f", "d"]
                       )
    return innerIndices, cutoffPoints, cutoffHolePoints, proceedFlags
end

@everywhere function momentumSpaceRG(
        size_BZ::Int64,
        bareCouplings::Dict;
        progressbarEnabled=false,
        loadData::Bool=false,
        saveData::Bool=true,
    )
    kvals = map1DTo2D.(1:size_BZ^2, size_BZ)
    kxVals = first.(kvals)
    kyVals = last.(kvals)
    omega_by_t = bareCouplings["omega_by_t"]
    μ = 0 # bareCouplings["μ"]
    W = Dict("f" => bareCouplings["Wf"], "d" => bareCouplings["Wd"])

    saveJLD = joinpath(SAVEDIR, SavePath("rgflow", size_BZ, bareCouplings, "jld2"))
    mkpath(SAVEDIR)
    if isfile(saveJLD) && loadData
        return load(saveJLD)
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
    couplings = Dict("f" => initialiseKondoJ(size_BZ, div(size_BZ + 1, 2), bareCouplings["Jf"]),
                     "d" => initialiseKondoJ(size_BZ, div(size_BZ + 1, 2), bareCouplings["Jd"]),
                     "⟂" => bareCouplings["J⟂"],
                )
    if couplings["⟂"] == 0
        couplings["⟂"] = 1e-3
    end

    initSigns = Dict(k => sign.(v) for (k, v) in couplings)

    # define flags to track whether the RG flow for a particular J_{k1, k2} needs to be stopped 
    # (perhaps because it has gone to zero, or its denominator has gone to zero). These flags are
    # initialised to one, which means that by default, the RG can proceed for all the momenta.
    proceedFlags = Dict("f" => fill(true, size_BZ^2, size_BZ^2), "d" => fill(true, size_BZ^2, size_BZ^2), "⟂" => [true])

    # Run the RG flow starting from the maximum energy, down to the penultimate energy (ΔE), in steps of ΔE

    initDeltaSigns = Dict("d" => repeat([1.], size_BZ^2, size_BZ^2),
                          "f" => repeat([1.], size_BZ^2, size_BZ^2),
                          "⟂" => 1.
                         )
    #=initDeltaSignPerp = 1=#

    GMatrix = Dict(k => 0 .* couplings[k] for k in ["f", "d"])
    WMatrix = 0.5 .* (cos.(kxVals' .- kxVals) .+ cos.(kyVals' .- kyVals))
    node = map2DTo1D(π/2, π/2, size_BZ)
    for (stepIndex, energyCutoff) in enumerate(cutOffEnergies[1:end-1])
        #=println(couplings["d"][node, node], node)=#
        deltaEnergy = abs(cutOffEnergies[stepIndex+1] - cutOffEnergies[stepIndex])

        # if there are no enabled flags (i.e., all are zero), stop the RG flow
        if !any([any(v) for v in values(proceedFlags)])
            break
        end

        innerIndices, cutoffPoints, cutoffHolePoints, proceedFlags = highLowSeparation(dispersionArray, energyCutoff, proceedFlags, size_BZ)
        GMatrix["f"] .= 0.
        GMatrix["d"] .= 0.
        GVector = zeros(length(cutoffPoints))
        for (i_q, (q, qbar)) in enumerate(zip(cutoffPoints, cutoffHolePoints))
            for k in ["f", "d"]
                GMatrix[k][q, q] = densityOfStates[q] * ((1/8) / (omega_by_t * HOP_T - energyCutoff / 2 + μ / 2 + couplings[k][q, q] / 4 + W[k] / 2 + 0.75 * couplings["⟂"]) 
                                                         + (3/8) / (omega_by_t * HOP_T - energyCutoff / 2 + μ / 2 + couplings[k][q, q] / 4 + W[k] / 2 - 0.25 * couplings["⟂"])
                                                         + (1/8) / (omega_by_t * HOP_T - energyCutoff / 2 - μ / 2 + couplings[k][qbar, qbar] / 4 + W[k] / 2 + 0.75 * couplings["⟂"]) 
                                                         + (3/8) / (omega_by_t * HOP_T - energyCutoff / 2 - μ / 2 + couplings[k][qbar, qbar] / 4 + W[k] / 2 - 0.25 * couplings["⟂"])
                                                        )
            end
            GVector[i_q] = sum([densityOfStates[q] / (omega_by_t * HOP_T - 0.5 * (dispersionArray[q] - dispersionArray[qbar]) + couplings[k][q, q] / 2 + W[k] - 0.25 * couplings["⟂"]) for k in ["f", "d"]])
        end
        traceGprime = Dict(k => sum([GMatrix[k][q, q] * couplings[k][q, qbar] for (q,qbar) in zip(cutoffPoints, cutoffHolePoints)]) for k in ["f", "d"])
        JVector = [couplings["f"][q, qbar] * couplings["d"][qbar, q] for (q,qbar) in zip(cutoffPoints, cutoffHolePoints)]

        delta = Dict()
        for k in ["f", "d"]
            delta[k] = (couplings[k][innerIndices[k], cutoffPoints] * GMatrix[k][cutoffPoints, cutoffPoints] * couplings[k][cutoffPoints, innerIndices[k]] .+ W[k] * traceGprime[k] .* WMatrix[innerIndices[k], innerIndices[k]])
        end
        delta["⟂"] = 0.5 * (JVector' * GVector)
        if step == 1
            initDeltaSigns["f"] = sign.(delta["f"])
            initDeltaSigns["d"] = sign.(delta["d"])
            initDeltaSigns["⟂"] = sign(delta["⟂"])
        else
            for k in ["f", "d"]
                initDeltaSigns[k][innerIndices[k], innerIndices[k]][sign.(delta[k]) .* initDeltaSigns[k][innerIndices[k], innerIndices[k]] .< 0] .= 0.
                delta[k][initDeltaSigns[k][innerIndices[k], innerIndices[k]] .== 0] .= 0.
            end
            if sign(delta["⟂"]) * initDeltaSigns["⟂"] < 0
                initDeltaSigns["⟂"] = 0.
                delta["⟂"] = 0.
            end
        end
        for k in ["f", "d"]
            couplings[k][innerIndices[k], innerIndices[k]] .+= delta[k]
            proceedFlags[k][innerIndices[k], innerIndices[k]] .= couplings[k][innerIndices[k], innerIndices[k]] .* initSigns[k][innerIndices[k], innerIndices[k]] .≤ 0
            couplings[k][sign.(couplings[k]) .* initSigns[k] .< 0] .= 0.
        end
        couplings["⟂"] += delta["⟂"]
        if sign(couplings["⟂"]) * initSigns["⟂"] < 0
            proceedFlags["⟂"] = [false]
            couplings["⟂"] = 0.
        end
    end
    if saveData
        save(saveJLD, couplings)
    end
    return couplings
end
