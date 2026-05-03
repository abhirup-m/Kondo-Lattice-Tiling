using LinearAlgebra

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
    couplings = Dict(k => initialiseKondoJ(size_BZ, div(size_BZ + 1, 2), bareCouplings[k]) for k in ["Jf", "Jd", "Kf", "Kd"])
    couplings["J⟂"] = bareCouplings["J⟂"]
    # if couplings["⟂"] == 0
    #     couplings["⟂"] = 1e-3
    # end

    initSigns = Dict(k => sign.(v) for (k, v) in couplings)

    # define flags to track whether the RG flow for a particular J_{k1, k2} needs to be stopped 
    # (perhaps because it has gone to zero, or its denominator has gone to zero). These flags are
    # initialised to one, which means that by default, the RG can proceed for all the momenta.
    proceedFlags = Dict(k => fill(true, size_BZ^2, size_BZ^2) for k in ["Jf", "Jd", "Kf", "Kd"])
    proceedFlags["J⟂"] = [true]

    # Run the RG flow starting from the maximum energy, down to the penultimate energy (ΔE), in steps of ΔE

    initDeltaSigns = Dict(k=> repeat([1.], size_BZ^2, size_BZ^2) for k in ["Jf", "Jd", "Kf", "Kd"])
    initDeltaSigns["J⟂"] = 1

    # GMatrix = Dict(k => 0 .* couplings[k] for k in ["f", "d"])
    WMatrix = 0.5 .* (cos.(kxVals' .- kxVals) .+ cos.(kyVals' .- kyVals))

    for (stepIndex, energyCutoff) in enumerate(cutOffEnergies[1:end-1])
        deltaEnergy = abs(cutOffEnergies[stepIndex+1] - cutOffEnergies[stepIndex])

        # if there are no enabled flags (i.e., all are zero), stop the RG flow
        if !any([any(v) for v in values(proceedFlags)])
            break
        end

        innerIndices, cutoffPoints, cutoffHolePoints, proceedFlags = highLowSeparation(dispersionArray, energyCutoff, proceedFlags, size_BZ)
        G_g = Dict(k => Dict(p => 0 .* couplings["Jf"] for p in ["+", "-"]) for k in ["Jf", "Jd", "Kf", "Kd"])
        G_aa = Dict(k => 0 .* couplings["Jf"] for k in ["Jf", "Jd", "Kf", "Kd"])
        G_JK = Dict(k => 0 .* couplings["Jf"] for k in ["f", "d"])
        di = diagind(couplings["Jf"])[cutoffPoints]
        for (g, k, kbar) in zip(["Jf", "Jd", "Kf", "Kd"], ["f", "d", "d", "f"], ["d", "f", "f", "d"])
            G[g]["+"][di] = 1 ./ (omega_by_t * HOP_T - energyCutoff / 2 + couplings[g][di] / 4 + W[k] / 2 + 0.75 * couplings["J⟂"])
            G[g]["-"][di] = 1 ./ (omega_by_t * HOP_T - energyCutoff / 2 + couplings[g][di] / 4 + W[k] / 2 - 0.25 * couplings["J⟂"])
            G_aa[g][di] = 1 ./ (omega_by_t * HOP_T - energyCutoff + couplings[g][di] / 2 + W[k] - 0.25 * couplings["J⟂"])
            G_JK[k][di] = 1 ./ (omega_by_t * HOP_T - energyCutoff / 2 + couplings["J"*k][di] / 4 + couplings["K"*kbar][di] / 4 + W[k] / 2 - 0.25 * couplings["J⟂"])
        end
        couplingqqbar = Dict(k => zeros(size_BZ^2) for k in ["Jf", "Jd", "Kf", "Kd"])
        for (q, qbar) in zip(cutoffPoints, cutoffHolePoints)
            for k in ["Jf", "Jd", "Kf", "Kd"]
                couplingqqbar[k][q] = couplings[k][q, qbar]
            end
        end
        dos = diagm(densityOfStates[1:size_BZ^2])
        #=G_inv = Dict(kbar => omega_by_t * HOP_T - energyCutoff / 2 + couplings[k][di[cutoffPoints]] / 4 + W[kbar] / 2 =#
        #=             for (k, kbar) in zip(["Jf", "Jd", "Kf", "Kd"], ["f", "d", "d", "f"]))=#
        #=G_JK_inv = Dict(kbar => omega_by_t * HOP_T - energyCutoff / 2 + couplings[J][di[cutoffPoints]] / 4 + couplings[K][di[cutoffPoints]] / 4 + W[kbar] / 2=#
        #=                for ((J, K), kbar) in zip([("Jf", "Kd"), ("Jd", "Kf")], ["f", "d"]))=#
        #=G_aa_inv = Dict(kbar => omega_by_t * HOP_T - energyCutoff + couplings[k][di[cutoffPoints]] / 2 + W[kbar] =#
        #=                for (k, kbar) in zip(["Jf", "Jd", "Kf", "Kd"], ["f", "d", "d", "f"]))=#
        #=for k in ["Jf", "Jd", "Kf", "Kd"]=#
        #=    G_1[k][di[cutoffPoints]] = densityOfStates[cutoffPoints] * [0.25 ./ (G_inv[k] + 0.75 * couplings["⟂"]) .+ 0.75 ./ (G_inv[k] - 0.25 * couplings["⟂"])]=#
        #=    G_2[k][di[cutoffPoints]] = densityOfStates[cutoffPoints] * [0.25 ./ (G_inv[k] + 0.75 * couplings["⟂"]) .+ 0.75 ./ (G_inv[k] - 0.25 * couplings["⟂"])]=#
        #=    G_alpha_alpha[k][di[cutoffPoints]] = densityOfStates[cutoffPoints] * [0.25 ./ (G_inv[k] + 0.75 * couplings["⟂"]) .+ 0.75 ./ (G_inv[k] - 0.25 * couplings["⟂"])]=#
        #=end=#





        #=for (i_q, q) in enumerate(cutoffPoints)=#
        #=    G_inv = Dict(kbar => omega_by_t * HOP_T - energyCutoff / 2 + couplings[k][q, q] / 4 + W[kbar] / 2 for (k, kbar) in zip(["Jf", "Jd", "Kf", "Kd"], ["f", "d", "d", "f"]))=#
        #=    G_JK_inv = Dict(kbar => omega_by_t * HOP_T - energyCutoff / 2 + couplings[J][q, q] / 4 + couplings[K][q, q] / 4 + W[kbar] / 2 for ((J, K), kbar) in zip([("Jf", "Kd"), ("Jd", "Kf")], ["f", "d", "d", "f"]))=#
        #=    G_aa_inv = Dict(kbar => omega_by_t * HOP_T - energyCutoff + couplings[k][q, q] / 2 + W[kbar] for (k, kbar) in zip(["Jf", "Jd", "Kf", "Kd"], ["f", "d", "d", "f"]))=#
        #=    for k in ["f", "d"]=#
        #=        GMatrix[k][q, q] = densityOfStates[q] * ((1/8) / (omega_by_t * HOP_T - energyCutoff / 2 + μ / 2 + couplings[k][q, q] / 4 + W[k] / 2 + 0.75 * couplings["⟂"]) =#
        #=                                                 + (3/8) / (omega_by_t * HOP_T - energyCutoff / 2 + μ / 2 + couplings[k][q, q] / 4 + W[k] / 2 - 0.25 * couplings["⟂"])=#
        #=                                                 + (1/8) / (omega_by_t * HOP_T - energyCutoff / 2 - μ / 2 + couplings[k][qbar, qbar] / 4 + W[k] / 2 + 0.75 * couplings["⟂"]) =#
        #=                                                 + (3/8) / (omega_by_t * HOP_T - energyCutoff / 2 - μ / 2 + couplings[k][qbar, qbar] / 4 + W[k] / 2 - 0.25 * couplings["⟂"])=#
        #=                                                )=#
        #=    end=#
        #=    GVector[i_q] = sum([densityOfStates[q] / (omega_by_t * HOP_T - 0.5 * (dispersionArray[q] - dispersionArray[qbar]) + couplings[k][q, q] / 2 + W[k] - 0.25 * couplings["⟂"]) for k in ["f", "d"]])=#
        #=end=#
        traceGprime = Dict(k => sum([GMatrix[k][q, q] * couplings[k][q, qbar] for (q,qbar) in zip(cutoffPoints, cutoffHolePoints)]) for k in ["f", "d"])
        JVector = [couplings["f"][q, qbar] * couplings["d"][qbar, q] for (q,qbar) in zip(cutoffPoints, cutoffHolePoints)]

        delta = Dict()
        for (k, kbar) in zip(("f", "d"), ("d", "f"))
            Jalpha = "J" * k
            JalphaBar = "J" * kbar
            Kalpha = "K" * k
            KalphaBar = "K" * kbar
            delta[Jalpha] = -deltaEnergy * (
                                            couplings[Jalpha][innerIndices[k], cutoffPoints] * dos * (0.25 .* G_g[Jalpha]["+"] .+ 0.75 .* G_g[Jalpha]["-"]) * couplings[Jalpha][cutoffPoints, innerIndices[k]] 
                                            .- 4 * tr(couplingqqbar[Jalpha] * dos * (0.25 .* G_g[Jalpha]["+"] .+ 0.75 .* G_g[Jalpha]["-"])) * W[k] * WMatrix[innerIndices[k], innerIndices[k]] 
                                            .- 0.5 * couplings[KalphaBar][innerIndices[k], cutoffPoints] * dos * (G_g[Kalpha]["+"] .+ G_g[Kalpha]["-"]) * couplings[KalphaBar][cutoffPoints, innerIndices[k]] 
                                            .- 2 * tr(couplingqqbar[KalphaBar] * dos * (G_g[Kalpha]["+"] .+ G_g[Kalpha]["-"])) * W[k] * WMatrix[innerIndices[k], innerIndices[k]]
                                           )
            delta[Kalpha] = -deltaEnergy * (
                                            couplings[Kalpha][innerIndices[kbar], cutoffPoints] * dos * (0.25 .* G_g[KalphaBar]["+"] .+ 0.75 .* G_g[KalphaBar]["-"]) * couplings[Kalpha][cutoffPoints, innerIndices[kbar]] 
                                            .- 4 * tr(couplingqqbar[Kalpha] * dos * (0.25 .* G_g[KalphaBar]["+"] .+ 0.75 .* G_g[KalphaBar]["-"])) * W[kbar] * WMatrix[innerIndices[k], innerIndices[k]] 
                                            .- 0.5 * couplings[JalphaBar][innerIndices[kbar], cutoffPoints] * dos * (G_g[Jalphabar]["+"] .+ G_g[JalphaBar]["-"]) * couplings[JalphaBar][cutoffPoints, innerIndices[kbar]] 
                                            .- 2 * tr(couplingqqbar[JalphaBar] * dos * (G_g[Jalphabar]["+"] .+ G_g[JalphaBar]["-"])) * W[kbar] * WMatrix[innerIndices[k], innerIndices[k]] 
                                           )
        end
        delta["J⟂"] = 0 
        delta["J⟂"] += -0.25 * (1 / size_BZ^2) * tr(dos * couplings["Jf"][innerIndices["f"], cutoffPoints] * dos * (G_g["Jf"]["+"] .+ G_g["Jf"]["-"]) * couplings["Jf"][cutoffPoints, innerIndices["f"]])
        delta["J⟂"] += -0.25 * (1 / size_BZ^2) * tr(dos * couplings["Jd"][innerIndices["d"], cutoffPoints] * dos * (G_g["Jd"]["+"] .+ G_g["Jd"]["-"]) * couplings["Jd"][cutoffPoints, innerIndices["d"]])
        delta["J⟂"] += -0.25 * (1 / size_BZ^2) * tr(dos * couplings["Kf"][innerIndices["f"], cutoffPoints] * dos * (G_g["Kd"]["+"] .+ G_g["Kd"]["-"]) * couplings["Kf"][cutoffPoints, innerIndices["f"]])
        delta["J⟂"] += -0.25 * (1 / size_BZ^2) * tr(dos * couplings["Kd"][innerIndices["d"], cutoffPoints] * dos * (G_g["Kf"]["+"] .+ G_g["Kf"]["-"]) * couplings["Kd"][cutoffPoints, innerIndices["d"]])
        delta["J⟂"] += (1 / size_BZ^2) * tr(dos * couplings["Jf"][innerIndices["f"], cutoffPoints] * dos * G_JK["f"] * couplings["Kd"][cutoffPoints, innerIndices["d"]])
        delta["J⟂"] += (1 / size_BZ^2) * tr(dos * couplings["Jd"][innerIndices["d"], cutoffPoints] * dos * G_JK["d"] * couplings["Kf"][cutoffPoints, innerIndices["f"]])
        delta["J⟂"] += 0.5 * tr(dos * couplingqqbar["Jf"] * (G_aa["Jf"] + G_aa["Jd"]) * couplingqqbar["Jd"])
        delta["J⟂"] += 0.5 * tr(dos * couplingqqbar["Kf"] * (G_aa["Kf"] + G_aa["Kd"]) * couplingqqbar["Kd"])
        #=delta["⟂"] = 0.5 * (JVector' * GVector)=#
        if step == 1
            initDeltaSigns["Jf"] = sign.(delta["Jf"])
            initDeltaSigns["Jd"] = sign.(delta["Jd"])
            initDeltaSigns["Kf"] = sign.(delta["Kf"])
            initDeltaSigns["Kd"] = sign.(delta["Kd"])
            initDeltaSigns["J⟂"] = sign(delta["J⟂"])
        else
            for (g, k) in zip(["Jf", "Jd", "Kf", "Kd"], ["f", "d", "d", "f"])
                initDeltaSigns[g][innerIndices[k], innerIndices[k]][sign.(delta[g]) .* initDeltaSigns[g][innerIndices[k], innerIndices[k]] .< 0] .= 0.
                delta[g][initDeltaSigns[k][innerIndices[k], innerIndices[k]] .== 0] .= 0.
            end
            if sign(delta["J⟂"]) * initDeltaSigns["J⟂"] < 0
                initDeltaSigns["J⟂"] = 0.
                delta["J⟂"] = 0.
            end
        end
        for (g, k) in zip(["Jf", "Jd", "Kf", "Kd"], ["f", "d", "d", "f"])
            couplings[g][innerIndices[k], innerIndices[k]] .+= delta[g]
            proceedFlags[g][innerIndices[k], innerIndices[k]] .= couplings[g][innerIndices[k], innerIndices[k]] .* initSigns[g][innerIndices[k], innerIndices[k]] .≤ 0
            couplings[g][sign.(couplings[g]) .* initSigns[g] .< 0] .= 0.
        end
        couplings["J⟂"] += delta["J⟂"]
        if sign(couplings["J⟂"]) * initSigns["J⟂"] < 0
            proceedFlags["J⟂"] = [false]
            couplings["J⟂"] = 0.
        end
    end
    if saveData
        save(saveJLD, couplings)
    end
    return couplings
end
