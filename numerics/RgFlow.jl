using LinearAlgebra, Serialization

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
    for key in ["Jd", "Jf"]
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
                        for key in ["Jd", "Jf"]
                       )
    return innerIndices, cutoffPoints, cutoffHolePoints, proceedFlags
end

function momentumSpaceRGFull(
        bareCouplings::Dict;
        progressbarEnabled=false,
        loadData::Bool=false,
        saveData::Bool=true,
    )
    size_BZ = bareCouplings["size_BZ"]
    kvals = map1DTo2D.(1:size_BZ^2, size_BZ)
    kxVals = first.(kvals)
    kyVals = last.(kvals)
    omega_by_t = bareCouplings["omega_by_t"]
    W = Dict("f" => bareCouplings["Wf"], "d" => bareCouplings["Wd"])

    savePath = joinpath(SAVEDIR, "RG-$(hash(bareCouplings))")
    mkpath(SAVEDIR)
    if isfile(savePath) && loadData
        return deserialize(savePath)
    end
    # @assert false loadData


    # ensure that [0, \pi] has odd number of states, so 
    # that the nodal point is well-defined.
    @assert (size_BZ - 5) % 4 == 0 "Size of Brillouin zone must be of the form N = 4n+5, n=0,1,2..., so that all the nodes and antinodes are well-defined."

    densityOfStates, dispersionArray = getDensityOfStates(tightBindDisp, size_BZ)

    cutOffEnergies = getCutOffEnergy(size_BZ)

    # Kondo coupling must be stored in a 3D matrix. Two of the dimensions store the 
    # incoming and outgoing momentum indices, while the third dimension stores the 
    # behaviour along the RG flow. For example, J[i][j][k] reveals the value of J 
    # for the momentum pair (i,j) at the k^th Rg step.
    couplings = Dict{String, Union{Float64, Matrix{Float64}}}(k => initialiseKondoJ(size_BZ, div(size_BZ + 1, 2), bareCouplings[k]) for k in ["Jf", "Jd", "Kf", "Kd"])
    couplings["J⟂"] = bareCouplings["J⟂"]

    initSigns = Dict(k => sign.(v) for (k, v) in couplings)

    # define flags to track whether the RG flow for a particular J_{k1, k2} needs to be stopped 
    # (perhaps because it has gone to zero, or its denominator has gone to zero). These flags are
    # initialised to one, which means that by default, the RG can proceed for all the momenta.
    proceedFlags = Dict{String, Matrix{Bool}}(k => fill(true, size_BZ^2, size_BZ^2) for k in ["Jf", "Jd", "Kf", "Kd"])
    proceedFlags["J⟂"] = [true;;]

    # Run the RG flow starting from the maximum energy, down to the penultimate energy (ΔE), in steps of ΔE

    initDeltaSigns = Dict{String, Union{Float64, Matrix{Float64}}}(k=> repeat([1.], size_BZ^2, size_BZ^2) for k in ["Jf", "Jd", "Kf", "Kd"])
    initDeltaSigns["J⟂"] = 1.

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
            G_g[g]["+"][di] = 1 ./ (omega_by_t * HOP_T - energyCutoff / 2 / 4 + W[k] / 2 + 0.75 * couplings["J⟂"] .+ couplings[g][di])
            G_g[g]["-"][di] = 1 ./ (omega_by_t * HOP_T - energyCutoff / 2 + W[k] / 2 - 0.25 * couplings["J⟂"] .+ couplings[g][di] / 4)
            G_aa[g][di] = 1 ./ (omega_by_t * HOP_T - energyCutoff + W[k] - 0.25 * couplings["J⟂"] .+ couplings[g][di] / 2)
            G_JK[k][di] = 1 ./ (omega_by_t * HOP_T - energyCutoff / 2 + W[k] / 2 - 0.25 * couplings["J⟂"] .+ couplings["J"*k][di] / 4 .+ couplings["K"*kbar][di] / 4)
        end
        couplingqqbar = Dict(k => zeros(size_BZ^2, size_BZ^2) for k in ["Jf", "Jd", "Kf", "Kd"])
        for (q, qbar) in zip(cutoffPoints, cutoffHolePoints)
            for k in ["Jf", "Jd", "Kf", "Kd"]
                couplingqqbar[k][q, q] = couplings[k][q, qbar]
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
        # traceGprime = Dict(k => sum([GMatrix[k][q, q] * couplings[k][q, qbar] for (q,qbar) in zip(cutoffPoints, cutoffHolePoints)]) for k in ["f", "d"])
        # JVector = [couplings["f"][q, qbar] * couplings["d"][qbar, q] for (q,qbar) in zip(cutoffPoints, cutoffHolePoints)]

        allInner = vcat([innerIndices[g] for g in ["Jf", "Jd", "Kf", "Kd"]]...)
        delta = Dict()
        for (k, kbar) in zip(("f", "d"), ("d", "f"))
            Jalpha = "J" * k
            JalphaBar = "J" * kbar
            Kalpha = "K" * k
            KalphaBar = "K" * kbar
            delta[Jalpha] = -deltaEnergy * (
                                            couplings[Jalpha][allInner, :] * dos * (0.25 .* G_g[Jalpha]["+"] .+ 0.75 .* G_g[Jalpha]["-"]) * couplings[Jalpha][:, allInner] 
                                            .- 4 * tr(couplingqqbar[Jalpha] * dos * (0.25 .* G_g[Jalpha]["+"] .+ 0.75 .* G_g[Jalpha]["-"])) * W[k] * WMatrix[allInner, allInner] 
                                            .- 0.5 * couplings[KalphaBar][allInner, :] * dos * (G_g[Kalpha]["+"] .+ G_g[Kalpha]["-"]) * couplings[KalphaBar][:, allInner] 
                                            .- 2 * tr(couplingqqbar[KalphaBar] * dos * (G_g[Kalpha]["+"] .+ G_g[Kalpha]["-"])) * W[k] * WMatrix[allInner, allInner]
                                           )
            delta[Kalpha] = - deltaEnergy * (
                                            couplings[Kalpha][allInner, :] * dos * (0.25 .* G_g[KalphaBar]["+"] .+ 0.75 .* G_g[KalphaBar]["-"]) * couplings[Kalpha][:, allInner] 
                                            .- 4 * tr(couplingqqbar[Kalpha] * dos * (0.25 .* G_g[KalphaBar]["+"] .+ 0.75 .* G_g[KalphaBar]["-"])) * W[kbar] * WMatrix[allInner, allInner] 
                                            .- 0.5 * couplings[JalphaBar][allInner, :] * dos * (G_g[JalphaBar]["+"] .+ G_g[JalphaBar]["-"]) * couplings[JalphaBar][:, allInner] 
                                            .- 2 * tr(couplingqqbar[JalphaBar] * dos * (G_g[JalphaBar]["+"] .+ G_g[JalphaBar]["-"])) * W[kbar] * WMatrix[allInner, allInner] 
                                           )
        end
        delta["J⟂"] = 0 
        for (g, gbar) in zip(["Jf", "Jd", "Kf", "Kd"], ["Jf", "Jd", "Kd", "Kf"])
            delta["J⟂"] += -0.25 * (1 / size_BZ^2) * tr(dos[innerIndices[g], innerIndices[g]] * couplings[g][innerIndices[g], :] * dos * (G_g[gbar]["+"] .+ G_g[gbar]["-"]) * couplings[g][:, innerIndices[g]])
        end
        # delta["J⟂"] += -0.25 * (1 / size_BZ^2) * tr(dos * couplings["Jd"][innerIndices["Jd"], cutoffPoints] * dos * (G_g["Jd"]["+"] .+ G_g["Jd"]["-"]) * couplings["Jd"][cutoffPoints, innerIndices["Jd"]])
        # delta["J⟂"] += -0.25 * (1 / size_BZ^2) * tr(dos * couplings["Kf"][innerIndices["Kf"], cutoffPoints] * dos * (G_g["Kd"]["+"] .+ G_g["Kd"]["-"]) * couplings["Kf"][cutoffPoints, innerIndices["Kf"]])
        # delta["J⟂"] += -0.25 * (1 / size_BZ^2) * tr(dos * couplings["Kd"][innerIndices["d"], cutoffPoints] * dos * (G_g["Kf"]["+"] .+ G_g["Kf"]["-"]) * couplings["Kd"][cutoffPoints, innerIndices["d"]])
        delta["J⟂"] += (1 / size_BZ^2) * tr(dos[vcat(innerIndices["Jf"], innerIndices["Kd"]), vcat(innerIndices["Jf"], innerIndices["Kd"])] * couplings["Jf"][vcat(innerIndices["Jf"], innerIndices["Kd"]), :] * dos * G_JK["f"] * couplings["Kd"][:, vcat(innerIndices["Jf"], innerIndices["Kd"])])
        delta["J⟂"] += (1 / size_BZ^2) * tr(dos[vcat(innerIndices["Jd"], innerIndices["Kf"]), vcat(innerIndices["Jd"], innerIndices["Kf"])] * couplings["Jd"][vcat(innerIndices["Jd"], innerIndices["Kf"]), :] * dos * G_JK["d"] * couplings["Kf"][:, vcat(innerIndices["Jd"], innerIndices["Kf"])])
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
            for g in ["Jf", "Jd", "Kf", "Kd"]
                initDeltaSigns[g][allInner, allInner][sign.(delta[g]) .* initDeltaSigns[g][allInner, allInner] .< 0] .= 0.
                delta[g][initDeltaSigns[g][allInner, allInner] .== 0] .= 0.
            end
            if sign(delta["J⟂"]) * initDeltaSigns["J⟂"] < 0
                initDeltaSigns["J⟂"] = 0.
                delta["J⟂"] = 0.
            end
        end
        for g in ["Jf", "Jd", "Kf", "Kd"]
            couplings[g][allInner, allInner] .+= delta[g]
            proceedFlags[g][allInner, allInner] .= couplings[g][allInner, allInner] .* initSigns[g][allInner, allInner] .≤ 0
            couplings[g][sign.(couplings[g]) .* initSigns[g] .< 0] .= 0.
        end
        couplings["J⟂"] += delta["J⟂"]
        if sign(couplings["J⟂"]) * initSigns["J⟂"] < 0
            proceedFlags["J⟂"][1] = false
            couplings["J⟂"] = 0.
        end
    end
    serialize(savePath, couplings)
    return couplings
end

function momentumSpaceRG(
        bareCouplings::Dict;
        progressbarEnabled=false,
        loadData::Bool=false,
        saveData::Bool=true,
    )
    size_BZ = bareCouplings["size_BZ"]
    kvals = map1DTo2D.(1:size_BZ^2, size_BZ)
    kxVals = first.(kvals)
    kyVals = last.(kvals)
    omega_by_t = bareCouplings["omega_by_t"]
    W = Dict("f" => bareCouplings["Wf"], "d" => bareCouplings["Wd"])

    savePath = joinpath(SAVEDIR, "RG-$(hash(bareCouplings))")
    mkpath(SAVEDIR)
    if isfile(savePath) && loadData
        return deserialize(savePath)
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
    couplings = Dict("Jf" => initialiseKondoJ(size_BZ, div(size_BZ + 1, 2), bareCouplings["Jf"]),
                     "Jd" => initialiseKondoJ(size_BZ, div(size_BZ + 1, 2), bareCouplings["Jd"]),
                     "J⟂" => bareCouplings["J⟂"],
                )
    if couplings["J⟂"] == 0
        couplings["J⟂"] = 1e-3
    end

    initSigns = Dict(k => sign.(v) for (k, v) in couplings)

    # define flags to track whether the RG flow for a particular J_{k1, k2} needs to be stopped 
    # (perhaps because it has gone to zero, or its denominator has gone to zero). These flags are
    # initialised to one, which means that by default, the RG can proceed for all the momenta.
    proceedFlags = Dict("Jf" => fill(true, size_BZ^2, size_BZ^2), "Jd" => fill(true, size_BZ^2, size_BZ^2), "J⟂" => [true])

    # Run the RG flow starting from the maximum energy, down to the penultimate energy (ΔE), in steps of ΔE

    initDeltaSigns = Dict("Jd" => repeat([1.], size_BZ^2, size_BZ^2),
                          "Jf" => repeat([1.], size_BZ^2, size_BZ^2),
                          "J⟂" => 1.
                         )
    #=initDeltaSignPerp = 1=#

    GMatrix = Dict(k => 0 .* couplings["Jf"] for k in ["Jf", "Jd", "J⟂f", "J⟂d"])
    WMatrix = 0.5 .* (cos.(kxVals' .- kxVals) .+ cos.(kyVals' .- kyVals))
    for (stepIndex, energyCutoff) in enumerate(cutOffEnergies[1:end-1])
        deltaEnergy = abs(cutOffEnergies[stepIndex+1] - cutOffEnergies[stepIndex])

        # if there are no enabled flags (i.e., all are zero), stop the RG flow
        if !any([any(v) for v in values(proceedFlags)])
            break
        end

        innerIndices, cutoffPoints, cutoffHolePoints, proceedFlags = highLowSeparation(dispersionArray, energyCutoff, proceedFlags, size_BZ)
        GMatrix["Jf"] .= 0.
        GMatrix["Jd"] .= 0.
        GMatrix["J⟂f"] .= 0.
        GMatrix["J⟂d"] .= 0.
        GVector = zeros(length(cutoffPoints))
        for (i_q, (q, qbar)) in enumerate(zip(cutoffPoints, cutoffHolePoints))
            for k in ["Jf", "Jd"]
                t = replace(k, "J" => "")
                GMatrix[k][q, q] = densityOfStates[q] * ((1/8) / (omega_by_t * HOP_T - energyCutoff / 2 + couplings[k][q, q] / 4 + W[t] / 2 + 0.75 * couplings["J⟂"]) 
                                                         + (3/8) / (omega_by_t * HOP_T - energyCutoff / 2 + couplings[k][q, q] / 4 + W[t] / 2 - 0.25 * couplings["J⟂"])
                                                         + (1/8) / (omega_by_t * HOP_T - energyCutoff / 2 + couplings[k][qbar, qbar] / 4 + W[t] / 2 + 0.75 * couplings["J⟂"]) 
                                                         + (3/8) / (omega_by_t * HOP_T - energyCutoff / 2 + couplings[k][qbar, qbar] / 4 + W[t] / 2 - 0.25 * couplings["J⟂"])
                                                        )
                kperp = "J⟂" * k[2]
                GMatrix[kperp][q, q] = densityOfStates[q] * (1 / (omega_by_t * HOP_T - energyCutoff / 2 + couplings[k][q, q] / 4 + W[t] / 2 + 0.75 * couplings["J⟂"]) 
                                                         + 1 / (omega_by_t * HOP_T - energyCutoff / 2 + couplings[k][q, q] / 4 + W[t] / 2 - 0.25 * couplings["J⟂"])
                                                    )
            end
            GVector[i_q] = sum([densityOfStates[q] / (omega_by_t * HOP_T - 0.5 * (dispersionArray[q] - dispersionArray[qbar]) + couplings[k][q, q] / 2 + W[replace(k, "J" => "")] - 0.25 * couplings["J⟂"]) for k in ["Jf", "Jd"]])
        end
        traceGprime = Dict(k => sum([GMatrix[k][q, q] * couplings[k][q, qbar] for (q,qbar) in zip(cutoffPoints, cutoffHolePoints)]) for k in ["Jf", "Jd"])
        JVector = [couplings["Jf"][q, qbar] * couplings["Jd"][qbar, q] for (q,qbar) in zip(cutoffPoints, cutoffHolePoints)]

        delta = Dict()
        for k in ["Jf", "Jd"]
            delta[k] = -deltaEnergy * (couplings[k][innerIndices[k], cutoffPoints] * GMatrix[k][cutoffPoints, cutoffPoints] * couplings[k][cutoffPoints, innerIndices[k]] .- 4 * W[replace(k, "J" => "")] * traceGprime[k] .* WMatrix[innerIndices[k], innerIndices[k]])
        end
        delta["J⟂"] = 0.5 * (JVector' * GVector) - 0.25 * sum([densityOfStates[innerIndices[k]]' * diag(couplings[k][innerIndices[k], cutoffPoints] * GMatrix["J⟂"*k[2]][cutoffPoints, cutoffPoints] * couplings[k][innerIndices[k], cutoffPoints]') / length(innerIndices[k]) for k in ["Jf", "Jd"]])
        if step == 1
            initDeltaSigns["Jf"] = sign.(delta["Jf"])
            initDeltaSigns["Jd"] = sign.(delta["Jd"])
            initDeltaSigns["J⟂"] = sign(delta["J⟂"])
        else
            for k in ["Jf", "Jd"]
                initDeltaSigns[k][innerIndices[k], innerIndices[k]][sign.(delta[k]) .* initDeltaSigns[k][innerIndices[k], innerIndices[k]] .< 0] .= 0.
                delta[k][initDeltaSigns[k][innerIndices[k], innerIndices[k]] .== 0] .= 0.
            end
            if sign(delta["J⟂"]) * initDeltaSigns["J⟂"] < 0
                initDeltaSigns["J⟂"] = 0.
                delta["J⟂"] = 0.
            end
        end
        for k in ["Jf", "Jd"]
            couplings[k][innerIndices[k], innerIndices[k]] .+= delta[k]
            proceedFlags[k][innerIndices[k], innerIndices[k]] .= couplings[k][innerIndices[k], innerIndices[k]] .* initSigns[k][innerIndices[k], innerIndices[k]] .≤ 0
            couplings[k][sign.(couplings[k]) .* initSigns[k] .< 0] .= 0.
        end
        couplings["J⟂"] += delta["J⟂"]
        if sign(couplings["J⟂"]) * initSigns["J⟂"] < 0
            proceedFlags["J⟂"] = [false]
            couplings["J⟂"] = 0.
        end
    end
    serialize(savePath, couplings)
    return couplings
end
