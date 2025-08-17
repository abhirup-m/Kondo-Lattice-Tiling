@everywhere function PoleFraction(
        size_BZ::Int64,
        couplings::Dict{String, Float64};
        poleFracData::Dict{String,Float64}=Dict(),
        loadData::Bool=false,
    )
    if string(couplings["Wf"]) ∈ keys(poleFracData) && loadData
        return poleFracData[string(couplings["Wf"])]
    end
    kondoJArray, _, dispersion = momentumSpaceRG(size_BZ, couplings; saveData=false, loadData=true)
    fermiPoints = unique(getIsoEngCont(dispersion, 0.0))
    @assert length(fermiPoints) == 2 * size_BZ - 2
    @assert all(==(0), dispersion[fermiPoints])
    averageKondoScale = sum(abs.(kondoJArray[:, :, 1])) / length(kondoJArray[:, :, 1])
    @assert averageKondoScale > RG_RELEVANCE_TOL
    kondoJArray[:, :, end] .= ifelse.(abs.(kondoJArray[:, :, end]) ./ averageKondoScale .> RG_RELEVANCE_TOL, kondoJArray[:, :, end], 0)
    scattProbBool = ScattProb(size_BZ, kondoJArray, dispersion)[2]
    polesFraction = count(>(0), scattProbBool[fermiPoints])/length(fermiPoints)
    poleFracData[string(couplings["Wf"])] = polesFraction
    return polesFraction
end


@everywhere function PerpFlag(
        size_BZ::Int64,
        couplings::Dict{String, Float64};
        perpFlagData::Dict{String,Float64}=Dict(),
        loadData::Bool=false,
    )
    if string(couplings["Wf"]) ∈ keys(perpFlagData) && loadData
        return perpFlagData[string(couplings["Wf"])]
    end
    _, kondoPerpArray, _ = momentumSpaceRG(size_BZ, couplings; saveData=false, loadData=true)
    perpFlag = kondoPerpArray[end] > kondoPerpArray[1] ? 3. : (kondoPerpArray[end] > RG_RELEVANCE_TOL ? 2 : 1)
    perpFlagData[string(couplings["Wf"])] = kondoPerpArray[end] / kondoPerpArray[1]
    return kondoPerpArray[end] / kondoPerpArray[1]
end


@everywhere function CriticalWf(
        size_BZ::Int64,
        WfLims::NTuple{2, Float64},
        WfSpacing::Float64,
        couplings::Dict{String, Float64};
        loadData::Bool=false,
        maxIter::Int64=100,
    )
    fracToIndex(f) = ifelse(f == 1, 1, ifelse(f > 0, 2, 3))
    perpRatioToIndex(f) = ifelse(f ≥ 1.2, 3, ifelse(f ≥ 0.5, 2, 1))

    savePathCrit = joinpath(SAVEDIR, SavePath("Wfcrit", size_BZ, couplings, "json"))
    criticalWfData = Dict{String,Dict{String, Vector{Float64}}}()
    if isfile(savePathCrit) && loadData
        merge!(criticalWfData, JSON3.read(read(savePathCrit, String), typeof(criticalWfData)))
        if string(WfSpacing) ∈ keys(criticalWfData) && (Inf ∉ abs.(criticalWfData[string(WfSpacing)]["J"]) || (minimum(WfLims) > minimum(criticalWfData[string(WfSpacing)]["lims"]) && maximum(WfLims) < maximum(criticalWfData[string(WfSpacing)]["lims"])))
            return criticalWfData[string(WfSpacing)]["Jf"], criticalWfData[string(WfSpacing)]["J"]
        end
    end
    if string(WfSpacing) ∉ keys(criticalWfData)
        criticalWfData[string(WfSpacing)] = Dict{String, Vector{Float64}}()
    end

    @assert WfSpacing > 0
    criticalWf_Jf = Float64[]
    @assert issorted(WfLims, rev=true)

    savePathPolefrac = joinpath(SAVEDIR, SavePath("PoleFrac", size_BZ, couplings, "json"))
    poleFracData = Dict{String,Float64}()
    if isfile(savePathPolefrac)
        merge!(poleFracData, JSON3.read(read(savePathPolefrac, String), typeof(poleFracData)))
    end
    for phaseBoundType in [(1, 2), (2, 3)]
        currentTransitionWindow = collect(WfLims)
        currentPoleFractions = [PoleFraction(size_BZ, merge(couplings, Dict("Wf" => Wf)); poleFracData=poleFracData, loadData=loadData) for Wf in currentTransitionWindow]
        currentPhaseIndices = map(fracToIndex, currentPoleFractions)
        numIter = 1
        while abs(currentTransitionWindow[1] - currentTransitionWindow[2]) > WfSpacing && numIter < maxIter
            updatedEdge = 0.5 * sum(currentTransitionWindow)
            newPoleFraction = PoleFraction(size_BZ, merge(couplings, Dict("Wf" => updatedEdge)); poleFracData=poleFracData, loadData=loadData)
            newPhaseIndex = fracToIndex(newPoleFraction)
            if newPhaseIndex == currentPhaseIndices[1] || newPhaseIndex == phaseBoundType[1]
                currentPhaseIndices[1] = newPhaseIndex
                currentTransitionWindow[1] = updatedEdge
            else
                currentPhaseIndices[2] = newPhaseIndex
                currentTransitionWindow[2] = updatedEdge
            end
            numIter += 1
        end
        push!(criticalWf_Jf, 0.5 * sum(currentTransitionWindow))
    end

    criticalWfData[string(WfSpacing)]["Jf"] = criticalWf_Jf
    open(savePathPolefrac, "w") do file JSON3.write(file, poleFracData) end

    criticalWf_J = Float64[]
    savePathPerpFlags = joinpath(SAVEDIR, SavePath("PerpFlags", size_BZ, couplings, "json"))
    perpFlagData = Dict{String, Float64}()
    if isfile(savePathPerpFlags)
        merge!(perpFlagData, JSON3.read(read(savePathPerpFlags, String), typeof(perpFlagData)))
    end
    for phaseBoundType in [(1, 2), (2, 3)]
        currentTransitionWindow = collect(WfLims)
        currentPerpFlags = [PerpFlag(size_BZ, merge(couplings, Dict("Wf" => Wf)); perpFlagData=perpFlagData, loadData=loadData) for Wf in currentTransitionWindow]
        currentIndices = perpRatioToIndex.(currentPerpFlags)
        if minimum(currentIndices) > phaseBoundType[1]
            push!(criticalWf_J, -9e9)
            continue
        end
        if maximum(currentIndices) < phaseBoundType[2]
            push!(criticalWf_J, 9e9)
            continue
        end

        numIter = 1
        while abs(currentTransitionWindow[1] - currentTransitionWindow[2]) > WfSpacing && numIter < maxIter
            updatedEdge = 0.5 * sum(currentTransitionWindow)
            newPerpFlag = PerpFlag(size_BZ, merge(couplings, Dict("Wf" => updatedEdge)); perpFlagData=perpFlagData, loadData=loadData)
            newIndex = perpRatioToIndex(newPerpFlag)
            if newIndex == currentIndices[1] || newIndex == phaseBoundType[1]
                currentIndices[1] = newIndex
                currentTransitionWindow[1] = updatedEdge
            else
                currentIndices[2] = newIndex
                currentTransitionWindow[2] = updatedEdge
            end
            numIter += 1
        end
        push!(criticalWf_J, 0.5 * sum(currentTransitionWindow))
    end

    open(savePathPerpFlags, "w") do file JSON3.write(file, perpFlagData) end

    criticalWfData[string(WfSpacing)]["J"] = criticalWf_J

    if "lims" ∉ keys(criticalWfData[string(WfSpacing)]) || (minimum(WfLims) ≤ minimum(criticalWfData[string(WfSpacing)]["lims"]) && maximum(WfLims) ≥ maximum(criticalWfData[string(WfSpacing)]["lims"]))
        criticalWfData[string(WfSpacing)]["lims"] = collect(WfLims)
        open(savePathCrit, "w") do file JSON3.write(file, criticalWfData) end
    end
    return criticalWf_Jf, criticalWf_J
end


function PhaseDiagram(
        size_BZ::Int64,
        kondo_perpLims::NTuple{2, Float64}, 
        kondo_perpSpacing::Float64,
        WfLims::NTuple{2, Float64}, 
        WfSpacing::Float64,
        couplings::Dict{String, Float64};
        loadData::Bool=false,
        fillPG::Bool=false,
    )

    mkpath(SAVEDIR)
    kondo_perpVals = collect(minimum(kondo_perpLims):kondo_perpSpacing:maximum(kondo_perpLims))
    WfVals = collect(minimum(WfLims):WfSpacing:maximum(WfLims))
    phaseDiagram_Jf = fill(0., (length(kondo_perpVals), length(WfVals)))
    phaseDiagram_J = fill(0., (length(kondo_perpVals), length(WfVals)))
    criticalWfResults = @showprogress pmap(kondo_perp -> CriticalWf(size_BZ, WfLims, WfSpacing, merge(couplings, Dict("kondo_perp" => kondo_perp)); loadData=loadData), kondo_perpVals)
    for (i, ((PGStart, PGStop), (crit1, crit2))) in enumerate(criticalWfResults)
        phaseDiagram_Jf[i, WfVals .≥ PGStart] .= 1.
        phaseDiagram_Jf[i, PGStop .≥ WfVals] .= 0.
        phaseDiagram_Jf[i, PGStop .< WfVals .< PGStart] .= 0.5
        phaseDiagram_J[i, WfVals .≥ crit2] .= 3
        phaseDiagram_J[i, crit1 .≥ WfVals] .= 1
        phaseDiagram_J[i, crit1 .< WfVals .< crit2] .= 2
    end

    return phaseDiagram_Jf, phaseDiagram_J
end
