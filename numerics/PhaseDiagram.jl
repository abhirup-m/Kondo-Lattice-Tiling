@everywhere function PoleFraction(
        size_BZ::Int64,
        couplings::Dict{String, Float64};
        availableData::Dict{String,Float64}=Dict(),
        loadData::Bool=false,
    )
    if string(couplings["Wf"]) ∈ keys(availableData) && loadData
        return availableData[string(couplings["Wf"])]
    end
    kondoJArray, kondoPerpArray, dispersion = momentumSpaceRG(size_BZ, couplings; saveData=false, loadData=true)
    fermiPoints = unique(getIsoEngCont(dispersion, 0.0))
    @assert length(fermiPoints) == 2 * size_BZ - 2
    @assert all(==(0), dispersion[fermiPoints])
    averageKondoScale = sum(abs.(kondoJArray[:, :, 1])) / length(kondoJArray[:, :, 1])
    @assert averageKondoScale > RG_RELEVANCE_TOL
    kondoJArray[:, :, end] .= ifelse.(abs.(kondoJArray[:, :, end]) ./ averageKondoScale .> RG_RELEVANCE_TOL, kondoJArray[:, :, end], 0)
    scattProbBool = ScattProb(size_BZ, kondoJArray, dispersion)[2]
    polesFraction = count(>(0), scattProbBool[fermiPoints])/length(fermiPoints)
    availableData[string(couplings["Wf"])] = polesFraction
    return polesFraction
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

    savePathCrit = joinpath(SAVEDIR, SavePath("Wfcrit", size_BZ, couplings, "json"))
    criticalWfData = Dict{String,Vector{Float64}}()
    if isfile(savePathCrit) && loadData
        merge!(criticalWfData, JSON3.read(read(savePathCrit, String), typeof(criticalWfData)))
        if string(WfSpacing) ∈ keys(criticalWfData)
            return criticalWfData[string(WfSpacing)]
        end
    end
    @assert WfSpacing > 0
    criticalWf = Float64[]
    @assert issorted(WfLims, rev=true)

    savePath = joinpath(SAVEDIR, SavePath("PoleFrac", size_BZ, couplings, "json"))
    availableData = Dict{String,Float64}()
    if isfile(savePath)
        merge!(availableData, JSON3.read(read(savePath, String), typeof(availableData)))
    end
    for phaseBoundType in [(1, 2), (2, 3)]
        currentTransitionWindow = collect(WfLims)
        currentPoleFractions = [PoleFraction(size_BZ, merge(couplings, Dict("Wf" => Wf)); availableData=availableData, loadData=loadData) for Wf in currentTransitionWindow]
        currentPhaseIndices = map(fracToIndex, currentPoleFractions)
        numIter = 1
        while abs(currentTransitionWindow[1] - currentTransitionWindow[2]) > WfSpacing && numIter < maxIter
            updatedEdge = 0.5 * sum(currentTransitionWindow)
            newPoleFraction = PoleFraction(size_BZ, merge(couplings, Dict("Wf" => updatedEdge)); availableData=availableData, loadData=loadData)
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
        push!(criticalWf, 0.5 * sum(currentTransitionWindow))
    end
    criticalWfData[string(WfSpacing)] = criticalWf

    open(savePath, "w") do file JSON3.write(file, availableData) end
    open(savePathCrit, "w") do file JSON3.write(file, criticalWfData) end

    return criticalWf
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
    phaseDiagram = fill(0., (length(kondo_perpVals), length(WfVals)))
    criticalWfResults = @showprogress pmap(kondo_perp -> CriticalWf(size_BZ, WfLims, WfSpacing, merge(couplings, Dict("kondo_perp" => kondo_perp)); loadData=loadData), kondo_perpVals)
    for (i, (PGStart, PGStop)) in enumerate(criticalWfResults)
        phaseDiagram[i, WfVals .≥ PGStart] .= 1.
        phaseDiagram[i, PGStop .≥ WfVals] .= 0.
    end

    pgFillComplete = @showprogress @distributed (v1, v2) -> vcat(v1, v2) for i in eachindex(criticalWfResults)
        PGStart, PGStop = criticalWfResults[i]
        pgFill = zeros(length(WfVals[PGStart .≥ WfVals .≥ PGStop]))
        pgFill .= 0.5
        [pgFill]
    end
    for (i, (PGStart, PGStop)) in enumerate(criticalWfResults)
        phaseDiagram[i, PGStart .≥ WfVals .≥ PGStop] .= pgFillComplete[i]
    end

    return phaseDiagram
end
