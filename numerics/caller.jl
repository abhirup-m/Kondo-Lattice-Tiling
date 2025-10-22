#!/bin/env julia-beta
#SBATCH --job-name=SC25                     # Job name
#SBATCH --output=job.%J.out                    # Standard output file
#SBATCH --error=job.%J.err                     # Standard error file
#SBATCH --partition=hm                         # Partition or queue name
#SBATCH --nodes=3                              # Number of nodes
#SBATCH --ntasks-per-node=48                   # Number of tasks per node
#SBATCH --ntasks=144                           # Total Number of tasks
#SBATCH --time=24:00:00                        # Maximum runtime (D-HH:MM:SS)

ENV["PYCALL_GC_FINALIZE"] = "0"
using Distributed, SlurmClusterManager, PDFmerger, Fermions, ProgressMeter, PyPlot
if "SLURM_SUBMIT_DIR" in keys(ENV)
    addprocs(SlurmManager())
end

global impU = 0.

@everywhere submitDir = pwd() * "/"
@everywhere if "SLURM_SUBMIT_DIR" in keys(ENV)
    submitDir = ENV["SLURM_SUBMIT_DIR"] * "/"
end

@everywhere include(submitDir * "Constants.jl")
@everywhere include(submitDir * "Helpers.jl")
@everywhere include(submitDir * "RgFlow.jl")
@everywhere include(submitDir * "Models.jl")
@everywhere include(submitDir * "PhaseDiagram.jl")
@everywhere include(submitDir * "Probes.jl")
@everywhere include(submitDir * "PltStyle.jl")

function RGFlow(
        Wf_arr::Vector{Float64},
        kondoPerp::Float64,
        W::Float64,
        size_BZ::Int64;
        loadData::Bool=false,
    )
    couplings(Wf) = Dict{String, Float64}(
                                      "omega_by_t" => OMEGA_BY_t,
                                      "kondoF" => kondoF,
                                      "kondoPerp" => kondoPerp,
                                      "Wf" => Wf,
                                      "epsilonF" => epsilonF,
                                      "mu_c" => mu_c,
                                      "W" => W,
                                      "lightBandFactor" => lightBandFactor,
                                     )
    results = @showprogress desc="rg flow" pmap(Wf -> momentumSpaceRG(size_BZ, couplings(Wf); loadData=loadData), Wf_arr)

    dispersion = results[1][3]
    kondoJArrays = Dict{Float64, Array{Float64, 2}}()
    kondoPerpArrays = Dict{Float64, Vector{Float64}}()
    for (result, Wf) in zip(results, Wf_arr)
        averageKondoScale = sum(abs.(result[1][:, :, 1])) / length(result[1][:, :, 1])
        @assert averageKondoScale > RG_RELEVANCE_TOL
        result[1][:, :, end] .= ifelse.(abs.(result[1][:, :, end]) ./ averageKondoScale .> RG_RELEVANCE_TOL, result[1][:, :, end], 0)
        kondoJArrays[Wf] = result[1][:, :, end]
        kondoPerpArrays[Wf] = result[2]
    end
    return kondoJArrays, kondoPerpArrays, dispersion
end


function ScattProb(
        size_BZ::Int64,
        couplingsRange::Dict{String, Vector{Float64}}, 
        axLab::Vector{String};
        loadData::Bool=false,
    )
    nonAxLabs = sort([k for k in keys(couplingsRange) if k ∉ axLab])
    basicCouplings = Dict{String, Float64}(k => v[1] for (k, v) in couplingsRange if length(v) == 1)
    basicCouplings["omega_by_t"] = OMEGA_BY_t
    basicCouplings["impU"] = impU
    rangedCouplings = Dict{String, Vector{Float64}}(k => FillIn(v) for (k, v) in couplingsRange if length(v) == 3)
    parameterSpace = Iterators.product([rangedCouplings[ax] for ax in axLab]...)
    paths = String[]
    for couplings in parameterSpace
        allCouplings = merge(basicCouplings, Dict(axLab .=> couplings))
        kondoJArray, kondoPerpArray, dispersion = momentumSpaceRG(size_BZ, allCouplings; loadData=loadData, saveData=true)

        averageKondoScale = sum(abs.(kondoJArray[:, :, 1])) / length(kondoJArray[:, :, 1])
        @assert averageKondoScale > RG_RELEVANCE_TOL
        kondoJArray[:, :, end] .= ifelse.(abs.(kondoJArray[:, :, end]) ./ averageKondoScale .> RG_RELEVANCE_TOL, kondoJArray[:, :, end], 0)
        kondoJArray[abs.(dispersion) .> 0.2 * HOP_T, :, end] .= 0
        kondoJArray[:, abs.(dispersion) .> 0.2 * HOP_T, end] .= 0

        node = map2DTo1D(π/2, π/2, size_BZ)
        antinode = map2DTo1D(π/1, 0., size_BZ)
        fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
        hm1 = ax[1].imshow(reshape(kondoJArray[node, :, end], size_BZ, size_BZ)', origin="lower")
        hm2 = ax[2].imshow(reshape(kondoJArray[antinode, :, end], size_BZ, size_BZ)', origin="lower")
        fig.colorbar(hm1)
        fig.colorbar(hm2)
        push!(paths, "SP-$(size_BZ)-" * ExtendedSaveName(allCouplings) * ".pdf")
        fig.savefig(paths[end])
    end
    merge_pdfs(paths, "SP.pdf", cleanup=true)
end


function PhaseDiagram(
        size_BZ::Int64,
        couplingsRange::Dict{String, Vector{Float64}}, 
        axLab::Vector{String},
        cbarLabels;
        loadData::Bool=false,
    )
    nonAxLabs = sort([k for k in keys(couplingsRange) if k ∉ axLab])
    basicCouplings = Dict{String, Float64}(k => v[1] for (k, v) in couplingsRange if length(v) == 1)
    basicCouplings["omega_by_t"] = OMEGA_BY_t
    basicCouplings["impU"] = impU
    rangedCouplings = Dict{String, Vector{Float64}}(k => FillIn(v) for (k, v) in couplingsRange if length(v) == 3)
    parameterSpace = Iterators.product([rangedCouplings[ax] for ax in axLab]...)
    saveName = "PD-$(size_BZ)-" * ExtendedSaveName(couplingsRange)

    heatmapF = Dict{NTuple{4, Float64}, Float64}()
    heatmapS = Dict{NTuple{4, Float64}, Float64}()
    try
        if !loadData
            @assert false
        end
        loadedData = JSON3.read(read(joinpath(SAVEDIR, saveName * "-f.json"), String), Dict{String, Any})
        for (k, v) in loadedData
            heatmapF[eval(Meta.parse(k))] = v
        end
        loadedData = JSON3.read(read(joinpath(SAVEDIR, saveName * "-s.json"), String), Dict{String, Any})
        for (k, v) in loadedData
            heatmapS[eval(Meta.parse(k))] = v
        end
    catch e

        combinedResults = @showprogress pmap(couplings -> FixedPointVals(
                                                                         size_BZ,
                                                                         merge(basicCouplings, Dict(axLab .=> couplings));
                                                                         loadData=loadData,
                                                        ),
                                             parameterSpace
                                            )

        for (index, key) in enumerate(parameterSpace)
            heatmapF[key] = abs(combinedResults[index]["f-pf"])
            heatmapS[key] = abs(combinedResults[index]["J"])
        end
        open(joinpath(SAVEDIR, saveName * "-f.json"), "w") do file JSON3.write(file, heatmapF) end
        open(joinpath(SAVEDIR, saveName * "-s.json"), "w") do file JSON3.write(file, heatmapS) end
    end
    suptitle = latexstring(join(["$(AXES_LABELS[lab])=$(couplingsRange[lab][1])" for lab in nonAxLabs], ", "))
    return HM4D(heatmapF, heatmapS, rangedCouplings, axLab, saveName, cbarLabels, suptitle)
end

function AuxiliaryCorrelations(
        size_BZ::Int64,
        maxSize::Int64,
        couplingsRange::Dict{String, Vector{Float64}}, 
        axLab::Vector{String},
        cbarLabels::Dict{},
        saveNamePrefix::String;
        loadData::Bool=false,
    )
    nonAxLabs = sort([k for k in keys(couplingsRange) if k ∉ axLab])
    suptitle = latexstring(join(["$(AXES_LABELS[lab])=$(couplingsRange[lab][1])" for lab in nonAxLabs], ", "))
    basicCouplings = Dict{String, Float64}(k => v[1] for (k, v) in couplingsRange if length(v) == 1)
    basicCouplings["omega_by_t"] = OMEGA_BY_t
    basicCouplings["impU"] = impU
    rangedCouplings = Dict{String, Vector{Float64}}(k => FillIn(v) for (k, v) in couplingsRange if length(v) == 3)
    parameterSpace = Iterators.product([rangedCouplings[ax] for ax in axLab]...)

    node = map2DTo1D(π/2, π/2, size_BZ)
    antinode = map2DTo1D(3π/4, π/4, size_BZ)
    #=microCorrelation = Dict(=#
    #=                    "SF-d0-f" => ("f", (k1, k2, points) -> k1 ≥ k2, (i, j, k1, k2; factor = 1) -> [("+-+-", [1, 2, i+1, j], 3 * factor * NNFunc(k1, k2, size_BZ) / 4), ("+-+-", [2, 1, i, j+1], 3 * factor * NNFunc(k1, k2, size_BZ) / 4)]),=#
    #=                    "SF-d0-N" => ("f", (k1, k2, points) -> k1 == node && k2 == node, (i, j, k1, k2; factor = 1) -> [("+-+-", [1, 2, i+1, j], 3 * factor / 4), ("+-+-", [2, 1, i, j+1], 3 * factor / 4)]),=#
    #=                    "SF-d0-AN" => ("f", (k1, k2, points) -> k1 == antinode && k2 == antinode, (i, j, k1, k2; factor = 1) -> [("+-+-", [1, 2, i+1, j], 3 * factor / 4), ("+-+-", [2, 1, i, j+1], 3 * factor / 4)]),=#
    #=                    "SF-d0-s" => ("s", nothing, (i, j; factor = 1) -> [("+-+-", [1, 2, i+1, j], 3 * factor / 4), ("+-+-", [2, 1, i, j+1], 3 * factor / 4)]),=#
    #=                    "Sdz" => ("i", nothing, [("n", [1], 0.5), ("n", [2], -0.5)]),=#
    #=                   )=#
    microCorrelation = Dict(
                        "SF-dk1k1-f" => ("f", (k1, k2, points) -> k1 ≥ k2, (i, j, k1, k2; factor = 1) -> [("+-+-", [1, 2, i+1, j], 3 * factor / 4), ("+-+-", [2, 1, i, j+1], 3 * factor / 4)]),
                        "SF-d0-s" => ("s", nothing, (i, j; factor = 1) -> [("+-+-", [1, 2, i+1, j], 3 * factor / 4), ("+-+-", [2, 1, i, j+1], 3 * factor / 4)]),
                        "Sdz" => ("i", nothing, [("n", [1], 0.5), ("n", [2], 0.5)]),
                       )
            
    dos, dispersion = getDensityOfStates(tightBindDisp, size_BZ)
    northEastFermiPoints = filter(p -> all(map1DTo2D(p, size_BZ) .≥ 0), getIsoEngCont(dispersion, 0.0))
    @assert issorted(northEastFermiPoints)
    combinedResults = @showprogress pmap(couplings -> AuxiliaryCorrelations(size_BZ,
                                                      merge(basicCouplings, Dict(axLab .=> couplings)),
                                                      microCorrelation,
                                                      northEastFermiPoints,
                                                      maxSize;
                                                      loadData=loadData,
                                                    ),
                                         parameterSpace
                                        )

    momentumPairs = vec(collect(Iterators.product(northEastFermiPoints, northEastFermiPoints)))
    correlations = Dict(
                        "SF-d0-f" => cR -> abs(sum([cR["SF-dk1k1-f"][index] * NNFunc(k1, k2, size_BZ) for (index, (k1, k2)) in enumerate(momentumPairs)])),
                        "SF-d0-s" => cR -> abs(cR["SF-d0-s"]),
                        "SF-dNN" => cR -> abs(sum([cR["SF-dk1k1-f"][index] for (index, (k1, k2)) in enumerate(momentumPairs) if node in (k1, k2)])),
                        "SF-dAA" => cR -> abs(sum([cR["SF-dk1k1-f"][index] for (index, (k1, k2)) in enumerate(momentumPairs) if antinode in (k1, k2)])),
                        "Sdz" => cR -> abs(cR["Sdz"]),
                        "PF" => cR -> 0. + count(v -> abs(v) > 1e-5, [cR["SF-dk1k1-f"][index] for (index, (k1, k2)) in enumerate(momentumPairs) if k1 == k2])
                       )
    figPaths = []
    plottableResults = Dict(name => Dict() for name in keys(correlations))
    for (name, func) in correlations
        for (cR, key) in zip(combinedResults, parameterSpace)
            plottableResults[name][key] = func(cR)
        end
    end
    saveName = "$(saveNamePrefix)-$(size_BZ)-$(maxSize)" * ExtendedSaveName(couplingsRange)
    #=heatmapF = Dict()=#
    #=heatmapS = Dict()=#
    #=heatmapN = Dict()=#
    #=heatmapA = Dict()=#
    #=heatmapSdz = Dict()=#
    #=heatmapPF = Dict()=#
    #=for (index, key) in enumerate(parameterSpace)=#
    #=    heatmapF[key] = abs(combinedResults[index]["SF-d0-f"])=#
    #=    heatmapS[key] = abs(combinedResults[index]["SF-d0-s"])=#
    #=    heatmapSdz[key] = abs(combinedResults[index]["Sdz"])=#
    #=    heatmapPF[key] = abs(combinedResults[index]["PF"])=#
    #=    heatmapN[key] = abs(combinedResults[index]["SF-dNN"])=#
    #=    heatmapA[key] = abs(combinedResults[index]["SF-dAA"])=#
    #=end=#
    HM4D(plottableResults["SF-d0-f"], plottableResults["SF-d0-s"], rangedCouplings, axLab, "loc-$(saveName)", [cbarLabels["SF-d0-f"], cbarLabels["SF-d0-s"]], suptitle)
    HM4D(plottableResults["SF-dNN"], plottableResults["SF-dAA"], rangedCouplings, axLab, "k-$(saveName)", [cbarLabels["SF-dNN"], cbarLabels["SF-dAA"]], suptitle)
    HM4D(plottableResults["Sdz"], plottableResults["PF"], rangedCouplings, axLab, "Sdz-$(saveName)", [cbarLabels["Sdz"], cbarLabels["PF"]], suptitle)
    Lines(Dict("PF" => plottableResults["PF"]), rangedCouplings, axLab, saveName, cbarLabels, suptitle)
    return figPaths
end

namesCorr = String[]
namesPD = String[]
for (kondoF, U) in Iterators.product([0.5], [5.,])
    global impU = U
    couplings = Dict(
                     "mu_c" => [0.3, 0.2, 0.3],
                     "W" => [-1.5, 0.2, 0.5],
                     "kondoF" => [kondoF],
                     "Wf" => [-0.4, 0.01, -0.2],
                     "epsilonF" => [-0.5 * impU],
                     "lightBandFactor" => [1.5],
                     "kondoPerp" => [0.1, 0.05, 0.1],
                    )
    axLab = ["kondoPerp", "mu_c", "Wf", "W"]
    @time append!(namesCorr,
                AuxiliaryCorrelations(
                    25,
                    1501,
                    couplings,
                    axLab,
                    Dict("SF-d0-f"=>L"\langle S_f \cdot S_{f}^\prime \rangle_\text{NN}", "SF-d0-s" => L"\langle S_f \cdot S_c \rangle", "Sdz" => L"S^z_\text{imp}", "SF-dNN"=>L"\langle S_f \cdot S_{N}\rangle", "SF-dAA" => L"\langle S_f \cdot S_{AN} \rangle", "PF" => "PF"),
                    "sc-mz-";
                    loadData=true,
                   )
               )

    #=@time push!(namesPD, PhaseDiagram(=#
    #=                25,=#
    #=                couplings,=#
    #=                axLab,=#
    #=                Dict(["PF", L"J^*"]);=#
    #=                loadData=false,=#
    #=               ))=#

    #=@time ScattProb(25,=#
    #=                couplings,=#
    #=                axLab;=#
    #=                loadData=true,=#
    #=               )=#
end
if !isempty(namesPD); merge_pdfs(namesPD, "PD.pdf", cleanup=true); end
if !isempty(namesCorr); merge_pdfs(namesCorr, "Corr.pdf", cleanup=true); end
