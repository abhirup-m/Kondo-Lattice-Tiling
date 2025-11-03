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
using Distributed
@everywhere using ProgressMeter, PyPlot, LaTeXStrings
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
#=@everywhere include(submitDir * "PhaseDiagram.jl")=#
@everywhere include(submitDir * "Probes.jl")
@everywhere include(submitDir * "PltStyle.jl")

@everywhere function insertWfJ(couplings, Wf, J, μ)
    couplings["Wf"] = Wf
    couplings["J⟂"] = J
    couplings["μ"] = μ
    return couplings
end

function RGFlow(
        couplings,
        Wf,
        J,
        μ,
        size_BZ::Int64;
        loadData::Bool=false,
    )
    sparseWf = length(Wf) ≥ 20 ? Wf[1:div(length(Wf), 10):end] : Wf
    sparseJ = length(J) ≥ 20 ? J[1:div(length(J), 10):end] : J
    parameters = Iterators.product(Wf, J)
    dos, dispersion = getDensityOfStates(tightBindDisp, size_BZ)
    FSpoints = getIsoEngCont(dispersion, 0.0)
    fig, axes = subplots(ncols=3, nrows=length(μ), figsize=(20, 3.5 * length(μ)))
    fig.tight_layout()
    boolTitles = Dict("f" => L"Pole fraction: $J_f$", "d" => L"Pole fraction: $J_d$", "⟂" => L"Rel/Irrelevance: $J_\perp$")
    for (j, μ_i) in enumerate(μ)
        results = @showprogress desc="rg flow" pmap(p -> momentumSpaceRG(size_BZ, insertWfJ(couplings, p[1], p[2], μ_i); loadData=loadData), parameters)
        for (i, key) in enumerate(["f", "d", "⟂"])
            ax = length(μ) > 1 ? axes[j, i] : axes[i]
            if key == "f" || key == "d"
                keyData = [maximum(diag(abs.(r[key])[FSpoints, FSpoints])) for r in results]
                boolData = [count(>(1e-3), abs.(diag(r[key][FSpoints, FSpoints]))) / length(FSpoints) for r in results]
            else
                keyData = [r[key] for r in results]
                boolData = [ifelse(r[key]/J > 0.8, 1, ifelse(r[key]/J > 0., 0.5, 0)) for (r, (Wf, J)) in zip(results, Iterators.product(Wf, J))]
            end
            hm = ax.imshow(reshape(keyData, length(Wf), length(J)), origin="lower", extent=(extrema(J)..., extrema(-Wf)...), cmap=CMAP, aspect="auto")
            sparseColors = [b for (b, (Wfi, Ji)) in zip(boolData, parameters) if Wfi ∈ sparseWf && Ji ∈ sparseJ] # map(p -> ifelse(p[2][1] ∈ sparseWf && p[2][2] ∈ sparseJ, p[1], nothing), zip(boolData, parameters))
            sparseX = [Ji for (Wfi, Ji) in parameters if Wfi ∈ sparseWf && Ji ∈ sparseJ]
            sparseY = [-Wfi for (Wfi, Ji) in parameters if Wfi ∈ sparseWf && Ji ∈ sparseJ]
            sc = ax.scatter(sparseX, sparseY, c=sparseColors, s=10, cmap=CMAP, alpha=0.8)
            #=sc.set_facecolor("none")=#
            #=sc.set_edgecolor("black")=#
            ax.set_xlabel(L"$J$")
            ax.set_ylabel(L"$|Wf|$")
            if j == 1
                ax.set_title(L"RG flow of $J_%$(key)$", pad=10)
            end
            if i == 1
                ax.text(-1.0, 0.5, L"$\mu=%$(μ_i)$", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes)
            end
            ax.set_aspect((maximum(J) - minimum(J)) / (maximum(-Wf) - minimum(-Wf)))
            fig.colorbar(hm, shrink=0.5, pad=-0.3, label=L"$g^*$")
            fig.colorbar(sc, location="left", shrink=0.5, pad=0.22, label=L"$g > 0$",)
        end
    end
    fig.tight_layout()
    fig.savefig("PD.pdf", bbox_inches="tight")
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
        couplings,
        Wf,
        J,
        size_BZ,
        maxSize,
        cbarLabels::Dict;
        loadData::Bool=false,
    )
    parameterSpace = Iterators.product(Wf, J)

    node = map2DTo1D(π/2, π/2, size_BZ)
    antinode = map2DTo1D(3π/4, π/4, size_BZ)
    microCorrelation = Dict(
                            "SF-fkk" => ("f", (i, j; factor = 1) -> [("+-+-", [1, 2, i+1, j], factor / 2), ("+-+-", [2, 1, i, j+1], factor / 2), ("n+-", [1, i, j], factor / 4), ("n+-", [1, i+1, j+1], -factor / 4), ("n+-", [2, i, j], -factor / 4), ("n+-", [2, i+1, j+1], factor / 4)]),
                            "SF-dkk" => ("d", (i, j; factor = 1) -> [("+-+-", [3, 4, i+1, j], factor / 2), ("+-+-", [4, 3, i, j+1], factor / 2), ("n+-", [3, i, j], factor / 4), ("n+-", [3, i+1, j+1], -factor / 4), ("n+-", [4, i, j], -factor / 4), ("n+-", [4, i+1, j+1], factor / 4)]),
                            "SF-fdpm" => ("i", [("+-+-", [1, 2, 4, 3], 1 / 2), ("+-+-", [2, 1, 3, 4], 1 / 2)]),
                            "SF-fdzz" => ("i", [("nn", [1, 3], 1 / 4), ("nn", [1, 4], -1 / 4), ("nn", [2, 3], -1 / 4), ("nn", [2, 4], 1 / 4)]),
                       )
            
    dos, dispersion = getDensityOfStates(tightBindDisp, size_BZ)
    fermiShell = argmin(abs.([dispersion[i] - couplings["μ"] for i in 1:div(size_BZ+1, 2)]))
    momentumPoints = getIsoEngCont(dispersion, dispersion[fermiShell])
    @assert issorted(momentumPoints)
    combinedResults = @showprogress pmap(WfJ -> AuxiliaryCorrelations(size_BZ,
                                                      insertWfJ(couplings, WfJ..., couplings["μ"]),
                                                      microCorrelation,
                                                      momentumPoints,
                                                      maxSize;
                                                      loadData=loadData,
                                                    ),
                                         parameterSpace
                                        )

    momentumPairs = vec(collect(Iterators.product(momentumPoints, momentumPoints)))
    correlations = Dict(
                        "SF-fmax" => cR -> maximum(abs.(cR["SF-fkk"])),
                        "SF-dmax" => cR -> maximum(abs.(cR["SF-dkk"])),
                        "SF-f0" => cR -> sum([abs(cR["SF-fkk"][index] * NNFunc(k1, k2, size_BZ)) for (index, (k1, k2)) in enumerate(momentumPairs)]) / length(momentumPoints),
                        "SF-d0" => cR -> sum([abs(cR["SF-dkk"][index] * NNFunc(k1, k2, size_BZ)) for (index, (k1, k2)) in enumerate(momentumPairs)]) / length(momentumPoints),
                        "SF-fdpm" => cR -> abs(cR["SF-fdpm"]),
                        "SF-fdzz" => cR -> abs(cR["SF-fdzz"]),
                        "SF-fPF" => cR -> count(>(0), abs.([cR["SF-fkk"][index] for (index, (k1, k2)) in enumerate(momentumPairs) if k1 == k2])) / length(momentumPoints),
                        "SF-dPF" => cR -> count(>(0), abs.([cR["SF-dkk"][index] for (index, (k1, k2)) in enumerate(momentumPairs) if k1 == k2])) / length(momentumPoints),
                        #="SF-dNN" => cR -> abs(sum([cR["SF-dk1k1-f"][index] for (index, (k1, k2)) in enumerate(momentumPairs) if node in (k1, k2)])),=#
                        #="SF-dAA" => cR -> abs(sum([cR["SF-dk1k1-f"][index] for (index, (k1, k2)) in enumerate(momentumPairs) if antinode in (k1, k2)])),=#
                        #="Sdz" => cR -> abs(cR["Sdz"]),=#
                        #="PF" => cR -> 0. + count(v -> abs(v) > 1e-5, [cR["SF-dk1k1-f"][index] for (index, (k1, k2)) in enumerate(momentumPairs) if k1 == k2])=#
                       )
    figPaths = []
    plottableResults = Dict{String, Vector{Float64}}()
    for (name, func) in correlations
        plottableResults[name] = [func(cR) for cR in vec(collect(combinedResults))]
    end
    #=saveName = "$(saveNamePrefix)-$(size_BZ)-$(maxSize)" * ExtendedSaveName(couplingsRange)=#
    RowPlots(plottableResults, collect(parameterSpace), [("SF-f0", "SF-d0"), ("SF-fdpm", "SF-fdzz"), ("SF-fPF", "SF-dPF")], [(L"$\langle {S_f\cdot S_{f0}}\rangle$",L"$\langle {S_d\cdot S_{d0}}\rangle$"), (L"$\langle {S_f^+S_d^- + \text{h.c.}}\rangle$", L"$\langle {S_f^zS_d^z}\rangle$"), ("f-PF", "d-PF")], ["in-plane correlation", "out-of-plane correlation", "Pole Fraction"], ("J", "Wf"), "locCorr.pdf")
    return figPaths
end

size_BZ = 33
J = 0.0:0.01:0.1
Wf = -0.0:-0.02:-0.2
#=RGFlow(Dict("omega_by_t" => -2., "μ" => 0.0, "Jf" => 0.1, "Jd" => 0.1, "J⟂" => 0., "Wd" => -0.0, "Wf" => 0.), Wf, J, 0:1.5:0, size_BZ; loadData=true)=#
AuxiliaryCorrelations(Dict("omega_by_t" => -2., "μ" => 0.0, "Jf" => 0.1, "Jd" => 0.1, "J⟂" => 0., "Wd" => -0.0, "Wf" => 0.), Wf, J, size_BZ, 802, Dict("SF-f0"=>"SF-f0", "SF-d0"=>"SF-d0"); loadData=false)
