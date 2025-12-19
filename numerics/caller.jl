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
@everywhere using ProgressMeter, PyPlot, LaTeXStrings, PDFmerger
if "SLURM_SUBMIT_DIR" in keys(ENV)
    addprocs(SlurmManager())
end

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

@everywhere function insertWfJ(couplings, Wf, J, μd)
    couplings["Wf"] = Wf
    couplings["Wd"] = Wf * couplings["Wd_by_Wf"]
    couplings["J⟂"] = J
    couplings["μd"] = μd
    couplings["omega_by_t"] = OMEGA_BY_t
    return couplings
end

function RGFlow(
        couplings,
        Wf,
        Jp,
        μ,
        size_BZ::Int64;
        loadData::Bool=false,
    )
    sparseWf = length(Wf) ≥ 20 ? Wf[1:div(length(Wf), 10):end] : Wf
    sparseJp = length(Jp) ≥ 20 ? Jp[1:div(length(Jp), 10):end] : Jp
    parameters = Iterators.product(Wf, Jp)
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
                boolData = [ifelse(r[key]/J > 0.8, 1, ifelse(r[key]/J > 0., 0.5, 0)) for (r, (Wf, J)) in zip(results, Iterators.product(Wf, Jp))]
            end
            hm = ax.imshow(reshape(keyData, length(Wf), length(Jp)), origin="lower", extent=(extrema(Jp)..., extrema(-Wf)...), cmap=CMAP, aspect="auto")
            sparseColors = [b for (b, (Wfi, Ji)) in zip(boolData, parameters) if Wfi ∈ sparseWf && Ji ∈ sparseJp] # map(p -> ifelse(p[2][1] ∈ sparseWf && p[2][2] ∈ sparseJ, p[1], nothing), zip(boolData, parameters))
            sparseX = [Ji for (Wfi, Ji) in parameters if Wfi ∈ sparseWf && Ji ∈ sparseJp]
            sparseY = [-Wfi for (Wfi, Ji) in parameters if Wfi ∈ sparseWf && Ji ∈ sparseJp]
            sc = ax.scatter(sparseX, sparseY, c=sparseColors, s=10, cmap=CMAP, alpha=0.8)
            #=sc.set_facecolor("none")=#
            #=sc.set_edgecolor("black")=#
            ax.set_xlabel(L"$J_\perp$")
            ax.set_ylabel(L"$|Wf|$")
            if j == 1
                ax.set_title(L"RG flow of $J_%$(key)$", pad=10)
            end
            if i == 1
                phAssymetry = round(μ_i - couplings["Ud"]/2, digits=3)
                ax.text(-1.2, 0.5, L"$\eta_d=%$(phAssymetry)$", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes)
            end
            ax.set_aspect((maximum(Jp) - minimum(Jp)) / (maximum(-Wf) - minimum(-Wf)))
            fig.colorbar(hm, shrink=0.5, pad=-0.7, label=L"$g^* (BG)$")
            fig.colorbar(sc, location="left", shrink=0.5, pad=0.11, label=L"$g > 0$ (FG)",)
        end
    end
    fig.tight_layout()
    fig.savefig("PD_$(size_BZ).pdf", bbox_inches="tight")
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
                            "CF-dkk" => ("d", (i, j; factor = 1) -> [("++--", [3, 4, i+1, j], factor / 2), ("++--", [4, 3, i, j+1], factor / 2)]),
                            "CF-fkk" => ("f", (i, j; factor = 1) -> [("++--", [1, 2, i+1, j], factor / 2), ("--++", [2, 1, i, j+1], factor / 2)]),
                            "ndu" => ("i", [("n", [3], 1.)]),
                            "ndd" => ("i", [("n", [4], 1.)]),
                            "nfu" => ("i", [("n", [1], 1.)]),
                            "nfd" => ("i", [("n", [2], 1.)]),
                       )
            
    dos, dispersion = getDensityOfStates(tightBindDisp, size_BZ)
    #=fermiShell = Dict(k => argmin(abs.([dispersion[i] - couplings["μ$k"] for i in 1:div(size_BZ+1, 2)])) for k in ["f", "d"])=#
    momentumPoints = Dict(k => getIsoEngCont(dispersion, 0.) for k in ["f", "d"])
    entanglement = String[]
    combinedResults = @showprogress pmap(WfJ -> AuxiliaryCorrelations(size_BZ,
                                                      insertWfJ(couplings, WfJ..., couplings["μd"]),
                                                      microCorrelation,
                                                      momentumPoints,
                                                      Dict(),
                                                      maxSize;
                                                      #=entanglement=entanglement,=#
                                                      loadData=loadData,
                                                    ),
                                         parameterSpace
                                        )

    momentumPairs = Dict(k => vec(collect(Iterators.product(momentumPoints[k], momentumPoints[k]))) for k in ["f", "d"])
    correlations = Dict(
                        "SF-fmax" => cR -> maximum(abs.(cR["SF-fkk"])),
                        "SF-dmax" => cR -> maximum(abs.(cR["SF-dkk"])),
                        "SF-f0" => cR -> -sum([abs(cR["SF-fkk"][index]) * NNFunc(k1, k2, size_BZ) for (index, (k1, k2)) in enumerate(momentumPairs["f"])]) / length(momentumPoints["f"]),
                        #="SF-d0" => cR -> maximum(abs.(cR["SF-dkk"])),=#
                        #="SF-d0" => cR -> sum(abs.(cR["SF-dkk"])) / length(momentumPoints["d"]),=#
                        "SF-d0" => cR -> -sum([abs(cR["SF-dkk"][index]) * NNFunc(k1, k2, size_BZ) for (index, (k1, k2)) in enumerate(momentumPairs["d"])]) / length(momentumPoints["d"]),
                        "SF-fd" => cR -> -(cR["SF-fdpm"] + cR["SF-fdzz"]),
                        "CF-d0" => cR -> sum([abs(cR["CF-dkk"][index]) * NNFunc(k1, k2, size_BZ) for (index, (k1, k2)) in enumerate(momentumPairs["d"])]) / length(momentumPoints["d"]),
                        "CF-f0" => cR -> -sum([abs(cR["CF-fkk"][index]) * NNFunc(k1, k2, size_BZ) for (index, (k1, k2)) in enumerate(momentumPairs["f"])]) / length(momentumPoints["f"]),
                        "SF-fdzz" => cR -> abs(cR["SF-fdzz"]),
                        "SF-fPF" => cR -> count(>(0), abs.([cR["SF-fkk"][index] for (index, (k1, k2)) in enumerate(momentumPairs["f"]) if k1 == k2])) / length(momentumPoints["f"]),
                        "SF-dPF" => cR -> count(>(0), abs.([cR["SF-dkk"][index] for (index, (k1, k2)) in enumerate(momentumPairs["d"]) if k1 == k2])) / length(momentumPoints["d"]),
                        "Sdz" => cR -> 0.5 * abs(cR["ndu"] - cR["ndd"]),
                        "Sfz" => cR -> 0.5 * abs(cR["nfu"] - cR["nfd"]),
                        "nd" => cR -> 0.5 * abs(cR["ndu"] + cR["ndd"]),
                        "nf" => cR -> 0.5 * abs(cR["nfu"] + cR["nfd"]),
                       )
    if !isempty(entanglement)
        correlations["SEE-f"] = cR -> cR["SEE-f"]
        correlations["SEE-d"] = cR -> cR["SEE-d"]
        correlations["I2-f-d"] = cR -> cR["I2-f-d"]
        correlations["I2-f-max"] = cR -> cR["I2-f-max"]
        correlations["I2-d-max"] = cR -> cR["I2-d-max"]
    end
    figPaths = []
    plottableResults = Dict{String, Vector{Float64}}()
    for (name, func) in correlations
        plottableResults[name] = [func(cR) for cR in vec(collect(combinedResults))]
    end
    eta = round(-couplings["μd"] + couplings["Ud"]/2, digits=3)
    return RowPlots(plottableResults,
                    collect(parameterSpace),
                    [("SF-f0", "SF-d0"), ("SF-fd", "SF-fPF"), ("CF-d0", "CF-f0"),],
                    #=[("SF-f0", "SF-d0"), ("SF-fd", "SF-fPF"), ("CF-d0", "CF-f0"), ("SEE-f", "I2-f-d"), ("I2-f-max", "I2-d-max")],=#
                    [(L"$-\langle {S_f\cdot S_{f0}}\rangle$",L"$-\langle {S_d\cdot S_{d0}}\rangle$"), (L"$-\langle S_f\cdot S_d\rangle$", "f-PF"), (L"CF", L"n_d"), ],
                    ["in-plane correlation", L"$Sd.Sf$, PF", "charge",],
                    ("J", "Wf"),
                    "locCorr-$(eta).pdf",
                    L"$\eta_d = %$(eta)$",
                    []
                   )
end

@everywhere function SpecFuncKspace(
        couplings,
        size_BZ,
        maxSize;
        loadData::Bool=false,
    )
    dos, dispersion = getDensityOfStates(tightBindDisp, size_BZ)
    momentumPoints = Dict(k => getIsoEngCont(dispersion, 0.) for k in ["f", "d"])

    antinode = map2DTo1D(π/1, 0., size_BZ)
    node = map2DTo1D(π/2, π/2, size_BZ)
    Ad = Dict("d" => [("+", [3], 1.), ("+", [4], 1.)], 
              "f" => [("+", [1], 1.), ("+", [2], 1.)]
             )
    siam_d(i_d, i_f) = Dict("d" => [("-+", [3, i_d], 1.), ("-+", [4, i_d + 1], 1.),],
                            "f" => [("-+", [1, i_d], 1.), ("-+", [2, i_d + 1], 1.),]
                           )
    siam_f(i_d, i_f, sgn) = Dict("d" => [("-+", [3, i_f], sgn * 1.), ("-+", [4, i_f + 1], sgn * 1.),],
                                 "f" => [("-+", [1, i_f], sgn * 1.), ("-+", [2, i_f + 1], sgn * 1.),]
                                )
    siam_charge_d(i_d, i_f) = Dict("d" => [("--+", [3, 4, i_d,], 1.), ("--+", [3, 4, i_d+1,], 1.),],
                                   "f" => [("--+", [1, 2, i_d,], 1.), ("--+", [1, 2, i_d+1,], 1.),]
                                  )
    siam_charge_f(i_d, i_f, sgn) = Dict("d" => [("--+", [3, 4, i_f,], sgn * 1.), ("--+", [3, 4, i_f+1,], sgn * 1.),],
                                        "f" => [("--+", [1, 2, i_f,], sgn * 1.), ("--+", [1, 2, i_f+1,], sgn * 1.),]
                                       )

    kondo_d(i_d, i_f) = Dict("d" => [("+-+", [3, 4, i_d], 1.), ("+-+", [4, 3, i_d + 1], 1.),],
                             "f" => [("+-+", [1, 2, i_d], 1.), ("+-+", [2, 1, i_d + 1], 1.),]
                            )
    kondo_f(i_d, i_f, sgn) = Dict("d" => [("+-+", [3, 4, i_f + 1], sgn * 1.), ("+-+", [4, 3, i_f], sgn * 1.)],
                                  "f" => [("+-+", [1, 2, i_f + 1], sgn * 1.), ("+-+", [2, 1, i_f], sgn * 1.)]
                                 )
    Ad_Siam_bonding(i_d, i_f) = Dict(k => vcat(siam_d(i_d, i_f)[k], siam_f(i_d, i_f, 1)[k]) for k in ["d", "f"])
    Ad_Siam_antibonding(i_d, i_f) = Dict(k => vcat(siam_d(i_d, i_f)[k], siam_f(i_d, i_f, -1)[k]) for k in ["d", "f"])
    Ad_Siam_charge_bonding(i_d, i_f) = Dict(k => vcat(siam_charge_d(i_d, i_f)[k], siam_charge_f(i_d, i_f, 1)[k]) for k in ["d", "f"])
    Ad_Siam_charge_antibonding(i_d, i_f) = Dict(k => vcat(siam_charge_d(i_d, i_f)[k], siam_charge_f(i_d, i_f, -1)[k]) for k in ["d", "f"])
    Ad_Kondo_bonding(i_d, i_f) = Dict(k =>  vcat(kondo_d(i_d, i_f)[k], kondo_f(i_d, i_f, 1)[k]) for k in ["d", "f"])
    Ad_Kondo_antibonding(i_d, i_f) = Dict(k => vcat(kondo_d(i_d, i_f)[k], kondo_f(i_d, i_f, -1)[k]) for k in ["d", "f"])

    specFuncReqs = Dict(
                        "Ad_Siam" => ("i", Ad["d"]),
                        "Ad_Siam_+_in" => ("bond", ==(node), (i_d, i_f) -> Ad_Siam_bonding(i_d, i_f)["d"]),
                        "Ad_Siam_-_in" => ("bond", ==(node), (i_d, i_f) -> Ad_Siam_antibonding(i_d, i_f)["d"]),
                        "Ad_Siam_+_out" => ("bond", ==(node), (i_d, i_f) -> Ad_Siam_bonding(i_d, i_f)["d"]),
                        "Ad_Siam_-_out" => ("bond", ==(node), (i_d, i_f) -> Ad_Siam_antibonding(i_d, i_f)["d"]),
                        "Ad_Kondo_+" => ("bond", ==(node), (i_d, i_f) -> Ad_Kondo_bonding(i_d, i_f)["d"]),
                        "Ad_Kondo_-" => ("bond", ==(node), (i_d, i_f) -> Ad_Kondo_antibonding(i_d, i_f)["d"]),
                        "Ad_charge_+" => ("bond", ==(node), (i_d, i_f) -> Ad_Siam_charge_bonding(i_d, i_f)["d"]),
                        "Ad_charge_-" => ("bond", ==(node), (i_d, i_f) -> Ad_Siam_charge_antibonding(i_d, i_f)["d"]),
                        "Af_Siam" => ("i", Ad["f"]),
                        "Af_Siam_+_in" => ("bond", ==(node), (i_d, i_f) -> Ad_Siam_bonding(i_d, i_f)["f"]),
                        "Af_Siam_-_in" => ("bond", ==(node), (i_d, i_f) -> Ad_Siam_antibonding(i_d, i_f)["f"]),
                        "Af_Siam_+_out" => ("bond", ==(node), (i_d, i_f) -> Ad_Siam_bonding(i_d, i_f)["f"]),
                        "Af_Siam_-_out" => ("bond", ==(node), (i_d, i_f) -> Ad_Siam_antibonding(i_d, i_f)["f"]),
                        "Af_Kondo_+" => ("bond", ==(node), (i_d, i_f) -> Ad_Kondo_bonding(i_d, i_f)["f"]),
                        "Af_Kondo_-" => ("bond", ==(node), (i_d, i_f) -> Ad_Kondo_antibonding(i_d, i_f)["f"]),
                        "Af_charge_+" => ("bond", ==(node), (i_d, i_f) -> Ad_Siam_charge_bonding(i_d, i_f)["f"]),
                        "Af_charge_-" => ("bond", ==(node), (i_d, i_f) -> Ad_Siam_charge_antibonding(i_d, i_f)["f"]),
                       )

    results, fpCouplings = AuxiliaryCorrelations(size_BZ,
                                  couplings,
                                  Dict(),
                                  momentumPoints,
                                  specFuncReqs,
                                  maxSize;
                                  entanglement=String[],
                                  loadData=loadData,
                                 )
    freqVals = collect(-10:0.001:10)
    #=specFuncReqs["η"] = 0.01 .* ones(length(specFuncReqs["ω"]))=#
    specFuncResults = Dict()

    phAssymetry = couplings["μd"] - couplings["Ud"]/2
    for k in keys(specFuncReqs)
        if k == "ω" || k == "η"
            continue
        end
        specFuncResults[k] = 0 .* freqVals
        for specCoeffs in results[k]
            if endswith(k, "+_in") || endswith(k, "-_in") || endswith(k, "+_out") || endswith(k, "-_out")
                bandFilter = endswith(k, "+_in") || endswith(k, "-_in") ? p -> abs(p) < couplings["Ud"]/4 : p -> abs(p) ≥ couplings["Ud"]/4
                shift = endswith(k, "+") ? couplings["J⟂"] : -couplings["J⟂"]
                greaterPoles = Dict(last.(specCoeffs) .=> last.(specCoeffs) .- shift .- phAssymetry)
                broadening = [ifelse(haskey(greaterPoles, ω), 0.5, 0.05) for ω in freqVals]
                specFunc_greater = count(p -> p > 0 && bandFilter(p), last.(specCoeffs)) > 0 ? SpecFunc([(w, greaterPoles[p]) for (w, p) in filter(p -> p[2] > 0 && bandFilter(p[2]), specCoeffs)], freqVals, broadening; normalise=false) : 0
                lesserPoles = Dict(last.(specCoeffs) .=> last.(specCoeffs) .+ shift .+ phAssymetry)
                broadening = [ifelse(haskey(lesserPoles, ω), 0.5, 0.05) for ω in freqVals]
                specFunc_lesser = count(p -> p < 0 && bandFilter(p), last.(specCoeffs)) > 0 ? SpecFunc([(w, lesserPoles[p]) for (w, p) in filter(p -> p[2] < 0 && bandFilter(p[2]), specCoeffs)], freqVals, broadening; normalise=false) : 0
                specFunc = specFunc_greater .+ specFunc_lesser
            elseif endswith(k, "+") || endswith(k, "-")
                shift = endswith(k, "+") ? couplings["J⟂"] : -couplings["J⟂"]
                greaterPoles = Dict(last.(specCoeffs) .=> last.(specCoeffs) .- shift .- phAssymetry)
                broadening = [ifelse(haskey(greaterPoles, ω), 0.5, 0.05) for ω in freqVals]
                specFunc_greater = count(>(0), last.(specCoeffs)) > 0 ? SpecFunc([(w, greaterPoles[p]) for (w, p) in filter(p -> p[2] > 0, specCoeffs)], freqVals, broadening; normalise=false) : 0
                lesserPoles = Dict(last.(specCoeffs) .=> last.(specCoeffs) .+ shift .+ phAssymetry)
                broadening = [ifelse(haskey(lesserPoles, ω), 0.5, 0.05) for ω in freqVals]
                specFunc_lesser = count(<(0), last.(specCoeffs)) > 0 ? SpecFunc([(w, lesserPoles[p]) for (w, p) in filter(p -> p[2] < 0, specCoeffs)], freqVals, broadening; normalise=false) : 0
                specFunc = specFunc_greater .+ specFunc_lesser
            else
                broadening = [ifelse(ω ∈ last.(specCoeffs), 0.5, 0.05) for ω in freqVals]
                specFunc = SpecFunc(specCoeffs, freqVals, broadening; normalise=false)
                #=specFunc = SpecFunc(filter(p -> abs(p[2]) < 0.5, specCoeffs), freqVals, specFuncReqs["η"]; normalise=false)=#
                #=specFunc += SpecFunc(filter(p -> abs(p[2]) > 0.5, specCoeffs), freqVals, specFuncReqs["η"]; normalise=false)=#
            end
            specFuncResults[k] .+= specFunc
        end
        if maximum(specFuncResults[k]) > 1e-2
            norm = abs(sum(specFuncResults[k]) * (freqVals[2] - freqVals[1]))
            #=println(norm)=#
            #=specFuncResults[k] /= norm=#
        end
        #=println(specFuncResults[k])=#
    end

    averageKondoScale = maximum(fpCouplings["d"][momentumPoints["d"], momentumPoints["d"]]) / couplings["Jd"]
    specFuncResults["node-d"] = (0 * specFuncResults["Ad_Siam"] +
                                 √averageKondoScale * couplings["Ud"] * specFuncResults["Ad_Siam_+_in"] +
                                 1 * specFuncResults["Ad_Siam_+_out"] +
                                 0.5 * fpCouplings["d"][node] * specFuncResults["Ad_Kondo_+"] +
                                 √averageKondoScale * couplings["Ud"] * specFuncResults["Ad_Siam_-_in"] +
                                 1 * specFuncResults["Ad_Siam_-_out"] +
                                 0.5 * fpCouplings["d"][node] * specFuncResults["Ad_Kondo_-"]
                                )
    norm = abs(sum(specFuncResults["node-d"]) * (freqVals[2] - freqVals[1]))
    specFuncResults["node-d"] /= norm

    averageKondoScale = maximum(fpCouplings["f"][momentumPoints["f"], momentumPoints["f"]]) / couplings["Jf"]
    specFuncResults["node-f"] = (0 * specFuncResults["Af_Siam"] +
                                 √averageKondoScale * couplings["Uf"] * specFuncResults["Af_Siam_+_in"] +
                                 1 * specFuncResults["Af_Siam_+_out"] +
                                 0.5 * averageKondoScale * specFuncResults["Af_Kondo_+"] +
                                 #=specFuncResults["Af_charge_+"] +=#
                                 √averageKondoScale * couplings["Uf"] * specFuncResults["Af_Siam_-_in"] +
                                 1 * specFuncResults["Af_Siam_-_out"] +
                                 0.5 * averageKondoScale * specFuncResults["Af_Kondo_-"]
                                 #=specFuncResults["Af_charge_-"]=#
                                )
    norm = abs(sum(specFuncResults["node-f"]) * (freqVals[2] - freqVals[1]))
    specFuncResults["node-f"] /= norm

    return specFuncResults, freqVals
end


function AuxSpecFunc(
        couplings,
        size_BZ,
        maxSize;
        loadData::Bool=false,
    )
    Ad_Siam = Dict("create" => [("+", [3,], 1.), ("+", [4,], 1.)],
                  "destroy" => [("-", [4], 1.), ("-", [3,], 1.)]
                 )

    Af_Siam = Dict("create" => [("+", [1,], 1.), ("+", [2,], 1.)],
                  "destroy" => [("-", [2,], 1.), ("-", [1,], 1.)]
                 )
    Afd_sym = Dict("create" => [("+-+", [1,2,4], 1.), ("+-+", [2,1,3], 1.)],
                  "destroy" => [("+--", [2,1,4], 1.), ("+--", [1,2,3,], 1.)]
                 )
    Ad_Kondo(i) = Dict("create" => [("+-+", [3,4,i+1], 1.), ("+-+", [4,3,i], 1.)],
                  "destroy" => [("+--", [4,3,i+1], 1.), ("+--", [3,4,i], 1.)]
                 )

    Af_Kondo(i) = Dict("create" => [("+-+", [1,2,i+1], 1.), ("+-+", [2,1,i], 1.)],
                  "destroy" => [("+--", [2,1,i+1], 1.), ("+--", [1,2,i], 1.)]
                 )
    Afd_asym(i,j) = Dict("create" => [("+-+", [1,2,i+1], 1.), ("+-+", [2,1,i], 1.)],
                  "destroy" => [("+--", [4,3,j+1], 1.), ("+--", [3,4,j], 1.)]
                 )
    specFunc = Dict(
                    "Ad_Siam" => ("i", Ad_Siam),
                    "Ad_Kondo" => ("d", i -> Ad_Kondo(i)),
                    "Af_Siam" => ("i", Af_Siam),
                    "Af_Kondo" => ("f", i -> Af_Kondo(i)),
                    "Afd" => ("i", Afd_sym),
                    "Afd_asym" => ("fd", (i, j) -> Afd_asym(i, j)),
                    "ω" => collect(-12:0.001:12),
                       )
    specFunc["η"] = 0.05 .+ abs.(specFunc["ω"] / maximum(specFunc["ω"]))
            
    dos, dispersion = getDensityOfStates(tightBindDisp, size_BZ)
    momentumPoints = Dict(k => getIsoEngCont(dispersion, 0.) for k in ["f", "d"])
    specFuncResults = AuxiliaryCorrelations(size_BZ,
                                  couplings,
                                  Dict(),
                                  momentumPoints,
                                  specFunc,
                                  maxSize;
                                  entanglement=String[],
                                  loadData=loadData,
                                 )

    for k in keys(specFunc)
        if k == "ω" || k == "η"
            continue
        end
        specFunc_k = 0 .* specFunc["ω"] 
        for r in specFuncResults[k]
            A_r = SpecFunc(r, specFunc["ω"], specFunc["η"]; normalise=false)
            norm = sum(A_r) * (specFunc["ω"][2] - specFunc["ω"][1])
            if maximum(A_r) > 1e-2
                A_r /= norm
            end
            specFunc_k .+= A_r
        end
        norm = sum(specFunc_k) * (specFunc["ω"][2] - specFunc["ω"][1])
        if maximum(specFunc_k) > 1e-2
            specFunc_k /= norm
        end
        specFuncResults[k] = specFunc_k
    end

    specFuncResults["Ad"] = (2/3) * (specFuncResults["Ad_Siam"] + 0.5 * specFuncResults["Ad_Kondo"])
    specFuncResults["Af"] = (2/3) * (specFuncResults["Af_Siam"] + 0.5 * specFuncResults["Af_Kondo"])

    return specFuncResults, specFunc["ω"]
end

function PhaseDiagram(
        size_BZ::Int64,
        couplingsRange::Dict{String, Any}, 
        axLab::Vector{String},
        cbarLabels;
        loadData::Bool=false,
    )
    nonAxLabs = sort([k for k in keys(couplingsRange) if k ∉ axLab])
    constCouplings = Dict{String, Float64}(k => v[1] for (k, v) in couplingsRange if length(v) == 1)
    constCouplings["omega_by_t"] = OMEGA_BY_t
    #=constCouplings["impU"] = impU=#
    rangedCouplings = Dict{String, Vector{Float64}}(k => v for (k, v) in couplingsRange if length(v) > 1)
    parameterSpace = Iterators.product([rangedCouplings[ax] for ax in axLab]...)
    saveName = "PD-$(size_BZ)-" * ExtendedSaveName(couplingsRange)

    heatmapF = Dict{NTuple{4, Float64}, Float64}()
    heatmapS = Dict{NTuple{4, Float64}, Float64}()
    try
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
                                                                         merge(constCouplings, Dict(axLab .=> couplings));
                                                                         loadData=loadData,
                                                        ),
                                             parameterSpace
                                            )

        for (index, key) in enumerate(parameterSpace)
            heatmapF[key] = abs(combinedResults[index]["Jf"])
            heatmapS[key] = abs(combinedResults[index]["J⟂"])
        end
        open(joinpath(SAVEDIR, saveName * "-f.json"), "w") do file JSON3.write(file, heatmapF) end
        open(joinpath(SAVEDIR, saveName * "-s.json"), "w") do file JSON3.write(file, heatmapS) end
    end
    suptitle = latexstring(join(["$(AXES_LABELS[lab])=$(couplingsRange[lab][1])" for lab in nonAxLabs], ", "))
    return HM4D(heatmapF, heatmapS, rangedCouplings, axLab, saveName, cbarLabels, suptitle)
end


function RealSpaceCorr()
    size_BZ = 13
    maxSize = 1199
    J = 0.0:0.05:0.2
    Wf = -0.0:-0.05:-0.15
    paths = String[]
    couplings = Dict("omega_by_t" => -2.,
                     "Uf" => 10.,
                     "Jf" => 0.1,
                     "Jd" => 0.1,
                     "J⟂" => 0.,
                     "Wd" => -0.0,
                     "Wd_by_Wf" => 1.0,
                    )
    couplings["Ud"] = couplings["Uf"] * couplings["Wd_by_Wf"]
    couplings["μf"] = couplings["Uf"]/2
    for μd in [1.0, 0.6, 0.2] .* couplings["Ud"] / 2
        couplings["μd"] = μd
        path = AuxiliaryCorrelations(couplings, Wf, J, size_BZ, maxSize, 
                                     Dict("SF-f0"=>"SF-f0", "SF-d0"=>"SF-d0");
                                     loadData=true)
        push!(paths, path)
    end
    merge_pdfs(paths, "locCorr_$(size_BZ)_$(maxSize).pdf", cleanup=true)
end

function RealSpaceSF()
    size_BZ = 33
    Wf_vals = [-0., -0.1, -0.15,]
    Jp_vals = [0., 0.05, 0.1]
    fig1, ax1 = plt.subplots(ncols=length(Jp_vals), nrows=length(Wf_vals), figsize=(8 * length(Jp_vals), 6 * length(Wf_vals)))
    fig2, ax2 = plt.subplots(ncols=length(Jp_vals), nrows=length(Wf_vals), figsize=(8 * length(Jp_vals), 6 * length(Wf_vals)))
    fig3, ax3 = plt.subplots(ncols=length(Jp_vals), nrows=length(Wf_vals), figsize=(8 * length(Jp_vals), 6 * length(Wf_vals)))
    counter = 1
    couplings = Dict("omega_by_t" => -2.,
                     "Uf" => 4.,
                     "Jf" => 0.1,
                     "Wd_by_Wf" => 0.5,
                    )
    couplings["Ud"] = couplings["Uf"] * couplings["Wd_by_Wf"]
    couplings["μf"] = couplings["Uf"]/2
    couplings["Jd"] = couplings["Jf"]
    for (Wf, Jp) in Iterators.product(Wf_vals, Jp_vals)
        couplings["J⟂"] = Jp
        couplings["Wf"] = Wf
        couplings["Wd"] = 0 * couplings["Wf"] * couplings["Wd_by_Wf"]
        μdValues = [1.0, 0.6, 1.4] .* couplings["Ud"] / 2
        axs = Dict(k => length(Wf_vals) * length(Jp_vals) > 1 ? ax[counter] : ax for (k, ax) in zip(["Ad", "Af", "Afd"], [ax1, ax2, ax3]))
        ins = Dict(k => axv.inset_axes([0.55,0.5,0.45,0.4]) for (k, axv) in axs)
        for (i, μd) in enumerate(μdValues)
            couplings["μd"] = μd
            results, ω = AuxSpecFunc(couplings,
                        size_BZ,
                        1301;
                        loadData=false,
                       )
            for (k, axv) in axs
                axv.plot(ω[abs.(ω) .< 4], results[k][abs.(ω) .< 4], label=L"η_d=%$(-μd + couplings[\"Ud\"]/2)")
                ins[k].plot(ω[abs.(ω) .< 1.5], results[k][abs.(ω) .< 1.5], label=L"η_d=%$(-μd + couplings[\"Ud\"]/2)")
                ins[k].set_yticks([])
            end
        end
        for k in ["Ad", "Af", "Afd"]
            axs[k].legend(loc="center left")
            axs[k].text(0.05, 0.95,
                             L"""
                             $J_d=%$(couplings[\"Jd\"])$
                             $J_f=%$(couplings[\"Jf\"])$
                             $J_⟂=%$(Jp)$
                             $W_f=%$(Wf)$
                             $W_d=%$(couplings["Wd"])$
                             """,
                             horizontalalignment="left", verticalalignment="top", transform=axs[k].transAxes)
        end
        global counter += 1
    end
    fig1.savefig("specFunc_Ad.pdf", bbox_inches="tight")
    fig2.savefig("specFunc_Af.pdf", bbox_inches="tight")
    fig3.savefig("specFunc_Afd.pdf", bbox_inches="tight")
    plt.close()
end

function KspaceSF()
    size_BZ = 33
    maxSize = 1707
    Wf_vals = [-0., -0.15]
    Jp_vals = [0., 0.1]
    fig1, ax1 = plt.subplots(ncols=length(Jp_vals), nrows=length(Wf_vals), figsize=(8 * length(Jp_vals), 6 * length(Wf_vals)))
    fig2, ax2 = plt.subplots(ncols=length(Jp_vals), nrows=length(Wf_vals), figsize=(8 * length(Jp_vals), 6 * length(Wf_vals)))
    counter = 1
    couplings = Dict("omega_by_t" => -2.,
                     "Uf" => 10.,
                     "Jf" => 0.1,
                     "Wd_by_Wf" => 1.,
                    )
    couplings["Ud"] = couplings["Uf"] * couplings["Wd_by_Wf"]
    couplings["μf"] = couplings["Uf"]/2
    couplings["Jd"] = couplings["Jf"]
    couplings["Wd"] = 0.
    μdValues = [1.] .* couplings["Ud"] / 2
    couplingSets = Iterators.product(Wf_vals, Jp_vals, μdValues)
    resultsPooled = Dict(couplingSets .=> pmap(trip -> SpecFuncKspace(merge(Dict("Wf" => trip[1], "J⟂" => trip[2], "μd" => trip[3]), couplings), size_BZ, maxSize; loadData=false), couplingSets))
    for (Wf, Jp) in Iterators.product(Wf_vals, Jp_vals)
        couplings["J⟂"] = Jp
        couplings["Wf"] = Wf
        axs = Dict(k => length(Wf_vals) * length(Jp_vals) > 1 ? ax[counter] : ax for (k, ax) in zip(["node-d", "node-f"], [ax1,ax2]))
        ins = Dict(k => axv.inset_axes([0.55,0.5,0.45,0.4]) for (k, axv) in axs)
        for (i, μd) in enumerate(μdValues)
            couplings["μd"] = μd
            results, ω = resultsPooled[(Wf, Jp, μd)]
            phAssymetry = round(couplings["μd"] - couplings["Ud"]/2, digits=3)
            xvals = ω .- 5 * phAssymetry / 2
            for (k, axv) in axs
                axv.plot(xvals[abs.(xvals) .< 9], results[k][abs.(xvals) .< 9], label=L"η_d=%$(phAssymetry)")
                ins[k].plot(xvals[abs.(xvals) .< 1.5], results[k][abs.(xvals) .< 1.5], label=L"η_d=%$(phAssymetry)")
                ins[k].set_yticks([])
            end
        end
        for k in ["node-d", "node-f"]
            axs[k].legend(loc="center left")
            axs[k].text(0.05, 0.95,
                             L"""
                             $J_d=%$(couplings[\"Jd\"])$
                             $J_f=%$(couplings[\"Jf\"])$
                             $J_⟂=%$(Jp)$
                             $W_f=%$(Wf)$
                             $W_d=%$(couplings["Wd"])$
                             """,
                             horizontalalignment="left", verticalalignment="top", transform=axs[k].transAxes)
        end
        counter += 1
    end
    fig1.savefig("specFunc_node_d_$(size_BZ)_$(maxSize).pdf", bbox_inches="tight")
    fig2.savefig("specFunc_node_f_$(size_BZ)_$(maxSize).pdf", bbox_inches="tight")
    plt.close()
end

#=RGFlow(=#
#=     Dict("Jd" => 0.1,=#
#=          "Jf" => 0.1,=#
#=          "Wd_by_Wf" => 0.,=#
#=          "Uf" => 10.,=#
#=          "Ud" => 10.,=#
#=          "μf" => 5.,=#
#=         ),=#
#=        0:-0.02:-0.25,=#
#=        0:0.05:0.4,=#
#=        [0.85 * 5],=#
#=        33;=#
#=        loadData=false,=#
#=    )=#


@time KspaceSF()
