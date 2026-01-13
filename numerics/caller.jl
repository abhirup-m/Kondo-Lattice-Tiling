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

@everywhere function insertCouplings(couplings, Wf, J)
    couplings["Wf"] = Wf
    couplings["Wd"] = Wf * couplings["Wd_by_Wf"]
    couplings["J⟂"] = J
    couplings["omega_by_t"] = OMEGA_BY_t
    return couplings
end

@everywhere const COUPLINGS = Dict("omega_by_t" => -2.,
                                   "Uf" => 9.,
                                   "Ud" => 9.,
                                   "Jf" => 0.1,
                                   "Jd" => 0.1,
                                   "Wd" => -0.0,
                                   "ηf" => 0.,
                                   "Wd_by_Wf" => 1.0,
                    )

function RGFlow(
        couplings::Dict,
        Wf,
        Jp,
        μ,
        size_BZ::Int64;
        loadData::Bool=false,
        disabled::Bool=true,
    )
    if disabled
        return
    end
    sparseWf = length(Wf) ≥ 20 ? Wf[1:div(length(Wf), 12):end] : Wf
    sparseJp = length(Jp) ≥ 20 ? Jp[1:div(length(Jp), 12):end] : Jp
    parameters = Iterators.product(Wf, Jp)
    dos, dispersion = getDensityOfStates(tightBindDisp, size_BZ)
    FSpoints = getIsoEngCont(dispersion, 0.0)
    fig, axes = subplots(ncols=3, nrows=length(μ), figsize=(18, 3.5 * length(μ)))
    fig.tight_layout()
    boolTitles = Dict("f" => L"Pole fraction: $J_f$", "d" => L"Pole fraction: $J_d$", "⟂" => L"Rel/Irrelevance: $J_\perp$")
    for (j, μ_i) in enumerate(μ)
        results = @showprogress desc="rg flow" pmap(p -> momentumSpaceRG(size_BZ, insertWfJ(couplings, p[1], p[2], μ_i); loadData=loadData), parameters)
        for (i, key) in enumerate(["f", "d", "⟂"])
            ax = length(μ) > 1 ? axes[j, i] : axes[i]
            fgLabel = L"\mathrm{FS~frac.~(FG)}" 
            if key == "f" || key == "d"
                keyData = [sum(diag(abs.(r[key])[FSpoints, FSpoints])) for r in results]
                boolData = [count(>(1e-3), abs.(diag(r[key][FSpoints, FSpoints]))) / length(FSpoints) for r in results]
            else
                keyData = [r[key] for r in results]
                boolData = [ifelse(r[key]/J > 0.8, 1, ifelse(r[key]/J > 0., 0.5, 0)) for (r, (Wf, J)) in zip(results, Iterators.product(Wf, Jp))]
                fgLabel = L"$g^* > 0?~$(FG)" 
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
            #=if i == 1=#
            #=    phAssymetry = round(μ_i - couplings["Ud"]/2, digits=3)=#
            #=    ax.text(-1.2, 0.5, L"$\eta_d=%$(phAssymetry)$", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes)=#
            #=end=#
            ax.set_aspect((maximum(Jp) - minimum(Jp)) / (maximum(-Wf) - minimum(-Wf)))
            fig.colorbar(hm, shrink=0.5, pad=-1.1, label=L"$g^*~$(BG)", format=matplotlib.ticker.FormatStrFormatter("%.2f"))
            fig.colorbar(sc, location="left", shrink=0.5, pad=0.09, label=fgLabel,)
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


function RealCorr(
        size_BZ,
        maxSize,
        J,
        Wf,
        ηd;
        loadData=false,
    )
    couplings = copy(COUPLINGS)
    couplings["ηd"] = ηd
    parameterSpace = Iterators.product(Wf, J)

    node = map2DTo1D(π/2, π/2, size_BZ)
    antinode = map2DTo1D(3π/4, π/4, size_BZ)
    microCorrelation = Dict(
                            "SF-fkk" => ("f", (i, j) -> [("+-+-", [1, 2, i+1, j], 0.5), 
                                                         ("+-+-", [2, 1, i, j+1], 0.5),
                                                         ("n+-", [1, i, j], 0.25), 
                                                         ("n+-", [1, i+1, j+1], -0.25),
                                                         ("n+-", [2, i, j], -0.25),
                                                         ("n+-", [2, i+1, j+1], 0.25)]),
                            #="SF-dkk" => ("d", (i, j) -> [("+-+-", [3, 4, i+1, j], 0.5), =#
                            #=                             ("+-+-", [4, 3, i, j+1], 0.5), =#
                            #=                             ("n+-", [3, i, j], 0.25), =#
                            #=                             ("n+-", [3, i+1, j+1], -0.25), =#
                            #=                             ("n+-", [4, i, j], -0.25), =#
                            #=                             ("n+-", [4, i+1, j+1], 0.25)]),=#
                            "SF-fdpm" => ("i", [("+-+-", [1, 2, 4, 3], 1 / 2), 
                                                ("+-+-", [2, 1, 3, 4], 1 / 2)]),
                            "SF-fdzz" => ("i", [("nn", [1, 3], 1 / 4), 
                                                ("nn", [1, 4], -1 / 4), 
                                                ("nn", [2, 3], -1 / 4), 
                                                ("nn", [2, 4], 1 / 4)]),
                            #="CF-dkk" => ("d", (i, j) -> [("++--", [3, 4, i+1, j], 0.5), =#
                            #=                             ("++--", [4, 3, i, j+1], 0.5)]),=#
                            #="CF-fkk" => ("f", (i, j) -> [("++--", [1, 2, i+1, j], 0.5), =#
                            #=                             ("--++", [2, 1, i, j+1], 0.5)]),=#
                            #="ndu" => ("i", [("n", [3], 1.)]),=#
                            #="ndd" => ("i", [("n", [4], 1.)]),=#
                            #="nfu" => ("i", [("n", [1], 1.)]),=#
                            #="nfd" => ("i", [("n", [2], 1.)]),=#
                       )
            
    dos, dispersion = getDensityOfStates(tightBindDisp, size_BZ)
    momentumPoints = Dict(k => getIsoEngCont(dispersion, 0.) for k in ["f", "d"])
    entanglement =  Dict(
                         #="SEE-d" => [3, 4],=#
                         #="SEE-f" => [1, 2],=#
                         "I2-f-d" => ([1, 2], [3, 4]),
                         "I2-f-max" => (f, d) -> ([1, 2], [f, f+1]),
                         #="I2-d-max" => (f, d) -> ([3, 4], [d, d+1]),=#
                        )
    pooledResults = @showprogress pmap(WfJ -> AuxiliaryCorrelations(size_BZ,
                                                      insertCouplings(couplings, WfJ...),
                                                      microCorrelation,
                                                      momentumPoints,
                                                      entanglement,
                                                      Dict(),
                                                      maxSize;
                                                      loadData=loadData,
                                                    ),
                                         parameterSpace
                                        )
    combinedResults = first.(pooledResults)

    momentumPairs = Dict(k => vec(collect(Iterators.product(momentumPoints[k], momentumPoints[k]))) for k in ["f", "d"])

    correlations = Dict(
                        #="SF-fmax" => cR -> maximum(abs.(cR["SF-fkk"])),=#
                        #="SF-dmax" => cR -> maximum(abs.(cR["SF-dkk"])),=#
                        "SF-f0" => cR -> sum([abs(cR["SF-fkk"][index] * NNFunc(k1, k2, size_BZ)) for (index, (k1, k2)) in enumerate(momentumPairs["f"]) if k1 == k2]) / length(momentumPoints["f"])^0.5,
                        #="SF-d0" => cR -> sum([abs(cR["SF-dkk"][index] * NNFunc(k1, k2, size_BZ)) for (index, (k1, k2)) in enumerate(momentumPairs["d"])]) / length(momentumPoints["d"]),=#
                        "SF-fd" => cR -> -(cR["SF-fdpm"] + cR["SF-fdzz"]),
                        #="CF-d0" => cR -> sum([abs(cR["CF-dkk"][index]) * NNFunc(k1, k2, size_BZ) for (index, (k1, k2)) in enumerate(momentumPairs["d"])]) / length(momentumPoints["d"]),=#
                        #="CF-f0" => cR -> -sum([abs(cR["CF-fkk"][index]) * NNFunc(k1, k2, size_BZ) for (index, (k1, k2)) in enumerate(momentumPairs["f"])]) / length(momentumPoints["f"]),=#
                        #="SF-fdzz" => cR -> abs(cR["SF-fdzz"]),=#
                        #="SF-fPF" => cR -> count(>(1e-4), [sum([abs(cR["SF-fkk"][index]) for (index, (k1, k2)) in enumerate(momentumPairs["f"]) if k1 == q || k2 == q]) for q in momentumPoints["f"]]) / length(momentumPoints["f"]),=#
                        #="Sdz" => cR -> 0.5 * abs(cR["ndu"] - cR["ndd"]),=#
                        #="Sfz" => cR -> 0.5 * abs(cR["nfu"] - cR["nfd"]),=#
                        #="SEE-d" => cR -> cR["SEE-d"],=#
                        #="SEE-f" => cR -> cR["SEE-f"],=#
                        "I2-f-d" => cR -> cR["I2-f-d"],
                        "I2-f-max" => cR -> cR["I2-f-max"],
                        #="I2-d-max" => cR -> cR["I2-d-max"],=#
                       )
    for k in keys(entanglement)
        correlations[k] = cR -> cR[k]
    end
    figPaths = []
    plottableResults = Dict{String, Vector{Float64}}()
    for (name, func) in correlations
        plottableResults[name] = [func(cR) for cR in vec(collect(combinedResults))]
    end
    return RowPlots(plottableResults,
                    collect(parameterSpace),
                    [
                     ("SF-f0", "SF-fd"), 
                     ("I2-f-max", "I2-f-d")
                     #=("SF-fd", "SF-fPF"),=#
                     #=("CF-d0", "CF-f0"),=#
                    ],
                    [
                     (L"$-\langle {S_f\cdot S_{f0}}\rangle$",L"$-\langle {S_d\cdot S_{f}}\rangle$"),
                     (L"I_2(f:f_0)", L"I_2(f:d)"),
                     #=(L"CF", L"n_d"), =#
                    ],
                    [
                     "",
                     "",
                     #=L"$Sd.Sf$, PF",=#
                     #="charge",=#
                    ],
                    (L"J_\perp", L"W_f"),
                    "locCorr_$(size_BZ)_$(maxSize)_$(ηd).pdf",
                    "",
                    []
                   )
end

@everywhere function MomentumSpecFunc(
        couplings,
        size_BZ,
        maxSize;
        loadData::Bool=false,
    )
    dos, dispersion = getDensityOfStates(tightBindDisp, size_BZ)
    momentumPoints = Dict(k => getIsoEngCont(dispersion, 0.) for k in ["f", "d"])

    antinode = map2DTo1D(π/1, 0., size_BZ)
    node = map2DTo1D(π/2, π/2, size_BZ)
    impSites = Dict("f" => [1, 2], "d" => [3, 4])
    siam(mom, coeff) = Dict(impOrb => vcat([[("-+", [impSites[impOrb][1], mom[bathOrb]], coeff[bi]),
                                             ("-+", [impSites[impOrb][2], mom[bathOrb] + 1], coeff[bi]),
                                             ("-+", [impSites[impOrb][2], mom[bathOrb]], coeff[bi]),
                                             ("-+", [impSites[impOrb][1], mom[bathOrb] + 1], coeff[bi])
                                            ] 
                                            for (bi, bathOrb) in enumerate(["d", "f"]) if coeff[bi] ≠ 0]...)
                            for impOrb in ["f", "d"])

    kondo(mom, coeff) = Dict(impOrb => vcat([[("+-+", [impSites[impOrb][1], impSites[impOrb][2], mom[bathOrb] + 1], coeff[bi]),
                                              ("+-+", [impSites[impOrb][2], impSites[impOrb][1], mom[bathOrb]], coeff[bi])]
                                            for (bi, bathOrb) in enumerate(["d", "f"]) if coeff[bi] ≠ 0]...)
                             for impOrb in ["f", "d"])

    charge(mom, coeff) = Dict(impOrb => vcat([[("--+", [impSites[impOrb][1], impSites[impOrb][2], mom[bathOrb] + 1], coeff[bi]),
                                              ("--+", [impSites[impOrb][1], impSites[impOrb][2], mom[bathOrb]], coeff[bi])]
                                            for (bi, bathOrb) in enumerate(["d", "f"]) if coeff[bi] ≠ 0]...)
                             for impOrb in ["f", "d"])



    specFuncReqs = Dict(
                        "d_Siam_+" => ("+-", ==(node), (mom, J) -> siam(mom, (1, 1))["d"]),
                        "d_Siam_-" => ("+-", ==(node), (mom, J) -> siam(mom, (1, -1))["d"]),
                        "d_Kondo_+" => ("+-", ==(node), (mom, J) -> kondo(mom, (J["d"], J["f"]))["d"]),
                        "d_Kondo_-" => ("+-", ==(node), (mom, J) -> kondo(mom, (J["d"], -J["f"]))["d"]),
                        "d_Charge_+" => ("+-", ==(node), (mom, J) -> charge(mom, (1, 1))["d"]),
                        "d_Charge_-" => ("+-", ==(node), (mom, J) -> charge(mom, (1, -1))["d"]),
                        "f_Siam_+" => ("+-", ==(node), (mom, J) -> siam(mom, (1, 1))["f"]),
                        "f_Siam_-" => ("+-", ==(node), (mom, J) -> siam(mom, (1, -1))["f"]),
                        "f_Kondo_+" => ("+-", ==(node), (mom, J) -> kondo(mom, (J["d"], J["f"]))["f"]),
                        "f_Kondo_-" => ("+-", ==(node), (mom, J) -> kondo(mom, (J["d"], -J["f"]))["f"]),
                        "f_Charge_+" => ("+-", ==(node), (mom, J) -> charge(mom, (1, 1))["f"]),
                        "f_Charge_-" => ("+-", ==(node), (mom, J) -> charge(mom, (1, -1))["f"]),
                        "Kondo_+" => ("+-", ==(node), (mom, J) -> vcat(kondo(mom, (J["d"], 0))["d"], kondo(mom, (0, J["f"]))["f"])),
                        "Kondo_-" => ("+-", ==(node), (mom, J) -> vcat(kondo(mom, (J["d"], 0))["d"], kondo(mom, (0, -J["f"]))["f"])),
                        "Siam_+" => ("+-", ==(node), (mom, J) -> vcat(siam(mom, (1, 0))["d"], siam(mom, (0, 1))["f"])),
                        "Siam_-" => ("+-", ==(node), (mom, J) -> vcat(siam(mom, (1, 0))["d"], siam(mom, (0, -1))["f"])),
                        "Charge_+" => ("+-", ==(node), (mom, J) -> vcat(charge(mom, (1, 0))["d"], charge(mom, (0, 1))["f"])),
                        "Charge_-" => ("+-", ==(node), (mom, J) -> vcat(charge(mom, (1, 0))["d"], charge(mom, (0, -1))["f"])),

                        #="Ad_Siam_loc" => ("id", [("+", [3,], 1.), ("+", [4,], 1.)]),=#
                        #="Af_Siam_loc" => ("if", [("+", [1,], 1.), ("+", [2,], 1.)]),=#
                        #="Ad_Kondo_loc" => ("d", i->[("+-+", [3,4,i+1], 1.), ("+-+", [4,3,i], 1.)]),=#
                        #="Af_Kondo_loc" => ("f", i->[("+-+", [1,2,i+1], 1.), ("+-+", [2,1,i], 1.)]),=#
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
    additionalKeys = ["Ad_Siam_loc", "Af_Siam_loc", "d_Siam_+", "d_Siam_-", "f_Siam_+", "f_Siam_-", "Siam_+", "Siam_-"]
    for k in filter(k -> haskey(results, k), additionalKeys)
        results["in_$(k)"] = results[k]
        results["out_$(k)"] = results[k]
    end

    freqVals = collect(-15:0.001:15)
    specFuncResults = Dict()

    phAssymetry = couplings["μd"] - couplings["Ud"]/2
    xvalsShift = 0.
    effHybridisation = √(fpCouplings["⟂"] * 0.5 * (couplings["Ud"] + couplings["Uf"]))
    gap = √(effHybridisation^2 + 0.25 * phAssymetry^2)
    println("G = ", gap)
    for k in keys(results)
        if k ∉ keys(specFuncReqs) && k ∉ "in_" .* additionalKeys && k ∉ "out_" .* additionalKeys
            continue
        end
        specFuncResults[k] = 0 .* freqVals
        for specCoeffs in results[k]
            if isempty(specCoeffs)
                continue
            end
            if endswith(k, "+") || endswith(k, "-") || endswith(k, "loc")
                bandEnergy = endswith(k, "+") ? (gap - phAssymetry) : (-gap - phAssymetry)
                if abs(bandEnergy) > xvalsShift
                    xvalsShift = bandEnergy
                end
                shiftedFrequency = freqVals .- bandEnergy
                broadening = 0.1 .+ 4 .* abs.(shiftedFrequency / maximum(shiftedFrequency)).^1.5
                if occursin("Kondo", k) || startswith(k, "in_")
                    resonancePoles = filter(p -> abs(p[2]) < couplings["Jd"], specCoeffs)
                    specFunc = SpecFunc(resonancePoles, shiftedFrequency, broadening; normalise=false)
                elseif startswith(k, "out_") || occursin("Charge", k)
                    resonancePoles = filter(p -> abs(p[2]) > couplings["Ud"]/3, specCoeffs)
                    specFunc = SpecFunc(resonancePoles, shiftedFrequency, broadening; normalise=false)
                else
                    specFunc = SpecFunc(specCoeffs, shiftedFrequency, broadening; normalise=false)
                end
            else
                broadening = 0.1
                specFunc = SpecFunc(specCoeffs, freqVals, broadening; normalise=false)
            end
            specFuncResults[k] .+= specFunc
        end
        specFuncResults[k] = Normalise(specFuncResults[k], freqVals; tolerance=0)
    end

    avgKondo_d = sum(abs.(fpCouplings["d"][momentumPoints["d"], momentumPoints["d"]])) / (length(momentumPoints["d"])^2 * couplings["Jd"])
    specFuncResults["node_d"] = (
                                 specFuncResults["in_d_Siam_+"]
                                 + avgKondo_d * (specFuncResults["d_Kondo_+"] + specFuncResults["in_d_Siam_+"])
                                 + specFuncResults["d_Charge_+"]
                                 + specFuncResults["out_d_Siam_-"]
                                 + avgKondo_d * (specFuncResults["d_Kondo_-"] + specFuncResults["in_d_Siam_-"])
                                 + specFuncResults["d_Charge_-"]
                                )

    avgKondo_f = sum(abs.(fpCouplings["f"][momentumPoints["f"], momentumPoints["f"]])) / (length(momentumPoints["f"])^2 * couplings["Jd"])
    specFuncResults["node_f"] = (
                                 specFuncResults["in_f_Siam_+"]
                                 + avgKondo_f * (specFuncResults["f_Kondo_+"] + specFuncResults["in_f_Siam_+"])
                                 + specFuncResults["f_Charge_+"]
                                 + specFuncResults["out_f_Siam_-"]
                                 + avgKondo_f * (specFuncResults["f_Kondo_-"] + specFuncResults["in_f_Siam_-"])
                                 + specFuncResults["f_Charge_-"]
                                )

    specFuncResults["node_+"] = 0.5 * (avgKondo_d + avgKondo_f) * (specFuncResults["Kondo_+"] .+ specFuncResults["in_Siam_+"]) + 1 * specFuncResults["out_Siam_+"] + 1 * specFuncResults["Charge_+"]
    specFuncResults["node_-"] = 0.5 * (avgKondo_d + avgKondo_f) * (specFuncResults["Kondo_-"] .+ specFuncResults["in_Siam_-"]) + specFuncResults["out_Siam_-"] + specFuncResults["Charge_-"]

    #=specFuncResults["Ad"] = avgKondo_d * (specFuncResults["Ad_Kondo_loc"] .+ specFuncResults["in_Ad_Siam_loc"]) + 1 * specFuncResults["out_Ad_Siam_loc"]=#
    #=specFuncResults["Af"] = avgKondo_f * (specFuncResults["Af_Kondo_loc"] .+ specFuncResults["in_Af_Siam_loc"]) + 1 * specFuncResults["out_Af_Siam_loc"]=#

    for (k, v) in specFuncResults
        specFuncResults[k] = Normalise(v, freqVals)
    end
    return specFuncResults, freqVals, xvalsShift
end


#=function RealCorr(size_BZ, maxSize, loadData, J, Wf, ηf)=#
#=    paths = String[]=#
#=    couplings = copy(COUPLINGS)=#
#=    couplings["μf"] = ηf + couplings["Uf"]/2=#
#=    for μd in [1.0,] .* couplings["Ud"] / 2=#
#=        couplings["μd"] = μd=#
#=        path = RealCorr(couplings, =#
#=                        Wf,=#
#=                        J,=#
#=                        size_BZ,=#
#=                        maxSize;=#
#=                        loadData=loadData=#
#=                       )=#
#=        push!(paths, path)=#
#=    end=#
#=    merge_pdfs(paths, "locCorr_$(size_BZ)_$(maxSize).pdf", cleanup=true)=#
#=end=#


function MomentumSpecFunc(size_BZ, maxSize, loadData)
    Wf_vals = [0., -0.15]
    Jp_vals = [0., 0.2]
    names = Dict(
                 "node_d" => L"A_d",
                 "node_f" => L"A_f",
                 "node_+" => L"A_+",
                 "node_-" => L"A_-",
                )
    counter = 1
    couplings = Dict("omega_by_t" => -2.,
                     "Uf" => 9.,
                     "Jf" => 0.1,
                     "Wd_by_Wf" => 1.,
                    )
    couplings["Ud"] = couplings["Uf"] * couplings["Wd_by_Wf"]
    couplings["μf"] = couplings["Uf"]/2
    couplings["Jd"] = couplings["Jf"]
    couplings["Wd"] = 0.
    μdValues = [1.] .* couplings["Ud"] / 2
    plots = Dict(name => plt.subplots(ncols=length(Jp_vals), nrows=length(μdValues), figsize=(8 * length(Jp_vals), 6 * length(μdValues))) for name in keys(names))
    couplingSets = Iterators.product(Wf_vals, Jp_vals, μdValues)
    resultsPooled = Dict(couplingSets .=> pmap(trip -> MomentumSpecFunc(merge(Dict("Wf" => trip[1], "J⟂" => trip[2], "μd" => trip[3]), couplings), size_BZ, maxSize; loadData=loadData), couplingSets))
    for (Jp, μd) in Iterators.product(Jp_vals, μdValues)
        couplings["J⟂"] = Jp
        couplings["μd"] = μd
        phAssymetry = round(couplings["μd"] - couplings["Ud"]/2, digits=3)
        for (name, ylabel) in names
            ax = length(μdValues) * length(Jp_vals) > 1 ? plots[name][2][counter] : plots[name][2]
            #=inset = ax.inset_axes([0.55,0.5,0.45,0.4])=#
            for Wf in Wf_vals
                couplings["Wf"] = Wf
                results, xvals, xvalsShift = resultsPooled[(Wf, Jp, μd)]
                shiftedXvals = findall(x -> x < 10 + xvalsShift && x > -10 + xvalsShift, xvals)
                ax.plot(xvals[shiftedXvals], results[name][shiftedXvals], label=L"W_f=%$(Wf)")
                #=inset.plot(xvals[abs.(xvals) .< 0.5], results[name][abs.(xvals) .< 0.5], label=L"W_f=%$(Wf)")=#
                #=inset.set_yscale("log")=#
            end
            ax.set_xlabel(L"\omega", fontsize=25)
            ax.set_ylabel(ylabel, fontsize=25)
            ax.legend(loc="upper right", fontsize=25, handlelength=1.0)
            ax.tick_params(labelsize=25)
            ax.text(0.05,
                    0.95,
                    L"""
                    $J_d=%$(couplings[\"Jd\"])$
                    $J_f=%$(couplings[\"Jf\"])$
                    $J_⟂=%$(Jp)$
                    $η_d=%$(phAssymetry)$
                    $W_d=%$(couplings["Wd"])$
                    """,
                    horizontalalignment="left", 
                    verticalalignment="top",
                    transform=ax.transAxes,
                    size=25,
                   )
        end
        counter += 1
    end
    for (name, (fig, _)) in plots
        fig.savefig("specFunc_$(name)_$(size_BZ)_$(maxSize).pdf", bbox_inches="tight")
    end
    plt.close()
end

RGFlow(
     Dict("Jd" => 0.1,
          "Jf" => 0.1,
          "Wd_by_Wf" => 0.,
          "Uf" => 9.,
          "Ud" => 9.,
          "μf" => 4.5,
         ),
     -0.07:-0.005:-0.15,
     0:0.01:0.4,
     [4.5],
     33;
     loadData=true,
     disabled=true,
    )

@time RealCorr(33, 0896, 0.05:0.025:0.40, -0.0:-0.02:-0.160, 0.; loadData=false)
#=@time MomentumSpecFunc(33, 531, false)=#
