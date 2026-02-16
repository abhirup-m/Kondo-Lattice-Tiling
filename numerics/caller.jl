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
    couplings["J⟂"] = J
    couplings["omega_by_t"] = OMEGA_BY_t
    return couplings
end

@everywhere const COUPLINGS = Dict("omega_by_t" => -2.,
                                   "Ud" => 4.0,
                                   "Uf" => 8.0,
                                   "Jf" => 0.2,
                                   "Jd" => 0.1,
                                   "Wd" => -0.0,
                                   "ηf" => 0.,
                                   "U_by_W" => 200.,
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

function MomentumSpecFunc(
        size_BZ::Int64,
        maxSize::Int64,
        Jp_vals::Vector,
        Wf_vals::Vector,
        ηd::Number;
        loadData=false
    )
    names = Dict(
                 "node_d" => L"A_d",
                 "node_f" => L"A_f",
                 "node_+" => L"A_+",
                 "node_-" => L"A_-",
                )

    dos, dispersion = getDensityOfStates(tightBindDisp, size_BZ)
    momentumPoints = Dict(k => getIsoEngCont(dispersion, 0.) for k in ["f", "d"])
    freqVals = collect(-15:0.001:15)

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

                       )

    counter = 1
    couplings = copy(COUPLINGS)
    couplings["ηd"] = ηd
    plots = Dict(name => plt.subplots(ncols=length(Jp_vals), figsize=(8 * length(Jp_vals), 6)) for name in keys(names))
    couplingSets = Iterators.product(Wf_vals, Jp_vals)

    resultsPooled = @distributed merge for (Wf, Jp) in collect(couplingSets)
        merge!(couplings, Dict("Wf" => Wf,
                               "J⟂" => Jp,
                               "Uf" => impCorr(Wf, couplings["U_by_W"]),
                              )
              )

        results, fpCouplings = AuxiliaryCorrelations(size_BZ,
                                      couplings,
                                      Dict(),
                                      momentumPoints,
                                      Dict(),
                                      specFuncReqs,
                                      maxSize;
                                      loadData=loadData,
                                     )
        for k in filter(k -> contains(k, "Siam"), keys(specFuncReqs))
            results["in_$(k)"] = results[k]
            results["out_$(k)"] = results[k]
            delete!(results, k)
        end

        freqVals = collect(-15:0.001:15)
        specFuncResults = Dict()

        xvalsShift = 0.
        effHybridisation = √(fpCouplings["⟂"] * (couplings["Ud"] + couplings["Uf"]))
        gap = √(effHybridisation^2 + 0.25 * couplings["ηd"]^2)
        println("hybridisation gap = ", gap)
        for k in keys(results)
            if k ∉ keys(specFuncReqs) && !contains(k, "in_") && !contains(k, "out_")
                continue
            end
            specFuncResults[k] = 0 .* freqVals
            for specCoeffs in results[k]
                if isempty(specCoeffs)
                    continue
                end
                bandEnergy = endswith(k, "+") ? (gap - couplings["ηd"]) : (-gap - couplings["ηd"])
                if abs(bandEnergy) > xvalsShift
                    xvalsShift = bandEnergy
                end
                shiftedFrequency = freqVals .- bandEnergy
                broadening = 0.1 .+ 4 .* abs.(shiftedFrequency / maximum(shiftedFrequency)).^1.5
                @assert occursin("Kondo", k) || startswith(k, "in_") || startswith(k, "out_") || occursin("Charge", k)
                if occursin("Kondo", k) || startswith(k, "in_")
                    resonancePoles = filter(p -> abs(p[2]) < couplings["Jd"], specCoeffs)
                    specFunc = SpecFunc(resonancePoles, shiftedFrequency, broadening; normalise=false)
                else
                    resonancePoles = filter(p -> abs(p[2]) > couplings["Ud"]/3, specCoeffs)
                    specFunc = SpecFunc(resonancePoles, shiftedFrequency, broadening; normalise=false)
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

        for (k, v) in specFuncResults
            specFuncResults[k] = Normalise(v, freqVals)
        end
        Dict((Wf, Jp) => (specFuncResults, freqVals, xvalsShift))
    end

    for Jp in Jp_vals
        for (name, ylabel) in names
            ax = length(Jp_vals) > 1 ? plots[name][2][counter] : plots[name][2]
            for Wf in Wf_vals
                results, xvals, xvalsShift = resultsPooled[(Wf, Jp)]
                shiftedXvals = findall(x -> x < 10 + xvalsShift && x > -10 + xvalsShift, xvals)
                ax.plot(xvals[shiftedXvals], results[name][shiftedXvals], label=L"W_f=%$(Wf)")
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
                    $η_d=%$(ηd)$
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


function RealSpecFunc(
        size_BZ::Int64,
        maxSize::Int64,
        Jp_vals::Vector,
        Wf_vals::Vector,
        ηd::Number;
        loadData=false
    )
    freqVals = collect(-15:0.001:15)
    names = Dict(
                 "Ad" => L"A_d",
                 "Af" => L"A_f",
                 "A+" => L"A_+",
                 "A-" => L"A_-",
                )

    dos, dispersion = getDensityOfStates(tightBindDisp, size_BZ)
    momentumPoints = Dict(k => getIsoEngCont(dispersion, 0.) for k in ["f", "d"])
    freqVals = collect(-15:0.001:15)

    antinode = map2DTo1D(π/1, 0., size_BZ)
    node = map2DTo1D(π/2, π/2, size_BZ)
    impSites = Dict("f" => [1, 2], "d" => [3, 4])

    counter = 1
    couplings = copy(COUPLINGS)
    couplings["ηd"] = ηd
    plots = Dict(name => plt.subplots(ncols=length(Jp_vals), figsize=(8 * length(Jp_vals), 6)) for name in keys(names))
    couplingSets = Iterators.product(Wf_vals, Jp_vals)

    resultsPooled = @distributed merge for (Wf, Jp) in collect(couplingSets)
        merge!(couplings, Dict("Wf" => Wf,
                               "J⟂" => Jp,
                               "Uf" => impCorr(Wf, couplings["U_by_W"]),
                               #="Ud" => impCorr(couplings["Wd"], couplings["U_by_W"]),=#
                              )
              )

        effHybridisation = 2√(Jp * (COUPLINGS["Ud"] + COUPLINGS["Uf"]))
        hybridMatrix = [[0, -effHybridisation] [-effHybridisation, -ηd]]
        # matrix is in the basis of [f, d]; first eigenvector is (1, 1), second is (1, -1)
        gap, V = eigen(hybridMatrix)
        hybCoeffs = Dict(k => Dict(["f", "d"] .=> all(≤(0), V[:, i]) ? -V[:, i] : V[:, i]) for (i, k) in enumerate(["+", "-"]))
        display(hybCoeffs)

        specFuncReqs = Dict()
        siam(coeff, orb) = [("+", [i], coeff) for i in impSites[orb]]
        kondo(coeff, orb, bath) = [("+-+", [imp..., bath+2-i], coeff) for (i,imp) in enumerate([impSites[orb], reverse(impSites[orb])])]

        for band in ["+", "-"]
            for k in ["d", "f"]
                specFuncReqs["Siam_$(band)_$(k)_loc"] = ("i", 
                           Dict(
                                "create" => siam(1., k),
                                "destroy" => Dagger(vcat(siam(hybCoeffs[band]["f"], "f"), siam(hybCoeffs[band]["d"], "d"))), 
                                #=[("-", [1,],hybCoeffs["+"]["f"]), =#
                                #=              ("-", [2,],hybCoeffs["+"]["f"]),=#
                                #=              ("-", [3,],hybCoeffs["+"]["d"]),=#
                                #=              ("-", [4,],hybCoeffs["+"]["d"]),=#
                                #=             ],=#
                               ),
                          )
                specFuncReqs["Siam_$(k)_$(band)_loc"] = ("i", 
                           Dict(
                                "create" => vcat(siam(hybCoeffs[band]["f"], "f"), siam(hybCoeffs[band]["d"], "d")), 
                                "create" => Dagger(siam(1., k)),
                                #=[("-", [1,],hybCoeffs["+"]["f"]), =#
                                #=              ("-", [2,],hybCoeffs["+"]["f"]),=#
                                #=              ("-", [3,],hybCoeffs["+"]["d"]),=#
                                #=              ("-", [4,],hybCoeffs["+"]["d"]),=#
                                #=             ],=#
                               ),
                          )
                specFuncReqs["Kondo_$(band)0_0$(k)_loc"] = ("i$(k)",
                     lastInd -> Dict(
                                     "create" => kondo(1., k, 5),
                                     #[("+-+", [impSites[k]...,6], 1.), ("+-+", [reverse(impSites[k])...,5], 1.),],
                                     "destroy" => Dagger(vcat(kondo(hybCoeffs[band][k], k, 5), 
                                                              kondo(hybCoeffs[band][k=="d" ? "f" : "d"], k, lastInd)
                                                             ))
                                      #=[=#
                                      #=              ("+--",[impSites[k]..., 5], hybCoeffs["+"][k]),=#
                                      #=              ("+--",[reverse(impSites[k])..., 6], hybCoeffs["+"][k]),=#
                                      #=              ("+--",[impSites[k]..., lastInd], hybCoeffs["+"][k=="d" ? "f" : "d"]),=#
                                      #=              ("+--",[reverse(impSites[k])..., lastInd + 1], hybCoeffs["+"][k=="d" ? "f" : "d"]),=#
                                      #=             ],=#
                                 ),
                    )
                specFuncReqs["Kondo_0$(k)_$(band)0_loc"] = ("i$(k)",
                     lastInd -> Dict(
                                     "create" => vcat(kondo(hybCoeffs[band][k], k, 5), 
                                                      kondo(hybCoeffs[band][k=="d" ? "f" : "d"], k, lastInd)
                                                     ),
                                     "destroy" => Dagger(kondo(1., k, 5)),
                                      #=[=#
                                      #=              ("+--",[impSites[k]..., 5], hybCoeffs["+"][k]),=#
                                      #=              ("+--",[reverse(impSites[k])..., 6], hybCoeffs["+"][k]),=#
                                      #=              ("+--",[impSites[k]..., lastInd], hybCoeffs["+"][k=="d" ? "f" : "d"]),=#
                                      #=              ("+--",[reverse(impSites[k])..., lastInd + 1], hybCoeffs["+"][k=="d" ? "f" : "d"]),=#
                                      #=             ],=#
                                 ),
                    )
            end
            specFuncReqs["Siam_$(band)_$(band)_loc"] = ("i", vcat(siam(1., "d"), siam(1., "f")))
            specFuncReqs["Kondo_$(band)_$(band)_d_loc"] = ("id", lastInd -> vcat(kondo(1., "d", 5), kondo(1., "f", lastInd)))
            specFuncReqs["Kondo_$(band)_$(band)_f_loc"] = ("if", lastInd -> vcat(kondo(1., "f", 5), kondo(1., "d", lastInd)))
        end

        results, fpCouplings = AuxiliaryCorrelations(size_BZ,
                                      couplings,
                                      Dict(),
                                      momentumPoints,
                                      Dict(),
                                      specFuncReqs,
                                      maxSize;
                                      loadData=loadData,
                                     )
        avgKondo = Dict(k => minimum((1, sum(abs.(fpCouplings[k][momentumPoints[k], momentumPoints[k]])) / (length(momentumPoints[k])^1.0 * couplings["J$(k)"]))) for k in ["f", "d"])
        #=avgKondo = Dict(k => minimum((Inf, sum(abs.(fpCouplings[k][momentumPoints[k], momentumPoints[k]])) / (length(momentumPoints[k])^1.0 * couplings["J$(k)"]))) for k in ["f", "d"])=#
        println(Wf, avgKondo)

        specFuncResults = Dict()

        for k in filter(k -> contains(k, "Siam"), keys(specFuncReqs))
            for dict in [results, specFuncReqs]
                dict["in_$(k)"] = dict[k]
                dict["out_$(k)"] = dict[k]
                delete!(dict, k)
            end
        end

        xvalsShift = couplings["ηd"]
        for k in keys(specFuncReqs)
            if k ∉ keys(specFuncReqs) && !contains(k, "in_") && !contains(k, "out_")
                continue
            end
            specFuncResults[k] = 0 .* freqVals
            bandEnergy = 0
            @assert contains(k, "_+_") || contains(k, "_+0_") || contains(k, "_-_") || contains(k, "_-0_") k
            if contains(k, "_+_") || contains(k, "_+0_")
                bandEnergy = gap[2]
            else
                bandEnergy = gap[1]
            end

            broadening = 0.1 * ones(length(freqVals))
            specCoeffs = vcat(results[k]...)
            if !isempty(specCoeffs)

                partition = (Inf, 0)
                if contains(k, "_d_") || contains(k, "_0d_")
                    partition = (couplings["Ud"], couplings["Ud"]/3)
                elseif contains(k, "_f_") || contains(k, "_0f_")
                    partition = (couplings["Uf"], couplings["Uf"]/3)
                end
                if occursin("Kondo", k) || startswith(k, "in_")
                    filter!(p -> abs(p[2]) < 1., specCoeffs)
                    map!(p -> (p[1], p[2] + bandEnergy), specCoeffs)
                    if maximum(partition) == 0
                        broadening .+= 2 * (abs.(freqVals .- bandEnergy)./maximum(freqVals)).^0.5
                    end
                else
                    map!(p -> (p[1], p[2] + bandEnergy), specCoeffs)
                    filter!(p -> partition[1] > abs(p[2]) > partition[2]/3, specCoeffs)
                    broadening[abs.(freqVals) .> minimum(partition) + bandEnergy] .+= 1 * ((abs.(freqVals) .- minimum(partition) .- bandEnergy)[abs.(freqVals) .> minimum(partition) + bandEnergy]./maximum(freqVals)).^0.5
                end
                specFuncResults[k] = SpecFunc(specCoeffs, freqVals, broadening; normalise=false)
                specFuncResults[k] = Normalise(specFuncResults[k], freqVals; tolerance=1e-4)
            else
                println("$(k) is empty!")
            end
        end

        specFuncResults["Ad"] = abs.(avgKondo["d"] * (
                                                  0
                                                 .+ hybCoeffs["+"]["d"] * specFuncResults["in_Siam_+_d_loc"] 
                                                 .+ hybCoeffs["-"]["d"] * specFuncResults["in_Siam_-_d_loc"]
                                                 .+ hybCoeffs["+"]["d"] * specFuncResults["Kondo_+0_0d_loc"] 
                                                 .+ hybCoeffs["-"]["d"] * specFuncResults["Kondo_-0_0d_loc"] 
                                                 .+ hybCoeffs["+"]["d"] * specFuncResults["in_Siam_d_+_loc"] 
                                                 .+ hybCoeffs["-"]["d"] * specFuncResults["in_Siam_d_-_loc"]
                                                 .+ hybCoeffs["+"]["d"] * specFuncResults["Kondo_0d_+0_loc"] 
                                                 .+ hybCoeffs["-"]["d"] * specFuncResults["Kondo_0d_-0_loc"] 
                                                ) 
                                 .+ hybCoeffs["+"]["d"] * specFuncResults["out_Siam_+_d_loc"] 
                                 .+ hybCoeffs["-"]["d"] * specFuncResults["out_Siam_-_d_loc"]
                                 .+ hybCoeffs["+"]["d"] * specFuncResults["out_Siam_d_+_loc"] 
                                 .+ hybCoeffs["-"]["d"] * specFuncResults["out_Siam_d_-_loc"]
                                )
        specFuncResults["Af"] = abs.(avgKondo["f"] * (
                                                  0
                                                 .+ hybCoeffs["+"]["f"] * specFuncResults["in_Siam_+_f_loc"] 
                                                 .+ hybCoeffs["-"]["f"] * specFuncResults["in_Siam_-_f_loc"]
                                                 .+ hybCoeffs["+"]["f"] * specFuncResults["Kondo_+0_0f_loc"] 
                                                 .+ hybCoeffs["-"]["f"] * specFuncResults["Kondo_-0_0f_loc"] 
                                                 .+ hybCoeffs["+"]["f"] * specFuncResults["in_Siam_f_+_loc"] 
                                                 .+ hybCoeffs["-"]["f"] * specFuncResults["in_Siam_f_-_loc"]
                                                 .+ hybCoeffs["+"]["f"] * specFuncResults["Kondo_0f_+0_loc"] 
                                                 .+ hybCoeffs["-"]["f"] * specFuncResults["Kondo_0f_-0_loc"] 
                                                 ) 
                                 .+ hybCoeffs["+"]["f"] * specFuncResults["out_Siam_+_f_loc"] 
                                 .+ hybCoeffs["-"]["f"] * specFuncResults["out_Siam_-_f_loc"]
                                 .+ hybCoeffs["+"]["f"] * specFuncResults["out_Siam_f_+_loc"] 
                                 .+ hybCoeffs["-"]["f"] * specFuncResults["out_Siam_f_-_loc"]
                                )
        specFuncResults["Afd"] = abs.(0.5 * (avgKondo["d"] + avgKondo["f"]) * (
                                                  0
                                                 .+ hybCoeffs["+"]["f"] * specFuncResults["in_Siam_+_d_loc"] 
                                                 .+ hybCoeffs["-"]["f"] * specFuncResults["in_Siam_-_d_loc"]
                                                 .+ hybCoeffs["+"]["d"] * specFuncResults["in_Siam_f_+_loc"] 
                                                 .+ hybCoeffs["-"]["d"] * specFuncResults["in_Siam_f_-_loc"]
                                                 .+ hybCoeffs["+"]["f"] * specFuncResults["Kondo_+0_0d_loc"] 
                                                 .+ hybCoeffs["-"]["f"] * specFuncResults["Kondo_-0_0d_loc"] 
                                                 .+ hybCoeffs["+"]["d"] * specFuncResults["Kondo_0f_+0_loc"] 
                                                 .+ hybCoeffs["-"]["d"] * specFuncResults["Kondo_0f_-0_loc"] 
                                                ) 
                                         .+ hybCoeffs["+"]["f"] * specFuncResults["out_Siam_+_d_loc"] 
                                         .+ hybCoeffs["-"]["f"] * specFuncResults["out_Siam_-_d_loc"]
                                         .+ hybCoeffs["+"]["d"] * specFuncResults["out_Siam_f_+_loc"] 
                                         .+ hybCoeffs["-"]["d"] * specFuncResults["out_Siam_f_-_loc"]
                                )
        specFuncResults["Adf"] = specFuncResults["Afd"]
        specFuncResults["A+"] = (
                                 avgKondo["d"] * avgKondo["f"] * (
                                                                          specFuncResults["in_Siam_+_+_loc"]
                                                                         .+ specFuncResults["Kondo_+_+_d_loc"]
                                                                         .+ specFuncResults["Kondo_+_+_f_loc"]
                                                                        )
                                 .+ specFuncResults["out_Siam_+_+_loc"]
                                )
        specFuncResults["A-"] = (
                                 avgKondo["d"] * avgKondo["f"] * (
                                                                          specFuncResults["in_Siam_-_-_loc"]
                                                                         .+ specFuncResults["Kondo_-_-_d_loc"]
                                                                         .+ specFuncResults["Kondo_-_-_f_loc"]
                                                                        )
                                 .+ specFuncResults["out_Siam_-_-_loc"]
                                )

        for (k, v) in specFuncResults
            specFuncResults[k] = Normalise(v, freqVals)
            #=if Jp == 0=#
            #=    specFuncResults[k] = 0.5 .* (specFuncResults[k] .+ reverse(specFuncResults[k]))=#
            #=end=#
        end
        Dict((Wf, Jp) => (specFuncResults, freqVals, xvalsShift))
    end

    for Jp in Jp_vals
        for (name, ylabel) in names
            ax = length(Jp_vals) > 1 ? plots[name][2][counter] : plots[name][2]
            ins = ax.inset_axes([0.6,0.6,0.4,0.4])
            for Wf in Wf_vals
                results, xvals, xvalsShift = resultsPooled[(Wf, Jp)]
                shiftedXvals = findall(x -> x < 10 + xvalsShift && x > -10 + xvalsShift, xvals)
                shiftedXvalsSmall = findall(x -> 0 < x < 1.0 + xvalsShift, xvals)
                ax.plot(xvals[shiftedXvals], results[name][shiftedXvals], label=L"W_f=%$(Wf)")
                ins.plot(xvals[shiftedXvalsSmall], results[name][shiftedXvalsSmall], label=L"W_f=%$(Wf)")
                ins.set_yscale("log")
                #=ins.set_xscale("log")=#
            end
            ax.set_xlabel(L"\omega", fontsize=25)
            ax.set_ylabel(ylabel, fontsize=25)
            #=ax.legend(loc="upper right", fontsize=25, handlelength=1.0)=#
            ax.tick_params(labelsize=25)
            ax.text(0.05,
                    0.95,
                    L"""
                    $J_d=%$(couplings[\"Jd\"])$
                    $J_f=%$(couplings[\"Jf\"])$
                    $J_⟂=%$(Jp)$
                    $η_d=%$(ηd)$
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

#=@time RealCorr(33, 0896, 0.05:0.025:0.40, -0.0:-0.02:-0.160, 0.; loadData=false)=#
#=@time MomentumSpecFunc(33, 1500, [0.0, 0.2], [-0., -0.2], 0.0; loadData=false)=#
@time RealSpecFunc(33, 1610, [0.2, 0.4], [-0., -0.06, -0.1409, -0.147, -0.152, -0.1535], 0.0; loadData=true)
#=@time RealSpecFunc(33, 1610, [0.2, 0.4], [-0., -0.06, -0.12, -0.1409, -0.147, -0.152, -0.1535, -0.16], 0.0; loadData=true)=#
