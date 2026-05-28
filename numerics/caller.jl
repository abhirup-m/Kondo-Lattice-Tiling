using Distributed, Serialization
@everywhere include("initialise.jl")
@everywhere COUPLINGS = Dict(
              "bw" => 1.0,
              "Jf" => 0.1,
              "Jd" => 0.1,
              "Wd" => -0.0,
              "Vf" => 1.,
              "Vd" => 0.1,
              "Uf" => 8.,
              "Ud" => 0.,
              "ηf" => 0.,
              "ηd" => 0.,
              "hop_t" => 0.1,
              "omega_by_t" => -2.
             )

function RGFlow(
        Wf_vals,
        Jp_vals,
        size_BZ::Int64;
        loadData::Bool=false,
    )
    parameters = Iterators.product(Wf_vals, Jp_vals)
    dos, dispersion = getDensityOfStates(tightBindDisp, size_BZ)
    FSpoints = getIsoEngCont(dispersion, 0.0)
    results = @showprogress @distributed vcat for p in collect(parameters)
        r = momentumSpaceRG(merge(COUPLINGS, Dict("J⟂" => p[2], "Wf" => p[1], "Kf" => -1e-5 * p[2], "Kd" => -1e-5 * p[2], "size_BZ" => size_BZ)); loadData=loadData)
        [r]
    end
    for (i, key) in enumerate(["Jf", "J⟂"])
        fgLabel = L"\mathrm{FS~frac.~(FG)}" 
        if key == "Jf" || key == "Jd"
            fig, ax = subplots(figsize=(7, 3.5))
            scData = [count(>(1e-3), diag(abs.(r[key])[FSpoints, FSpoints])) / length(FSpoints) for r in results]
            keyData = [maximum([sum(abs.(r[key][p, FSpoints])) for p in FSpoints]) for r in results]
            # println(maximum(keyData))
            x = [Jp ./ COUPLINGS["Jf"] for Jp in Jp_vals for Wf in Wf_vals]
            y = [-Wf ./ COUPLINGS["Jf"] for Jp in Jp_vals for Wf in Wf_vals]
            sc = ax.scatter(x, y, c=scData, marker="o", cmap="magma", s=4)
            fig.colorbar(sc, location="left", label=fgLabel, pad=-0.4)
        else
            fig, ax = subplots(figsize=(5, 3.5))
            keyData = [r[key] for r in results]
            fgLabel = L"$g^* > 0?~$(FG)" 
        end
        hm = ax.imshow(reshape(keyData, length(Wf_vals), length(Jp_vals)), origin="lower", cmap="magma_r", aspect="auto", extent=vcat(extrema(Jp_vals)..., extrema(-Wf_vals)...) ./ COUPLINGS["Jf"])
        ax.set_xlabel(L"$J_\perp / J_f$")
        ax.set_ylabel(L"$-W_f/J_f$")
        ax.grid(false)
        ax.set_box_aspect(1)
        fig.colorbar(hm, label=L"$g^*~$(BG)", location="right", format=matplotlib.ticker.FormatStrFormatter("%.2f"))
        fig.tight_layout()
        fig.savefig("PD_$(size_BZ)-$(i).pdf", bbox_inches="tight")
    end
end


function MomentumCorr(
        Wf_vals,
        Jp_vals,
        size_BZ,
        maxSize;
        loadData=false
    )
    couplings = copy(COUPLINGS)
    parameterSpace = Iterators.product(Wf_vals, Jp_vals)
    corrDef = Dict(
                   "SF-f-fk" => ("f", (i, j) -> [("+-+-", [1, 2, i+1, j], 0.5), 
                                                ("+-+-", [2, 1, i, j+1], 0.5),
                                                ("n+-", [1, i, j], 0.25), 
                                                ("n+-", [1, i+1, j+1], -0.25),
                                                ("n+-", [2, i, j], -0.25),
                                                ("n+-", [2, i+1, j+1], 0.25)
                                               ]
                               ),
                   # "SF-f-dk" => ("d", (i, j) -> [("+-+-", [1, 2, i+1, j], 0.5), 
                   #                              ("+-+-", [2, 1, i, j+1], 0.5),
                   #                              ("n+-", [1, i, j], 0.25), 
                   #                              ("n+-", [1, i+1, j+1], -0.25),
                   #                              ("n+-", [2, i, j], -0.25),
                   #                              ("n+-", [2, i+1, j+1], 0.25)
                   #                             ]
                   #             ),
                   # "SF-d-dk" => ("d", (i, j) -> [("+-+-", [3, 4, i+1, j], 0.5), 
                   #                              ("+-+-", [4, 3, i, j+1], 0.5),
                   #                              ("n+-", [3, i, j], 0.25), 
                   #                              ("n+-", [3, i+1, j+1], -0.25),
                   #                              ("n+-", [4, i, j], -0.25),
                   #                              ("n+-", [4, i+1, j+1], 0.25)
                   #                             ]
                   #             ),
                   "SF-d-fk" => ("f", (i, j) -> [("+-+-", [3, 4, i+1, j], 0.5), 
                                                ("+-+-", [4, 3, i, j+1], 0.5),
                                                ("n+-", [3, i, j], 0.25), 
                                                ("n+-", [3, i+1, j+1], -0.25),
                                                ("n+-", [4, i, j], -0.25),
                                                ("n+-", [4, i+1, j+1], 0.25)
                                               ]
                               ),
                  )
    dos, dispersion = getDensityOfStates(tightBindDisp, size_BZ)
    momentumPoints = Dict(k => getIsoEngCont(dispersion, 0.) for k in ["f", "d"])
    pooledResults = @showprogress @distributed vcat for (Wf, Jp) in collect(parameterSpace)
        params = merge(couplings, Dict("J⟂" => Jp, "Wf" => Wf, "size_BZ" => size_BZ, "maxSize" => maxSize))
        results = AuxiliaryCorrelations(
                                        params,
                                        corrDef,
                                        momentumPoints,
                                        Dict(),
                                        Dict();
                                        loadData=loadData,
                                       )
        [results]
    end
    # println(pooledResults[1])
    node = map2DTo1D(π/2, π/2, size_BZ)
    antinode = map2DTo1D(π/1, 0., size_BZ)
    SFresults = Dict(
                     "f-N" => [sum(r["SF-f-fk-$node"].^2)^0.5 for r in pooledResults],
                     "f-AN" => [sum(r["SF-f-fk-$antinode"].^2)^0.5 for r in pooledResults],
                     "d-N" => [sum(r["SF-d-fk-$node"].^2)^0.5 for r in pooledResults],
                     "d-AN" => [sum(r["SF-d-fk-$antinode"].^2)^0.5 for r in pooledResults],
                    )
    # SFresults = [count(>(0), abs.([r["SF-f-k"][(p, p)] for p in momentumPoints["f"]])) for r in pooledResults]
    # println(SFresults)
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 8))
    for (i, r) in zip([(1,1), (2,1), (1,2), (2,2)], ["f-N", "f-AN", "d-N", "d-AN"])
        hm = ax[i...].imshow(reshape(SFresults[r], length(Wf_vals), length(Jp_vals)), origin="lower", cmap="magma_r", aspect="auto", extent=vcat(extrema(Jp_vals)..., extrema(-Wf_vals)...) ./ COUPLINGS["Jf"]) #, norm=matplotlib[:colors][:LogNorm](vmin=1e-3,))
        ax[i...].set_ylabel(L"-W_f/J_f")
        ax[i...].set_xlabel(L"J_\perp/J_f")
        fig.colorbar(hm, label=r, location="right")
        # fig.colorbar(hm, label=L"\sum_{k}|\chi_{k_\mathrm{N}, k}|", location="right")
    end
    # hm = ax[1, 2].imshow(reshape(SFresults["d-N"], length(Wf_vals), length(Jp_vals)), origin="lower", cmap="magma_r", aspect="auto", extent=vcat(extrema(Jp_vals)..., extrema(-Wf_vals)...) ./ COUPLINGS["Jf"])
    # ax[2, 2].imshow(reshape(SFresults["d-AN"], length(Wf_vals), length(Jp_vals)), origin="lower", cmap="magma_r", aspect="auto", extent=vcat(extrema(Jp_vals)..., extrema(-Wf_vals)...) ./ COUPLINGS["Jf"])
    # ax[1, 2].set_ylabel(L"-W_f/J_f")
    # ax[1, 2].set_xlabel(L"J_\perp/J_f")
    # fig.colorbar(hm, label=L"\sum_{k}|\chi_{k_\mathrm{AN}, k}|", location="right")
    fig.tight_layout()
    fig.savefig("RC-$(size_BZ)-$(maxSize).pdf", bbox_inches="tight")
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
                            "SF-f-k" => ("f", (i, j) -> [("+-+-", [1, 2, i+1, j], 0.5), 
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
                            # "SF-fdpm" => ("i", [("+-+-", [1, 2, 4, 3], 1 / 2), 
                            #                     ("+-+-", [2, 1, 3, 4], 1 / 2)]),
                            # "SF-fdzz" => ("i", [("nn", [1, 3], 1 / 4), 
                            #                     ("nn", [1, 4], -1 / 4), 
                            #                     ("nn", [2, 3], -1 / 4), 
                            #                     ("nn", [2, 4], 1 / 4)]),
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
                         # "I2-f-d" => ([1, 2], [3, 4]),
                         # "I2-f-max" => (f, d) -> ([1, 2], [f, f+1]),
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
        Jp_vals,
        Wf_vals;
        loadData=false,
        map=false
    )
    dos, dispersion = getDensityOfStates(tightBindDisp, size_BZ)
    momentumPoints = getIsoEngCont(dispersion, 0.)
    ω = collect(-20:0.001:20)
    ωp = abs.(ω) .< 1.0
    σ = 0.01

    node = map2DTo1D(π/2, π/2, size_BZ)
    antinode = map2DTo1D(π/1, 0., size_BZ)
    impSites = Dict("f" => [1, 2], "d" => [3, 4])

    siam(i) = Dict("create" => [("+", [i], 1.0), ("+", [i+1], 1.0)])
    kondo(i, j) = Dict("create" => [
                                    ("+-+", [i, i+1, j+1], 1.0),
                                    ("+-+", [i+1, i, j], 1.0),
                                   ]
                      )
    specFuncReqs = Dict("Af_f" => ("f", mom -> siam(1)), "Af_f0" => ("f", mom -> kondo(1, mom)), "Ad_f0" => ("f", mom -> kondo(3, mom)))
    counter = 1
    plots = Dict(name => plt.subplots(ncols=length(Jp_vals), figsize=(8 * length(Jp_vals), 6)) for name in keys(names))
    couplingSets = Iterators.product(Wf_vals, Jp_vals)

    calculateAt = [node, antinode] # filter(p -> map1DTo2D(p, size_BZ)[1] ≥ 0 && map1DTo2D(p, size_BZ)[2] ≥ 0, momentumPoints)
    resultsPooled = Dict(c => Dict("SF" => Dict(), "kondoFactor" => 0.0, "perpFactor" => 0.0) for c in couplingSets)
    @showprogress desc="outer" Threads.@threads for (i, (Wf, Jp)) in enumerate(couplingSets) |> collect
        params = Dict{String, Any}(merge(COUPLINGS, Dict("J⟂" => Jp, "Wf" => Wf, "size_BZ" => size_BZ, "maxSize" => maxSize)))
        params["calculateAt"] = calculateAt
        results, couplings = AuxiliaryCorrelations(
                                        params,
                                        Dict(),
                                        momentumPoints,
                                        Dict(),
                                        specFuncReqs;
                                        loadData=loadData,
                                       )
        resultsPooled[(Wf, Jp)] = Dict(
              "SF" => results,
              "node" => clamp(couplings["Jf"][node, node] / COUPLINGS["Jf"], 0., 1.)^0.5,
              "antinode" => clamp(couplings["Jf"][antinode, antinode] / COUPLINGS["Jf"], 0., 1.)^0.5,
              "perpFactor" => Jp == 0 ? 0 : clamp(couplings["J⟂"][end] / Jp - 1, 0., 1.)
             )
        # rm("data-iterdiag", force=true, recursive=true)
    end
    if !map
        fig, ax = plt.subplots(nrows=length(Wf_vals), ncols=2, figsize=(13, 4.5 * length(Wf_vals)))
        for (i, Wf) in enumerate(Wf_vals)
            axTitle = round(Wf/COUPLINGS["Jf"], digits=2)
            for Jp in Jp_vals 
                for (axi, pivot, name) in zip(ax[i, :], [node, antinode], ["node", "antinode"])
                    Aff = SpecFunc(resultsPooled[(Wf, Jp)]["SF"]["Af_f-$pivot"], ω, σ)
                    # Aff = 0 .* ω
                    Aff0 = resultsPooled[(Wf, Jp)][name] .* SpecFunc(resultsPooled[(Wf, Jp)]["SF"]["Af_f0-$pivot"], ω, σ)
                    Adf0 = resultsPooled[(Wf, Jp)]["perpFactor"] .* SpecFunc(resultsPooled[(Wf, Jp)]["SF"]["Ad_f0-$pivot"], ω, σ)
                    axi.plot(ω[ωp], (Aff + Aff0 + Adf0)[ωp], label=L"J_\perp = %$Jp", lw=3)
                    axi.legend()
                    axi.set_title(L"%$name: $W_f/J_f=%$(axTitle)$")
                    # axi.set_yscale("log")
                end
            end
        end
        fig.tight_layout()
        fig.savefig("SFK-$size_BZ-$maxSize.pdf", bbox_inches="tight")
    end
    if map
        weights = Dict("node" => Dict(couplingSets .=> 0.), "antinode" => Dict(couplingSets .=> 0.))
        innerRange = abs.(ω) .< 0.1
        @showprogress Threads.@threads for (i, c) in couplingSets |> enumerate |> collect
            for (pivot, name) in zip([node, antinode], ["node", "antinode"])
                Aff = SpecFunc(resultsPooled[c]["SF"]["Af_f-$pivot"], ω, σ)
                # Aff = 0 .* ω
                Aff0 = resultsPooled[c][name] .* SpecFunc(resultsPooled[c]["SF"]["Af_f0-$pivot"], ω, σ)
                Adf0 = resultsPooled[c][name] * resultsPooled[c]["perpFactor"] .* SpecFunc(resultsPooled[c]["SF"]["Ad_f0-$pivot"], ω, σ)
                A = Aff + Aff0 + Adf0
                weights[name][c] = sum(A[innerRange]) / sum(A)
            end
        end
        fig, ax = plt.subplots(figsize=(5.5, 4))
        lims = extrema(vcat(collect(values(weights["node"])), collect(values(weights["antinode"]))))
        hm = ax.imshow(reshape([abs(weights["node"][c] - weights["antinode"][c]) for c in couplingSets], length(Wf_vals), length(Jp_vals)), origin="lower", cmap="magma_r", aspect="auto", extent=vcat(extrema(Jp_vals)..., extrema(-Wf_vals)...) ./ COUPLINGS["Jf"], norm=matplotlib[:colors][:LogNorm](vmin=lims[1], vmax=lims[2]))
        cb = fig.colorbar(hm, pad=0.2, location="left")
        cb.ax.set_title(L"A_N - A_\text{AN}") 
        # sc = ax.scatter(last.(couplingSets) ./ COUPLINGS["Jf"], -1 .* first.(couplingSets) ./ COUPLINGS["Jf"], c=[weights["antinode"][c] for c in couplingSets], cmap="magma_r", norm=matplotlib[:colors][:LogNorm](vmin=lims[1], vmax=lims[2]))
        # sc.set_edgecolor("black")

        ax.set_xlabel(L"J_\perp/J_f")
        ax.set_ylabel(L"-W_f/J_f")
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top") 
        fig.tight_layout()
        fig.savefig("SFK-MAP-$size_BZ-$maxSize.pdf", bbox_inches="tight")
    end
end


function MomentumSpecFuncMap(
        size_BZ::Int64,
        maxSize::Int64,
        Jp_vals,
        Wf_vals;
        loadData=false,
    )

    dos, dispersion = getDensityOfStates(tightBindDisp, size_BZ)
    momentumPoints = getIsoEngCont(dispersion, 0.)
    ω = collect(-20:0.001:20)
    ωp = abs.(ω) .< 1.0
    σ = 0.01

    impSites = Dict("f" => [1, 2], "d" => [3, 4])

    siam(i) = Dict("create" => [("+", [i], 1.0), ("+", [i+1], 1.0)])
    kondo(i, j) = Dict("create" => [
                                    ("+-+", [i, i+1, j+1], 1.0),
                                    ("+-+", [i+1, i, j], 1.0),
                                   ]
                      )
    specFuncReqs = Dict("Af_f" => ("f", mom -> siam(1)), "Af_f0" => ("f", mom -> kondo(1, mom)), "Ad_f0" => ("f", mom -> kondo(3, mom)))
    counter = 1
    couplingSets = Iterators.product(Wf_vals, Jp_vals)
    pivotPoints = filter(p -> map1DTo2D(p, size_BZ)[1] ≥ 0 && map1DTo2D(p, size_BZ)[2] ≥ π/2,
                         momentumPoints
                        )

    resultsPooled = Dict(c => Dict("SF" => Dict(), "kondoFactor" => 0.0, "perpFactor" => 0.0) for c in couplingSets)
    for (i, (Wf, Jp)) in enumerate(couplingSets) |> collect
    # resultsPooled = @distributed merge for (i, (Wf, Jp)) in enumerate(couplingSets) |> collect
        params = Dict{String, Any}(merge(COUPLINGS, Dict("J⟂" => Jp, "Wf" => Wf, "size_BZ" => size_BZ, "maxSize" => maxSize)))
        params["calculateAt"] = pivotPoints
        results, couplings = AuxiliaryCorrelations(
                                        params,
                                        Dict(),
                                        momentumPoints,
                                        Dict(),
                                        specFuncReqs;
                                        loadData=loadData,
                                       )
        resultsPooled[(Wf, Jp)] = merge(Dict("SF" => results, 
                                             "perpFactor" => Jp == 0 ? 0 : clamp(couplings["J⟂"][end] / Jp - 1, 0., 1.)
                                            ), 
                                        Dict("$p" => clamp(couplings["Jf"][p, p] / COUPLINGS["Jf"], 0., 1.)^0.5 
                                             for p in params["calculateAt"]
                                            )
                              )
        # Dict((Wf, Jp) => merge(Dict("SF" => results, "perpFactor" => Jp == 0 ? 0 : clamp(couplings["J⟂"][end] / Jp - 1, 0., 1.)), 
        #                        Dict("$p" => clamp(couplings["Jf"][p, p] / COUPLINGS["Jf"], 0., 1.)^0.5 for p in params["calculateAt"])
        #                       )
        #     )
        # rm("data-iterdiag", force=true, recursive=true)
    end
    # fig, ax = plt.subplots(nrows=length(Wf_vals), ncols=length(Jp_vals), figsize=(6.5 * length(Jp_vals), 4.5 * length(Wf_vals)))
    fig, ax = plt.subplots()
    innerRange = abs.(ω) .< 0.1
    plotData = zeros(length(couplingSets), length(pivotPoints))
    @showprogress for (i, (Wf, Jp)) in enumerate(couplingSets)
        specFuncMap = Dict()
        Threads.@threads for pivot in pivotPoints
            Aff = SpecFunc(resultsPooled[(Wf, Jp)]["SF"]["Af_f-$pivot"], ω, σ)
            Aff0 = resultsPooled[(Wf, Jp)]["$pivot"] .* SpecFunc(resultsPooled[(Wf, Jp)]["SF"]["Af_f0-$pivot"], ω, σ)
            Adf0 = resultsPooled[(Wf, Jp)]["perpFactor"] .* SpecFunc(resultsPooled[(Wf, Jp)]["SF"]["Ad_f0-$pivot"], ω, σ)
            specFuncMap[pivot] = sum((Aff + Aff0 + Adf0)[innerRange]) / sum(Aff + Aff0 + Adf0)
            if specFuncMap[pivot] < 5e-3
                specFuncMap[pivot] = NaN
            end
            # kx, ky = map1DTo2D(pivot, size_BZ)
            # specFuncMap[map2DTo1D(π - kx, π - ky, size_BZ)] = specFuncMap[pivot]
            # axi = length(Wf_vals) * length(Jp_vals) > 1 ? ax[i + (j-1) * length(Wf_vals)] : ax
            # allPoints = map1DTo2D.(1:size_BZ^2, size_BZ)
            # southEast = first.(allPoints) .≥ 0 .&& last.(allPoints) .≥ 0
            # hm = axi.imshow(reshape(specFuncMap[southEast], div(size_BZ + 1, 2), div(size_BZ + 1, 2)), extent = (0, 1, -1, 0))
            # axi.set_xlabel(L"k_x/\pi")
            # axi.set_ylabel(L"k_x/\pi")
            # fig.colorbar(hm)
        end
        plotData[i, :] .= [specFuncMap[p] for p in sort(pivotPoints, by=p -> map1DTo2D(p, size_BZ)[1])]
    end
    hm = ax.imshow(plotData) #, norm=matplotlib[:colors][:LogNorm]())
    fig.colorbar(hm, label=L"A_{k, f}", shrink=0.8)
    ycoup = length(Wf_vals) == 1 ? (Jp_vals ./ COUPLINGS["Jf"], L"J_\perp/J_f") : (Wf_vals ./ COUPLINGS["Jf"], L"W_f/J_f")
    titleCoup = length(Wf_vals) == 1 ? L"W_f / J_f =%$(trunc(Wf_vals[1] / COUPLINGS[\"Jf\"], digits=2))" : L"J_\perp / J_f =%$(trunc(Jp_vals[1] / COUPLINGS[\"Jf\"], digits=2))"
    ax.set_yticks(0:(length(couplingSets)-1), trunc.(ycoup[1], digits=3))
    ax.set_xticks(0:2:(length(pivotPoints)-1), first.(map1DTo2D.(sort(pivotPoints[1:2:end], by=p -> map1DTo2D(p, size_BZ)[1]), size_BZ)) ./ π)
    ax.set_xlabel(L"k_x/\pi")
    ax.set_ylabel(ycoup[2])
    ax.grid(false)
    ax.set_title(titleCoup, color="dimgray")
    fig.tight_layout()
    fig.savefig("SFK-FSMAP-$size_BZ-$maxSize.pdf", bbox_inches="tight")
end


function RealSpecFunc(
        size_BZ::Int64,
        maxSize::Int64,
        Jp_vals,
        Wf_vals;
        loadData=false,
    )
    names = Dict(
                 "node_d" => L"A_d",
                 "node_f" => L"A_f",
                 "node_+" => L"A_+",
                 "node_-" => L"A_-",
                )

    dos, dispersion = getDensityOfStates(tightBindDisp, size_BZ)
    momentumPoints = getIsoEngCont(dispersion, 0.)
    ω = collect(-20:0.001:20)
    ωp = abs.(ω) .< 1.0
    σ = 0.08

    impSites = Dict("f" => [1, 2], "d" => [3, 4])

    siam(i) = Dict("create" => [("+", [i], 1.0), ("+", [i+1], 1.0)])
    kondo(i, j) = Dict("create" => [
                                    ("+-+", [i, i+1, j+1], 1.0),
                                    ("+-+", [i+1, i, j], 1.0),
                                   ]
                      )
    specFuncReqs = Dict(
                        "Af_f_loc" => ("f", z -> siam(1)),
                        "Af_f0_loc" => ("f", z -> kondo(1, z)),
                        "Ad_d_loc" => ("d", z -> siam(3)),
                        "Ad_d0_loc" => ("d", z -> kondo(3, z)),
                        "Af_d_loc" => ("i", z -> kondo(1, 3)),
                       )
    counter = 1
    plots = Dict(name => plt.subplots(ncols=length(Jp_vals), figsize=(8 * length(Jp_vals), 6)) for name in keys(names))
    couplingSets = Iterators.product(Wf_vals, Jp_vals)

    resultsPooled = Dict(c => Dict("SF" => Dict(), "kondoFactor" => 0.0, "perpFactor" => 0.0) for c in couplingSets)
    fermiSurface = getIsoEngCont(dispersion, 0.)
    @showprogress desc="outer" Threads.@threads for (i, (Wf, Jp)) in enumerate(couplingSets) |> collect
        params = Dict{String, Any}(merge(COUPLINGS, Dict("J⟂" => Jp, "Wf" => Wf, "size_BZ" => size_BZ, "maxSize" => maxSize)))
        results, couplings = AuxiliaryCorrelations(
                                        params,
                                        Dict(),
                                        momentumPoints,
                                        Dict(),
                                        specFuncReqs;
                                        loadData=loadData,
                                       )
        resultsPooled[(Wf, Jp)] = Dict(
              "SF" => results,
              "fFactor" => clamp(maximum(couplings["Jf"][fermiSurface,fermiSurface]) / COUPLINGS["Jf"], 0., 1.)^0.5,
              "dFactor" => clamp(maximum(couplings["Jd"][fermiSurface,fermiSurface]) / COUPLINGS["Jf"], 0., 1.)^0.5,
              "perpFactor" => Jp == 0 ? 0 : couplings["J⟂"][end] / Jp
             )
        # rm("data-iterdiag", force=true, recursive=true)
    end
    fig, ax = plt.subplots(nrows=length(Wf_vals), figsize=(7, 4.5 * length(Wf_vals)))
    SF = Any[Nothing for _ in couplingSets]
    Threads.@threads for (i, (Wf, Jp)) in couplingSets |> enumerate |> collect 
        t = "f"
        σ1 = σ # map(ωi -> minimum(abs.(last.(vcat(resultsPooled[(Wf, Jp)]["SF"]["Af_d_loc"]...)) .- ωi)) < 1e-2 ? σ : 0.0, ω)
        Af_d = resultsPooled[(Wf, Jp)]["perpFactor"] .* SpecFunc(vcat(resultsPooled[(Wf, Jp)]["SF"]["Af_d_loc"]...), ω, σ1)
        σ2 = σ # map(ωi -> minimum(abs.(last.(vcat(resultsPooled[(Wf, Jp)]["SF"]["A$(t)_$(t)0_loc"]...)) .- ωi)) < 1e-2 ? σ : 0.0, ω)
        Att = SpecFunc(vcat(resultsPooled[(Wf, Jp)]["SF"]["A$(t)_$(t)_loc"]...), ω, σ)
        At_t0 = resultsPooled[(Wf, Jp)][t*"Factor"] .* SpecFunc(vcat(resultsPooled[(Wf, Jp)]["SF"]["A$(t)_$(t)0_loc"]...), ω, σ2)
        SF[i] = (Att, At_t0, Af_d)
    end
    for ((Att, At_t0, Af_d), (Wf, Jp)) in zip(SF, couplingSets)
        axi = ax[findfirst(==(Wf), Wf_vals)]
        axTitle = L"W_f = %$(round(Wf/COUPLINGS[\"Jf\"], digits=2))"
        # axi.plot(ω[ωp], Att[ωp], label=L"J_\perp = %$Jp", lw=3)
        axi.plot(ω[ωp], Normalise(Att .+ Af_d .+ At_t0, ω)[ωp], label=L"J_\perp = %$Jp", lw=3)
        # axi.plot(ω[ωp], , label=L"J_\perp = %$Jp", lw=3)
        axi.legend()
        axi.set_title(axTitle)
        axi.set_xlabel(L"\omega")
        axi.set_ylabel(L"A(\omega)")
        # axi.set_yscale("log")
    end
    fig.tight_layout()
    fig.savefig("SFR-$size_BZ-$maxSize.pdf", bbox_inches="tight")
end


function RealSpecFuncOld(
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
    momentumPoints = getIsoEngCont(dispersion, 0.)
    freqVals = collect(-15:0.001:15)

    antinode = map2DTo1D(π/1, 0., size_BZ)
    node = map2DTo1D(π/2, π/2, size_BZ)
    impSites = Dict("f" => [1, 2], "d" => [3, 4])

    counter = 1
    couplings = copy(COUPLINGS)
    couplings["ηd"] = ηd
    plots = Dict(name => plt.subplots(ncols=length(Jp_vals), figsize=(8 * length(Jp_vals), 6)) for name in keys(names))
    couplingSets = Iterators.product(Wf_vals, Jp_vals)

    siam(coeff, orb) = [("+", [i], coeff) for i in impSites[orb]]
    kondo(coeff, orb, bath) = [("+-+", [imp..., bath+2-i], coeff) for (i,imp) in enumerate([impSites[orb], reverse(impSites[orb])])]
    resultsPooled = @distributed merge for (Wf, Jp) in collect(couplingSets)
        couplings = Dict{String, Any}(merge(COUPLINGS, Dict("J⟂" => Jp, "Wf" => Wf, "size_BZ" => size_BZ, "maxSize" => maxSize)))
        # merge!(couplings, Dict("Wf" => Wf,
        #                        "J⟂" => Jp,
        #                        "Uf" => impCorr(Wf, couplings["U_by_W"]),
        #                        #="Ud" => impCorr(couplings["Wd"], couplings["U_by_W"]),=#
        #                       )
        #       )

        effHybridisation = 2√(Jp * (COUPLINGS["Ud"] + COUPLINGS["Uf"]))
        hybridMatrix = [[0, -effHybridisation] [-effHybridisation, -ηd]]
        # matrix is in the basis of [f, d]; first eigenvector is (1, 1), second is (1, -1)
        gap, V = eigen(hybridMatrix)
        hybCoeffs = Dict(k => Dict(["f", "d"] .=> all(≤(0), V[:, i]) ? -V[:, i] : V[:, i]) for (i, k) in enumerate(["+", "-"]))
        display(hybCoeffs)

        specFuncReqs = Dict()

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

        results, fpCouplings = AuxiliaryCorrelations(
                                      couplings,
                                      Dict(),
                                      momentumPoints,
                                      Dict(),
                                      specFuncReqs;
                                      loadData=loadData,
                                     )
        avgKondo = Dict(k => minimum((1, sum(abs.(fpCouplings[k][momentumPoints[k], momentumPoints[k]])) / (length(momentumPoints[k])^1.0 * couplings["J$(k)"]))) for k in ["f", "d"])
        #=avgKondo = Dict(k => minimum((Inf, sum(abs.(fpCouplings[k][momentumPoints[k], momentumPoints[k]])) / (length(momentumPoints[k])^1.0 * couplings["J$(k)"]))) for k in ["f", "d"])=#

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

# RGFlow(-0.1:-0.0025:-0.17, 0:0.05:1.0, 33; loadData=true)
# RGFlow(-0.0:-0.05:-0.2, 0:0.01:0.01, 13; loadData=false,)
# MomentumCorr(-0.16:-0.0025:-0.18, 0.00:0.025:0.2, 33, 399; loadData=true)
#=@time RealCorr(33, 0896, 0.05:0.025:0.40, -0.0:-0.02:-0.160, 0.; loadData=false)=#
# @time MomentumSpecFunc(33, 399, [0.0, 0.2, 0.4], [-0., -0.165, -0.174]; loadData=true)
# @time MomentumSpecFunc(33, 399, 0.025:0.00625:0.3, 0:-0.005:-0.18; loadData=true, map=true)
# @time MomentumSpecFuncMap(33, 598, [0.0,], [-0.165, -0.166, -0.167, -0.17, -0.172, -0.174, -0.1741]; loadData=true)
# @time MomentumSpecFuncMap(33, 599, [0.11, 0.124, 0.1245, 0.1247, 0.1249, 0.125,], [-0.166,]; loadData=true)
# @time MomentumSpecFuncMap(33, 601, 0.15:0.0005:0.154, [-0.1,]; loadData=true)
# @time MomentumSpecFuncMap(33, 600, [0.32, 0.36, 0.37, 0.4, 0.41], [-0.05,]; loadData=true)
@time RealSpecFunc(33, 609, [0., 0.3, 0.6], [-0.0, -0.1, -0.16, -0.17, -0.18]; loadData=true)
#=@time RealSpecFunc(33, 1610, [0.2, 0.4], [-0., -0.06, -0.12, -0.1409, -0.147, -0.152, -0.1535, -0.16], 0.0; loadData=true)=#
