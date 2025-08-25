using Distributed, Combinatorics, Serialization, PDFmerger, PyPlot, CodecZlib
using Fermions

@everywhere using FileIO, JSON3, LinearAlgebra, ProgressMeter, JLD2

@everywhere include("Constants.jl")
@everywhere include("Helpers.jl")
@everywhere include("RgFlow.jl")
@everywhere include("Models.jl")
include("PhaseDiagram.jl")
include("Probes.jl")
include("PltStyle.jl")

global kondoF = 0.1
global kondoPerp = 0.0
global lightBandFactor = 2.
global epsilonF = -2 * HOP_T
global mu_c = 0.2 * HOP_T
global W = 0.5 * HOP_T
maxSize = 1000
WmaxSize = 500

@everywhere NiceValues(size_BZ) = Dict{Int64, Vector{Float64}}(
                         13 => -1.0 .* [0., 1., 1.5, 1.55, 1.6, 1.61] ./ size_BZ,
                         41 => -1.0 .* [0, 3.5, 7.13, 7.3, 7.5, 7.564, 7.6, 8.] ./ size_BZ,
                         77 => -1.0 .* [0., 7., 14.04, 14.6, 14.99, 15.0] ./ size_BZ,
                        )[size_BZ]
@everywhere pseudogapStart(size_BZ) = Dict{Int64, Float64}(
                         13 => -1.0 * 1.5 / size_BZ,
                         41 => -1.0 * 7.13 / size_BZ,
                         77 => -1.0 * 14.04 / size_BZ,
                        )[size_BZ]
@everywhere pseudogapEnd(size_BZ) = Dict{Int64, Float64}(
                         13 => -1.0 * 1.6 / size_BZ,
                         41 => -1.0 * 7.564 / size_BZ,
                         77 => -1.0 * 14.99 / size_BZ,
                        )[size_BZ]

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


function PhaseDiagram(
        size_BZ::Int64,
        kondoFRange::NTuple{3, Float64},
        WfRange::NTuple{3, Float64},
        kondoPerpRange::NTuple{3, Float64},
        WRange::NTuple{3, Float64};
        loadData::Bool=false,
        fillPG::Bool=false,
    )
    @assert minimum(kondoPerpRange) > 0
    @assert minimum(kondoFRange) > 0
    #=epsilonF = 0.0 * HOP_T=#
    #=mu_c = 0.0 * HOP_T=#
    #=kondoPerpVals = (0.05:0.2:0.65) .* HOP_T=#
    #=W_arr = (0.0:-0.2:-0.6) .* HOP_T=#
    WfVals, WVals, kondoPerpVals, kondoFVals = FillIn.((WfRange, WRange, kondoPerpRange, kondoFRange))

    fig, axes = PyPlot.subplots(nrows=length(kondoFVals), ncols=length(WVals), figsize=(8 * length(WVals), 6 * length(kondoFVals)))
    for (i, kondoF) in enumerate(kondoFVals)
        for (j, W) in enumerate(WVals)
            phaseDiagram_Jf, phaseDiagram_J = PhaseDiagram(size_BZ, kondoPerpVals, WfRange, 
                                                           Dict("omega_by_t"=>OMEGA_BY_t, "W"=>W, "kondoF"=>kondoF, 
                                        "epsilonF"=>epsilonF, "mu_c"=>mu_c, "lightBandFactor"=>lightBandFactor);
                                   loadData=loadData, fillPG=fillPG
                                  )
            hmap = axes[i,j].imshow(phaseDiagram_Jf, aspect="auto", origin="lower", 
                                    extent=(WfRange[1],
                                            WfRange[3],
                                            kondoPerpRange[1],
                                            kondoPerpRange[3],
                                           ),
                                    cmap = matplotlib.colors.ListedColormap(plt.get_cmap(CMAP)(60:220)),
                                   )
            for i in 1:length(kondoPerpVals)
                if i % 10 ≠ 0
                    phaseDiagram_J[i, :] .= 0
                    continue
                end
                phaseDiagram_J[i,((1:length(WfVals)) .% 10) .≠ 0] .= 0
            end
            colors = ["black", "midnightblue", "purple"]
            markers = ["x", ".", "P"]
            for flag in [1, 2, 3]
                pairs = findall(==(flag), phaseDiagram_J)
                axes[i,j].scatter([WfVals[p[2]] for p in pairs], [kondoPerpVals[p[1]] for p in pairs], marker=markers[flag], color=colors[flag], s=150)
            end

            if j == 1
                axes[i, 1].set_ylabel(L"J")
                axes[i, j].text(-0.45, 0.5, "\$J_f=$(round(kondoF, digits=2))\$", horizontalalignment="center", verticalalignment="center", transform=axes[i,j].transAxes, bbox=Dict("facecolor"=>"red", "alpha"=>0.1, "boxstyle"=>"Round,pad=0.2"))
            end
            if i == 1
                axes[i, j].text(0.5, 1.2, "\$W=$(round(W, digits=2))\$", horizontalalignment="center", verticalalignment="center", transform=axes[i,j].transAxes, bbox=Dict("facecolor"=>"red", "alpha"=>0.1, "boxstyle"=>"Round,pad=0.2"))
            end
            if i == length(kondoPerpVals)
                axes[length(kondoPerpVals), j].set_xlabel(L"W_f")
            end
        end
    end
    fig.tight_layout()
    savefig("PD-$(size_BZ)-$(epsilonF)-$(mu_c)-$(lightBandFactor).pdf", bbox_inches="tight"); PyPlot.close()
end

function AuxiliaryMomentumCorrelations(
        size_BZ::Int64,
        maxSize::Int64,
        kondoFRange::NTuple{3, Float64},
        WfRange::NTuple{3, Float64},
        kondoPerpRange::NTuple{3, Float64},
        WRange::NTuple{3, Float64};
        loadData::Bool=false,
    )

    WfVals, WVals, kondoPerpVals, kondoFVals = FillIn.((WfRange, WRange, kondoPerpRange, kondoFRange))

    parameterSpace = Iterators.product(WfVals, kondoPerpVals, WVals, kondoFVals)
    results = @distributed vcat for (Wf, kondoPerp, W, kondoF) in parameterSpace |> collect
        couplings = Dict("omega_by_t" => OMEGA_BY_t,
                         "W"=>W,
                         "Wf"=>Wf,
                         "kondoF"=>kondoF, 
                         "kondoPerp"=>kondoPerp, 
                         "epsilonF"=> epsilonF,
                         "mu_c"=> mu_c,
                         "lightBandFactor"=>lightBandFactor
                        )
        correlations = Dict(
                            "SF-dkk" => (1, (i, j) -> [("+-+-", [1, 2, i+1, j], 0.25), ("+-+-", [2, 1, i, j+1], 0.25)]),
                           )
        results = AuxiliaryMomentumCorrelations(size_BZ, couplings, correlations, maxSize; loadData=loadData, silent=true)
        kspaceResult = sum([v for (k,v) in results if k ≠ "SF-dkk-0-0"]) / length(filter(≠("SF-dkk-0-0"), keys(results)))
        zeroSiteResult = results["SF-dkk-0-0"] / length(filter(≠("SF-dkk-0-0"), keys(results)))
        [(kspaceResult, zeroSiteResult),]
    end

    kspaceResults = Dict(parameters => 0. for parameters in parameterSpace)
    zeroSiteResults = Dict(parameters => 0. for parameters in parameterSpace)
    for (key, (kspaceResult, zeroSiteResult)) in zip(parameterSpace, results)
        kspaceResults[key] = kspaceResult
        zeroSiteResults[key] = zeroSiteResult
    end

    fig, axes = plt.subplots(nrows=length(kondoFVals), ncols=length(WVals), figsize=(16 * length(kondoFVals), 8 * length(WVals)))
    fig.tight_layout()
    for ((row, kondoF), (col, W)) in Iterators.product(enumerate.((kondoFVals, WVals))...)
        kspaceResult = zeros(length(kondoPerpVals), length(WfVals))
        zeroSiteResult = zeros(length(kondoPerpVals) * length(WfVals))
        for ((y, kondoPerp), (x, Wf)) in Iterators.product(enumerate.((kondoPerpVals, WfVals))...)
            key = (Wf, kondoPerp, W, kondoF)
            kspaceResult[y, x] = kspaceResults[key]
            zeroSiteResult[y * (length(WfVals) - 1) + x] = zeroSiteResults[key]
        end
        if length(kondoFVals) * length(WVals) > 1
            ax = axes[row, col]
        else
            ax = axes
        end
        zeroSiteResult = abs.(zeroSiteResult)
        kspaceResult = abs.(kspaceResult)
        if length(zeroSiteResult[zeroSiteResult .> 0]) > 0
            zeroSiteResult[zeroSiteResult .== 0] .= minimum(zeroSiteResult[zeroSiteResult .> 0])/10
        end
        hm = ax.imshow(kspaceResult, extent=(minimum(WfVals), maximum(WfVals),minimum(kondoPerpVals), maximum(kondoPerpVals)), aspect="auto", cmap=matplotlib.colors.ListedColormap(plt.get_cmap(CMAP)(60:220)))
        if length(zeroSiteResult[zeroSiteResult .> 0]) > 0
            sc = ax.scatter(repeat(WfVals, outer=length(kondoPerpVals)), repeat(kondoPerpVals, inner=length(WfVals)), c=zeroSiteResult, cmap="tab20b", norm="log", s=100)
        else
            sc = ax.scatter(repeat(WfVals, outer=length(kondoPerpVals)), repeat(kondoPerpVals, inner=length(WfVals)), c=zeroSiteResult, cmap="tab20b", s=100, vmin=0, vmax=0.1)
        end
        cb1 = fig.colorbar(hm, shrink=0.5, pad=0.12, location="left")
        cb1.set_label(L"\overline{\langle S_d \cdot S_k \rangle}", labelpad=-80, y=1.1, rotation="horizontal")
        cb2 = fig.colorbar(sc, shrink=0.5, pad=0.05, location="right")
        cb2.set_label(L"\langle S_d \cdot S_0 \rangle", labelpad=-30, y=1.2, rotation="horizontal")
        ax.set_xlabel(L"W_f")
        ax.set_ylabel(L"J")

        colors = ["black", "midnightblue", "purple"]
        markers = ["x", ".", "P"]

        if col == 1
            ax.text(-0.45, 0.5, "\$J_f=$(round(kondoF, digits=2))\$", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, bbox=Dict("facecolor"=>"red", "alpha"=>0.1, "boxstyle"=>"Round,pad=0.2"))
        end
        if row == 1
            ax.text(0.5, 1.2, "\$W=$(round(W, digits=2))\$", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, bbox=Dict("facecolor"=>"red", "alpha"=>0.1, "boxstyle"=>"Round,pad=0.2"))
        end
    end
    fig.savefig("auxCorr-$(size_BZ)-$(maxSize)-$(kondoF)-$(W)-$(epsilonF)-$(mu_c)-$(lightBandFactor).pdf", bbox_inches="tight")
end

@time AuxiliaryMomentumCorrelations(33, 100, (0.05, 0.7, 0.65), (-0.6, 0.6, -0.0), (0.01, 0.8, 0.81), (0., -0.7, -0.6); loadData=false)

#=PhaseDiagram(33, (0.05, 0.2, 0.65), (-0.6, 0.06, -0.0), (0.01, 0.07, 0.8), (0., -0.2, -0.6); loadData=false)=#

#=postProcess(data) = reshape(data, (size_BZ, size_BZ)) .|> abs=#
#==#
#=node = map2DTo1D(π/2, π/2, size_BZ)=#
#=midway = map2DTo1D(3π/4, π/4, size_BZ)=#
#=antinode = map2DTo1D(π/1, 0., size_BZ)=#
#=size_BZ = 41=#
#=Wf_arr = NiceValues(size_BZ)[[1, 7, 8]]=#
#=for Wc in [0.0, -0.2, -0.3] * kondoF=#
#=    pdfPaths = String[]=#
#=    for kondoPerp_i in [0., 1.2, 2.] .* kondoF=#
#=        kondoJArrays, kondoPerpArrays, dispersion = RGFlow(Wf_arr, kondoPerp_i, Wc, size_BZ; loadData=true)=#
#==#
#=        fig, axes = plt.subplots(nrows=length(Wf_arr), ncols=3, figsize=(14, 9), gridspec_kw=Dict("width_ratios"=> [1,1,0.9]))=#
#=        plt.subplots_adjust(wspace=0.5, hspace=0.5)=#
#=        for (i, Wf) in enumerate(Wf_arr)=#
#=            hmap = axes[i,1].imshow(postProcess(kondoJArrays[Wf][node, :, end]), aspect="auto", norm=matplotlib[:colors][:LogNorm]())=#
#=            axes[i,1].set_xlabel(L"RG flow $\longrightarrow$")=#
#=            axes[i,1].set_ylabel(L"$J_f~(k_\mathrm{N}, k)$")=#
#=            fig.colorbar(hmap)=#
#=            hmap = axes[i,2].imshow(postProcess(kondoJArrays[Wf][antinode, :, end]), aspect="auto", norm=matplotlib[:colors][:LogNorm]())=#
#=            axes[i,2].set_xlabel(L"RG flow $\longrightarrow$")=#
#=            axes[i,2].set_ylabel(L"$J_f~(k_\mathrm{AN}, k)$")=#
#=            fig.colorbar(hmap)=#
#=            axes[i,3].plot(kondoPerpArrays[Wf], label=L"$W_f = %$(round(Wf, digits=2))$")=#
#=            axes[i,3].set_xlabel(L"RG flow $\longrightarrow$")=#
#=            axes[i,3].set_ylabel(L"$J$")=#
#=            axes[i,3].legend()=#
#=        end=#
#=        fig.suptitle(L"Varying $W_f$. $~ J/J_f=%$(kondoPerp_i / kondoF)~ W/J_f=%$(Wc/kondoF)$", y=0.93)=#
#=        path = "rgflow-$(size_BZ)-$(Wc)-$(kondoPerp_i).pdf"=#
#=        savefig(path, bbox_inches="tight"); PyPlot.close()=#
#=        push!(pdfPaths, path)=#
#=    end=#
#=    merge_pdfs(pdfPaths, "rgflow-$(size_BZ)-$(Wc).pdf"; cleanup=true)=#
#=end=#
