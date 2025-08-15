using Distributed, CSV, CodecZlib, Combinatorics, Serialization, PDFmerger, PyPlot
using Fermions

@everywhere using FileIO, JSON3, LinearAlgebra, ProgressMeter, JLD2

@everywhere include("Constants.jl")
@everywhere include("Helpers.jl")
@everywhere include("RgFlow.jl")
include("PhaseDiagram.jl")
include("Probes.jl")
include("PltStyle.jl")

global kondo_f = 0.1
global kondo_perp = 0.0
global lightBandFactor = 2.
global epsilon_f = 0.5 * HOP_T
global mu_c = 0.5 * HOP_T
global Wc = 0.5 * HOP_T
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
        kondo_perp::Float64,
        Wc::Float64,
        size_BZ::Int64;
        loadData::Bool=false,
    )
    kondoJArrays = Dict{Float64, Array{Float64, 2}}()
    kondoPerpArrays = Dict{Float64, Vector{Float64}}()
    couplings(Wf) = Dict{String, Float64}(
                                      "omega_by_t" => OMEGA_BY_t,
                                      "kondo_f" => kondo_f,
                                      "kondo_perp" => kondo_perp,
                                      "Wf" => Wf,
                                      "epsilon_f" => epsilon_f,
                                      "mu_c" => mu_c,
                                      "Wc" => Wc,
                                      "lightBandFactor" => lightBandFactor,
                                     )
    results = @showprogress desc="rg flow" pmap(Wf -> momentumSpaceRG(size_BZ, couplings(Wf); loadData=loadData), Wf_arr)
    dispersion = results[1][3]
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
        kondo_perpLims::NTuple{2, Float64}, 
        kondo_perpSpacing::Float64,
        WfLims::NTuple{2, Float64}, 
        WfSpacing::Float64;
        loadData::Bool=false,
        fillPG::Bool=false,
    )
    @assert minimum(kondo_perpLims) > 0
    epsilon_f = 0.5 * HOP_T
    kondo_f_arr = (0.1:0.1:0.2) .* HOP_T
    Wc_arr = (0.0:-0.2:-0.2) .* HOP_T
    WfVals = collect(minimum(WfLims):WfSpacing:maximum(WfLims))
    JperpVals = collect(minimum(kondo_perpLims):kondo_perpSpacing:maximum(kondo_perpLims))
    fig, axes = PyPlot.subplots(nrows=length(kondo_f_arr), ncols=length(Wc_arr), figsize=(12 * length(Wc_arr), 12 * length(kondo_f_arr)))
    for (i, kondo_f) in enumerate(kondo_f_arr)
        for (j, Wc) in enumerate(Wc_arr)
            phaseDiagram_Jf, phaseDiagram_J = PhaseDiagram(size_BZ, kondo_perpLims, kondo_perpSpacing, WfLims, WfSpacing, 
                                   Dict("omega_by_t"=>OMEGA_BY_t, "Wc"=>Wc, "kondo_f"=>kondo_f, 
                                        "epsilon_f"=>epsilon_f, "mu_c"=>mu_c, "lightBandFactor"=>lightBandFactor);
                                   loadData=loadData, fillPG=fillPG
                                  )
            hmap = axes[i,j].imshow(phaseDiagram_Jf, aspect="auto", origin="lower", 
                                    extent=(minimum(WfLims), 
                                            maximum(WfLims), 
                                            minimum(kondo_perpLims), 
                                            maximum(kondo_perpLims)
                                           ),
                                    cmap = plt.get_cmap(CMAP, 3),
                                   )
            for flag in [2, 3]
                pairs = findall(==(flag), phaseDiagram_J)[1:10:end]
                axes[i,j].scatter([WfVals[p[1]] for p in pairs], [JperpVals[p[2]] for p in pairs], marker=["x","o"][flag - 1], color="white", s=6)
            end

            if j == 1
                axes[i, 1].set_ylabel(L"J")
                axes[i, j].text(-0.45, 0.5, "\$J_f=$(round(kondo_f, digits=2))\$", horizontalalignment="center", verticalalignment="center", transform=axes[i,j].transAxes, bbox=Dict("facecolor"=>"red", "alpha"=>0.1, "boxstyle"=>"Round,pad=0.2"))
            end
            if i == 1
                axes[i, j].text(0.5, 1.2, "\$W=$(round(Wc, digits=2))\$", horizontalalignment="center", verticalalignment="center", transform=axes[i,j].transAxes, bbox=Dict("facecolor"=>"red", "alpha"=>0.1, "boxstyle"=>"Round,pad=0.2"))
            end
            if i == length(kondo_f_arr)
                axes[length(kondo_f_arr), j].set_xlabel(L"W_f")
            end
            if j == length(Wc_arr)
                fig.colorbar(hmap, ticks=[0, 0.5, 1])
            end
        end
    end
    fig.tight_layout()
    savefig("PD-$(size_BZ)-$(epsilon_f)-$(mu_c)-$(lightBandFactor).pdf", bbox_inches="tight"); PyPlot.close()
end


PhaseDiagram(13, (0.01, 1.0), 0.005, (-0.01, -1.0), 0.005; loadData=false)

#=postProcess(data) = reshape(data, (size_BZ, size_BZ)) .|> abs=#
#==#
#=node = map2DTo1D(π/2, π/2, size_BZ)=#
#=midway = map2DTo1D(3π/4, π/4, size_BZ)=#
#=antinode = map2DTo1D(π/1, 0., size_BZ)=#
#=size_BZ = 41=#
#=Wf_arr = NiceValues(size_BZ)[[1, 7, 8]]=#
#=for Wc in [0.0, -0.2, -0.3] * kondo_f=#
#=    pdfPaths = String[]=#
#=    for kondo_perp_i in [0., 1.2, 2.] .* kondo_f=#
#=        kondoJArrays, kondoPerpArrays, dispersion = RGFlow(Wf_arr, kondo_perp_i, Wc, size_BZ; loadData=true)=#
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
#=        fig.suptitle(L"Varying $W_f$. $~ J/J_f=%$(kondo_perp_i / kondo_f)~ W/J_f=%$(Wc/kondo_f)$", y=0.93)=#
#=        path = "rgflow-$(size_BZ)-$(Wc)-$(kondo_perp_i).pdf"=#
#=        savefig(path, bbox_inches="tight"); PyPlot.close()=#
#=        push!(pdfPaths, path)=#
#=    end=#
#=    merge_pdfs(pdfPaths, "rgflow-$(size_BZ)-$(Wc).pdf"; cleanup=true)=#
#=end=#
