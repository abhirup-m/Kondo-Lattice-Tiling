using Distributed

@everywhere using LinearAlgebra, CSV, JLD2, FileIO, CodecZlib, ProgressMeter

@everywhere include("constants.jl")
@everywhere include("helpers.jl")
@everywhere include("rgFlow.jl")
@everywhere include("pltStyle.jl")

global J_val = 0.1
global Jperp_val = 0.0
global lightBandFactor = 2.
global epsilon_d = 0.1 * HOP_T
global mu_c = 0.1 * HOP_T
global Wc_val = 0.
maxSize = 1000
WmaxSize = 500

@everywhere NiceValues(size_BZ) = Dict{Int64, Vector{Float64}}(
                         13 => -1.0 .* [0., 1., 1.5, 1.55, 1.6, 1.61] ./ size_BZ,
                         41 => -1.0 .* [0, 3.5, 7.13, 7.3, 7.5, 7.564, 7.6] ./ size_BZ,
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
        W_val_arr::Vector{Float64},
        Jperp_val::Float64,
        Wc_val::Float64,
        size_BZ::Int64;
        loadData::Bool=false,
    )
    kondoJArrays = Dict{Float64, Array{Float64, 2}}()
    kondoPerpArrays = Dict{Float64, Vector{Float64}}()
    couplings(W_val) = Dict{String, Float64}(
                                      "omega_by_t" => OMEGA_BY_t,
                                      "J_val" => J_val,
                                      "Jperp_val" => Jperp_val,
                                      "W_val" => W_val,
                                      "epsilon_d" => epsilon_d,
                                      "mu_c" => mu_c,
                                      "Wc_val" => Wc_val,
                                     )
    results = @showprogress desc="rg flow" pmap(w -> momentumSpaceRG(size_BZ, couplings(w), lightBandFactor; loadData=loadData), W_val_arr)
    dispersion = results[1][3]
    for (result, W_val) in zip(results, W_val_arr)
        averageKondoScale = sum(abs.(result[1][:, :, 1])) / length(result[1][:, :, 1])
        @assert averageKondoScale > RG_RELEVANCE_TOL
        result[1][:, :, end] .= ifelse.(abs.(result[1][:, :, end]) ./ averageKondoScale .> RG_RELEVANCE_TOL, result[1][:, :, end], 0)
        kondoJArrays[W_val] = result[1][:, :, end]
        kondoPerpArrays[W_val] = result[2]
    end
    return kondoJArrays, kondoPerpArrays, dispersion
end

postProcess(data) = reshape(data, (size_BZ, size_BZ)) .|> abs

node = map2DTo1D(π/2, π/2, size_BZ)
midway = map2DTo1D(3π/4, π/4, size_BZ)
antinode = map2DTo1D(π/1, 0., size_BZ)
size_BZ = 41
W_val_arr = NiceValues(size_BZ)[[1, 4, 5]]
for Wc_val in [0.0, -0.2, -0.3] * J_val
    pdfPaths = String[]
    for Jperp_val in [0., 1.2, 2.] .* J_val
        kondoJArrays, kondoPerpArrays, dispersion = RGFlow(W_val_arr, Jperp_val, Wc_val, size_BZ; loadData=true)

        fig, axes = plt.subplots(nrows=length(W_val_arr), ncols=3, figsize=(14, 9), gridspec_kw=Dict("width_ratios"=> [1,1,0.9]))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        for (i, W_val) in enumerate(W_val_arr)
            hmap = axes[i,1].imshow(postProcess(kondoJArrays[W_val][node, :, end]), aspect="auto", norm=matplotlib[:colors][:LogNorm]())
            axes[i,1].set_xlabel(L"RG flow $\longrightarrow$")
            axes[i,1].set_ylabel(L"$J_f~(k_\mathrm{N}, k)$")
            fig.colorbar(hmap)
            hmap = axes[i,2].imshow(postProcess(kondoJArrays[W_val][antinode, :, end]), aspect="auto", norm=matplotlib[:colors][:LogNorm]())
            axes[i,2].set_xlabel(L"RG flow $\longrightarrow$")
            axes[i,2].set_ylabel(L"$J_f~(k_\mathrm{AN}, k)$")
            fig.colorbar(hmap)
            axes[i,3].plot(kondoPerpArrays[W_val], label=L"$W_f = %$(round(W_val, digits=2))$")
            axes[i,3].set_xlabel(L"RG flow $\longrightarrow$")
            axes[i,3].set_ylabel(L"$J$")
            axes[i,3].legend()
        end
        fig.suptitle(L"Varying $W_f$. $~ J/J_f=%$(Jperp_val / J_val)~ W/J_f=%$(Wc_val/J_val)$", y=0.93)
        savefig("rgflow-$(size_BZ)-$(Jperp_val).pdf", bbox_inches="tight"); PyPlot.close()
        push!(pdfPaths, "rgflow-$(size_BZ)-$(Jperp_val).pdf")
    end
    merge_pdfs(pdfPaths, "rgflow-$(size_BZ)-$(Wc_val).pdf"; cleanup=true)
end
