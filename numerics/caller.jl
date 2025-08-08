using Distributed, Random, ProgressMeter, Plots
using LsqFit

@everywhere using LinearAlgebra, CSV, JLD2, FileIO, CodecZlib

@everywhere include("constants.jl")
@everywhere include("helpers.jl")
@everywhere include("rgFlow.jl")

global J_val = 0.1
global Jperp_val = 0.1
global narrowBandFactor = 2.
maxSize = 1000
WmaxSize = 500

@everywhere NiceValues(size_BZ) = Dict{Int64, Vector{Float64}}(
                         13 => -1.0 .* [0., 1., 1.5, 1.55, 1.6, 1.61] ./ size_BZ,
                         41 => -1.0 .* [0, 3.5, 7.13, 7.3, 7.5, 7.564, 7.6] ./ size_BZ,
                         77 => -1.0 .* [0., 7., 14.04, 14.6, 14.99, 15.0] ./ size_BZ,
                        )[size_BZ]
@everywhere pseudogapEnd(size_BZ) = Dict{Int64, Float64}(
                         13 => -1.0 * 1.6 / size_BZ,
                         41 => -1.0 * 7.564 / size_BZ,
                         77 => -1.0 * 14.99 / size_BZ,
                        )[size_BZ]
@everywhere pseudogapStart(size_BZ) = Dict{Int64, Float64}(
                         13 => -1.0 * 1.5 / size_BZ,
                         41 => -1.0 * 7.13 / size_BZ,
                         77 => -1.0 * 14.04 / size_BZ,
                        )[size_BZ]

function RGFlow(
        W_val_arr::Vector{Float64},
        size_BZ::Int64;
        loadData::Bool=false,
    )
    kondoJArrays = Dict{Float64, Array{Float64, 2}}()
    kondoPerpArrays = Dict{Float64, Vector{Float64}}()
    results = @showprogress desc="rg flow" pmap(w -> momentumSpaceRG(size_BZ, OMEGA_BY_t, J_val, Jperp_val, w, narrowBandFactor; loadData=loadData), W_val_arr)
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

size_BZ = 33
antinode = map2DTo1D(π/1, 0., size_BZ)
W_val_arr = [0., -0.2]
kondoJArrays, kondoPerpArrays, dispersion = RGFlow(W_val_arr, size_BZ)
plots = []
for W in W_val_arr
    push!(plots, Plots.heatmap(reshape(kondoJArrays[W][antinode, :], (size_BZ, size_BZ)), label="$(W)"))
end
p = Plots.plot(plots...)
display(p)
