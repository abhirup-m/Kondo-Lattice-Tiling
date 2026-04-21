using Distributed#, PyPlot
#=using PlotlyLight=#
#=preset.template.ggplot2!()=#
using Plots, Measures
include("libs.jl")
#=const PLOTS_DEFAULTS = Dict(:theme => :wong, :fontfamily => "Computer Modern",=#
#=    :label => nothing, :dpi => 600=#
#=)=#
theme(:vibrant)
# scalefontsizes(1.5)
default(thickness_scaling=1.4, legendfontsize=12, margin=-2mm, widen=true)

@everywhere RG_TOL = 10^(-5)
@everywhere PARAMS = Dict(
              "bw" => 1.0,
              "Jf" => 0.1,
              "Jd" => 0.1,
              "Wf" => nothing,
              "Wd" => -0.0,
              "Kf" => nothing,
              "Kd" => nothing,
              "Vf" => 0.0,
              "Vd" => 0.0,
              "Uf" => 15.,
              "Ud" => 0.,
              "ηf" => 0.,
              "ηd" => 0.,
              "hop_t" => 0.1,
              "scale" => 0.999,
              "upFactor" => 1.,
             )

function RgRunner()
    WfValues = round.(PARAMS["Jf"] .* (0., -0.26, -0.28, -0.3), digits=3)
    plottables = ["Jf", "Jd",]
    fig, axes = subplots(ncols=length(plottables), figsize=(3. * length(plottables), 2.2))
    for Wf in WfValues
        params = copy(PARAMS)
        params["J⟂"] = 0.
        params["Wf"] = Wf
        params["Kf"] = 0.
        params["Kd"] = 0.
        rgflows = rgFlow(params, D -> -D/2; loadData=false)
        for (i, key) in enumerate(plottables)
            axes[i].plot(rgflows["bw"], rgflows[key] / rgflows[key][1], label="\$W_f=$(Wf)\$")
            axes[i].set_xscale("log")
            axes[i].set_title(key)
            axes[i].legend()
        end
    end
    fig.tight_layout()
    plt.rcParams["text.usetex"] = true
    savefig("RGF.pdf", bbox_inches="tight")
end
#=RgRunner()=#

function PhaseDiagRunner()
    WfValues = sort(-1 .* 10 .^ (-2:0.05:0.0))
    #=WfValues = sort(-0.0:-0.1:-1.0)=#
    JValues = sort(10 .^ (-3:0.1:-1))
    titleDict = Dict("Jf" => "\$J_f^*/J_f\$", "Jd" => "\$J_d\$", "Kf" => "\$-K_f\$", "Kd" => "\$-K_d\$", "J⟂" => "\$J_\\perp\$")
    plottables = ["Jf", "Jd", "J⟂"]
    #=plottables = ["Kf", "Kd",]=#
    for (j, (Kf, Kd)) in enumerate([(0., 0.)])
    #=for (j, (Kf, Kd)) in enumerate([(-1e-5, -1e-5)])=#
    #=for (j, (Kf, Kd)) in enumerate([(0., 0.), (-1e-5, -1e-5)])=#
        fig, axes = subplots(ncols=length(plottables), figsize=(2.6 * length(plottables), 2))
        FixedPoint, bareParams = getFixedPointData(WfValues, JValues, Kf, Kd; loadData=true)
        for (i, key) in enumerate(plottables)
            title = titleDict[key]
            data = FixedPoint[key]
            #=data = abs.(FixedPoint[key] ./ bareParams[key])=#
            #=println(bareParams[key], key)=#
            #=data[sortperm(data)[end]] += 1e-6=#
            ax = axes[i]
            #=x = =#
            hm = ax.pcolormesh(JValues,
                               -WfValues,
                               reshape(data, length(WfValues), length(JValues)),
                               #=edgecolors="none",=#
                               rasterized=true,
                               #=shading="nearest",=#
                              )
            #=ax.axhline(0.25 * PARAMS["Jf"], 0, 1)=#
            #=hm = ax.imshow(reshape(data, length(WfValues), length(JValues)), origin="lower", extent=(extrema(JValues)..., -1 .* extrema(WfValues)...), cmap="inferno", aspect="auto")#, norm=matplotlib.colors.LogNorm())=#
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("\$J_\\perp\$")
            ax.set_ylabel("\$-Wf\$")
            ax.set_title("Fixed-point $(title)", pad=10)
            yPoints = repeat(-WfValues, outer=length(JValues))
            xPoints = repeat(JValues, inner=length(WfValues))
            fig.colorbar(hm)
        end
        fig.tight_layout()
        plt.rcParams["text.usetex"] = true
        savefig("bilayerHubbard_$(j).pdf", bbox_inches="tight")
    end
end
#=PhaseDiagRunner()=#


function RealCorrRunner()
    Kf = -1e-4
    Kd = -1e-4
    size = 10
    states = 4000
    Jp_values = collect(0.00:0.01:0.15)
    # Jp_values = 10 .^ (-4:0.02:0.0)
    correlationDef = Dict("Sf.Sd" => SpinCorr(1, 3), "Sf.sf" => SpinCorr(1, 5), "Sd.sd" => SpinCorr(3, 7))
    mutInfoDef = Dict("fd" => ([1, 2], [3, 4]), "Ff" => ([1, 2], [5, 6]))
    corrResultsArr = Dict(k => 0 .* Jp_values for k in keys(correlationDef))
    mutInfoResultsArr = Dict(k => 0 .* Jp_values for k in keys(mutInfoDef))
    for Wf in [-0.03]
        p = Progress(length(Jp_values))
        # pl = plot(xscale=:log)
        pl = plot(ylabel="Correlation", xlabel="\$J_\\perp/J_f\$", legend=:right)
        @showprogress Threads.@threads for (i, Jp) in collect(enumerate(Jp_values))
            params = Dict("Kf" => Kf * Jp, "Kd" => Kd * Jp, "Wf" => Wf, "J⟂" => Jp, "size" => size, "states" => states)
            results = RealCorr(
                               params,
                               correlationDef,
                               mutInfoDef;
                               loadData=true
                              )
            for k in keys(correlationDef)
                corrResultsArr[k][i] = -results[k]
            end
            next!(p)
        end
        finish!(p)
        scatter!(pl, Jp_values ./ PARAMS["Jf"], corrResultsArr["Sf.Sd"], label="\$\\langle S_f \\cdot S_d \\rangle\$")
        scatter!(pl, Jp_values ./ PARAMS["Jf"], corrResultsArr["Sf.sf"], label="\$\\langle S_f \\cdot s_f \\rangle\$")
        # plot!(pl, Jp_values ./ PARAMS["Jf"], corrResultsArr["Sd.sd"], label="\$\\langle S_d \\cdot s_d \\rangle\$")
        Plots.pdf("BL-RC-$(size)-$(states)-$(Wf)")
    end
end
# RealCorrRunner()



function SFRunner()
    size = 11
    states = 501
    ω = collect(-2:0.001:2)
    σ = 0.04
    Jp_values = [0.0, 0.06, 0.08, 0.3]
    for Wf in [-0.0, -0.1, -0.2, -0.3] * PARAMS["Jf"]
    # for Wf in [-0.0,] * PARAMS["Jf"]
        p1 = plot(xlabel="\$\\omega\$", ylabel="\$A(ω)\$")
        p2 = plot(xlabel="\$\\omega\$", ylabel="\$A(ω)\$")
        for (i, Jp) in collect(enumerate(Jp_values))
            Kf = -1e-4 * Jp
            Kd = -1e-4 * Jp
            params = Dict("Kf" => Kf, "Kd" => Kd, "Wf" => Wf, "J⟂" => Jp, "size" => size, "states" => states)
            results = RealSpecFunc(
                                   params,
                                   ω,
                                   σ;
                                   loadData=true,
                )
            rm("data-iterdiag"; recursive=true, force=true)
            #=println(results["Ad0"])=#
            #=figax2[2].plot(ω, results["Aff"], label="Aff: Jp=$(Jp)")=#
            #=figax2[2].plot(ω, results["Af0"], label="Af0: Jp=$(Jp)")=#
            #=figax1[2].plot(ω, results["Add"], label="Add: Jp=$(Jp)")=#
            #=figax1[2].plot(ω, results["Ad0"], label="Ad0: Jp=$(Jp)")=#
            plot!(p1, ω, Norm(results["Ad0"] .+ 0 .* results["Add"], ω), label="\$J_\\perp/J_f=$(round(Jp/PARAMS["Jf"], digits=1))\$")
            #=plot!(p2, ω, results["Afd"] - results["Af0"] - results["Ad0"], label="\$J_\\perp/J_f=$(round(Jp/PARAMS["Jf"], digits=1))\$")=#
            plot!(p2, ω, Norm(results["Af0"], ω), label="\$J_\\perp/J_f=$(round(Jp/PARAMS["Jf"], digits=1))\$")
            #=plot!(p2, ω, Norm(results["Af0"] .+ 0 .* results["Aff"] .+ 0 .* results["Afd"], ω), label="\$J_\\perp/J_f=$(round(Jp/PARAMS["Jf"], digits=1))\$")=#
            #=plot!(p2, ω, results["Afd"], label="Afd: Jp=$(Jp)")=#
            #=ax.plot(ω, Norm(results["Ad0"] .+ results["Add"], ω), label="Ad: Jp=$(Jp)")=#
            #=ax.plot(ω, Norm(results["Af0"] .+ results["Aff"], ω), label="Af: Jp=$(Jp)")=#
        end
        Plots.pdf(p1, "BL-SF-$(size)-$(states)-$(Wf)-d")
        Plots.pdf(p2, "BL-SF-$(size)-$(states)-$(Wf)-f")
    end
end
SFRunner()


####### TODO
# Compute real correlation for more sites
