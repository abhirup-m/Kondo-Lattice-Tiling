using Distributed#, PyPlot
#=using PlotlyLight=#
#=preset.template.ggplot2!()=#
using PyPlot, Measures
include("libs.jl")

@everywhere RG_TOL = 10^(-5)
@everywhere PARAMS = Dict(
              "bw" => 1.0,
              "Jf" => 0.1,
              "Jd" => 0.1,
              "Wf" => nothing,
              "Wd" => -0.0,
              "Kf" => nothing,
              "Kd" => nothing,
              "Vf" => 0.1,
              "Vd" => 0.1,
              "Uf" => 10.,
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
    size = 13
    states = 751
    Jp_values = 10 .^ (-2.0:0.2:0)
    correlationDef = Dict("Sf.Sd" => SpinCorr(1, 3, 3), "Sf.sf" => SpinCorr(1, 5, 5), "Sd.sd" => SpinCorr(3, 7, 7), "t_ff0" => [("+-", [1, 5], 1.0), ("+-", [2, 6], 1.0),])
    mutInfoDef = Dict() 
    mutInfoDef = Dict("d-f" => sites -> ([3, 4],[1, 2]), "d-ff0" => sites -> ([3, 4],[1, 2, 5, 6]), "d-f0" => sites -> ([3, 4],[5, 6]), "f-d0" => sites -> ([1, 2],[7, 8]), "f-dd0" => sites -> ([1, 2],[3, 4, 7, 8]))
    corrResultsArr = Dict(k => 0 .* Jp_values for k in keys(correlationDef))
    mutInfoResultsArr = Dict(k => 0 .* Jp_values for k in keys(mutInfoDef))
    plots1 = []
    plots2 = []
    for Wf in [-0.0, -0.3] .* PARAMS["Jf"]
        p = Progress(length(Jp_values))
        # pl = plot(xscale=:log)
        pl1 = plot(ylabel="Correlation", xlabel="\$J_\\perp/J_f\$", legend=:right, xscale=:log)
        pl2 = plot(ylabel="Correlation", xlabel="\$J_\\perp/J_f\$", legend=:right, xscale=:log)
        Threads.@threads for (i, Jp) in collect(enumerate(Jp_values))
            params = Dict("Kf" => Kf * Jp, "Kd" => Kd * Jp, "Wf" => Wf, "J⟂" => Jp, "size" => size, "states" => states)
            results = RealCorr(
                               params,
                               correlationDef,
                               mutInfoDef;
                               loadData=true,
                              )
            for k in keys(correlationDef)
                corrResultsArr[k][i] = -results[k]
            end
            for k in keys(mutInfoDef)
                mutInfoResultsArr[k][i] = results[k]
            end
            next!(p)
        end
        finish!(p)
        scatter!(pl1, Jp_values ./ PARAMS["Jf"], abs.(corrResultsArr["Sf.Sd"]), label="\$\\langle S_f \\cdot S_d \\rangle\$")
        if -Wf / PARAMS["Jf"] ≥ 0.25
            scatter!(pl1, Jp_values ./ PARAMS["Jf"], abs.(corrResultsArr["Sf.sf"]), label="\$\\langle S_f \\cdot s_f \\rangle\$")
        else
            scatter!(pl1, Jp_values ./ PARAMS["Jf"], corrResultsArr["t_ff0"], label="\$\\langle d^\\dagger_{\\sigma} f_{\\sigma} + \\text{h.c.} \\rangle\$")
        end
        scatter!(pl2, Jp_values ./ PARAMS["Jf"], mutInfoResultsArr["d-f0"], label="\$I_2(d:f_0)\$")
        scatter!(pl2, Jp_values ./ PARAMS["Jf"], -(mutInfoResultsArr["d-f"] + mutInfoResultsArr["f-d0"] - mutInfoResultsArr["f-dd0"]), label="\$I_3(f:d:d_0)\$")
        scatter!(pl2, Jp_values ./ PARAMS["Jf"], -(mutInfoResultsArr["d-f"] + mutInfoResultsArr["d-f0"] - mutInfoResultsArr["d-ff0"]), label="\$I_3(d:f:f_0)\$")
        push!(plots1, pl1)
        push!(plots2, pl2)
    end
    p = plot(plots1..., size=(1000, 400))
    Plots.pdf(p, "BL-RC-$(size)-$(states)")
    p = plot(plots2..., size=(1000, 400))
    Plots.pdf(p, "BL-I2-$(size)-$(states)")
end
RealCorrRunner()

function CorrPDRunner()
    Kf = -1e-4
    Kd = -1e-4
    size = 31
    states = 2998
    Jp_values = 10 .^ (-2.0:0.05:0)
    Wf_values = (-10 .^ (-2.0:0.05:-0)) .* PARAMS["Jf"]
    correlationDef = Dict("Sf.Sd" => SpinCorr(1, 3, 3), "Sf.sf" => SpinCorr(1, 5, 5), "Sd.sd" => SpinCorr(3, 7, 7), "t_ff0" => [("+-", [1, 5], 1.0), ("+-", [5, 1], 1.0), ("+-", [2, 6], 1.0), ("+-", [6, 2], 1.0),])
    mutInfoDef = Dict() 
    PD1 = zeros(length(Jp_values), length(Wf_values))
    PD2 = zeros(length(Jp_values), length(Wf_values))
    PD3 = zeros(length(Jp_values), length(Wf_values))
    p = Progress(length(Jp_values) * length(Wf_values))
    Threads.@threads for ((i, Wf), (j, Jp)) in collect(Iterators.product(enumerate(Wf_values), enumerate(Jp_values)))
        params = Dict("Kf" => Kf * Jp, "Kd" => Kd * Jp, "Wf" => Wf, "J⟂" => Jp, "size" => size, "states" => states)
        results = RealCorr(
                           params,
                           correlationDef,
                           mutInfoDef;
                           loadData=false,
                          )
        PD1[j, i] = -results["t_ff0"]
        PD2[j, i] = -results["Sf.sf"]
        PD3[j, i] = -results["Sf.Sd"]
        next!(p)
    end
    finish!(p)
    #=maps = ["cool", "autumn", "winter"]=#
    maps = ["viridis", "inferno", "cividis"]
    fig, ax = plt.subplots()
    params = [[], [], []]
    vals = [[], [], []]
    @time for ((i, Wf), (j, Jp)) in Iterators.product(enumerate(Wf_values), enumerate(Jp_values))
        max = argmax([PD1[j, i], PD2[j, i], PD3[j, i]])
        push!(params[max], (-Wf, Jp))
        push!(vals[max], maximum([PD1[j, i], PD2[j, i], PD3[j, i]]))
    end
    #=bars = [plot(cb=:left), plot(cb=:top), plot(cb=:right)]=#
    for (i, (title, loc, marker)) in enumerate(zip(["\$\\sum_\\sigma\\langle f^\\dagger_\\sigma f^\\dagger_{0,\\sigma} + \\text{h.c.}\\rangle\$", "\$\\sum_\\sigma\\langle S_f \\cdot s_f\\rangle\$", "\$\\langle S_f\\cdot S_d\\rangle\$"], ["left", "right", "top"], ["o", "h", "s"]))
        sc = ax.scatter(params[i] .|> first, params[i] .|> last, c=vals[i], cmap=maps[i], marker=marker)
        cb = fig.colorbar(sc, ax=ax, location=loc, fraction=(loc == "top" ? 0.06 : 0.04), pad=(loc=="left" ? 0.15 : 0.05))
        cb.set_label(title)
        #=push!(bars, heatmap(rand(2,2), clims=(0,1), framestyle=:none, c=maps[i], cbar=true, lims=extrema(sets[i])))=#
    end
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("\$-W_f/J_f\$")
    ax.set_ylabel("\$J_\\perp\$")
    #=l = @layout [=#
    #=     a{0.1w} [b{0.1h}=#
    #=              c{0.9h}  ] d{0.1w}=#
    #=]=#
    #=p2 = heatmap(rand(2,2), clims=(0,1), framestyle=:none, c=cmap, cbar=true, lims=(-1,0))=#
    #=p1 = heatmap(-Wf_values ./ PARAMS["Jf"], Jp_values, PD1, yscale=:log10, xscale=:log10)=#
    #=p2 = heatmap(-Wf_values ./ PARAMS["Jf"], Jp_values, PD2, yscale=:log10, xscale=:log10)=#
    #=p3 = heatmap(-Wf_values ./ PARAMS["Jf"], Jp_values, PD3, yscale=:log10, xscale=:log10)=#
    #=p = plot(p1, p2, p3)=#
    #=Plots.pdf(plot(bars[1], bars[2], p, bars[3], layout=l), "BL-RCPD-$(size)-$(states)")=#
    plt.savefig("BL-RCPD-$(size)-$(states).pdf", bbox_inches="tight")
end
CorrPDRunner();


function SFRunner()
    size = 11
    states = 1001
    ω = collect(-2:0.001:2)
    σ = 0.04
    Jp_values = [0.0, 0.06, 0.08, 0.3]
    #=for Wf in [-0.0, -0.1, -0.2, -0.3] * PARAMS["Jf"]=#
    for Wf in [-0.0,] * PARAMS["Jf"]
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
#=SFRunner()=#


####### TODO
# Compute real correlation for more sites
