using Distributed, PyPlot
plt.style.use("ggplot")
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 20
# plt.style.use("seaborn-v0_8")
#=using PlotlyLight=#
#=preset.template.ggplot2!()=#
# using CairoMakie, Measures, Foresight, MakiePublication, MakieThemes
# foresight() |> Makie.set_theme!
# theme_ggthemr(:fresh) |> Makie.set_theme!
# update_theme!(fontsize=20, figure_padding = 0, xlabelsize=40)
# update_theme!(Axis = (
#     xgridvisible = false,
#     ygridvisible = false,
#     xlabelsize = 20,
#     ylabelsize = 20,
#     leftspinevisible=true,
#     bottomspinevisible=true
# ))
# update_theme!(Lines = (
#                        linewidth=2,
#    ))
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
              "Vf" => 1.,
              "Vd" => 0.1,
              "Uf" => 10.,
              "Ud" => 0.,
              "ηf" => 0.,
              "ηd" => 0.,
              "hop_t" => 0.1,
              "scale" => 0.9999,
              "upFactor" => 1.,
             )

function RgRunner()
    WfValues = round.(PARAMS["Jf"] .* (-0.3), digits=3)
    plottables = ["Jf", "Jd", "J⟂", "Kf", "Kd"]
    fig, axes = subplots(ncols=length(plottables), figsize=(3. * length(plottables), 2.2))
    for Wf in WfValues
        params = copy(PARAMS)
        params["J⟂"] = 0.
        params["Wf"] = Wf
        params["Kf"] = 0.
        params["Kd"] = 0.
        rgflows = rgFlow(params, D -> -D/2; loadData=false)
        for (i, key) in enumerate(plottables)
            axes[i].scatter(rgflows["bw"][1:length(rgflows[key])], rgflows[key], label="\$W_f=$(Wf)\$")
            axes[i].set_xscale("log")
            axes[i].set_title(key)
            axes[i].legend()
        end
    end
    fig.tight_layout()
    plt.rcParams["text.usetex"] = true
    savefig("RGF.pdf", bbox_inches="tight")
end
# RgRunner()

function PhaseDiagRunner()
    WfValues = sort(-1 .* 10 .^ (-2:0.05:0.0))
    #=WfValues = sort(-0.0:-0.1:-1.0)=#
    JValues = sort(10 .^ (-3:0.05:-1))
    titleDict = Dict("Jf" => "\$J_f^*\$", "Jd" => "\$J_d\$", "Kf" => "\$-K_f\$", "Kd" => "\$-K_d\$", "J⟂" => "\$J_\\perp\$")
    plottables = ["Jf", "Jd", "J⟂"]

    fig, axes = subplots(ncols=length(plottables), figsize=(6 * length(plottables), 4.5))
    FixedPoint, bareParams = getFixedPointData(WfValues, JValues)
    for (i, key) in enumerate(plottables)
        title = titleDict[key]
        data = FixedPoint[key]
        ax = axes[i]
        hm = ax.pcolormesh(JValues,
                           -WfValues,
                           reshape(data, length(WfValues), length(JValues)),
                           rasterized=true,
                          )
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
    savefig("BL-PD.pdf", bbox_inches="tight")
end
# PhaseDiagRunner()


function RealCorrRunner()
    Kf = -1e-4
    Kd = -1e-4
    size = 11
    states = 998
    Jp_values = 10 .^ (-1.5:0.15:0.5)
    for Wf in -[0.2, 0.3] .* PARAMS["Jf"]
        correlationDef = Dict("Sf.Sd" => SpinCorr(1, 3, 3))
        if -Wf / PARAMS["Jf"] ≥ 0.25
            correlationDef["Sf.sf"] = SpinCorr(1, 5, 5)
        else
            correlationDef["t_ff0"] = [("+-", [1, 5], 0.5), ("+-", [5, 1], 0.5), ("+-", [6, 2], 0.5), ("+-", [2, 6], 0.5),]
        end
        corrResultsArr = Dict(k => 0 .* Jp_values for k in keys(correlationDef))
        fig, ax = plt.subplots()
        p = Progress(length(Jp_values))
        # pl = plot(xscale=:log)
        # pl1 = plot(ylabel="Correlation", xlabel="\$J_\\perp/J_f\$", legend=:right, xscale=:log)
        # pl2 = plot(ylabel="Correlation", xlabel="\$J_\\perp/J_f\$", legend=:right, xscale=:log)
        Threads.@threads for (i, Jp) in collect(enumerate(Jp_values))
            params = Dict("Kf" => Kf * Jp, "Kd" => Kd * Jp, "Wf" => Wf, "J⟂" => Jp, "size" => size, "states" => states, "Uf" => abs(Wf * PARAMS["Uf_by_Wf"]))
            results = RealCorr(
                               params,
                               correlationDef,
                               Dict();
                               loadData=false,
                              )
            for k in keys(correlationDef)
                corrResultsArr[k][i] = -results[k]
            end
            next!(p)
        end
        finish!(p)
        ax.plot(Jp_values ./ PARAMS["Jf"], abs.(corrResultsArr["Sf.Sd"]), label=L"\langle S_f \cdot S_d \rangle", marker="o")
        if -Wf / PARAMS["Jf"] ≥ 0.25
            ax.plot(Jp_values ./ PARAMS["Jf"], abs.(corrResultsArr["Sf.sf"]), label=L"\langle S_f \cdot s_f \rangle", marker="o")
        else
            ax.plot(Jp_values ./ PARAMS["Jf"], corrResultsArr["t_ff0"], label=L"\frac{1}{2}\sum_\sigma \langle f^\dagger_{\sigma} f_{0,\sigma} + \text{h.c.} \rangle", marker="o")
        end
        ax.set_xlabel(L"J_\perp/J_f")
        ax.set_ylabel("Correlation")
        ax.set_xscale("log")
        ax.legend()
        plt.savefig("BL-RC-$(size)-$(states)-$(Wf).pdf", bbox_inches="tight")
    end
end
# RealCorrRunner()


function I2Runner()
    Kf = -1e-4
    Kd = -1e-4
    size = 11
    states = 2099
    Jp_values = 10 .^ (-2.0:0.2:0.5)
    correlationDef = Dict()
    # mutInfoDef = Dict() 
    mutInfoDef = Dict("f-d" => sites -> ([1,2],[3,4]))
    mutInfoResultsArr = Dict(k => 0 .* Jp_values for k in keys(mutInfoDef))
    fig, ax = plt.subplots()
    for Wf in -[0.0, 0.3] .* PARAMS["Jf"]
        p = Progress(length(Jp_values))
        Threads.@threads for (i, Jp) in collect(enumerate(Jp_values))
            params = Dict("Kf" => Kf * Jp, "Kd" => Kd * Jp, "Wf" => Wf, "J⟂" => Jp, "size" => size, "states" => states, "Uf" => abs(Wf * PARAMS["Uf_by_Wf"]))
            results = RealCorr(
                               params,
                               correlationDef,
                               mutInfoDef;
                               loadData=true,
                              )
            for k in keys(mutInfoDef)
                mutInfoResultsArr[k][i] = results[k]
            end
            next!(p)
        end
        finish!(p)
        ax.plot(Jp_values ./ PARAMS["Jf"], mutInfoResultsArr["f-d"], marker="o", label=L"W_f/J_f=%$(round(Wf, digits=2))")
    end
    ax.set_xlabel(L"J_\perp/J_f")
    ax.set_ylabel(L"I_2(d:f)")
    ax.legend()
    ax.set_xscale("log")
    plt.savefig("BL-I2-$(size)-$(states)-$(keys(mutInfoResultsArr) |> first).pdf", bbox_inches="tight")
end
# I2Runner()


function I3Runner()
    Kf = -1e-4
    Kd = -1e-4
    size = 7
    states = 2001
    Wf_values = -1 .* (10 .^ (-0.9:0.3:0.9)) .* PARAMS["Jf"]
    correlationDef = Dict()
    # mutInfoDef = Dict() 
    mutInfoDef = Dict("d-f" => sites -> ([3, 4],[1, 2]), "d-fd0" => sites -> ([3, 4],[1, 2, 7, 8]), "d-d0" => sites -> ([3, 4],[7, 8]))
    # mutInfoDef = Dict("d-d0f0" => sites -> ([3, 4],[5,6,7,8]))
    mutInfoResultsArr = Dict(k => 0 .* Wf_values for k in ["d-d0", "d-f", "d-fd0"])
    for Jp in [2.] .* PARAMS["Jf"]
        fig, ax = plt.subplots()
        p = Progress(length(Wf_values))
        Threads.@threads for (i, Wf) in collect(enumerate(Wf_values))
            params = Dict("Kf" => Kf * Jp, "Kd" => Kd * Jp, "Wf" => Wf, "J⟂" => Jp, "size" => size, "states" => states, "Uf" => abs(Wf * PARAMS["Uf_by_Wf"]))
            results = RealCorr(
                               params,
                               correlationDef,
                               mutInfoDef;
                               loadData=false,
                              )
            for k in keys(mutInfoResultsArr)
                mutInfoResultsArr[k][i] = results[k]
            end
            next!(p)
        end
        finish!(p)
        ax.plot(-Wf_values ./ PARAMS["Jf"], mutInfoResultsArr["d-f"] + mutInfoResultsArr["d-d0"] - mutInfoResultsArr["d-fd0"], marker="o")
        ax.axvline(0.25)
        # ax.plot(-Wf_values ./ PARAMS["Jf"], mutInfoResultsArr["d-f"] + mutInfoResultsArr["d-f0"] - mutInfoResultsArr["d-ff0"], marker="o")
        # ax.plot(-Wf_values ./ PARAMS["Jf"], mutInfoResultsArr["d-f0"] + mutInfoResultsArr["d-d0"] - mutInfoResultsArr["d-d0f0"], marker="o")
        ax.set_xlabel(L"-W_f/J_f")
        ax.set_ylabel(L"\langle S_f \cdot S_d \rangle")
        ax.set_xscale("log")
        plt.savefig("BL-I3-$(size)-$(states)-$(Jp).pdf")
    end
end
# I3Runner()


function CorrPDRunner()
    Kf = -1e-4
    Kd = -1e-4
    size = 13
    states = 2502
    Jp_values = 10 .^ (-2.0:0.05:0)
    Wf_values = -(10 .^ (-1.5:0.05:0.5)) .* PARAMS["Jf"]
    correlationDef = Dict("Sf.Sd" => SpinCorr(1, 3, 3), "Sf.sf" => SpinCorr(1, 5, 5), "Sd.sd" => SpinCorr(3, 7, 7), "t_ff0" => [("+-", [1, 5], 1.0), ("+-", [5, 1], 1.0), ("+-", [2, 6], 1.0), ("+-", [6, 2], 1.0),])
    mutInfoDef = Dict() 
    PD1 = zeros(length(Jp_values), length(Wf_values))
    PD2 = zeros(length(Jp_values), length(Wf_values))
    PD3 = zeros(length(Jp_values), length(Wf_values))
    # p = Progress(length(Jp_values) * length(Wf_values))
    collected = @showprogress @distributed merge for ((i, Wf), (j, Jp)) in collect(Iterators.product(enumerate(Wf_values), enumerate(Jp_values)))
    # Threads.@threads for ((i, Wf), (j, Jp)) in collect(Iterators.product(enumerate(Wf_values), enumerate(Jp_values)))
    params = Dict("Kf" => Kf * Jp, "Kd" => Kd * Jp, "Wf" => Wf, "J⟂" => Jp, "size" => size, "states" => states, "Uf" => 10.)
        results = RealCorr(
                           params,
                           correlationDef,
                           mutInfoDef;
                           loadData=false,
                          )
        Dict((j, i) => (-results["t_ff0"], -results["Sf.sf"], -results["Sf.Sd"]))
        # PD1[j, i] = -results["t_ff0"]
        # PD2[j, i] = -results["Sf.sf"]
        # PD3[j, i] = -results["Sf.Sd"]
        # next!(p)
    end
    for (loc, (r1, r2, r3)) in collected
        PD1[loc...] = r1
        PD2[loc...] = r2
        PD3[loc...] = r3
    end
    # finish!(p)
    maps = ["GnBu", "RdPu", "viridis"]
    fig, ax = plt.subplots()
    params = [[], [], []]
    vals = [[], [], []]
    @time for ((i, Wf), (j, Jp)) in Iterators.product(enumerate(Wf_values), enumerate(Jp_values))
        max = argmax([PD1[j, i], PD2[j, i], PD3[j, i]])
        push!(params[max], (-Wf / PARAMS["Jf"], Jp / PARAMS["Jf"]))
        push!(vals[max], maximum([PD1[j, i], PD2[j, i], PD3[j, i]]))
    end
    for (i, (title, loc, marker)) in enumerate(zip(["\$\\sum_\\sigma\\langle f^\\dagger_\\sigma f^\\dagger_{0,\\sigma} + \\text{h.c.}\\rangle\$", "\$\\langle S_f \\cdot S_{f0}\\rangle\$", "\$\\langle S_f\\cdot S_d\\rangle\$"], ["left", "right", "top"], ["s", "s", "s"]))
        sc = ax.scatter(params[i] .|> first, params[i] .|> last, c=vals[i], cmap=maps[i], marker=marker)
        cb = fig.colorbar(sc, ax=ax, location=loc, fraction=(loc == "left" ? 0.04 : 0.06), pad=(loc=="left" ? 0.15 : 0.05))
        cb.set_label(title)
    end
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("\$-W_f/J_f\$")
    ax.set_ylabel("\$J_\\perp\$")
    plt.savefig("BL-RCPD-$(size)-$(states).pdf", bbox_inches="tight")
end
CorrPDRunner()

function CorrPDRunner()
    Kf = -1e-4
    Kd = -1e-4
    size = 9
    states = 501
    Jp_values = 10 .^ (-2.0:0.1:0)
    Wf_values = -(10 .^ (-2.0:0.1:-0)) .* PARAMS["Jf"]
    correlationDef = Dict("Sf.Sd" => SpinCorr(1, 3, 3), "Sf.sf" => SpinCorr(1, 5, 5), "Sd.sd" => SpinCorr(3, 7, 7), "t_ff0" => [("+-", [1, 5], 1.0), ("+-", [5, 1], 1.0), ("+-", [2, 6], 1.0), ("+-", [6, 2], 1.0),])
    mutInfoDef = Dict() 
    PD1 = zeros(length(Jp_values), length(Wf_values))
    PD2 = zeros(length(Jp_values), length(Wf_values))
    PD3 = zeros(length(Jp_values), length(Wf_values))
    p = Progress(length(Jp_values) * length(Wf_values))
    Threads.@threads for ((i, Wf), (j, Jp)) in collect(Iterators.product(enumerate(Wf_values), enumerate(Jp_values)))
        params = Dict("Kf" => Kf * Jp, "Kd" => Kd * Jp, "Wf" => Wf, "J⟂" => Jp, "size" => size, "states" => states, "Uf" => 10.)
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
    maps = ["GnBu", "RdPu", "plasma"]
    fig, ax = plt.subplots()
    params = [[], [], []]
    vals = [[], [], []]
    @time for ((i, Wf), (j, Jp)) in Iterators.product(enumerate(Wf_values), enumerate(Jp_values))
        max = argmax([PD1[j, i], PD2[j, i], PD3[j, i]])
        push!(params[max], (-Wf / PARAMS["Jf"], Jp / PARAMS["Jf"]))
        push!(vals[max], maximum([PD1[j, i], PD2[j, i], PD3[j, i]]))
    end
    for (i, (title, loc, marker)) in enumerate(zip(["\$\\sum_\\sigma\\langle f^\\dagger_\\sigma f^\\dagger_{0,\\sigma} + \\text{h.c.}\\rangle\$", "\$\\langle S_f \\cdot S_{f0}\\rangle\$", "\$\\langle S_f\\cdot S_d\\rangle\$"], ["left", "right", "top"], ["s", "s", "s"]))
        sc = ax.scatter(params[i] .|> first, params[i] .|> last, c=vals[i], cmap=maps[i], marker=marker)
        cb = fig.colorbar(sc, ax=ax, location=loc, fraction=(loc == "left" ? 0.04 : 0.06), pad=(loc=="left" ? 0.15 : 0.05))
        cb.set_label(title)
    end
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("\$-W_f/J_f\$")
    ax.set_ylabel("\$J_\\perp\$")
    plt.savefig("BL-RCPD-$(size)-$(states).pdf", bbox_inches="tight")
end
# CorrPDRunner()

function SFRunner()
    size = 11
    states = 517
    ω = collect(-10:0.001:10)
    ω_p = -0.1 .< ω .< 2
    σ = Dict(
             "Add" => 0.5,
             "Aff" => 0.5,
             "Ad0" => 0.3, # map(w -> abs(w) > PARAMS["Uf"] / 4 ? 0.2 * clamp(exp((abs(w) - PARAMS["Uf"] / 4)), 0., 3.) : 0.2, ω),
             "Af0" => 0.1,
             "Afd" => 0.04
            )
    Jp_values = [0.0, 0.1, 0.3]
    for Wf in -[0.2, 0.3,] * PARAMS["Jf"]
        fig, ax = plt.subplots(ncols=2, figsize=(14, 6), dpi=600)
        insets = []
        for (t, axi) in zip(["d", "f"], ax)
            axi.set_xlabel(L"\omega")
            axi.set_ylabel(L"A_%$(t)(\omega)")
            # push!(insets, axi.inset_axes([0.4, 0.25, 0.6, 0.5]))
            # axi.axvline([0.], ls="--")
        end
        # ax1 = Axis(f1[1, 1], xlabel=L"\omega", ylabel=L"A(\omega)")
        # ax2 = Axis(f2[1, 1], xlabel=L"\omega", ylabel=L"A(\omega)")
        # ax11 = Axis(f1[1, 1], width=Relative(0.6), height=Relative(0.5), halign=1.0, valign=0.5, title="Full frequency range")
        # ax22 = Axis(f2[1, 1], width=Relative(0.6), height=Relative(0.5), halign=1.0, valign=0.5, title="Full frequency range")
        for (i, Jp) in collect(enumerate(Jp_values))
            Kf = -1e-4 * Jp
            Kd = -1e-4 * Jp
            params = Dict("Kf" => Kf, "Kd" => Kd, "Wf" => Wf, "J⟂" => Jp, "size" => size, "states" => states)

            height = 0.5
            heightTol = 1e-3
            results = RealSpecFunc(
                                   params,
                                   ω,
                                   σ,
                                   height,
                                   heightTol;
                                   loadData=false,
                )
            ax[1].plot(ω, Norm(results["Ad0"] .+ results["Add"], ω), label=L"J_\perp/J_f=%$(round(Jp/PARAMS[\"Jf\"], digits=1))")
            ax[2].plot(ω, Norm(results["Af0"] + results["Aff"], ω), label=L"J_\perp/J_f=%$(round(Jp/PARAMS[\"Jf\"], digits=1))")
            # insets[1].plot(ω, Norm(results["Ad0"] .+ results["Add"], ω))
            # insets[2].plot(ω, Norm(results["Af0"] + results["Aff"], ω))
            # lines!(ax22, ω, Norm(results["Af0"] .+ results["Aff"], ω))
        end
        ax[1].legend()
        ax[2].legend()
        fig.savefig("BL-SF-$(size)-$(states)-$(Wf).pdf", bbox_inches="tight")
        # axislegend(ax1)
        # axislegend(ax2)
        # trim!(f1.layout)
        # trim!(f2.layout)
        # save("BL-SF-$(size)-$(states)-$(Wf)-d.pdf", f1)
        # save("BL-SF-$(size)-$(states)-$(Wf)-f.pdf", f2)
    end
end
# SFRunner()


####### TODO
# Compute real correlation for more sites
# Compute spectral functions with full resolution in "factor" and "steps" and more sites and states
