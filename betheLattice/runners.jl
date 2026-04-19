using Distributed, PyPlot
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
              "Uf" => 15.,
              "Ud" => 5.,
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


function RealCorrRunner()
    Kf = -1e-5
    Kd = -1e-5
    Wf = -0.0
    size = 30
    states = 1000
    Jp_values = collect(0:0.005:0.3)
    #=Jp_values = 10 .^ (-2:0.02:-0.4)=#
    correlationDef = Dict("Sf.Sd" => SpinCorr(1, 3), "Sf.sf" => SpinCorr(1, 5), "Sd.sd" => SpinCorr(3, 7))
    mutInfoDef = Dict("fd" => ([1, 2], [3, 4]), "Ff" => ([1, 2], [5, 6]))
    corrResultsArr = Dict(k => 0 .* Jp_values for k in keys(correlationDef))
    mutInfoResultsArr = Dict(k => 0 .* Jp_values for k in keys(mutInfoDef))
    fig, ax = plt.subplots()
    @time Threads.@threads for (i, Jp) in collect(enumerate(Jp_values))
        results = RealCorr(
                     Kf,
                     Kd,
                     Wf,
                     Jp,
                     size,
                     correlationDef,
                     mutInfoDef,
                     states;
                     loadData=true
            )
        for k in keys(correlationDef)
            corrResultsArr[k][i] = -results[k]
        end
        #=for k in keys(mutInfoDef)=#
        #=    mutInfoResultsArr[k][i] = results[k]=#
        #=end=#
    end
    ax.plot(Jp_values, corrResultsArr["Sf.Sd"], label="-<Sf.Sd>")
    ax.plot(Jp_values, corrResultsArr["Sf.sf"], label="-<Sf.sf>")
    ax.plot(Jp_values, corrResultsArr["Sd.sd"], label="-<Sd.sd>")
    #=ax.plot(Jp_values, mutInfoResultsArr["fd"], label="I2(f:d)")=#
    #=ax.plot(Jp_values, mutInfoResultsArr["Ff"], label="I2(f:f0)")=#
    ax.set_xlabel("J_perp")
    ax.set_ylabel("real-space correlation")
    #=ax.set_xscale("log")=#
    ax.legend()
    fig.savefig("BL-RC-$(size)-$(states).pdf")
end
#=RealCorrRunner()=#


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


function SFRunner()
    size = 31
    states = 1001
    ω = collect(-16:0.1:16)
    σ = 0.1
    Jp_values = 0.0:0.5:1.0
    for Wf in [-0.0, -0.2, -0.3] * PARAMS["Jf"]
    #=for Wf in [-0.0, -0.2, -0.3] * PARAMS["Jf"]=#
        figax1 = plt.subplots()
        figax2 = plt.subplots()
        for (i, Jp) in collect(enumerate(Jp_values))
            Kf = -1e-4 * Jp
            Kd = -1e-4 * Jp
            println("J⟂=$(Jp), Wf=$(Wf)")
            results = RealSpecFunc(
                         Kf,
                         Kd,
                         Wf,
                         Jp,
                         ω,
                         σ,
                         size,
                         states;
                         loadData=true,
                )
            rm("data-iterdiag"; recursive=true, force=true)
            #=println(results["Ad0"])=#
            #=figax2[2].plot(ω, results["Aff"], label="Aff: Jp=$(Jp)")=#
            #=figax2[2].plot(ω, results["Af0"], label="Af0: Jp=$(Jp)")=#
            #=figax1[2].plot(ω, results["Add"], label="Add: Jp=$(Jp)")=#
            #=figax1[2].plot(ω, results["Ad0"], label="Ad0: Jp=$(Jp)")=#
            figax1[2].plot(ω, Norm(results["Ad0"] .+ results["Add"], ω), label="Ad: Jp=$(Jp)")
            figax2[2].plot(ω, Norm(results["Af0"] .+ results["Aff"], ω), label="Af: Jp=$(Jp)")
            figax2[2].plot(ω, Norm(results["Afd"], ω), label="Afd: Jp=$(Jp)")
            #=ax.plot(ω, Norm(results["Ad0"] .+ results["Add"], ω), label="Ad: Jp=$(Jp)")=#
            #=ax.plot(ω, Norm(results["Af0"] .+ results["Aff"], ω), label="Af: Jp=$(Jp)")=#
        end
        for (t, (fig, ax)) in zip(["d", "f"], [figax1, figax2])
            ax.set_xlabel("J_perp")
            ax.set_ylabel("A(ω)")
            ax.legend()
            fig.savefig("BL-SF-$(size)-$(states)-$(Wf)-$(t).pdf")
        end
    end
end
SFRunner()
