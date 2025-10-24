CMAP = "Spectral"
cmap_3 = PyPlot.matplotlib.cm.get_cmap(CMAP, 3) # Number of colors is one less than number of bounds
norm_3 = PyPlot.matplotlib.colors.BoundaryNorm([0, 0.33, 0.66, 1], cmap_3.N)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
PyPlot.plt.style.use("ggplot")
PyPlot.matplotlib.use("pdf")
rcParams["lines.linewidth"] = 1.5
rcParams["font.size"] = 15
#=rcParams["image.cmap"] = CMAP=#
rcParams["text.usetex"] = false
rcParams["font.family"] = "serif"
rcParams["grid.alpha"] = 0


#=function HM4D(=#
#=        heatmapF,=#
#=        heatmapS,=#
#=        rangedCouplings,=#
#=        axLab,=#
#=        saveName,=#
#=        cbarLabels,=#
#=        suptitle,=#
#=    )=#
#=    nr, nc = [length(rangedCouplings[axLab[2]]), length(rangedCouplings[axLab[1]])]=#
#=    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(160 * length(nr), 40 * length(nc)))=#
#=    fig.tight_layout()=#
#=    for (col, gX) in enumerate(rangedCouplings[axLab[1]])=#
#=        for (row, gY) in enumerate(rangedCouplings[axLab[2]])=#
#=            if length(rangedCouplings[axLab[1]]) > 1 && length(rangedCouplings[axLab[2]]) > 1=#
#=                ax = axes[row, col]=#
#=            elseif length(rangedCouplings[axLab[1]]) > 1=#
#=                ax = axes[col]=#
#=            elseif length(rangedCouplings[axLab[2]]) > 1=#
#=                ax = axes[row]=#
#=            else=#
#=                ax = axes=#
#=            end=#
#==#
#=            hmData = [heatmapF[(gX, gY, gx, gy)] for (gy, gx) in Iterators.product(rangedCouplings[axLab[4]], rangedCouplings[axLab[3]])]=#
#=            normHm = ifelse((maximum(abs.(hmData)) - minimum(abs.(hmData))) > 100, "log", "linear")=#
#=            hmData[argmax(hmData)] += 1e-5=#
#=            hmData[argmin(hmData)] -= 1e-5=#
#=            flatx = repeat(rangedCouplings[axLab[3]], outer=length(rangedCouplings[axLab[4]]))=#
#=            flaty = repeat(rangedCouplings[axLab[4]], inner=length(rangedCouplings[axLab[3]]))=#
#=            scData = [heatmapS[(gX, gY, x, y)] for (x, y) in zip(flatx, flaty)]=#
#=            normSc = ifelse((maximum(abs.(scData)) - minimum(abs.(scData))) > 100, "log", "linear")=#
#=            scData[argmax(scData)] += 1e-5=#
#=            scData[argmin(scData)] -= 1e-5=#
#=            hm_f = ax.imshow(=#
#=                           hmData,=#
#=                           extent=(extrema(rangedCouplings[axLab[3]])..., extrema(rangedCouplings[axLab[4]])...),=#
#=                           aspect="auto",=#
#=                           origin="lower", =#
#=                           cmap = "magma_r",=#
#=                           norm=normHm,=#
#=                          )=#
#=            sc_s = ax.scatter(flatx,=#
#=                            flaty,=#
#=                            c=scData,=#
#=                            cmap = "magma_r",=#
#=                            norm=normSc,=#
#=                            #=norm="log",=#=#
#=                            s=100=#
#=                           )=#
#=            cb_f = fig.colorbar(hm_f, shrink=0.5, pad=0.12, location="left")=#
#=            cb_f.set_label(cbarLabels[1], labelpad=-80, y=1.1, rotation="horizontal")=#
#=            cb_s = fig.colorbar(sc_s, shrink=0.5, pad=0.05, location="right")=#
#=            cb_s.set_label(cbarLabels[2], labelpad=-30, y=1.2, rotation="horizontal")=#
#=            ax.set_xlabel(L"%$(AXES_LABELS[axLab[3]])")=#
#=            ax.set_ylabel(L"%$(AXES_LABELS[axLab[4]])")=#
#==#
#=            if col == 1=#
#=                ax.set_title(L"%$(AXES_LABELS[axLab[2]])=%$(round(gY, digits=2))", loc="left")=#
#=                ax.text(-0.45, 0.5, "\$$(AXES_LABELS[axLab[2]])=$(round(gY, digits=2))\$", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, bbox=Dict("facecolor"=>"red", "alpha"=>0.1, "boxstyle"=>"Round,pad=0.2"))=#
#=            end=#
#=            if row == 1=#
#=                ax.set_title(L"%$(AXES_LABELS[axLab[1]])=%$(round(gX, digits=2))")=#
#=            end=#
#=        end=#
#=    end=#
#=    fig.suptitle(suptitle, y=1.1)=#
#=    fig.savefig(saveName * ".pdf", bbox_inches="tight")=#
#=    plt.close()=#
#=    return saveName * ".pdf"=#
#=end=#
#==#
#==#
#=function Lines(=#
#=        results,=#
#=        rangedCouplings,=#
#=        axLab,=#
#=        saveName,=#
#=        cbarLabels,=#
#=        suptitle,=#
#=    )=#
#=    xvals = sort(rangedCouplings[axLab[3]])=#
#=    gy = rangedCouplings[axLab[4]][1]=#
#=    nr, nc = [length(rangedCouplings[axLab[2]]), length(rangedCouplings[axLab[1]])]=#
#=    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(40 * length(nc), 40 * length(nr)))=#
#=    fig.tight_layout()=#
#=    for (col, gX) in enumerate(rangedCouplings[axLab[1]])=#
#=        for (row, gY) in enumerate(rangedCouplings[axLab[2]])=#
#=            if length(rangedCouplings[axLab[1]]) > 1 && length(rangedCouplings[axLab[2]]) > 1=#
#=                ax = axes[row, col]=#
#=            elseif length(rangedCouplings[axLab[1]]) > 1=#
#=                ax = axes[col]=#
#=            elseif length(rangedCouplings[axLab[2]]) > 1=#
#=                ax = axes[row]=#
#=            else=#
#=                ax = axes=#
#=            end=#
#=            for (name, vals) in results=#
#=                yvals = [vals[(gX, gY, gx, gy)] for gx in xvals]=#
#=                ax.scatter(xvals, yvals, label=cbarLabels[name])=#
#=            end=#
#=            ax.legend()=#
#=            ax.set_title(latexstring("$(AXES_LABELS[axLab[1]])=$(gX), $(AXES_LABELS[axLab[2]])=$(gY), $(AXES_LABELS[axLab[4]])=$(gy)"))=#
#=        end=#
#=    end=#
#=    fig.suptitle(suptitle, y=1.1)=#
#=    fig.savefig(saveName * ".pdf", bbox_inches="tight")=#
#=    plt.close(fig)=#
#=    return saveName * ".pdf"=#
#=end=#
