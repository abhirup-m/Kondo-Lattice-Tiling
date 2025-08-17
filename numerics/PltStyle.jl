const CMAP = "tab20c"
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
PyPlot.plt.style.use("ggplot")
PyPlot.matplotlib.use("pdf")
rcParams["lines.linewidth"] = 1.5
rcParams["font.size"] = 35
rcParams["image.cmap"] = CMAP
rcParams["text.usetex"] = true
rcParams["font.family"] = "serif"
rcParams["grid.alpha"] = 0
