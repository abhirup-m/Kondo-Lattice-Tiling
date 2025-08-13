using PyPlot, PyCall, PDFmerger
rcParams = PyDict(matplotlib["rcParams"])
PyPlot.plt.style.use("ggplot")
rcParams["lines.linewidth"] = 1.5
rcParams["font.size"] = 14
rcParams["image.cmap"] = "plasma_r"
rcParams["text.usetex"] = true
rcParams["font.family"] = "serif"
rcParams["grid.alpha"] = 0
