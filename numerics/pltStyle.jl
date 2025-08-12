using PyPlot, PyCall, PDFmerger
rcParams = PyDict(matplotlib["rcParams"])
PyPlot.plt.style.use("ggplot")
rcParams["lines.linewidth"] = 1.5
rcParams["font.family"] = "sans-serif"
rcParams["font.size"] = 14
rcParams["image.cmap"] = "plasma_r"
