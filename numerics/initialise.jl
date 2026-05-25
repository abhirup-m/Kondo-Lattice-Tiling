display("Initialising libraries and inclusions on $(length(workers())) workers ...")
using PyPlot
using LaTeXStrings
using ProgressMeter
plt.style.use("ggplot")
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 20
@everywhere include("Constants.jl")
@everywhere include("Helpers.jl")
@everywhere include("RgFlow.jl")
@everywhere include("Models.jl")
@everywhere include("Probes.jl")

display("Initialising complete.")
