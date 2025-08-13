(@isdefined SAVEDIR) || const SAVEDIR = "saveData"

(@isdefined PLANCK) || const PLANCK = 6.626e-34
(@isdefined MASS_ELE) || const MASS_ELE = 9.109e-31

# define the edges of the Brillouin zone
(@isdefined K_MIN) || const K_MIN = -pi
(@isdefined K_MAX) || const K_MAX = pi

# tolerance for identifying energy contours
(@isdefined TOLERANCE) || const TOLERANCE = 1e-8

(@isdefined GLOBALFIELD) || const GLOBALFIELD = 1e-5

# overlap integral 't' set to 1
(@isdefined HOP_T) || const HOP_T = 1.0

(@isdefined OMEGA_BY_t) || const OMEGA_BY_t = -2.

(@isdefined NODAL_POINTS) || const NODAL_POINTS = [(-π/2, -π/2), (-π/2, π/2), (π/2, π/2), (π/2, -π/2)]
(@isdefined ANTINODAL_POINTS) || const ANTINODAL_POINTS = [(-π, 0.), (0., π), (π, 0.), (0., -π)]
(@isdefined CENTRAL_POINTS) || const CENTRAL_POINTS = [0.5 .* (p1 .+ p2) for (p1, p2) in zip(NODAL_POINTS, ANTINODAL_POINTS)]

(@isdefined FIG_SIZE) || const FIG_SIZE = (700, 350)
(@isdefined RG_RELEVANCE_TOL) || const RG_RELEVANCE_TOL = 1e-2

(@isdefined ALLOWED_PROBES) || const ALLOWED_PROBES = ["scattProb", "kondoCoupNodeMap", "kondoCoupAntinodeMap", "kondoCoupOffNodeMap", "kondoCoupOffAntinodeMap", "spinFlipCorrMap"]
(@isdefined PLOT_SIZE) || const PLOT_SIZE = (400, 300)
(@isdefined ASPECT) || const ASPECT = PLOT_SIZE[2] / PLOT_SIZE[1]

# for testing. 
(@isdefined SIZE_BZ) || const SIZE_BZ = [5, 101, 1001]
(@isdefined KX_VALUES) || const KX_VALUES = [K_MIN, K_MAX, K_MIN, K_MAX, 0]
(@isdefined KY_VALUES) || const KY_VALUES = [K_MIN, K_MIN, K_MAX, K_MAX, 0]
POINTS(num_kspace) = [
    1,
    num_kspace,
    num_kspace^2 - num_kspace + 1,
    num_kspace^2,
    Int(0.5 * num_kspace^2 + 0.5),
]

(@isdefined FS_POINTS) || const FS_POINTS = [3, 7, 9, 11, 15, 17, 19, 23]
(@isdefined FS_POINTS_LEFT) || const FS_POINTS_LEFT = [3, 7, 11, 17, 23]
(@isdefined CENTER_POINTS) || const CENTER_POINTS = [13]
(@isdefined CORNER_POINTS) || const CORNER_POINTS = [1, 5, 21, 25]
(@isdefined OUTSIDE_POINTS) || const OUTSIDE_POINTS = [2, 4, 6, 10, 16, 20, 22, 24]
(@isdefined INSIDE_POINTS) || const INSIDE_POINTS = [8, 12, 14, 18]
(@isdefined PROBE_ENERGIES) || const PROBE_ENERGIES = [0, -2 * HOP_T, 2 * HOP_T, -4 * HOP_T, 4 * HOP_T]
(@isdefined DELTA_ENERGY) || const DELTA_ENERGY = PROBE_ENERGIES[3] - PROBE_ENERGIES[1]
(@isdefined PROCEED_FLAGS) || const PROCEED_FLAGS = fill(1, SIZE_BZ[1]^2, SIZE_BZ[1]^2)
(@isdefined INNER_INDICES_ONE) || const INNER_INDICES_ONE = [2, 3, 6, 7, 8, 11, 12, 16, 17, 18, 22, 23]
(@isdefined EXCLUDED_INDICES_ONE) || const EXCLUDED_INDICES_ONE = [24, 19, 20, 14, 15, 9, 10, 4]
