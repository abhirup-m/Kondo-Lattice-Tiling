function BilayerLEEReal(
        J::Matrix{Float64},
        K::Matrix{Float64},
        Jp::Float64,
        hybrid::Vector{Float64},
        η::Dict{String, Float64},
        impCorr::Dict{String, Float64},
        hop_t::Union{Float64, Dict{String,Float64}},
        layerSpecs::Vector{String},
        hop_step::Dict;
        globalField::Union{Vector{Float64}, Float64}=0.,
        couplingTolerance::Number=1e-15,
        heisenberg::Dict{String, Vector{Float64}}=Dict{String, Vector{Float64}}(),
    )
    #=@assert layerSpecs[1] ≠ layerSpecs[end]=#
    @assert size(J)[1] == size(J)[2] == length(layerSpecs) == length(hybrid)

    if isa(hop_t, Float64)
        hop_t = Dict("f" => hop_t, "d" => hop_t)
    end
    if isa(globalField, Vector)
        @assert length(globalField) == length(layerSpecs) + 2
    else
        globalField = repeat([globalField], length(layerSpecs) + 2)
    end

    #### Indexing convention ####
    # Sf   Sd   γ1   γ2  ...
    # 1,2, 3,4, 5,6, 7,8 ...
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]
    if abs(Jp) > couplingTolerance
        push!(hamiltonian,
              ("nn",  [1, 3], Jp / 4)
             ) # n_{d up, n_{0 up}
        push!(hamiltonian,
              ("nn",  [1, 4], -Jp / 4)
             ) # n_{d up, n_{0 down}
        push!(hamiltonian,
              ("nn",  [2, 3], -Jp / 4)
             ) # n_{d down, n_{0 up}
        push!(hamiltonian,
              ("nn",  [2, 4], Jp / 4)
             ) # n_{d down, n_{0 down}
        push!(hamiltonian,
              ("+-+-",  [1, 2, 4, 3], Jp / 2)
             ) # S_d^+ S_0^-
        push!(hamiltonian,
              ("+-+-",  [2, 1, 3, 4], Jp / 2)
             ) # S_d^- S_0^+
    end

    # kondo terms
    for (i, li) in enumerate(layerSpecs)
        bath_i = 3 + 2 * i
        imp_J = li == "f" ? 1 : 3
        imp_K = li == "f" ? 3 : 1
        for (j, lj) in enumerate(layerSpecs)
            if li ≠ lj
                continue
            end
            bath_j = 3 + 2 * j
            for (g_ij, imp) in zip([J[i,j], K[i,j]], [imp_J, imp_K])
                if abs(g_ij) > couplingTolerance
                    push!(hamiltonian, ("n+-",  [imp, bath_i, bath_j], g_ij / 4))
                    push!(hamiltonian, ("n+-",  [imp, bath_i + 1, bath_j + 1], -g_ij / 4))
                    push!(hamiltonian, ("n+-",  [imp + 1, bath_i, bath_j], -g_ij / 4))
                    push!(hamiltonian, ("n+-",  [imp + 1, bath_i + 1, bath_j + 1], g_ij / 4))
                    push!(hamiltonian, ("+-+-",  [imp, imp + 1, bath_i + 1, bath_j], g_ij / 2))
                    push!(hamiltonian, ("+-+-",  [imp + 1, imp, bath_i, bath_j + 1], g_ij / 2))
                end
            end
        end
        if abs(hybrid[i]) > couplingTolerance
            push!(hamiltonian, ("+-",  [imp_J, bath_i], hybrid[i])) # n_{d up, n_{0 up}
            push!(hamiltonian, ("+-",  [imp_J + 1, bath_i + 1], hybrid[i])) # n_{d up, n_{0 up}
            push!(hamiltonian, ("+-",  [bath_i, imp_J], hybrid[i])) # n_{d up, n_{0 up}
            push!(hamiltonian, ("+-",  [bath_i + 1, imp_J + 1], hybrid[i])) # n_{d up, n_{0 up}
        end
    end

    for (i, t) in enumerate(layerSpecs)
        bath_i = 3 + 2 * i
        j = findnext(==(t), layerSpecs, i+1)
        if !isnothing(j)
            bath_j = 3 + 2 * j
            push!(hamiltonian, ("+-",  [bath_i, bath_j], -hop_t[t]))
            push!(hamiltonian, ("+-",  [bath_j, bath_i], -hop_t[t]))
            push!(hamiltonian, ("+-",  [bath_i + 1, bath_j + 1], -hop_t[t]))
            push!(hamiltonian, ("+-",  [bath_j + 1, bath_i + 1], -hop_t[t]))
        end
    end

    for (site, k) in zip([1, 3], ["f", "d"])
        if abs(η[k]) > couplingTolerance
            push!(hamiltonian, ("n",  [site], -η[k]))
            push!(hamiltonian, ("n",  [site + 1], -η[k]))
        end
    end

    if abs(impCorr["f"]) > couplingTolerance
        push!(hamiltonian, ("nn",  [1, 2], impCorr["f"]))
        push!(hamiltonian, ("n",  [1], -0.5 * impCorr["f"]))
        push!(hamiltonian, ("n",  [2], -0.5 * impCorr["f"]))
    end
    if abs(impCorr["d"]) > couplingTolerance
        push!(hamiltonian, ("nn",  [3, 4], impCorr["d"]))
        push!(hamiltonian, ("n",  [3], -0.5 * impCorr["d"]))
        push!(hamiltonian, ("n",  [4], -0.5 * impCorr["d"]))
    end

    # global magnetic field (to lift any trivial degeneracy)
    for site in 1:(2 + length(layerSpecs))
        if abs(globalField[site]) > couplingTolerance
            push!(hamiltonian, ("n",  [2 * site - 1], globalField[site]/2))
            push!(hamiltonian, ("n",  [2 * site], -globalField[site]/2))
        end
    end
    if length(heisenberg) > 0
        for (t, imp) in zip(["f", "d"], [1, 2])
            sites = [[imp]; 2 .+ findall(==(t), layerSpecs)[1:end-1]]
            for i in 1:(length(sites)-1)
                if abs(heisenberg[t][i]) < couplingTolerance
                    continue
                end
                j = 2 * sites[i] - 1
                k = 2 * sites[i+1] - 1
                push!(hamiltonian, ("nn",  [j, k], heisenberg[t][sites[i]]/4))
                push!(hamiltonian, ("nn",  [j, k + 1], -heisenberg[t][sites[i]]/4))
                push!(hamiltonian, ("nn",  [j + 1, k], -heisenberg[t][sites[i]]/4))
                push!(hamiltonian, ("nn",  [j + 1, k + 1], heisenberg[t][sites[i]]/4))
                push!(hamiltonian, ("+-+-",  [j, j + 1, k + 1, k], heisenberg[t][sites[i]]/2))
                push!(hamiltonian, ("+-+-",  [j + 1, j, k, k + 1], heisenberg[t][sites[i]]/2))
            end
        end
    end

    @assert !isempty(hamiltonian) "Hamiltonian is empty!"

    return hamiltonian
end


function BilayerLEE(
        J::Matrix{Float64},
        K::Matrix{Float64},
        Jp::Float64,
        hybrid::Vector{Float64},
        η::Dict{String, Float64},
        impCorr::Dict{String, Float64},
        Ek::Vector{Float64},
        layerSpecs::Vector{String};
        globalField::Union{Vector{Float64}, Float64}=0.,
        couplingTolerance::Number=1e-15,
        heisenberg::Dict{String, Vector{Float64}}=Dict{String, Vector{Float64}}(),
    )
    #=@assert layerSpecs[1] ≠ layerSpecs[end]=#
    @assert size(J)[1] == size(J)[2] == length(layerSpecs) == length(hybrid)

    #=if isa(hop_t, Float64)=#
    #=    hop_t = Dict("f" => hop_t, "d" => hop_t)=#
    #=end=#
    if isa(globalField, Vector)
        @assert length(globalField) == length(layerSpecs) + 2
    else
        globalField = repeat([globalField], length(layerSpecs) + 2)
    end

    #### Indexing convention ####
    # Sf   Sd   γ1   γ2  ...
    # 1,2, 3,4, 5,6, 7,8 ...
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]
    if abs(Jp) > couplingTolerance
        push!(hamiltonian,
              ("nn",  [1, 3], Jp / 4)
             ) # n_{d up, n_{0 up}
        push!(hamiltonian,
              ("nn",  [1, 4], -Jp / 4)
             ) # n_{d up, n_{0 down}
        push!(hamiltonian,
              ("nn",  [2, 3], -Jp / 4)
             ) # n_{d down, n_{0 up}
        push!(hamiltonian,
              ("nn",  [2, 4], Jp / 4)
             ) # n_{d down, n_{0 down}
        push!(hamiltonian,
              ("+-+-",  [1, 2, 4, 3], Jp / 2)
             ) # S_d^+ S_0^-
        push!(hamiltonian,
              ("+-+-",  [2, 1, 3, 4], Jp / 2)
             ) # S_d^- S_0^+
    end

    # kondo terms
    for (i, li) in enumerate(layerSpecs)
        bath_i = 3 + 2 * i
        imp_J = li == "f" ? 1 : 3
        imp_K = li == "f" ? 3 : 1
        for (j, lj) in enumerate(layerSpecs)
            if li ≠ lj
                continue
            end
            bath_j = 3 + 2 * j
            for (g_ij, imp) in zip([J[i,j], K[i,j]], [imp_J, imp_K])
                if abs(g_ij) > couplingTolerance
                    push!(hamiltonian, ("n+-",  [imp, bath_i, bath_j], g_ij / 4))
                    push!(hamiltonian, ("n+-",  [imp, bath_i + 1, bath_j + 1], -g_ij / 4))
                    push!(hamiltonian, ("n+-",  [imp + 1, bath_i, bath_j], -g_ij / 4))
                    push!(hamiltonian, ("n+-",  [imp + 1, bath_i + 1, bath_j + 1], g_ij / 4))
                    push!(hamiltonian, ("+-+-",  [imp, imp + 1, bath_i + 1, bath_j], g_ij / 2))
                    push!(hamiltonian, ("+-+-",  [imp + 1, imp, bath_i, bath_j + 1], g_ij / 2))
                end
            end
        end
        if abs(hybrid[i]) > couplingTolerance
            push!(hamiltonian, ("+-",  [imp_J, bath_i], hybrid[i])) # n_{d up, n_{0 up}
            push!(hamiltonian, ("+-",  [imp_J + 1, bath_i + 1], hybrid[i])) # n_{d up, n_{0 up}
            push!(hamiltonian, ("+-",  [bath_i, imp_J], hybrid[i])) # n_{d up, n_{0 up}
            push!(hamiltonian, ("+-",  [bath_i + 1, imp_J + 1], hybrid[i])) # n_{d up, n_{0 up}
        end
    end

    for (i, t) in enumerate(layerSpecs)
        bath_i = 3 + 2 * i
        if abs(Ek[i]) > couplingTolerance
            push!(hamiltonian, ("n",  [bath_i], Ek[i]))
            push!(hamiltonian, ("n",  [bath_i + 1], Ek[i]))
        end
    end

    for (site, k) in zip([1, 3], ["f", "d"])
        if abs(η[k]) > couplingTolerance
            push!(hamiltonian, ("n",  [site], -η[k]))
            push!(hamiltonian, ("n",  [site + 1], -η[k]))
        end
    end

    if abs(impCorr["f"]) > couplingTolerance
        push!(hamiltonian, ("nn",  [1, 2], impCorr["f"]))
        push!(hamiltonian, ("n",  [1], -0.5 * impCorr["f"]))
        push!(hamiltonian, ("n",  [2], -0.5 * impCorr["f"]))
    end
    if abs(impCorr["d"]) > couplingTolerance
        push!(hamiltonian, ("nn",  [3, 4], impCorr["d"]))
        push!(hamiltonian, ("n",  [3], -0.5 * impCorr["d"]))
        push!(hamiltonian, ("n",  [4], -0.5 * impCorr["d"]))
    end

    # global magnetic field (to lift any trivial degeneracy)
    for site in 1:(2 + length(layerSpecs))
        if abs(globalField[site]) > couplingTolerance
            push!(hamiltonian, ("n",  [2 * site - 1], globalField[site]/2))
            push!(hamiltonian, ("n",  [2 * site], -globalField[site]/2))
        end
    end
    if length(heisenberg) > 0
        for (t, imp) in zip(["f", "d"], [1, 2])
            sites = [[imp]; 2 .+ findall(==(t), layerSpecs)[1:end-1]]
            for i in 1:(length(sites)-1)
                if abs(heisenberg[t][i]) < couplingTolerance
                    continue
                end
                j = 2 * sites[i] - 1
                k = 2 * sites[i+1] - 1
                push!(hamiltonian, ("nn",  [j, k], heisenberg[t][sites[i]]/4))
                push!(hamiltonian, ("nn",  [j, k + 1], -heisenberg[t][sites[i]]/4))
                push!(hamiltonian, ("nn",  [j + 1, k], -heisenberg[t][sites[i]]/4))
                push!(hamiltonian, ("nn",  [j + 1, k + 1], heisenberg[t][sites[i]]/4))
                push!(hamiltonian, ("+-+-",  [j, j + 1, k + 1, k], heisenberg[t][sites[i]]/2))
                push!(hamiltonian, ("+-+-",  [j + 1, j, k, k + 1], heisenberg[t][sites[i]]/2))
            end
        end
    end

    @assert !isempty(hamiltonian) "Hamiltonian is empty!"

    return hamiltonian
end

