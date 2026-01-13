function BilayerKondo(
        kondoGammaPlus::Matrix{Float64},
        kondoGammaMinus::Matrix{Float64},
        kondoPerp::Number,
        hoppingCbath::Number;
        globalField::Number=0,
        epsilonF::Number=0,
        cbathChemPot::Number=0,
        couplingTolerance::Number=1e-15,
    )
    @assert size(kondoGammaPlus) == size(kondoGammaMinus)
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]

    cbathUp = 2 .+ range(1, length=size(kondoGammaPlus)[1], step=6)
    plusUp = 2 .+ cbathUp
    minusUp = 2 .+ plusUp

    # kondo terms
    for index1 in 1:size(kondoGammaPlus)[1]
        for (kondoGamma, upIndices) in zip([kondoGammaPlus, kondoGammaMinus], [plusUp, minusUp])
            nonZeroTerms = findall(J -> abs(J) > couplingTolerance, kondoGamma[index1, 1:index1])
            if !isempty(nonZeroTerms)
                for (J_kq, index2) in zip(kondoGamma[index1, nonZeroTerms], (1:index1)[nonZeroTerms])
                    push!(hamiltonian, 
                          ("n+-",  [1, upIndices[index1], upIndices[index2]], J_kq/4)) # n_{d up, n_{0 up}
                    push!(hamiltonian, 
                          ("n+-",  [1, upIndices[index1]+1, upIndices[index2]+1], -J_kq/4)) # n_{d up, n_{0 down}
                    push!(hamiltonian, 
                          ("n+-",  [2, upIndices[index1], upIndices[index2]], -J_kq/4)) # n_{d down, n_{0 up}
                    push!(hamiltonian, 
                          ("n+-",  [2, upIndices[index1]+1, upIndices[index2]+1], J_kq/4)) # n_{d down, n_{0 down}
                    push!(hamiltonian, 
                          ("+-+-",  [1, 2, upIndices[index1]+1, upIndices[index2]], J_kq/2)) # S_d^+ S_0^-
                    push!(hamiltonian, 
                          ("+-+-",  [2, 1, upIndices[index1], upIndices[index2]+1], J_kq/2)) # S_d^- S_0^+
                end
            end
        end
        if index1 == 1
            push!(hamiltonian, 
                  ("n+-",  [1, 3, 3], kondoPerp/4)) # n_{d up, n_{0 up}
            push!(hamiltonian, 
                  ("n+-",  [1, 4, 4], -kondoPerp/4)) # n_{d up, n_{0 down}
            push!(hamiltonian, 
                  ("n+-",  [2, 3, 3], -kondoPerp/4)) # n_{d down, n_{0 up}
            push!(hamiltonian, 
                  ("n+-",  [2, 4, 4], kondoPerp/4)) # n_{d down, n_{0 down}
            push!(hamiltonian, 
                  ("+-+-",  [1, 2, 4, 3], kondoPerp/2)) # S_d^+ S_0^-
            push!(hamiltonian, 
                  ("+-+-",  [2, 1, 3, 4], kondoPerp/2)) # S_d^- S_0^+
        else
            push!(hamiltonian, 
                  ("+-",  [cbathUp[index1], cbathUp[index1 - 1]], -hoppingCbath)) # S_d^- S_0^+
            push!(hamiltonian, 
                  ("+-",  [cbathUp[index1 - 1], cbathUp[index1]], -hoppingCbath)) # S_d^- S_0^+
            push!(hamiltonian, 
                  ("+-",  [cbathUp[index1] + 1, cbathUp[index1 - 1] + 1], -hoppingCbath)) # S_d^- S_0^+
            push!(hamiltonian, 
                  ("+-",  [cbathUp[index1 - 1] + 1, cbathUp[index1] + 1], -hoppingCbath)) # S_d^- S_0^+
        end
        push!(hamiltonian, 
              ("n",  [cbathUp[index1]], -cbathChemPot)
             )
        push!(hamiltonian, 
              ("n",  [cbathUp[index1] + 1], -cbathChemPot)
             )

    end


    # global magnetic field (to lift any trivial degeneracy)
    if abs(globalField) > couplingTolerance
        for site in 1:(1 + 3 * size(kondoGammaMinus)[1])
            push!(hamiltonian, ("n",  [2 * site - 1], globalField/2))
            push!(hamiltonian, ("n",  [2 * site], -globalField/2))
        end
    end

    if abs(epsilonF) > couplingTolerance
        push!(hamiltonian, ("n",  [1], epsilonF))
        push!(hamiltonian, ("n",  [2], epsilonF))
    end

    @assert !isempty(hamiltonian) "Hamiltonian is empty!"

    return hamiltonian
end
export BilayerKondo


function BilayerKondo(
        kondoGamma::Matrix{Float64},
        kondoPerp::Number;
        globalField::Number=0,
        epsilonF::Number=0,
        impU::Number=0,
        cbathChemPot::Number=0,
        couplingTolerance::Number=1e-15,
    )
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]

    #=upIndices = 2 .+ range(1, length=size(kondoGamma)[1], step=2)=#
    #=cbathUp = size(kondoGamma)[1] > 0 ? upIndices[end] + 2 : 3=#
    upIndices = 4 .+ range(1, length=size(kondoGamma)[1], step=2)
    cbathUp = 3

    # kondo terms
    for (index1, Jk_arr) in enumerate(eachrow(kondoGamma))
        for (index2, J_kq) in enumerate(Jk_arr)
            if abs(J_kq) < couplingTolerance
                continue
            end
            push!(hamiltonian,
                  ("n+-",  [1, upIndices[index1], upIndices[index2]], J_kq/4)
                 ) # n_{d up, n_{0 up}
            push!(hamiltonian,
                  ("n+-",  [1, upIndices[index1]+1, upIndices[index2]+1], -J_kq/4)
                 ) # n_{d up, n_{0 down}
            push!(hamiltonian,
                  ("n+-",  [2, upIndices[index1], upIndices[index2]], -J_kq/4)
                 ) # n_{d down, n_{0 up}
            push!(hamiltonian,
                  ("n+-",  [2, upIndices[index1]+1, upIndices[index2]+1], J_kq/4)
                 ) # n_{d down, n_{0 down}
            push!(hamiltonian,
                  ("+-+-",  [1, 2, upIndices[index1]+1, upIndices[index2]], J_kq/2)
                 ) # S_d^+ S_0^-
            push!(hamiltonian,
                  ("+-+-",  [2, 1, upIndices[index1], upIndices[index2]+1], J_kq/2)
                 ) # S_d^- S_0^+
        end

    end

    if abs(kondoPerp) > couplingTolerance
        push!(hamiltonian, 
              ("n+-",  [1, cbathUp, cbathUp], kondoPerp/4))
        push!(hamiltonian, 
              ("n+-",  [1, cbathUp+1, cbathUp+1], -kondoPerp/4))
        push!(hamiltonian, 
              ("n+-",  [2, cbathUp, cbathUp], -kondoPerp/4))
        push!(hamiltonian, 
              ("n+-",  [2, cbathUp+1, cbathUp+1], kondoPerp/4))
        push!(hamiltonian, 
              ("+-+-",  [1, 2, cbathUp+1, cbathUp], kondoPerp/2))
        push!(hamiltonian, 
              ("+-+-",  [2, 1, cbathUp, cbathUp+1], kondoPerp/2))
    end
    if abs(cbathChemPot) > couplingTolerance
        push!(hamiltonian, 
              ("n",  [cbathUp], -cbathChemPot)
             )
        push!(hamiltonian, 
              ("n",  [cbathUp + 1], -cbathChemPot)
             )
    end

    # global magnetic field (to lift any trivial degeneracy)
    if abs(globalField) > couplingTolerance
        for site in 1:(2 + size(kondoGamma)[1])
            push!(hamiltonian, ("n",  [2 * site - 1], globalField/2))
            push!(hamiltonian, ("n",  [2 * site], -globalField/2))
        end
    end

    if abs(epsilonF) > couplingTolerance
        push!(hamiltonian, ("n",  [1], epsilonF))
        push!(hamiltonian, ("n",  [2], epsilonF))
    end
    if abs(impU) > couplingTolerance
        push!(hamiltonian, ("nn",  [1, 2], impU))
    end

    @assert !isempty(hamiltonian) "Hamiltonian is empty!"

    return hamiltonian
end
export BilayerKondo


function BilayerKondo(
        kondoGamma::Matrix{Float64};
        globalField::Number=0,
        epsilonF::Number=0,
        impU::Number=0,
        bathChemPot::Vector{Float64}=Float64[],
        couplingTolerance::Number=1e-15,
    )
    @assert isempty(bathChemPot) || length(bathChemPot) == size(kondoGamma)[1]
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]

    upIndices = 2 .+ range(1, length=size(kondoGamma)[1], step=2)

    # kondo terms
    for (index1, Jk_arr) in enumerate(eachrow(kondoGamma))
        for (index2, J_kq) in enumerate(Jk_arr)
            if abs(J_kq) < couplingTolerance
                continue
            end
            push!(hamiltonian,
                  ("n+-",  [1, upIndices[index1], upIndices[index2]], J_kq/4)
                 ) # n_{d up, n_{0 up}
            push!(hamiltonian,
                  ("n+-",  [1, upIndices[index1]+1, upIndices[index2]+1], -J_kq/4)
                 ) # n_{d up, n_{0 down}
            push!(hamiltonian,
                  ("n+-",  [2, upIndices[index1], upIndices[index2]], -J_kq/4)
                 ) # n_{d down, n_{0 up}
            push!(hamiltonian,
                  ("n+-",  [2, upIndices[index1]+1, upIndices[index2]+1], J_kq/4)
                 ) # n_{d down, n_{0 down}
            push!(hamiltonian,
                  ("+-+-",  [1, 2, upIndices[index1]+1, upIndices[index2]], J_kq/2)
                 ) # S_d^+ S_0^-
            push!(hamiltonian,
                  ("+-+-",  [2, 1, upIndices[index1], upIndices[index2]+1], J_kq/2)
                 ) # S_d^- S_0^+
        end
        if !isempty(bathChemPot) && abs(bathChemPot[index1]) > couplingTolerance
            push!(hamiltonian, 
                  ("n",  [upIndices[index1]], -bathChemPot[index1])
                 )
            push!(hamiltonian, 
                  ("n",  [upIndices[index1]+1], -bathChemPot[index1])
                 )
        end
    end


    # global magnetic field (to lift any trivial degeneracy)
    if abs(globalField) > couplingTolerance
        for site in 1:2*(1 + size(kondoGamma)[1])
            if site % 2 == 1
                push!(hamiltonian, ("n",  [site], globalField/2))
            else
                push!(hamiltonian, ("n",  [site], -globalField/2))
            end
        end
    end

    if abs(epsilonF) > couplingTolerance
        push!(hamiltonian, ("n",  [1], epsilonF))
        push!(hamiltonian, ("n",  [2], epsilonF))
    end
    if abs(impU) > couplingTolerance
        push!(hamiltonian, ("nn",  [1, 2], impU))
    end

    @assert !isempty(hamiltonian) "Hamiltonian is empty!"

    return hamiltonian
end
export BilayerKondo


function BilayerLEE(
        K_values::Vector{Tuple{Bool, Float64}},
        Jp::Float64;
        globalField::Number=0,
        couplingTolerance::Number=1e-15,
    )

    #### Indexing convention ####
    # Sf   Sd   γ1   γ2  ...
    # 1,2, 3,4, 5,6, 7,8 ...
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]
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

    # kondo terms
    for (i, (fLayer, K_i)) in enumerate(K_values)
        if abs(K_i) < couplingTolerance
            continue
        end
        if fLayer
            imp = 1
        else
            imp = 3
        end
        bath = 3 + 2 * i
        push!(hamiltonian,
              ("n+-",  [imp, bath, bath], K_i / 4)
             ) # n_{d up, n_{0 up}
        push!(hamiltonian,
              ("n+-",  [imp, bath + 1, bath + 1], -K_i / 4)
             ) # n_{d up, n_{0 down}
        push!(hamiltonian,
              ("n+-",  [imp + 1, bath, bath], -K_i / 4)
             ) # n_{d down, n_{0 up}
        push!(hamiltonian,
              ("n+-",  [imp + 1, bath + 1, bath + 1], K_i / 4)
             ) # n_{d down, n_{0 down}
        push!(hamiltonian,
              ("+-+-",  [imp, imp + 1, bath + 1, bath], K_i / 2)
             ) # S_d^+ S_0^-
        push!(hamiltonian,
              ("+-+-",  [imp + 1, imp, bath, bath + 1], K_i / 2)
             ) # S_d^- S_0^+
    end

    # global magnetic field (to lift any trivial degeneracy)
    if abs(globalField) > couplingTolerance
        for site in 1:2*(2 + length(K_values))
            if site % 2 == 1
                push!(hamiltonian, ("n",  [site], globalField/2))
            else
                push!(hamiltonian, ("n",  [site], -globalField/2))
            end
        end
    end

    @assert !isempty(hamiltonian) "Hamiltonian is empty!"

    return hamiltonian
end

function BilayerLEE(
        J::Matrix{Float64},
        Jp::Float64,
        hybrid::Vector{Float64},
        η::Dict{String, Float64},
        impCorr::Dict{String, Float64},
        layerSpecs::Vector{String};
        globalField::Number=0,
        couplingTolerance::Number=1e-15,
    )

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
    for i in 1:size(J)[1]
        if layerSpecs[i] == "f"
            imp = 1
        else
            imp = 3
        end
        bath_i = 3 + 2 * i
        for j in 1:size(J)[2]
            J_ij = J[i, j]
            if abs(J_ij) < couplingTolerance
                continue
            end
            bath_j = 3 + 2 * j
            push!(hamiltonian,
                  ("n+-",  [imp, bath_i, bath_j], J_ij / 4)
                 ) # n_{d up, n_{0 up}
            push!(hamiltonian,
                  ("n+-",  [imp, bath_i + 1, bath_j + 1], -J_ij / 4)
                 ) # n_{d up, n_{0 down}
            push!(hamiltonian,
                  ("n+-",  [imp + 1, bath_i, bath_j], -J_ij / 4)
                 ) # n_{d down, n_{0 up}
            push!(hamiltonian,
                  ("n+-",  [imp + 1, bath_i + 1, bath_j + 1], J_ij / 4)
                 ) # n_{d down, n_{0 down}
            push!(hamiltonian,
                  ("+-+-",  [imp, imp + 1, bath_i + 1, bath_j], J_ij / 2)
                 ) # S_d^+ S_0^-
            push!(hamiltonian,
                  ("+-+-",  [imp + 1, imp, bath_i, bath_j + 1], J_ij / 2)
                 ) # S_d^- S_0^+
        end
        if abs(hybrid[i]) > couplingTolerance
            push!(hamiltonian, ("+-",  [imp, bath_i], hybrid[i])) # n_{d up, n_{0 up}
            push!(hamiltonian, ("+-",  [imp + 1, bath_i + 1], hybrid[i])) # n_{d up, n_{0 up}
            push!(hamiltonian, ("+-",  [bath_i, imp], hybrid[i])) # n_{d up, n_{0 up}
            push!(hamiltonian, ("+-",  [bath_i + 1, imp + 1], hybrid[i])) # n_{d up, n_{0 up}
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
    if abs(globalField) > couplingTolerance
        for site in 1:2*(2 + size(J)[1])
            if site % 2 == 1
                push!(hamiltonian, ("n",  [site], globalField/2))
            else
                push!(hamiltonian, ("n",  [site], -globalField/2))
            end
        end
    end

    @assert !isempty(hamiltonian) "Hamiltonian is empty!"

    return hamiltonian
end

