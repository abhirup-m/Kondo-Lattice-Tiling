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
