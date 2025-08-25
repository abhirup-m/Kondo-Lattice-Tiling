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
        kondoGammaPlus::Matrix{Float64},
        kondoPerp::Number,
        hoppingCbath::Number;
        globalField::Number=0,
        epsilonF::Number=0,
        cbathChemPot::Number=0,
        couplingTolerance::Number=1e-15,
    )
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]

    cbathUp = 2 .+ range(1, length=size(kondoGammaPlus)[1], step=4)
    plusUp = 2 .+ cbathUp

    # kondo terms
    for index1 in 1:size(kondoGammaPlus)[1]
        kondoGamma = kondoGammaPlus 
        upIndices = plusUp
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
        for site in 1:(1 + 2 * size(kondoGammaPlus)[1])
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

