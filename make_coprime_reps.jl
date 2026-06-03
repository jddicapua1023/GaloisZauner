using Hecke
include("ray_unit_quotient.jl")
include("term2.jl")
include("enumeration_with_infinite_places.jl")

function make_coprime_representatives(O, m, mP, group_elems)
    OK = maximal_order(number_field(O))
    m_OK = OK * m
    fac = factor(m_OK)

    reps = []

    for g in group_elems
        I_O = mP(g)
        I = OK * I_O          # lift to OK

        correction_needed = Dict{AbsSimpleNumFieldOrderIdeal, Int}()
        for (p, _) in fac
            v = valuation(I, p)
            if v != 0
                correction_needed[p] = -v
            end
        end

        if isempty(correction_needed)
            push!(reps, I)    # I is already in OK, not I_O
            continue
        end

        K = number_field(O)
        alpha = one(K)
        for (p, exp) in correction_needed
            pi_p = elem_in_nf(uniformizer(p))
            alpha = alpha * pi_p^exp
        end

        J_OK = numerator(simplify(alpha * I))

        if !isone(J_OK + m_OK)
            error("Failed to make ideal coprime to m for group element ", g)
        end

        push!(reps, J_OK)     # no pullback to O
    end

    return reps
end