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
        I = mP(g)  # already lives in OK

        # Check valuations at each prime above m
        correction_needed = Dict{AbsSimpleNumFieldOrderIdeal, Int}()
        for (p, _) in fac
            v = valuation(I, p)
            if v != 0
                correction_needed[p] = -v
            end
        end

        if isempty(correction_needed)
            push!(reps, I)
            continue
        end

        # Build alpha in K* with required valuations to cancel bad primes
        K = number_field(O)
        alpha = one(K)
        for (p, exp) in correction_needed
            pi_p = elem_in_nf(uniformizer(p))
            alpha = alpha * pi_p^exp
        end

        # Form alpha * I as a fractional ideal, simplify, then take numerator
        J = numerator(simplify(alpha * I))

        # Sanity check
        if !isone(J + m_OK)
            error("Failed to make ideal coprime to m for group element ", g)
        end

        push!(reps, J)
    end

    return reps
end