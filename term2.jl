using Hecke

function residue_unit_quotient(O_prime, O, m_prime, m, delta, Sigma_prime, Sigma)

    K = number_field(O)
    @assert number_field(O_prime) === K "O and O_prime must lie in the same field"

    # Step 1: construct O'/delta and its unit group
    R_prime, proj_prime = quo(O_prime, delta)
    U_Rprime, mU_Rprime = unit_group(R_prime)

    # Step 2: construct O/(delta ∩ O) and its unit group
    delta_basis = basis(delta)
    delta_O_gens = AbsSimpleNumFieldOrderElem[]
    for b in delta_basis
        kb = K(b)
        try
            ob = O(kb)
            push!(delta_O_gens, ob)
        catch
        end
    end

    if isempty(delta_O_gens)
        delta_O = ideal(O, minimum(delta))
    else
        delta_O = ideal(O, delta_O_gens)
    end

    R, proj = quo(O, delta_O)
    U_R, mU_R = unit_group(R)

    # Step 3: compute O/m and O'/m' for membership tests
    R_m, proj_m = quo(O, m)
    R_m_prime, proj_m_prime = quo(O_prime, m_prime)

    # membership tests using quotient ring projections
    function in_U_m_prime(u_elem)
        lifted = O_prime(lift(mU_Rprime(u_elem)))
        return isone(proj_m_prime(lifted))
    end

    function in_U_m(u_elem)
        lifted = O(lift(mU_R(u_elem)))
        return isone(proj_m(lifted))
    end

    # Step 4: find U_{m'}(O'/delta) — kernel of U_Rprime -> (O'/m')×
    upper_gens = FinGenAbGroupElem[]
    divs_prime = elementary_divisors(U_Rprime)

    for i in 1:ngens(U_Rprime)
        g = U_Rprime[i]
        ord_g = Int(divs_prime[i])
        for k in 1:ord_g-1
            gk = k * g
            if in_U_m_prime(gk)
                push!(upper_gens, gk)
                break
            end
        end
    end

    if isempty(upper_gens)
        G_upper = abelian_group([1])
        inc_upper = x -> zero(U_Rprime)
    else
        G_upper, inc_upper = sub(U_Rprime, upper_gens)
    end

    # Step 5: find U_m(O/delta_O) — kernel of U_R -> (O/m)×
    lower_gens = FinGenAbGroupElem[]
    divs = elementary_divisors(U_R)

    for i in 1:ngens(U_R)
        g = U_R[i]
        ord_g = Int(divs[i])
        for k in 1:ord_g-1
            gk = k * g
            if in_U_m(gk)
                push!(lower_gens, gk)
                break
            end
        end
    end

    if isempty(lower_gens)
        G_lower = abelian_group([1])
    else
        G_lower, inc_lower = sub(U_R, lower_gens)
    end

    # Step 6: map G_lower into G_upper via O ⊆ O'
    lower_images_in_upper = FinGenAbGroupElem[]

    for g in lower_gens
        elem_O = O(lift(mU_R(g)))
        elem_Oprime = O_prime(K(elem_O))
        elem_Rprime = proj_prime(elem_Oprime)
        try
            u_in_Rprime = mU_Rprime \ elem_Rprime
            success, u_in_Gupper = haspreimage(inc_upper, u_in_Rprime)
            if success
                push!(lower_images_in_upper, u_in_Gupper)
            else
                @warn "Element not in G_upper"
            end
        catch e
            @warn "Could not find preimage in U_Rprime: $e"
        end
    end

    # Step 7: compute quotient G_upper / image(G_lower)
    if isempty(lower_images_in_upper)
        Q_finite = G_upper
    else
        H, inc_H = sub(G_upper, lower_images_in_upper)
        Q_finite, proj_Q = quo(G_upper, H)
    end

    # Step 8: build (Z/2Z)^|Sigma \ Sigma'|
    extra_places = setdiff(Sigma, Sigma_prime)
    n_extra = length(extra_places)

    if n_extra == 0
        Z2_part = abelian_group([1])
    else
        Z2_part = abelian_group(fill(2, n_extra))
    end

    # Step 9: direct product
    if order(Q_finite) == 1
        Q = Z2_part
    elseif order(Z2_part) == 1
        Q = Q_finite
    else
        Q, _ = direct_product(Q_finite, Z2_part)
    end

    structure = elementary_divisors(Q)

    println("  residue_unit_quotient: order(U_Rprime) = ", order(U_Rprime))
    println("  residue_unit_quotient: order(U_R) = ", order(U_R))
    println("  residue_unit_quotient: order(G_upper) = ", order(G_upper))
    println("  residue_unit_quotient: order(G_lower) = ", order(G_lower))
    println("  residue_unit_quotient: order(Q_finite) = ", order(Q_finite))
    println("  residue_unit_quotient: n_extra = ", n_extra)
    println("  residue_unit_quotient: order(Q) = ", order(Q))

    return Q, structure
end