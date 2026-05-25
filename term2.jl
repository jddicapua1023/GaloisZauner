using Hecke

function residue_unit_quotient(O_prime, O, m_prime, m, delta, Sigma_prime, Sigma)

    K = number_field(O)
    @assert number_field(O_prime) === K "O and O_prime must lie in the same field"

    # ----------------------------------------------------------------
    # Step 1: construct O'/delta and its unit group
    # ----------------------------------------------------------------
    R_prime, proj_prime = quo(O_prime, delta)
    U_Rprime, mU_Rprime = unit_group(R_prime)

    # ----------------------------------------------------------------
    # Step 2: construct O/(delta ∩ O) and its unit group
    # delta ∩ O: take the basis of delta, keep elements lying in O
    # ----------------------------------------------------------------
    delta_basis = basis(delta)
    delta_O_gens = AbsSimpleNumFieldOrderElem[]
    for b in delta_basis
        kb = K(b)
        try
            ob = O(kb)
            push!(delta_O_gens, ob)
        catch
            # b does not lie in O, skip
        end
    end

    # if no basis elements lie in O, use minimum(delta) as fallback
    if isempty(delta_O_gens)
        delta_O = ideal(O, minimum(delta))
    else
        delta_O = ideal(O, delta_O_gens)
    end

    R, proj = quo(O, delta_O)
    U_R, mU_R = unit_group(R)

    # ----------------------------------------------------------------
    # Step 3: membership tests using iszero(mod(...))
    # ----------------------------------------------------------------
    function in_U_m_prime(u_elem)
        lifted = O_prime(lift(mU_Rprime(u_elem)))
        return iszero(mod(lifted - one(O_prime), m_prime))
    end

    function in_U_m(u_elem)
        lifted = O(lift(mU_R(u_elem)))
        return iszero(mod(lifted - one(O), m))
    end

    # ----------------------------------------------------------------
    # Step 4: find U_{m'}(O'/delta) by iterating over ALL elements
    # of (O'/delta)× and checking the congruence condition.
    # We do this by finding which elements of U_Rprime satisfy it.
    # ----------------------------------------------------------------

    # Collect all elements of U_Rprime that satisfy in_U_m_prime
    upper_elems = FinGenAbGroupElem[]
    for g in U_Rprime
        if in_U_m_prime(g)
            push!(upper_elems, g)
        end
    end

    # The subgroup U_{m'}(O'/delta) inside U_Rprime
    if isempty(upper_elems)
        G_upper = abelian_group([1])
        inc_upper = x -> zero(U_Rprime)
    else
        # find generators: elements whose order in U_Rprime generates the subgroup
        G_upper, inc_upper = sub(U_Rprime, upper_elems)
    end

    # ----------------------------------------------------------------
    # Step 5: find U_m(O/delta_O) similarly
    # ----------------------------------------------------------------
    lower_elems = FinGenAbGroupElem[]
    for g in U_R
        if in_U_m(g)
            push!(lower_elems, g)
        end
    end

    if isempty(lower_elems)
        G_lower = abelian_group([1])
    else
        G_lower, inc_lower = sub(U_R, lower_elems)
    end

# ----------------------------------------------------------------
# Step 6: map G_lower into G_upper via O ⊆ O'
# ----------------------------------------------------------------
lower_images_in_upper = FinGenAbGroupElem[]

for g in lower_elems
    # lift to O
    elem_O = O(lift(mU_R(g)))
    # embed into O_prime
    elem_Oprime = O_prime(K(elem_O))
    # project to O'/delta
    elem_Rprime = proj_prime(elem_Oprime)
    # find preimage in U_Rprime
    try
        u_in_Rprime = mU_Rprime \ elem_Rprime
        # now express this element in terms of G_upper
        # inc_upper maps G_upper -> U_Rprime, so we need the preimage
        # under inc_upper
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

# ----------------------------------------------------------------
# Step 7: compute quotient G_upper / image(G_lower)
# ----------------------------------------------------------------
if isempty(lower_images_in_upper)
    Q_finite = G_upper
else
    H, inc_H = sub(G_upper, lower_images_in_upper)
    Q_finite, proj_Q = quo(G_upper, H)
end

    # ----------------------------------------------------------------
    # Step 8: build (Z/2Z)^|Sigma \ Sigma'|
    # ----------------------------------------------------------------
    extra_places = setdiff(Sigma, Sigma_prime)
    n_extra = length(extra_places)

    if n_extra == 0
        Z2_part = abelian_group([1])
    else
        Z2_part = abelian_group(fill(2, n_extra))
    end

    # ----------------------------------------------------------------
    # Step 9: direct product Q_finite x Z2_part
    # ----------------------------------------------------------------
    if order(Q_finite) == 1
        Q = Z2_part
    elseif order(Z2_part) == 1
        Q = Q_finite
    else
        Q, _ = direct_product(Q_finite, Z2_part)
    end

    structure = elementary_divisors(Q)
    total_order = order(Q)

    #=
    println("Group structure (elementary divisors): ", structure)
    println("Order of group: ", total_order)
    =#

    return Q, structure
end