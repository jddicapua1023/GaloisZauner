# unit_group_quotients.jl
#
# Two related, independently-computed constructions of unit-group quotients
# used to build explicit ray class group representatives:
#
#   - ray_unit_quotient:      computes U_{m',Sigma'}(O') / U_{m,Sigma}(O)
#                              explicitly, returning the abstract group,
#                              coset representatives, and the projection map.
#   - residue_unit_quotient:  computes the same quotient's abstract group
#                              structure via a residue-ring (O/delta) approach.
#                              Used as an independent cross-check on the
#                              order/structure produced by ray_unit_quotient.
#
# (Merged from ray_unit_quotient.jl and term2.jl.)

using Hecke

"""
    ray_unit_quotient(O_prime, O, m_prime, m, Sigma_prime, Sigma)

Computes the quotient group U_{m',Σ'}(O') / U_{m,Σ}(O) together with
coset representatives.

Inputs:
  O_prime    : order containing O (both orders in same real quadratic field K)
  O          : suborder
  m_prime    : ideal of O_prime with m*O_prime ⊆ m_prime
  m          : ideal of O
  Sigma_prime: subset of real places (the smaller set for O_prime)
  Sigma      : set of real places with Sigma_prime ⊆ Sigma (the larger set for O)

Returns:
  (Q, coset_reps, phi) where
    Q          : FinGenAbGroup isomorphic to the quotient
    coset_reps : Vector of elements of O_prime×, one per coset
    phi        : function sending a unit in U_{m',Σ'}(O') to its class in Q
"""
function ray_unit_quotient(O_prime, O, m_prime, m, Sigma_prime, Sigma)
    K = number_field(O)
    @assert number_field(O_prime) === K "O and O_prime must be orders in the same field"

    # ----------------------------------------------------------------
    # Step 1: get unit groups
    # ----------------------------------------------------------------
    U_prime, mU_prime = unit_group(O_prime)
    n_prime = ngens(U_prime)
    # generators of O_prime×: unit_gens_prime[1] = -1 (torsion), [2] = fund. unit
    unit_gens_prime = [mU_prime(U_prime[i]) for i in 1:n_prime]

    # ----------------------------------------------------------------
    # Step 2: membership tests
    # ----------------------------------------------------------------

    # Is alpha in U_{m', Sigma'}(O')?
    function in_upper(alpha)
        O_prime(alpha) - one(O_prime) in m_prime || return false
        return all(sign(K(alpha), p) > 0 for p in Sigma_prime)
    end

    # Is alpha in U_{m, Sigma}(O)? (the subgroup / kernel)
    function in_lower(alpha)
        # must actually lie in O
        try
            beta = O(alpha)
            beta - one(O) in m || return false
            return all(sign(K(alpha), p) > 0 for p in Sigma)
        catch
            return false
        end

    end

    # ----------------------------------------------------------------
    # Step 3: find the period for the free generator of O_prime×
    # modulo U_{m, Sigma}(O), so we can enumerate U_{m',Sigma'}(O')
    # The torsion generator (-1) has order 2.
    # We find the smallest k >= 1 s.t. epsilon^k in U_{m,Sigma}(O).
    # ----------------------------------------------------------------
    epsilon = unit_gens_prime[end]   # fundamental unit of O_prime
    period = 0
    for k in 1:10_000
        if in_lower(epsilon^k)
            period = k
            break
        end
    end
    period == 0 && error("Period not found within cap; increase bound")

    # ----------------------------------------------------------------
    # Step 4: enumerate U_{m',Sigma'}(O') / U_{m,Sigma}(O)
    # by ranging over e_torsion in {0,1}, e_free in {0,...,period-1}
    # ----------------------------------------------------------------
    coset_reps = []          # one representative per coset
    coset_exponents = []     # the (e_torsion, e_free) for each rep

    for e_torsion in 0:1
        for e_free in 0:(period - 1)
            alpha = mU_prime(e_torsion * U_prime[1] + e_free * U_prime[2])

            # Must be in the upper group
            in_upper(alpha) || continue

            # Check if this is a new coset
            is_new = true
            for rep in coset_reps
                ratio = alpha * inv(K(rep))
                if in_lower(ratio)
                    is_new = false
                    break
                end
            end

            if is_new
                push!(coset_reps, alpha)
                push!(coset_exponents, (e_torsion, e_free))
            end
        end
    end

    N = length(coset_reps)   # order of the quotient group

    # ----------------------------------------------------------------
    # Step 5: build the group structure as a FinGenAbGroup
    # Strategy: find generators of the quotient among coset_reps,
    # then express all reps in terms of those generators to get
    # the relation matrix for abelian_group().
    #
    # We build a multiplication table: for reps[i] * reps[j],
    # find which coset the product lands in. This gives us the
    # group law, from which we extract generators and relations
    # via Smith normal form.
    # ----------------------------------------------------------------

    # Helper: given alpha in U_{m',Sigma'}(O'), find its coset index (0-based)
    function coset_index(alpha)
        for (idx, rep) in enumerate(coset_reps)
            ratio = alpha * inv(K(rep))
            if in_lower(ratio)
                return idx
            end
        end
        error("Element not found in any coset — something is wrong")
    end

    # Multiplication table (1-indexed)
    mult_table = Matrix{Int}(undef, N, N)
    for i in 1:N
        for j in 1:N
            mult_table[i, j] = coset_index(coset_reps[i] * coset_reps[j])
        end
    end

    # From the multiplication table, identify the group structure.
    # Since the group is abelian and finite, we use the following:
    # - Find the order of each coset rep
    # - Build a relation matrix for abelian_group()
    #
    # Represent each coset_rep as a generator g_i.
    # Relations come from:
    #   (a) the order of each generator: ord(g_i) * e_i = 0
    #   (b) expressing products g_i * g_j in terms of existing generators
    #
    # Simplest approach: present as Z^N with relations from the group law,
    # then let Hecke's Smith normal form reduce it.

    # Build relation matrix: N generators, relations from order and products
    # Relation rows: for each pair (i,j) with i <= j,
    #   g_i + g_j - g_{mult_table[i,j]} = 0  (written additively)
    # Plus order relations: ord(g_i) * g_i = 0

    # First compute orders of each element
    function element_order(i)
        cur = i
        for k in 1:N
            if cur == 1  # index of identity
                return k
            end
            cur = mult_table[cur, i]
        end
        error("Order not found")
    end

    # Find identity element (index where mult_table[i,j] = j for all j)
    identity_idx = findfirst(i -> all(mult_table[i, j] == j for j in 1:N), 1:N)
    identity_idx === nothing && error("Identity not found in coset reps")

    # Reorder so identity is first
    if identity_idx != 1
        coset_reps[1], coset_reps[identity_idx] = coset_reps[identity_idx], coset_reps[1]
        # Rebuild mult_table after swap
        for i in 1:N, j in 1:N
            mult_table[i, j] = coset_index(coset_reps[i] * coset_reps[j])
        end
        identity_idx = 1
    end

    # Compute order of each element
    orders = [element_order(i) for i in 1:N]

    # Build relation matrix for abelian_group:
    # N generators e_1,...,e_N
    # For each i: orders[i] * e_i = 0
    # For each i,j: e_i + e_j = e_{mult_table[i,j]}
    #   i.e.  e_i + e_j - e_{mult_table[i,j]} = 0
    relations = Vector{Vector{Int}}()

    # Order relations
    for i in 1:N
        row = zeros(Int, N)
        row[i] = orders[i]
        push!(relations, row)
    end

    # Product relations
    for i in 1:N
        for j in i:N
            k = mult_table[i, j]
            row = zeros(Int, N)
            row[i] += 1
            row[j] += 1
            row[k] -= 1
            if any(!=(0), row)
                push!(relations, row)
            end
        end
    end

    M = matrix(ZZ, hcat(relations...)')  # relation matrix: rows = relations
    Q, proj = abelian_group(M), nothing

    # Recompute cleanly: abelian_group takes the matrix directly
    Q = abelian_group(M)

    # ----------------------------------------------------------------
    # Step 6: define phi: unit -> element of Q
    # ----------------------------------------------------------------
    function phi(alpha)
        idx = coset_index(alpha)
        # express coset_reps[idx] in terms of Q's generators
        # Since Q was built with generators = coset_reps, the idx-th
        # generator of Q corresponds to coset_reps[idx]
        return Q[idx]
    end

    #=
    println("Quotient group structure: ", elementary_divisors(Q))
    println("Order of quotient: ", N)
    println("Number of coset representatives: ", length(coset_reps))
    =#

    println("  ray_unit_quotient: period = ", period)
    println("  ray_unit_quotient: N (order of quotient) = ", N)
    println("  ray_unit_quotient: is -1 in lower? ", in_lower(mU_prime(U_prime[1])))
    println("  ray_unit_quotient: elementary_divisors = ", elementary_divisors(Q))

    return Q, coset_reps, phi
end

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