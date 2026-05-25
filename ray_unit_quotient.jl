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

    return Q, coset_reps, phi
end