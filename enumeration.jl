using Hecke
include("ray_unit_quotient.jl")
include("term2.jl")

function ray_class_group_order(OK, O, mp, m, delta, Sigma_prime, Sigma)
    Q_res, structure = residue_unit_quotient(OK, O, mp, m, delta, Sigma_prime, Sigma)
    N1 = prod(elementary_divisors(Q_res))

    Q_ray, reps, phi = ray_unit_quotient(OK, O, mp, m, Sigma_prime, Sigma)
    N2 = order(Q_ray)

    R, mR = ray_class_group(mp, Sigma_prime)
    N3 = order(R)

    return divexact(N3 * N1, N2)
end

#=
K, a = quadratic_field(5)
OK = maximal_order(K)
O6 = Order(K, [K(1), 3*a])

m   = ideal(O6, 5)
mp  = ideal(OK, 5)
delta = ideal(OK, 30)

plcs        = real_places(K)
Sigma       = InfPlc[]
Sigma_prime = InfPlc[]

result = ray_class_group_order(OK, O6, mp, m, delta, Sigma_prime, Sigma)
println("Result: ", result)

f = 6  # conductor of your order
mf = ideal(OK, f)

R, mR = ray_class_group(mf, InfPlc[])

println(elementary_divisors(R))
println(order(R))

K, a = quadratic_field(5)
OK = maximal_order(K)
O6 = Order(K, [K(1), 3*a])

# Get all ideal classes in Pic(O_f)
P, mP = picard_group(O6)
h = order(P)
println("h(O_f) = ", h)

picard_ideals = []

gens_P = gens(P)
divs = elementary_divisors(P)

ranges = [0:Int(d)-1 for d in divs]
for exps in Iterators.product(ranges...)
    if isempty(gens_P)
        g = zero(P)
    else
        g = sum(exps[i] * gens_P[i] for i in 1:length(gens_P))
    end
    I = mP(g)
    push!(picard_ideals, I)
end

println("picard_ideals = ", picard_ideals)

# Example 1: Q(sqrt(10)), order of conductor 2
K, a = quadratic_field(10)
OK = maximal_order(K)
O2 = Order(K, [K(1), 2*a])

P, mP = picard_group(O2)
println("h(O_f) for Q(sqrt(10)), conductor 2: ", order(P))
println("Elementary divisors: ", elementary_divisors(P))

# Example 2: Q(sqrt(5)), order of conductor 5
K2, a2 = quadratic_field(5)
OK2 = maximal_order(K2)
O5 = Order(K2, [K2(1), 5*a2])

P2, mP2 = picard_group(O5)
println("h(O_f) for Q(sqrt(5)), conductor 5: ", order(P2))
println("Elementary divisors: ", elementary_divisors(P2))

# Example 3: Q(sqrt(3)), order of conductor 6
K3, a3 = quadratic_field(3)
OK3 = maximal_order(K3)
O6 = Order(K3, [K3(1), 6*a3])

P3, mP3 = picard_group(O6)
println("h(O_f) for Q(sqrt(3)), conductor 6: ", order(P3))
println("Elementary divisors: ", elementary_divisors(P3))

# Print ideal representatives for whichever has non-trivial Picard group
for (P, mP, label) in [(P, mP, "Q(sqrt(10)) f=2"), (P2, mP2, "Q(sqrt(5)) f=5"), (P3, mP3, "Q(sqrt(3)) f=6")]
    if order(P) > 1
        println("\nIdeal class representatives for ", label, ":")
        gens_P = gens(P)
        divs = elementary_divisors(P)
        ranges = [0:Int(d)-1 for d in divs]
        for exps in Iterators.product(ranges...)
            g = isempty(gens_P) ? zero(P) : sum(exps[i] * gens_P[i] for i in 1:length(gens_P))
            I = mP(g)
            println("  exps=", exps, "  ideal=", I, "  norm=", norm(I))
        end
    end
end

using Hecke

K, a = quadratic_field(10)
OK = maximal_order(K)
O2 = Order(K, [K(1), 2*a])

P, mP = picard_group(O2)
println("h(O_f) for Q(sqrt(10)), conductor 2: ", order(P))
println("Elementary divisors: ", elementary_divisors(P))

let
    gens_P = gens(P)
    divs = elementary_divisors(P)
    ranges = [0:Int(d)-1 for d in divs]
    println("\nIdeal class representatives:")
    for exps in Iterators.product(ranges...)
        local g = isempty(gens_P) ? zero(P) : sum(exps[i] * gens_P[i] for i in 1:length(gens_P))
        local I = mP(g)
        println("  exps=", exps, "  ideal=", I, "  norm=", norm(I))
    end
end
=#

#=
old version 4_16_2026
function enumerate_principal_ray_classes(K, O, m, Sigma, N)
    N == 0 && return []  # early exit if nothing to find
    
    OK = maximal_order(K)
    B = basis(O)

    U, mU = unit_group(O)

    function units_mod_m()
        torsion = [one(K), -one(K)]
        eps = nothing
        for i in 1:ngens(U)
            u = elem_in_nf(mU(U[i]))
            if !(u == one(K) || u == -one(K))
                eps = u
                break
            end
        end
        eps === nothing && return torsion

        seen_powers = []
        current = one(K)
        for _ in 1:10000
            push!(seen_powers, current)
            current = current * eps
            in_order = true
            diff_O = try
                O(current - 1)
            catch
                in_order = false
                nothing
            end
            if in_order && diff_O in m
                break
            end
        end

        all_units = []
        for t in torsion, p in seen_powers
            push!(all_units, t * p)
        end
        return all_units
    end

    unit_reps = units_mod_m()
    println("Number of unit reps mod m: ", length(unit_reps))

    function is_invertible(I)
        try
            return ring_of_multipliers(I) == O
        catch
            return false
        end
    end

    function cong_one_mod_m(alpha)
        try
            diff_O = O(alpha - 1)
            return diff_O in m
        catch
            return false
        end
    end

    function positive_on_Sigma(alpha)
        for rho in Sigma
            evaluate(alpha, rho) > 0 || return false
        end
        return true
    end

    function coprime_to_m(gamma)
        try
            I = ideal(O, O(gamma))
            return isone(I + m)
        catch
            return false
        end
    end

    function are_equivalent(gamma, delta)
        try
            g = elem_in_nf(gamma)
            d = elem_in_nf(delta)
            I_g = ideal(O, O(g))
            I_d = ideal(O, O(d))
            I_g == I_d && return true
            base_ratio = g * inv(d)
            for u in unit_reps
                candidate = u * base_ratio
                if cong_one_mod_m(candidate) && positive_on_Sigma(candidate)
                    return true
                end
            end
            return false
        catch
            return false
        end
    end

    collected = []
    seen = []

    bound = 1
    max_bound = 10000  # safeguard against infinite loop
    while length(collected) < N && bound <= max_bound
        found_any = false
        for i in -bound:bound, j in -bound:bound
            # only process elements on the boundary of the current bound
            # to avoid reprocessing elements from smaller bounds
            (abs(i) == bound || abs(j) == bound) || continue

            gamma = i * B[1] + j * B[2]
            iszero(gamma) && continue
            abs(norm(gamma)) > bound^2 && continue
            coprime_to_m(elem_in_nf(gamma)) || continue

            I = try
                ideal(O, gamma)
            catch
                continue
            end

            is_invertible(I) || continue

            is_new = true
            if are_equivalent(gamma, one(O))
                is_new = false
            end
            if is_new
                for delta in seen
                    if are_equivalent(gamma, delta)
                        is_new = false
                        break
                    end
                end
            end

            if is_new
                found_any = true
                push!(seen, gamma)
                push!(collected, (I, gamma, abs(norm(gamma))))
                println("  Found class ", length(collected), ": generator=", gamma, " norm=", abs(norm(gamma)))
                length(collected) >= N && break
            end
        end
        bound += 1
    end

    if length(collected) < N
        println("Warning: only found ", length(collected), " of ", N, " requested classes at bound ", bound)
    end

    sort!(collected, by = x -> x[3])
    return collected
end
=#

#even more fixes on congruence checks 4/29
function enumerate_principal_ray_classes(K, ord, m, Sigma, N)
    N == 0 && return []

    OK = maximal_order(K)
    B = basis(ord)

    U, mU = unit_group(ord)

    function units_mod_m()
        torsion = [one(K), -one(K)]
        eps = nothing
        for i in 1:ngens(U)
            u = elem_in_nf(mU(U[i]))
            if !(u == one(K) || u == -one(K))
                eps = u
                break
            end
        end
        eps === nothing && return torsion

        seen_powers = []
        current = one(K)
        for _ in 1:10000
            push!(seen_powers, current)
            current = current * eps
            in_ord = true
            diff_O = try
                ord(current - 1)
            catch
                in_ord = false
                nothing
            end
            if in_ord && diff_O in m
                break
            end
        end

        all_units = []
        for t in torsion, p in seen_powers
            push!(all_units, t * p)
        end
        return all_units
    end

    unit_reps = units_mod_m()
    println("Number of unit reps mod m: ", length(unit_reps))

    function same_mod_m(alpha, beta)
        # alpha ≡ beta mod m iff alpha - beta ∈ m
        try
            diff = K(alpha) - K(beta)
            diff_ord = ord(diff)
            return diff_ord in m
        catch
            return false
        end
    end

    function positive_on_Sigma(alpha)
        for rho in Sigma
            evaluate(alpha, rho) > 0 || return false
        end
        return true
    end

    function coprime_to_m_ideal(I)
        try
            return isone(I + m)
        catch
            return false
        end
    end

    function are_equivalent(gamma, delta)
        try
            g = elem_in_nf(gamma)
            d = elem_in_nf(delta)
            # check ideal equality first
            I_g = ideal(ord, ord(g))
            I_d = ideal(ord, ord(d))
            I_g == I_d && return true
            # check if u*gamma ≡ delta mod m for some unit u
            # and positivity condition on u*gamma/delta
            for u in unit_reps
                candidate = u * g
                if same_mod_m(candidate, d) && positive_on_Sigma(candidate * inv(d))
                    return true
                end
            end
            return false
        catch
            return false
        end
    end

    collected = []
    seen = []

    function candidates_up_to_norm(norm_bound)
        results = []
        coord_bound = max(10, Int(ceil(sqrt(Float64(norm_bound)))) * 5)
        for i in -coord_bound:coord_bound, j in -coord_bound:coord_bound
            gamma = i * B[1] + j * B[2]
            iszero(gamma) && continue
            n = abs(norm(gamma))
            n > norm_bound && continue
            I = try ideal(ord, gamma) catch; continue end
            coprime_to_m_ideal(I) || continue
            push!(results, (gamma, I, n))
        end
        sort!(results, by = x -> x[3])
        return results
    end

    norm_bound = 200
    max_norm_bound = 10^7

    while length(collected) < N && norm_bound <= max_norm_bound
        println("Trying norm bound: ", norm_bound)
        candidates = candidates_up_to_norm(norm_bound)
        println("  Candidates to check: ", length(candidates))

        for (gamma, I, n) in candidates
            abs(norm(gamma)) == 1 && continue

            is_new = true
            if are_equivalent(gamma, one(ord))
                is_new = false
            end
            if is_new
                for delta in seen
                    if are_equivalent(gamma, delta)
                        is_new = false
                        break
                    end
                end
            end

            if is_new
                push!(seen, gamma)
                push!(collected, (I, gamma, n))
                println("  Found class ", length(collected),
                        ": generator=", gamma, " norm=", n)
                length(collected) >= N && break
            end
        end

        length(collected) >= N && break
        println("Expanding norm bound from ", norm_bound, " to ", norm_bound * 10)
        norm_bound *= 10
    end

    if length(collected) < N
        println("Warning: only found ", length(collected), " of ", N, " classes")
    end

    sort!(collected, by = x -> x[3])
    return collected
end

#=more fixes on congruence checks 4/29
function enumerate_principal_ray_classes(K, ord, m, Sigma, N)
    N == 0 && return []

    OK = maximal_order(K)
    B = basis(ord)

    U, mU = unit_group(ord)

    function units_mod_m()
        torsion = [one(K), -one(K)]
        eps = nothing
        for i in 1:ngens(U)
            u = elem_in_nf(mU(U[i]))
            if !(u == one(K) || u == -one(K))
                eps = u
                break
            end
        end
        eps === nothing && return torsion

        seen_powers = []
        current = one(K)
        for _ in 1:10000
            push!(seen_powers, current)
            current = current * eps
            in_ord = true
            diff_O = try
                ord(current - 1)
            catch
                in_ord = false
                nothing
            end
            if in_ord && diff_O in m
                break
            end
        end

        all_units = []
        for t in torsion, p in seen_powers
            push!(all_units, t * p)
        end
        return all_units
    end

    unit_reps = units_mod_m()
    println("Number of unit reps mod m: ", length(unit_reps))

    function same_mod_m(alpha, beta)
        # returns true iff alpha ≡ beta mod m
        # i.e. alpha - beta ∈ m
        try
            diff = alpha - beta
            diff_ord = ord(diff)
            return diff_ord in m
        catch
            return false
        end
    end

    function positive_on_Sigma(alpha)
        for rho in Sigma
            evaluate(alpha, rho) > 0 || return false
        end
        return true
    end

    function coprime_to_m_ideal(I)
        try
            return isone(I + m)
        catch
            return false
        end
    end

    function are_equivalent(gamma, delta)
        try
            g = elem_in_nf(gamma)
            d = elem_in_nf(delta)
            I_g = ideal(ord, ord(g))
            I_d = ideal(ord, ord(d))
            I_g == I_d && return true
            base_ratio = g * inv(d)
            for u in unit_reps
                candidate = u * base_ratio
                if same_mod_m(candidate, one(K)) && positive_on_Sigma(candidate)
                    return true
                end
            end
            return false
        catch
            return false
        end
    end

    collected = []
    seen = []

    function candidates_up_to_norm(norm_bound)
        results = []
        coord_bound = max(10, Int(ceil(sqrt(Float64(norm_bound)))) * 5)
        for i in -coord_bound:coord_bound, j in -coord_bound:coord_bound
            gamma = i * B[1] + j * B[2]
            iszero(gamma) && continue
            n = abs(norm(gamma))
            n > norm_bound && continue
            I = try ideal(ord, gamma) catch; continue end
            coprime_to_m_ideal(I) || continue
            push!(results, (gamma, I, n))
        end
        sort!(results, by = x -> x[3])
        return results
    end

    norm_bound = 200
    max_norm_bound = 10^7

    while length(collected) < N && norm_bound <= max_norm_bound
        println("Trying norm bound: ", norm_bound)
        candidates = candidates_up_to_norm(norm_bound)
        println("  Candidates to check: ", length(candidates))

        for (gamma, I, n) in candidates
            abs(norm(gamma)) == 1 && continue

            is_new = true
            if are_equivalent(gamma, one(ord))
                is_new = false
            end
            if is_new
                for delta in seen
                    if are_equivalent(gamma, delta)
                        is_new = false
                        break
                    end
                end
            end

            if is_new
                push!(seen, gamma)
                push!(collected, (I, gamma, n))
                println("  Found class ", length(collected),
                        ": generator=", gamma, " norm=", n)
                length(collected) >= N && break
            end
        end

        length(collected) >= N && break
        println("Expanding norm bound from ", norm_bound, " to ", norm_bound * 10)
        norm_bound *= 10
    end

    if length(collected) < N
        println("Warning: only found ", length(collected), " of ", N, " classes")
    end

    sort!(collected, by = x -> x[3])
    return collected
end
=#


#=revised version 4/29 attempt to fix congruence checks
function enumerate_principal_ray_classes(K, ord, m, Sigma, N)
    N == 0 && return []

    OK = maximal_order(K)
    B = basis(ord)

    U, mU = unit_group(ord)
    
    # Build the quotient ring ord/m and its unit group for congruence checks
    Q_ring, qmap = quo(ord, m)

    function units_mod_m()
        torsion = [one(K), -one(K)]
        eps = nothing
        for i in 1:ngens(U)
            u = elem_in_nf(mU(U[i]))
            if !(u == one(K) || u == -one(K))
                eps = u
                break
            end
        end
        eps === nothing && return torsion

        seen_powers = []
        current = one(K)
        for _ in 1:10000
            push!(seen_powers, current)
            current = current * eps
            in_ord = true
            diff_O = try
                ord(current - 1)
            catch
                in_ord = false
                nothing
            end
            if in_ord && diff_O in m
                break
            end
        end

        all_units = []
        for t in torsion, p in seen_powers
            push!(all_units, t * p)
        end
        return all_units
    end

    unit_reps = units_mod_m()
    println("Number of unit reps mod m: ", length(unit_reps))

    function cong_one_mod_m(alpha)
        # alpha ≡ 1 mod m iff alpha is in ord AND alpha - 1 ∈ m
        try
            alpha_O = ord(alpha)
            return qmap(alpha_O) == qmap(one(ord))
        catch
            return false
        end
    end

    function positive_on_Sigma(alpha)
        for rho in Sigma
            evaluate(alpha, rho) > 0 || return false
        end
        return true
    end

    function coprime_to_m_ideal(I)
        try
            return isone(I + m)
        catch
            return false
        end
    end

    function are_equivalent(gamma, delta)
        try
            g = elem_in_nf(gamma)
            d = elem_in_nf(delta)
            I_g = ideal(ord, ord(g))
            I_d = ideal(ord, ord(d))
            I_g == I_d && return true
            base_ratio = g * inv(d)
            for u in unit_reps
                candidate = u * base_ratio
                if cong_one_mod_m(candidate) && positive_on_Sigma(candidate)
                    return true
                end
            end
            return false
        catch
            return false
        end
    end

    collected = []
    seen = []

    function candidates_up_to_norm(norm_bound)
        results = []
        coord_bound = max(10, Int(ceil(sqrt(Float64(norm_bound)))) * 5)
        for i in -coord_bound:coord_bound, j in -coord_bound:coord_bound
            gamma = i * B[1] + j * B[2]
            iszero(gamma) && continue
            n = abs(norm(gamma))
            n > norm_bound && continue
            I = try ideal(ord, gamma) catch; continue end
            coprime_to_m_ideal(I) || continue
            push!(results, (gamma, I, n))
        end
        sort!(results, by = x -> x[3])
        return results
    end

    norm_bound = 200
    max_norm_bound = 10^7

    while length(collected) < N && norm_bound <= max_norm_bound
        println("Trying norm bound: ", norm_bound)
        candidates = candidates_up_to_norm(norm_bound)
        println("  Candidates to check: ", length(candidates))

        for (gamma, I, n) in candidates
            abs(norm(gamma)) == 1 && continue

            is_new = true
            if are_equivalent(gamma, one(ord))
                is_new = false
            end
            if is_new
                for delta in seen
                    if are_equivalent(gamma, delta)
                        is_new = false
                        break
                    end
                end
            end

            if is_new
                push!(seen, gamma)
                push!(collected, (I, gamma, n))
                println("  Found class ", length(collected), 
                        ": generator=", gamma, " norm=", n)
                length(collected) >= N && break
            end
        end

        length(collected) >= N && break
        println("Expanding norm bound from ", norm_bound, " to ", norm_bound * 10)
        norm_bound *= 10
    end

    if length(collected) < N
        println("Warning: only found ", length(collected), " of ", N, " classes")
    end

    sort!(collected, by = x -> x[3])
    return collected
end
=#


#=Hybrid version 1
function enumerate_principal_ray_classes(K, O, m, Sigma, N)
    N == 0 && return []

    OK = maximal_order(K)
    B = basis(O)

    U, mU = unit_group(O)

    function units_mod_m()
        torsion = [one(K), -one(K)]
        eps = nothing
        for i in 1:ngens(U)
            u = elem_in_nf(mU(U[i]))
            if !(u == one(K) || u == -one(K))
                eps = u
                break
            end
        end
        eps === nothing && return torsion

        seen_powers = []
        current = one(K)
        for _ in 1:10000
            push!(seen_powers, current)
            current = current * eps
            in_order = true
            diff_O = try
                O(current - 1)
            catch
                in_order = false
                nothing
            end
            if in_order && diff_O in m
                break
            end
        end

        all_units = []
        for t in torsion, p in seen_powers
            push!(all_units, t * p)
        end
        return all_units
    end

    unit_reps = units_mod_m()

    function is_invertible(I)
        try
            return ring_of_multipliers(I) == O
        catch
            return false
        end
    end

    function cong_one_mod_m(alpha)
        try
            diff_O = O(alpha - 1)
            return diff_O in m
        catch
            return false
        end
    end

    function positive_on_Sigma(alpha)
        for rho in Sigma
            evaluate(alpha, rho) > 0 || return false
        end
        return true
    end

    function coprime_to_m(gamma)
        try
            I = ideal(O, O(gamma))
            return isone(I + m)
        catch
            return false
        end
    end

    function are_equivalent(gamma, delta)
        try
            g = elem_in_nf(gamma)
            d = elem_in_nf(delta)
            I_g = ideal(O, O(g))
            I_d = ideal(O, O(d))
            I_g == I_d && return true
            base_ratio = g * inv(d)
            for u in unit_reps
                candidate = u * base_ratio
                if cong_one_mod_m(candidate) && positive_on_Sigma(candidate)
                    return true
                end
            end
            return false
        catch
            return false
        end
    end

    collected = []
    seen = []

    # collect all elements up to current coordinate bound,
    # sort by norm, process in order, expand if needed
    coord_bound = 10
    processed_up_to = 0

    while length(collected) < N
        # gather all candidates up to coord_bound
        candidates = []
        for i in -coord_bound:coord_bound, j in -coord_bound:coord_bound
            gamma = i * B[1] + j * B[2]
            iszero(gamma) && continue
            coprime_to_m(elem_in_nf(gamma)) || continue
            I = try ideal(O, gamma) catch; continue end
            is_invertible(I) || continue
            push!(candidates, (gamma, I, abs(norm(gamma))))
        end

        # sort by norm and process only new ones
        sort!(candidates, by = x -> x[3])

        for (gamma, I, h) in candidates
            # skip if norm is below what we already processed
            # (these were handled in a previous iteration)
            h <= processed_up_to && continue

            is_new = true
            if are_equivalent(gamma, one(O))
                is_new = false
            end
            if is_new
                for delta in seen
                    if are_equivalent(gamma, delta)
                        is_new = false
                        break
                    end
                end
            end

            if is_new
                push!(seen, gamma)
                push!(collected, (I, gamma, h))
                println("  Found class ", length(collected), ": generator=", gamma, " norm=", h)
                length(collected) >= N && break
            end
        end

        length(collected) >= N && break

        # track the maximum norm we processed so we don't redo work
        if !isempty(candidates)
            processed_up_to = maximum(x[3] for x in candidates)
        end

        println("Expanding coord bound from ", coord_bound, " to ", coord_bound * 2)
        coord_bound *= 2

        coord_bound > 100000 && (println("Warning: coord bound too large, stopping"); break)
    end

    sort!(collected, by = x -> x[3])
    return collected
end
=#

#New version (claude says this is faster 4_16_2026)
#=
function enumerate_principal_ray_classes(K, O, m, Sigma, N)
    N == 0 && return []

    OK = maximal_order(K)
    B = basis(O)

    U, mU = unit_group(O)

    function units_mod_m()
        torsion = [one(K), -one(K)]
        eps = nothing
        for i in 1:ngens(U)
            u = elem_in_nf(mU(U[i]))
            if !(u == one(K) || u == -one(K))
                eps = u
                break
            end
        end
        eps === nothing && return torsion

        seen_powers = []
        current = one(K)
        for _ in 1:10000
            push!(seen_powers, current)
            current = current * eps
            in_order = true
            diff_O = try
                O(current - 1)
            catch
                in_order = false
                nothing
            end
            if in_order && diff_O in m
                break
            end
        end

        all_units = []
        for t in torsion, p in seen_powers
            push!(all_units, t * p)
        end
        return all_units
    end

    unit_reps = units_mod_m()
    println("Number of unit reps mod m: ", length(unit_reps))

    function is_invertible(I)
        try
            return ring_of_multipliers(I) == O
        catch
            return false
        end
    end

    function cong_one_mod_m(alpha)
        try
            diff_O = O(alpha - 1)
            return diff_O in m
        catch
            return false
        end
    end

    function positive_on_Sigma(alpha)
        for rho in Sigma
            evaluate(alpha, rho) > 0 || return false
        end
        return true
    end

    function coprime_to_m(gamma)
        try
            I = ideal(O, O(gamma))
            return isone(I + m)
        catch
            return false
        end
    end

    function are_equivalent(gamma, delta)
        try
            g = elem_in_nf(gamma)
            d = elem_in_nf(delta)
            I_g = ideal(O, O(g))
            I_d = ideal(O, O(d))
            I_g == I_d && return true
            base_ratio = g * inv(d)
            for u in unit_reps
                candidate = u * base_ratio
                if cong_one_mod_m(candidate) && positive_on_Sigma(candidate)
                    return true
                end
            end
            return false
        catch
            return false
        end
    end

    collected = []
    seen = []

    # enumerate over a flat list of (i,j) pairs sorted by abs(norm(i*B[1]+j*B[2]))
    # use an expanding bound but collect ALL candidates up to each bound first,
    # then sort by norm before processing
    max_coord = 1000
    println("Building candidate list up to coord bound ", max_coord, "...")

    candidates = []
    for i in -max_coord:max_coord, j in -max_coord:max_coord
        gamma = i * B[1] + j * B[2]
        iszero(gamma) && continue
        coprime_to_m(elem_in_nf(gamma)) || continue
        I = try ideal(O, gamma) catch; continue end
        is_invertible(I) || continue
        push!(candidates, (gamma, I, abs(norm(gamma))))
    end

    println("Total candidates found: ", length(candidates))
    sort!(candidates, by = x -> x[3])

    for (gamma, I, h) in candidates
        is_new = true
        if are_equivalent(gamma, one(O))
            is_new = false
        end
        if is_new
            for delta in seen
                if are_equivalent(gamma, delta)
                    is_new = false
                    break
                end
            end
        end

        if is_new
            push!(seen, gamma)
            push!(collected, (I, gamma, h))
            println("  Found class ", length(collected), ": generator=", gamma, " norm=", h)
            length(collected) >= N && break
        end
    end

    if length(collected) < N
        println("Warning: only found ", length(collected), " of ", N, " classes within coord bound ", max_coord)
    end

    sort!(collected, by = x -> x[3])
    return collected
end
=#

#=
K, a = quadratic_field(10)
OK = maximal_order(K)
O2 = Order(K, [K(1), 2*a])

m = ideal(O2, 3)
plcs = real_places(K)
Sigma = InfPlc[]   # or plcs for totally positive condition

println("Sigma is: ", Sigma)

N = 10  # number of classes to find

result = enumerate_principal_ray_classes(K, O2, m, Sigma, N)

for (I, gamma, h) in result
    println("Generator: ", gamma, "  Norm: ", h, "  Ideal: ", I)
end
=#