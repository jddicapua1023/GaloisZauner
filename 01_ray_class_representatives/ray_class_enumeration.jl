# ray_class_enumeration.jl
#
# Enumerates explicit principal ray class representatives.
#
#   - ray_class_group_order:          order of a ray class group, computed
#                                      from unit_group_quotients.jl's two
#                                      quotient constructions plus Hecke's
#                                      built-in ray_class_group, as a
#                                      consistency check.
#   - enumerate_principal_ray_classes: explicit search for N principal ray
#                                      class representatives (elements
#                                      congruent to 1 mod m, positive at Sigma).
#   - ray_class_unit_reps, ray_class_same_mod_m, ray_class_same_sign_on_Sigma,
#     ray_class_are_equivalent, ray_class_is_trivial_class: standalone,
#     reusable versions of the machinery enumerate_principal_ray_classes
#     builds internally (L_prime, sign_at_place, same_mod_m, same_sign_on_Sigma,
#     are_equivalent, is_trivial_class). These duplicate the exact logic above
#     without touching enumerate_principal_ray_classes itself, so that other
#     code (e.g. computing the elementary divisors of the full ray class
#     group, in 02_galois_action_ghost_sics/galois_action_ghost_sics.jl) can
#     reuse the SAME equivalence test used to build the N distinct principal
#     ray classes in the first place, rather than reimplementing it
#     (incorrectly) from scratch. ray_class_unit_reps is a live dependency of
#     galois_action_ghost_sics.jl's ray_class_group_elementary_divisors.
#
# (Extracted from enumeration_with_infinite_places.jl. That file accumulated
# 8 earlier draft implementations of enumerate_principal_ray_classes as
# dated #= =# block comments during development; only the version below was
# ever actually live, since Julia only keeps the final method definition and
# the drafts were block-commented out anyway. The drafts were removed here
# for clarity per project owner's request; the original uploaded file is
# left untouched if that history is ever needed again.)

using Hecke
include("unit_group_quotients.jl")

function ray_class_group_order(OK, O, mp, m, delta, Sigma_prime, Sigma)
    Q_res, structure = residue_unit_quotient(OK, O, mp, m, delta, Sigma_prime, Sigma)
    N1 = prod(elementary_divisors(Q_res))

    Q_ray, reps, phi = ray_unit_quotient(OK, O, mp, m, Sigma_prime, Sigma)
    N2 = order(Q_ray)

    R, mR = ray_class_group(mp, Sigma_prime)
    N3 = order(R)

    return divexact(N3 * N1, N2)
end

function enumerate_principal_ray_classes(K, ord, m, Sigma, N)
    N == 0 && return []

    OK = maximal_order(K)
    B = basis(ord)

    U, mU = unit_group(OK)

    # Find fundamental unit of OK
    eps = nothing
    for i in 1:ngens(U)
        u = elem_in_nf(mU(U[i]))
        if !(u == one(K) || u == -one(K))
            eps = u
            break
        end
    end

    function sign_at_place(x, rho)
        emb = embedding(rho)
        for prec in [64, 128, 256, 512, 1024]
            try
                v = emb(x, prec)
                return sign(Int64, v)
            catch
                continue
            end
        end
        error("Could not determine sign even at high precision")
    end

    function build_L()
        k = 0
        current = one(K)
        for attempt in 1:100000
            current = current * eps
            in_ord = try ord(OK(current)); true catch; false end
            if in_ord
                diff_in_m = try ord(OK(current - 1)) in m catch; false end
                pos = try
                    all(rho -> sign_at_place(current, rho) > 0, Sigma)
                catch
                    false
                end
                if diff_in_m && pos
                    k = attempt
                    break
                end
            end
        end
        k == 0 && error("Could not find period k within bound")

        L = []
        current = one(K)
        for i in 0:k
            push!(L, current)
            push!(L, -current)
            current = current * eps
        end
        return L
    end

    L = build_L()

    # L' = subset of L whose elements are in ord
    L_prime = filter(u -> (try ord(OK(u)); true catch; false end), L)
    println("Size of L: ", length(L))
    println("Size of L': ", length(L_prime))

    function same_mod_m(alpha, beta)
        try
            diff = K(alpha) - K(beta)
            diff_OK = OK(diff)
            diff_ord = ord(diff_OK)
            return diff_ord in m
        catch
            return false
        end
    end

    function same_sign_on_Sigma(alpha, beta)
        for rho in Sigma
            if sign_at_place(K(alpha), rho) != sign_at_place(K(beta), rho)
                return false
            end
        end
        return true
    end

    function coprime_to_m_ideal(I)
        try
            return isone(OK * I + ideal(OK, 1) * norm(m))
        catch
            return false
        end
    end

    function is_trivial_class(alpha)
        g = elem_in_nf(alpha)
        I_g = try ideal(ord, ord(g)) catch; return false end
        for u in L_prime
            candidate = g * u
            if same_mod_m(candidate, one(K)) && same_sign_on_Sigma(candidate, one(K))
                return true
            end
        end
        return false
    end

    function are_equivalent(beta, gamma)
        g = elem_in_nf(beta)
        d = elem_in_nf(gamma)
        # NOTE: do NOT short-circuit on I_g == I_d (same ideal) or use it as a
        # necessary condition either. Ray-equivalence of two principal-in-O
        # generators gamma, delta depends only on whether delta/gamma is
        # ray-trivial up to a global unit -- i.e. exactly the congruence
        # gamma*u == delta (mod m) tested by the loop below. This holds
        # regardless of whether (gamma) and (delta) are the same ideal:
        # generators of genuinely different-norm principal ideals can still
        # be ray-equivalent, and generators of the SAME ideal can fail to be
        # ray-equivalent if they differ by a non-ray-trivial unit. The old
        # code's "I_g == I_d && return true" shortcut incorrectly forced the
        # former case to always merge, over-collapsing distinct ray classes.
        _ = (try ideal(ord, ord(g)) catch; return false end)
        _ = (try ideal(ord, ord(d)) catch; return false end)
        for u in L_prime
            candidate = g * u
            if same_mod_m(candidate, d) && same_sign_on_Sigma(candidate, d)
                return true
            end
        end
        return false
    end

    collected = []

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
            # NOTE (9/15): a principal ideal (gamma), gamma != 0, is ALWAYS
            # invertible (inverse (1/gamma)), so an is_invertible filter here
            # is a no-op; the earlier "even-norm generators are not invertible"
            # diagnosis was wrong. Even-norm gamma (coprime to m=15 but not to
            # the conductor prime 2) ARE in J_m^*(O). The real d=15 bug was the
            # consumer contracting gamma*OK back to O - see
            # ray_class_group_representatives in the main file.
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

            is_new = !is_trivial_class(gamma)

            if is_new
                for (_, delta, _) in collected
                    if are_equivalent(gamma, delta)
                        is_new = false
                        break
                    end
                end
            end

            if is_new
                # Return the O-ideal (gamma) itself, NOT OK*I: lifting to OK and
                # contracting back is not the identity when gamma meets the
                # conductor (d=15, even norms), and every consumer wants an
                # O-ideal anyway.
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

# ---------------------------------------------------------------------------
# Standalone, reusable versions of the machinery enumerate_principal_ray_classes
# builds internally (L_prime, sign_at_place, same_mod_m, same_sign_on_Sigma,
# are_equivalent, is_trivial_class). These are purely additive -- they
# duplicate the exact logic above without touching enumerate_principal_ray_classes
# itself -- so that other code (e.g. computing the elementary divisors of the
# full ray class group) can reuse the SAME equivalence test that was used to
# build the N distinct principal ray classes in the first place, rather than
# reimplementing it (incorrectly) from scratch.
# ---------------------------------------------------------------------------
function ray_class_unit_reps(K, ord, m, Sigma)
    OK = maximal_order(K)
    U, mU = unit_group(OK)

    eps = nothing
    for i in 1:ngens(U)
        u = elem_in_nf(mU(U[i]))
        if !(u == one(K) || u == -one(K))
            eps = u
            break
        end
    end
    eps === nothing && error("ray_class_unit_reps: could not find an infinite-order unit generator of OK")

    embs = Dict(rho => embedding(rho) for rho in Sigma)  # compute once, reuse

    function sign_at_place(x, rho)
        emb = embs[rho]
        for prec in [64, 128, 256, 512, 1024]
            try
                v = emb(x, prec)
                return sign(Int64, v)
            catch
                continue
            end
        end
        error("Could not determine sign even at high precision")
    end

    k = 0
    current = one(K)
    for attempt in 1:100000
        current = current * eps
        in_ord = try ord(OK(current)); true catch; false end
        if in_ord
            diff_in_m = try ord(OK(current - 1)) in m catch; false end
            pos = try
                all(rho -> sign_at_place(current, rho) > 0, Sigma)
            catch
                false
            end
            if diff_in_m && pos
                k = attempt
                break
            end
        end
    end
    k == 0 && error("ray_class_unit_reps: could not find period k within bound")

    L = []
    current = one(K)
    for i in 0:k
        push!(L, current)
        push!(L, -current)
        current = current * eps
    end

    L_prime = filter(u -> (try ord(OK(u)); true catch; false end), L)
    return L_prime, sign_at_place
end

function ray_class_same_mod_m(K, OK, ord, m, alpha, beta)
    try
        diff = K(alpha) - K(beta)
        diff_OK = OK(diff)
        diff_ord = ord(diff_OK)
        return diff_ord in m
    catch
        return false
    end
end

function ray_class_same_sign_on_Sigma(K, Sigma, sign_at_place, alpha, beta)
    for rho in Sigma
        if sign_at_place(K(alpha), rho) != sign_at_place(K(beta), rho)
            return false
        end
    end
    return true
end

function ray_class_are_equivalent(K, OK, ord, m, Sigma, L_prime, sign_at_place, beta, gamma)
    g = elem_in_nf(beta)
    d = elem_in_nf(gamma)
    # See NOTE in enumerate_principal_ray_classes' are_equivalent: same-ideal
    # is neither sufficient nor necessary for ray-equivalence, so we must not
    # short-circuit on I_g == I_d. The L_prime loop below is the correct and
    # complete test on its own.
    for u in L_prime
        candidate = g * u
        if ray_class_same_mod_m(K, OK, ord, m, candidate, d) &&
           ray_class_same_sign_on_Sigma(K, Sigma, sign_at_place, candidate, d)
            return true
        end
    end
    return false
end

function ray_class_is_trivial_class(K, OK, ord, m, Sigma, L_prime, sign_at_place, alpha)
    return ray_class_are_equivalent(K, OK, ord, m, Sigma, L_prime, sign_at_place, alpha, one(K))
end