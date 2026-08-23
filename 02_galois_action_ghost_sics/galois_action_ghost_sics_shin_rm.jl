using Random
#Random.seed!(305817264)
Random.seed!(672049183)
#Random.seed!(891357246)

# galois_action_ghost_sics.jl
#
# Uses the explicit ray class group representatives (from
# ../01_ray_class_representatives/) to describe the Galois action on
# ghost SICs: for each ray class representative gamma, computes the
# "twisted" ghost overlaps/character (twisted_chi_general) and compares
# against the untwisted ghost invariants.
#
# Current/canonical version (per project owner, Aug 2026): find_alpha_rhoB_s1_s2_O
# uses a TWO-STAGE JOINT search (a single primitive-vector search over the
# reduced basis of b*d*O finds alpha and rho_B together, after which s1, s2
# are solved for in closed form from an explicit generator of
# Jb_O * (alpha)^{-1}) rather than the three-separate-bounded-searches
# approach used in archive/galois_action_naive_search_OLD.jl.
#
# b_O selection uses DISC-LOG-FREE Picard-class verification
# (is_principal_O_verified / find_b_with_verified_principal_product /
# lattice_canonical / quad_conj), added after independent exact-arithmetic
# verification (d=21, D=44, q=3) showed the ORIGINAL selection
# (b_O = mP_O(-u_B), certified only via Hecke's picard_group discrete log)
# delivered a representative of the WRONG class ([J] instead of [J]^{-1}) -
# since the output lattice Z+Z*rho_B is always in b_O's class, this silently
# corrupted the final ray class representative. b_O is now selected by
# directly EXHIBITING a generator of J*b_O (maximal-order machinery + HNF
# lattice-equality only, no picard_group/inv() dependence), and
# find_alpha_rhoB_s1_s2_O independently re-verifies the full criterion
# "(s2*rho_B - s1)*(Z+Z*rho_B)^{-1} is in the Picard class of J" before
# returning, erroring loudly (pointing at b_O selection, not the (m,n)
# search) if it does not hold. A diagnostic print compares Hecke's
# picard_group discrete log against the verified-correct class on every run,
# to surface whether the disc log itself is inverted for a given order.
#
# Entry point: test_all_twists(d) - runs the twisted character computation
# over all (non-)principal ray class representatives for AdmissibleTuple(d).
# Also provides ray_class_group_elementary_divisors/ray_class_group_data for
# inspecting the ray class group structure itself, and
# test_ray_class_group_size(_first_n_tuples) for consistency-checking N1*N3/N2
# against the number of explicit representatives actually constructed.
#
# (Renamed from galois_twist_vector_lattice_search.jl; superseded
# archive/galois_action_naive_search_OLD.jl as canonical, Aug 2026 - see
# that file's header for why.)

#using Random
#Random.seed!(918273645)   # fixed seed - Hecke's picard_group/unit_group/ideal
                          # routines use randomized algorithms internally with
                          # no fixed seed by default, which is the most likely
                          # source of run-to-run variability otherwise seen in
                          # the vector map from multiplication. Other seeds
                          # confirmed to also work if this one ever needs to
                          # be swapped out: 427193856, 561904732, 738214905,
                          # 20260731, 918273646.

using Hecke
using Zauner
import Zauner: _ghost_invariants, _dualize, _approx_complex_intersection, roots, Vandermonde, pow_to_elem_sym_poly, shin_rm
using GenericFFT: fft
using LinearAlgebra
include("../01_ray_class_representatives/unit_group_quotients.jl")
include("../01_ray_class_representatives/ray_class_enumeration.jl")
include("../01_ray_class_representatives/coprime_representatives.jl")
# NOTE: num_principal_ray_classes and ray_class_group_representatives are
# NOT included from ../01_ray_class_representatives/ray_class_group.jl on
# purpose - this file defines its own, more robust versions of both further
# below (see the comment above num_principal_ray_classes), which shadow the
# ray_class_group.jl definitions if both were loaded together. This mirrors
# how the original galois_twist_vector_lattice_search.jl behaved: it never
# included full_rayclassgrp.jl either, defining these locally instead.


function ghost_overlaps(ψ::AbstractVector)
    φ = circshift(reverse(ψ), 1)
    d = length(ψ)
    scale = sqrt(eltype(ψ)(d + 1)) / (φ'ψ)
    return real.(scale .* [φ' * wh(p, q, ψ) for p = 0:d-1, q = 0:d-1])
end

function ghost_overlap_test_complex(ovs, d)
    max_dev = 0.0
    for p1 = 0:d-1
        for p2 = 0:d-1
            (p1 == 0 && p2 == 0) && continue
            ov_p = ovs[p1+1, p2+1]
            ov_mp = ovs[mod(-p1, d)+1, mod(-p2, d)+1]
            dev = abs(abs(ov_p * ov_mp) - 1/(d+1))
            max_dev = max(max_dev, dev)
        end
    end
    return max_dev
end

function express_in_basis(x, rho, D)
    p = coeff(x, 0)
    q_coeff = coeff(x, 1)
    c1 = coeff(rho, 0)
    c2 = coeff(rho, 1)
    coord2 = q_coeff / c2
    coord1 = p - q_coeff * c1 / c2
    return [coord1, coord2]
end

function multiplication_matrix_mod_d(gamma, rho, d, D)
    col1 = express_in_basis(gamma, rho, D)
    col2 = express_in_basis(gamma * rho, rho, D)
    to_int_mod(x) = Int(mod(ZZ(numerator(x)) * ZZ(invmod(ZZ(denominator(x)), ZZ(d))), ZZ(d)))
    M = [to_int_mod(col1[1]) to_int_mod(col2[1]);
         to_int_mod(col1[2]) to_int_mod(col2[2])]
    return M
end

function squarefree_part(n::Int)
    fac = factor(n)
    sq_part = 1
    for (p, e) in fac
        if isodd(e)
            sq_part *= p
        end
    end
    return sq_part
end

function rationals_by_height(max_h)
    # NOTE: rewritten for efficiency at large max_h. The previous version
    # escalated an internal "h" from 1 to max_h and, at every level, rescanned
    # ALL q in 1:h, p in -h:h AGAIN (redundant with smaller h's already done),
    # while also doing an O(n) linear "r in rats" scan per candidate to dedup.
    # That's effectively O(max_h^4)-ish work for what should be a single
    # O(max_h^2) sweep. Since gcd(p,q)==1 already guarantees each (p,q) gives
    # a fraction in lowest terms, and lowest-terms fractions are unique per
    # (p,q), no dedup scan is needed at all - just collect directly.
    rats = [QQ(0)]
    for q in 1:max_h
        for p in -max_h:max_h
            gcd(abs(p), q) == 1 || continue
            push!(rats, QQ(p, q))
        end
    end
    return rats
end

function to_du_mod_d(u_val, d)
    du = u_val
    k = floor(du / d)
    du_mod = du - d * k
    return du_mod
end

function find_class_representatives(O, F, K)
    D = Int(F.D)
    a = gen(K)

    Dr = squarefree_part(D)
    sqrt_Dr = a // isqrt(D ÷ Dr)

    P_O, mP_O = picard_group(O)
    d_val = Int(F.d)

    rho_reps = Dict{FinGenAbGroupElem, Any}()
    gQ_reps  = Dict{FinGenAbGroupElem, Any}()
    F_class  = nothing

    for g in P_O
        I_g = mP_O(g)
        b = basis(I_g)
        b1 = elem_in_nf(b[1])
        b2 = elem_in_nf(b[2])

        b1_rat = coeff(b1, 0)
        rho_g = b2 / b1_rat

        rho_reps[g] = rho_g

        c1 = coeff(rho_g, 0)
        c2 = coeff(rho_g, 1)
        p0 = c1^2 - c2^2 * D
        p1_form = -2 * c1
        p2_form = 1
        scale = lcm(Int(denominator(QQ(p0))),
                   Int(denominator(QQ(p1_form))), 1)
        a_int = Int(round(scale * p2_form))
        b_int = Int(round(scale * p1_form))
        c_int = Int(round(scale * p0))
        g_int = gcd(a_int, b_int, c_int)
        a_int ÷= g_int
        b_int ÷= g_int
        c_int ÷= g_int
        gQ_cand = binary_quadratic_form(a_int, b_int, c_int)
        V = reduced_hj_orbit(gQ_cand)
        gQ_reps[g] = minimal_hj_stabilizer(V, d_val)

        println("  class ", g, ": rho = ", rho_g, " gQ = ", gQ_reps[g])

        if gQ_reps[g] == F.Q
            F_class = g
            println("  -> this is the class of F.Q")
        end
    end

    D_Q = Int(discriminant(F.Q))
    _, f_Q = coredisc(D_Q)
    sqrt_DQ = f_Q * sqrt_Dr
    rho_F = (-F.Q.b + sqrt_DQ) / (2 * F.Q.a)

    if F_class === nothing
        for g in P_O
            if rho_reps[g] == rho_F
                F_class = g
                println("  F.Q class found by rho match: ", g)
                break
            end
        end
    end

    if F_class === nothing
        println("  WARNING: could not determine class of F.Q, defaulting to trivial")
        F_class = zero(P_O)
    end

    rho_reps[F_class] = rho_F
    gQ_reps[F_class]  = F.Q
    println("  F.Q class = ", F_class, " rho_F = ", rho_F)

    return P_O, mP_O, rho_reps, gQ_reps, F_class
end

function find_rho_j_and_coords(product, rho_j)
    p0 = coeff(product, 0)
    p1 = coeff(product, 1)
    c0 = coeff(rho_j, 0)
    c1 = coeff(rho_j, 1)
    iszero(c1) && error("rho_j has zero irrational part")
    u2 = p1 / c1
    u1 = u2 * c0 - p0
    return QQ(u2), QQ(u1)
end

function find_s1_s2(J, rho_B, rho, u_B, P_O, mP_O, O, K, OK, d_val, rho_j)
    J_OK  = extend(J, OK)
    b_OK  = extend(mP_O(-u_B), OK)
    dO_OK = ideal(OK, d_val)
    bdO_OK = b_OK * dO_OK
    Jb_OK = J_OK * b_OK

    is_princ_Jb, Jb_gen_fac = is_principal_with_data(Jb_OK)
    !is_princ_Jb && error("Jb_OK is not principal")
    Jb_gen = elem_in_nf(Jb_gen_fac)
    println("  Jb_gen = ", Jb_gen)
    println("  norm(Jb_gen) = ", norm(K(Jb_gen)))

    target_OK = bdO_OK * inv(fractional_ideal(OK, [K(1), rho_B]))
    is_princ, alpha_fac = is_principal_with_data(target_OK)
    !is_princ && error("b*dO / (Z+rho_B*Z) is not principal")
    alpha_base = elem_in_nf(alpha_fac)

    places = real_places(K)
    U_OK, mU_OK = unit_group(OK)
    eps_OK = elem_in_nf(mU_OK(U_OK[2]))

    best_alpha = nothing
    best_size = Inf
    ALPHA_SEARCH_RANGE = -150:150
    for k in ALPHA_SEARCH_RANGE
        for sgn in [1, -1]
            a_try = K(sgn) * alpha_base * eps_OK^k
            all(sign(a_try, p) > 0 for p in places) || continue
            sz = Float64(abs(coeff(a_try, 0))) + Float64(abs(coeff(a_try, 1)))
            if sz < best_size
                best_size = sz
                best_alpha = a_try
            end
        end
    end
    best_alpha === nothing && error("Could not find totally positive alpha in range ", ALPHA_SEARCH_RANGE)
    alpha = best_alpha

    # Explicit verification: alpha must be totally positive (positive under
    # every real embedding of K), since downstream code (e.g. is_valid's
    # sign convention, and alpha's role as "= d" in the ideal quotient) relies
    # on this. This should always hold given the filter above, but we check
    # it explicitly rather than assuming the search logic is bug-free.
    alpha_signs = [sign(alpha, p) for p in places]
    all(s -> s > 0, alpha_signs) ||
        error("Selected alpha is NOT totally positive! signs = ", alpha_signs,
              ", alpha = ", alpha)
    println("  alpha = ", alpha, " (= d)  [verified totally positive: signs = ", alpha_signs, "]")

    println("  norm(Jb_OK) = ", norm(Jb_OK))

    c_rhoB = coeff(rho_B, 1)
    println("  rho_B = ", rho_B)
    println("  c_rhoB = ", c_rhoB)

    emb_conj = embedding(places[1])
    rhoB_conj = Float64(emb_conj(rho_B))
    println("  conj(rho_B) = ", rhoB_conj)

    function to_float(q)
        return Float64(BigInt(numerator(q))) / Float64(BigInt(denominator(q)))
    end

    function is_valid(s1, s2)
        s1_f = to_float(s1)
        s2_f = to_float(s2)
        val = s2_f * rhoB_conj - s1_f
        return val > 0
    end

    function try_elem_with_alpha(elem, alpha)
        try
            check_ideal = fractional_ideal(OK, [elem])
            check_ideal == Jb_OK || return nothing
        catch
            return nothing
        end
        ratio = elem / alpha
        r0 = coeff(ratio, 0)
        r1 = coeff(ratio, 1)
        s1 = QQ(-r0)
        s2 = QQ(r1 / c_rhoB)
        return s1, s2
    end

    result = try_elem_with_alpha(Jb_gen, alpha)
    if result !== nothing
        s1, s2 = result
        if is_valid(s1, s2)
            println("  found s1=", s1, " s2=", s2, " j=0, sgn=1")
            return s1, s2
        end
    end
    result = try_elem_with_alpha(-Jb_gen, alpha)
    if result !== nothing
        s1, s2 = result
        if is_valid(s1, s2)
            println("  found s1=", s1, " s2=", s2, " j=0, sgn=-1")
            return s1, s2
        end
    end

    for j in 1:200
        for jj in [j, -j]
            for sgn in [1, -1]
                elem = K(sgn) * Jb_gen * eps_OK^jj
                result = try_elem_with_alpha(elem, alpha)
                result === nothing && continue
                s1, s2 = result
                if is_valid(s1, s2)
                    println("  found s1=", s1, " s2=", s2, " j=", jj, " sgn=", sgn)
                    return s1, s2
                end
            end
        end
    end

    error("Could not find valid s1, s2")
end

# =============================================================================
# O-NATIVE VERSION — works directly with ideals of the non-maximal order O
# rather than extending everything to OK. alpha, rho_B, and (s1, s2) are all
# found by a single JOINT search ordered by height, rather than three separate
# bounded searches run in sequence. This guarantees we don't miss a genuine
# solution just because rho_B or alpha happened to be "settled" early at a
# smaller height by an independent earlier pass.
# =============================================================================

# -----------------------------------------------------------------------------
# DISC-LOG-FREE Picard-class machinery (quadratic orders only).
#
# WHY THIS EXISTS. The correctness of find_alpha_rhoB_s1_s2_O hinges on ONE
# class condition: the lattice Z+Z*rho_B it outputs must lie in the INVERSE
# of J's Picard class (so that (s2*rho_B - s1) * (Z+Z*rho_B)^{-1} lands in the
# class of J). Since Z+Z*rho_B = (1/alpha)*b*d*O by construction, its class is
# exactly the class of b_O -- so everything reduces to b_O really being in
# class [J]^{-1}. Previously b_O's class was certified ONLY through Hecke's
# picard_group discrete log (mP_O \ I == -u_B). Independent verification in
# exact arithmetic (PARI + hand-checked F_9 residues, d=21 / D=44 / q=3 run)
# showed the produced lattice actually sat in class +[J], not [J]^{-1} --
# i.e. that certification chain delivered the WRONG class, and, since
# Pic(O) = Z/4 there, the answer produced was genuinely wrong (off by the
# order-2 element, exactly the F_9^x/F_3^x ratio w observed when checking
# (100/21*rho_B + 53/7)*l against <5, 3/2*sqrt(44)+3>).
#
# These helpers certify the ONE condition that matters -- "J * b_O is
# principal in O" -- by EXHIBITING an explicit generator, using only:
#   * element arithmetic + integer-matrix HNF (lattice equality),
#   * is_principal_with_data on the MAXIMAL order (battle-tested), and
#   * a bounded twist by units of O_K to descend the generator to O.
# No picard_group map, no non-maximal-order fractional-ideal ==, no inv().
# Any sign/direction convention of mP_O drops out entirely.
# -----------------------------------------------------------------------------

# Signals that a candidate rho_j (found in twisted_chi_general's outer
# retry loop) must be rejected ENTIRELY - not just at the specific (p1,p2)
# grid point where the failure was detected - because a per-grid-point
# invariant that must hold for EVERY (p1,p2) failed for this one. Caught in
# the outer retry loop, which then excludes this rho_j candidate and tries
# a different one, re-running the whole (p1,p2) grid from scratch.
struct RhoJGridRejected <: Exception
    msg::String
end

function _nf_elem_of(K, x)
    try
        return elem_in_nf(x)
    catch
    end
    try
        return K(x)
    catch
    end
    try
        return K(evaluate(x))   # FacElem results from is_principal_with_data
    catch
    end
    return x
end

# Galois conjugate a + b*gen(K) -> a - b*gen(K)  (K real quadratic)
function quad_conj(K, x)
    return K(coeff(x, 0)) - K(coeff(x, 1)) * gen(K)
end

# Canonical form (den, 2x2 upper-triangular ZZ HNF, common content removed) of
# the Z-lattice in K spanned by `gens`, in coordinates over (1, gen(K)).
# Two full-rank lattices are equal iff their canonical forms are equal.
function lattice_canonical(K, gens)
    n = length(gens)
    den = one(ZZ)
    coords = Vector{Tuple{QQFieldElem, QQFieldElem}}(undef, n)
    for (i, g) in enumerate(gens)
        c0 = QQ(coeff(g, 0)); c1 = QQ(coeff(g, 1))
        coords[i] = (c0, c1)
        den = lcm(den, denominator(c0))
        den = lcm(den, denominator(c1))
    end
    M = zero_matrix(ZZ, n, 2)
    for i in 1:n
        M[i, 1] = numerator(coords[i][1] * den)
        M[i, 2] = numerator(coords[i][2] * den)
    end
    H = hnf(M)
    (iszero(H[1, 1]) && iszero(H[1, 2])) &&
        error("lattice_canonical: zero lattice")
    (n < 2 || (iszero(H[2, 1]) && iszero(H[2, 2]))) &&
        error("lattice_canonical: lattice not of full rank 2")
    c = gcd(gcd(H[1, 1], H[1, 2]), gcd(H[2, 1], H[2, 2]))
    c = gcd(c, den)
    H2 = zero_matrix(ZZ, 2, 2)
    for i in 1:2, j in 1:2
        H2[i, j] = divexact(H[i, j], c)
    end
    return (divexact(den, c), H2)
end

lattices_equal(K, gensA, gensB) = lattice_canonical(K, gensA) == lattice_canonical(K, gensB)

# Z-generators of the product module A*B from Z-generators of A and B.
# (Valid because A and B are O-modules: the Z-span of pairwise products is
# already the full product module.)
lattice_product_gens(gensA, gensB) = [a * b for a in gensA for b in gensB]

# Certified principality test for an invertible fractional O-ideal given by
# Z-generators `gens` (K-elements): returns (true, gamma) with gamma an
# EXPLICIT verified generator (gamma * O equals the lattice, checked by HNF),
# or (false, nothing). Independent of picard_group / mP_O entirely.
function is_principal_O_verified(K, O, gens; unit_bound::Int = 60)
    OK = maximal_order(K)
    gens_nf = [_nf_elem_of(K, g) for g in gens]

    ext = try
        fractional_ideal(OK, gens_nf)
    catch e
        error("is_principal_O_verified: could not extend to OK: ", e)
    end
    fl, g0_raw = is_principal_with_data(ext)
    fl || return (false, nothing)   # not even principal over OK => not principal over O
    g0 = _nf_elem_of(K, g0_raw)

    U_OK, mU_OK = unit_group(OK)
    eps = nothing
    for i in 1:ngens(U_OK)
        cand = _nf_elem_of(K, mU_OK(U_OK[i]))
        (cand == one(K) || cand == -one(K)) && continue
        eps = cand
        break
    end
    eps === nothing && error("is_principal_O_verified: no infinite-order unit of OK found")

    b1 = _nf_elem_of(K, basis(O)[1])
    b2 = _nf_elem_of(K, basis(O)[2])

    canonA = lattice_canonical(K, gens_nf)
    for k in -unit_bound:unit_bound
        epsk = eps^k
        for s in (1, -1)
            cand = K(s) * g0 * epsk
            if lattice_canonical(K, [cand * b1, cand * b2]) == canonA
                return (true, cand)
            end
        end
    end
    # g0 generates the extension to OK but no unit twist of it generates the
    # O-lattice within the bound: either genuinely non-principal in O (the
    # obstruction lives in the conductor residue group, which has small
    # exponent, so the bound is ample) or unit_bound too small.
    return (false, nothing)
end

# Certified test for whether a fractional O-ideal (given by generators, not
# necessarily integral - is_principal_O_verified handles that) is principal
# AND generated by a TOTALLY POSITIVE element (positive at every real place
# of K). Built on is_principal_O_verified rather than searching separately.
#
# Why no extra search over sign is needed: O's fundamental unit here,
# eps = 10+theta, has norm N(eps) = +1 (verified earlier this session), so
# eps is itself totally positive (same sign at both places, and positive at
# the place where it's obviously large), hence every eps^k is totally
# positive and every -eps^k is totally negative - unit-twisting can NEVER
# turn a mixed-sign generator into a totally positive one. So: if the
# verified generator g has matching signs at both places, one global sign
# flip (g or -g) gives a totally positive generator; if g has MIXED signs,
# no totally positive generator exists for this ideal at all - it's a
# genuine rejection, not a search-depth issue.
function is_principal_totally_positive_verified(K, O, gens; unit_bound::Int = 60)
    ok, g = is_principal_O_verified(K, O, gens; unit_bound = unit_bound)
    ok || return (false, nothing)
    signs = [sign(g, p) for p in real_places(K)]
    if all(s -> s > 0, signs)
        return (true, g)
    elseif all(s -> s < 0, signs)
        return (true, -g)
    else
        return (false, nothing)
    end
end

# Like is_principal_totally_positive_verified, but for use where the
# generator alpha must ALSO satisfy an extra condition: alpha*companion must
# be an element of target_lattice. Since O's fundamental unit eps has norm
# +1 (verified earlier this session), every eps^k is totally positive, so
# there is a whole family {alpha_0*eps^k : k in Z} of totally positive
# generators of the SAME ideal - is_principal_totally_positive_verified only
# ever returns the first one found. This searches that whole family (within
# unit_bound) for one that ALSO satisfies the membership condition, rather
# than accepting the first totally positive one and erroring if membership
# happens to fail for that particular one.
function is_principal_totally_positive_verified_with_membership(K, O, gens, companion,
                                                                  target_lattice;
                                                                  unit_bound::Int = 200)
    OK = maximal_order(K)
    gens_nf = [_nf_elem_of(K, g) for g in gens]

    ext = try
        fractional_ideal(OK, gens_nf)
    catch e
        error("is_principal_totally_positive_verified_with_membership: could not extend to OK: ", e)
    end
    fl, g0_raw = is_principal_with_data(ext)
    fl || return (false, nothing)
    g0 = _nf_elem_of(K, g0_raw)

    U_OK, mU_OK = unit_group(OK)
    eps = nothing
    for i in 1:ngens(U_OK)
        cand = _nf_elem_of(K, mU_OK(U_OK[i]))
        (cand == one(K) || cand == -one(K)) && continue
        eps = cand
        break
    end
    eps === nothing && error("is_principal_totally_positive_verified_with_membership: ",
                              "no infinite-order unit of OK found")

    b1 = _nf_elem_of(K, basis(O)[1])
    b2 = _nf_elem_of(K, basis(O)[2])
    canonA = lattice_canonical(K, gens_nf)
    places = real_places(K)

    for k in 0:unit_bound
        for signed_k in (k, -k)
            epsk = eps^signed_k
            for s in (1, -1)
                cand = K(s) * g0 * epsk
                lattice_canonical(K, [cand * b1, cand * b2]) == canonA || continue

                signs = [sign(cand, p) for p in places]
                cand_tp = if all(x -> x > 0, signs)
                    cand
                elseif all(x -> x < 0, signs)
                    -cand
                else
                    continue   # mixed sign at this unit twist - skip, not a candidate
                end

                test_elem = cand_tp * companion
                in_target = try
                    test_elem in target_lattice
                catch
                    false
                end
                in_target && return (true, cand_tp)
            end
            k == 0 && break   # k=0 and -k=0 are the same, don't double-try
        end
    end
    return (false, nothing)
end


# a generator of J*b -- the actual condition the construction needs -- rather
# than by trusting a Picard discrete log. Same height-shell search pattern as
# find_coprime_picard_representative. Returns (b_O, mu) with mu a verified
# generator of J*b_O.
function find_b_with_verified_principal_product(O, K, J, moduli;
                                                 max_height::Int = 100,
                                                 unit_bound::Int = 60)
    b1 = _nf_elem_of(K, basis(O)[1])
    b2 = _nf_elem_of(K, basis(O)[2])
    J_gens_nf = [_nf_elem_of(K, x) for x in basis(J)]

    for height in 1:max_height
        for c1 in -height:height, c2 in -height:height
            for d1 in -height:height, d2 in -height:height
                (abs(c1) != height && abs(c2) != height &&
                 abs(d1) != height && abs(d2) != height) && continue

                alpha1 = c1 * b1 + c2 * b2
                alpha2 = d1 * b1 + d2 * b2
                iszero(alpha1) && iszero(alpha2) && continue

                I = try
                    gens = AbsSimpleNumFieldOrderElem[]
                    iszero(alpha1) || push!(gens, O(alpha1))
                    iszero(alpha2) || push!(gens, O(alpha2))
                    ideal(O, gens)
                catch
                    continue
                end

                cop = try
                    all(isone(I + mod_ideal) for mod_ideal in moduli)
                catch
                    false
                end
                cop || continue

                I_gens_nf = [_nf_elem_of(K, x) for x in basis(I)]
                prod_gens = lattice_product_gens(J_gens_nf, I_gens_nf)
                okp, mu = try
                    is_principal_O_verified(K, O, prod_gens; unit_bound = unit_bound)
                catch
                    (false, nothing)
                end
                okp || continue

                return I, mu
            end
        end
    end

    error("find_b_with_verified_principal_product: no coprime b with J*b verifiably ",
          "principal found within height ", max_height, " -- if this fires, raise ",
          "max_height/unit_bound before suspecting anything deeper.")
end

function find_coprime_picard_representative(O, mP_O, target_class, moduli; max_height::Int = 100)
    # Mirrors make_coprime_representatives (make_coprime_reps.jl): searches
    # O's own basis coordinates for TWO generators alpha1, alpha2 (height =
    # max of their four integer coordinates), builds the ideal they
    # generate, and accepts it once it's coprime to EVERY ideal in `moduli`
    # and lies in the target Picard class. Unlike make_coprime_representatives
    # (which collects representatives for many classes in one sweep), this
    # stops as soon as it finds one for the single target_class.
    b1 = elem_in_nf(basis(O)[1])
    b2 = elem_in_nf(basis(O)[2])

    for height in 1:max_height
        for c1 in -height:height
            for c2 in -height:height
                for d1 in -height:height
                    for d2 in -height:height
                        (abs(c1) != height && abs(c2) != height &&
                         abs(d1) != height && abs(d2) != height) && continue

                        alpha1 = c1 * b1 + c2 * b2
                        alpha2 = d1 * b1 + d2 * b2
                        iszero(alpha1) && iszero(alpha2) && continue

                        try
                            gens = AbsSimpleNumFieldOrderElem[]
                            iszero(alpha1) || push!(gens, O(alpha1))
                            iszero(alpha2) || push!(gens, O(alpha2))
                            isempty(gens) && continue

                            I = ideal(O, gens)

                            all(isone(I + mod_ideal) for mod_ideal in moduli) || continue
                            (mP_O \ I) == target_class || continue

                            return I
                        catch
                            continue
                        end
                    end
                end
            end
        end
    end

    error("Could not find a representative of the target Picard class coprime to all given ",
          "moduli within height ", max_height)
end

# Directly finds (alpha, rho_B) satisfying I*d*O = alpha*(Z+Z*rho_B), with
# alpha totally positive and rho_B > conj(rho_B) - using I ITSELF (not a
# representative of [I]^{-1}) as the basis source for the (m,n) joint
# search. This is the same (m,n) search find_alpha_rhoB_s1_s2_O runs
# internally, but that function always searches on b_O*d*O where b_O
# represents [J]^{-1} (found via find_b_with_verified_principal_product) -
# appropriate when the goal is a representative of J's OWN class. Here, per
# project owner, the goal for Stage 1 (finding new_rho_j from P_int) is
# P_int*d*O = alpha*(Z+Z*new_rho_j) directly, so I*d*O is used as-is with no
# inverse-class detour.
function find_alpha_rho_direct(I, K, O, d_val; mn_height::Int = 3000,
                                exclude::Vector = [], verbose::Bool = false)
    IdO = I * ideal(O, d_val)

    places   = real_places(K)
    emb_conj = embedding(places[1])
    emb_main = embedding(places[2])

    # _nf_elem_of (not raw elem_in_nf) since I may be a FRACTIONAL ideal
    # (e.g. P_int^{-1}), whose basis() elements aren't plain O-elements the
    # way an integral ideal's are.
    bd_basis_raw = basis(IdO)
    bd1_raw = _nf_elem_of(K, bd_basis_raw[1])
    bd2_raw = _nf_elem_of(K, bd_basis_raw[2])

    function embed_pair(x, plcs)
        return (Float64(embedding(plcs[1])(x)), Float64(embedding(plcs[2])(x)))
    end

    function lattice_reduce_2d(v1, v2, plcs)
        norm2(e) = e[1]^2 + e[2]^2
        dotp(ea, eb) = ea[1]*eb[1] + ea[2]*eb[2]

        e1 = embed_pair(v1, plcs)
        e2 = embed_pair(v2, plcs)

        for _ in 1:200
            if norm2(e2) < norm2(e1)
                v1, v2 = v2, v1
                e1, e2 = e2, e1
            end
            mu = round(Int, dotp(e1, e2) / norm2(e1))
            mu == 0 && break
            v2 = v2 - K(mu) * v1
            e2 = embed_pair(v2, plcs)
        end

        return v1, v2
    end

    bd1, bd2 = lattice_reduce_2d(bd1_raw, bd2_raw, places)
    e1 = embed_pair(bd1, places)
    e2 = embed_pair(bd2, places)
    verbose && println("  [direct] reduced I*dO basis: |bd1|^2 = ", e1[1]^2+e1[2]^2,
            "  |bd2|^2 = ", e2[1]^2+e2[2]^2)

    n_mn_tried = 0
    for h in 0:mn_height
        for m in -h:h, n in -h:h
            max(abs(m), abs(n)) == h || continue
            (m == 0 && n == 0) && continue
            gcd(abs(m), abs(n)) == 1 || continue

            n_mn_tried += 1

            alpha_cand = K(m) * bd1 + K(n) * bd2
            iszero(alpha_cand) && continue

            pos_ok = all(sign(alpha_cand, p) > 0 for p in places)
            neg_ok = all(sign(alpha_cand, p) < 0 for p in places)
            alpha = if pos_ok
                alpha_cand
            elseif neg_ok
                -alpha_cand
            else
                continue
            end

            g, x, y = gcdx(m, n)
            g == 1 || continue
            p_, q_ = -y, x
            beta = K(p_) * bd1 + K(q_) * bd2

            rho_B = beta / alpha_cand
            main_val = Float64(emb_main(rho_B))
            conj_val = Float64(emb_conj(rho_B))
            main_val > conj_val || continue

            (rho_B in exclude) && continue

            c1 = coeff(rho_B, 1)
            iszero(c1) && continue

            rhoB_lattice_O = try
                fractional_ideal(O, [K(1), rho_B])
            catch
                continue
            end
            order(rhoB_lattice_O) == O || continue

            alpha_ideal = try
                fractional_ideal(O, [alpha])
            catch
                continue
            end
            alpha_ideal * rhoB_lattice_O == IdO || continue

            verbose && println("  [direct] alpha (totally positive) = ", alpha,
                    "  rho_B = ", rho_B, "  from (m,n)=(", m, ",", n, ")",
                    "  ((m,n) tries=", n_mn_tried, ")")

            return alpha, rho_B
        end
    end

    error("find_alpha_rho_direct: no (alpha, rho_B) found within mn_height=", mn_height,
          " satisfying I*d*O = alpha*(Z+Z*rho_B)")
end

function find_alpha_rhoB_s1_s2_O(J, rho_B_init, rho, u_B, P_O, mP_O, O, K,
                                  d_val, rho_j, d_check;
                                  mn_height::Int = 3000,           # bound for the joint (alpha,rho_B) primitive-vector search
                                  q_gen_height::Int = 3000,        # bound for locating a generator of Q = Jb_O * (alpha)^{-1}
                                  exclude::Vector = [],            # rho_B candidates to skip (used to retry for a DIFFERENT valid rho_B)
                                  verbose::Bool = false)           # print internal [two-stage] search log (now called many times per outer retry loop, so off by default)
    # TWO-STAGE DESIGN. Once alpha, rho_B satisfy b*d*O = alpha*(Z+rho_B*Z)
    # with alpha totally positive, a valid (s1,s2) is GUARANTEED to exist and
    # can be solved for directly rather than searched for:
    #
    #   - b_O = mP_O(-u_B), i.e. b_O represents the INVERSE of J's ideal
    #     class. So Jb_O = J*b_O always has trivial class - it is
    #     automatically principal, regardless of which rho_B/alpha we use.
    #   - (alpha) = target_O is principal by construction (that's what the
    #     alpha search already verifies).
    #   - Q := Jb_O * (alpha)^{-1} is therefore a principal ideal divided by
    #     a principal ideal - also automatically principal. A generator g of
    #     Q is guaranteed to exist.
    #   - {1, rho_B} is a full Q-basis of K (rho_B irrational), so ANY
    #     generator of Q can be written as s2*rho_B - s1 for SOME rationals
    #     s1, s2 - no lattice constraint, just linear algebra. So once a
    #     generator of Q is found, (s1,s2) is a closed-form solve, not a
    #     search.
    #   - is_valid(s1,s2) depends only on the sign of elem2 = s2*rho_B - s1
    #     at ONE embedding (emb_conj), so a single global +/-1 flip on the
    #     found generator always suffices - no unit-power search needed
    #     there either.
    #   - Q does not depend on which totally-positive-unit representative of
    #     alpha we picked (they all generate the same ideal target_O), so
    #     there is no need to search over alpha's unit-power "height" either
    #     - the first totally positive alpha found is as good as any other.
    dO_O = ideal(O, d_val)
    is_inv_d, _ = Hecke.is_invertible(dO_O)
    is_inv_d || error("dO_O = ideal(O, d_val) is NOT invertible - this would be ",
                       "surprising for a principal ideal and likely indicates a ",
                       "Hecke API misuse rather than a genuine mathematical obstruction.")

    # b_O must represent [J]^{-1}: the whole construction (and the final
    # criterion "(s2*rho_B - s1) * (Z+Z*rho_B)^{-1} lies in the class of J")
    # stands or falls on this ONE class condition, because the output lattice
    # Z+Z*rho_B = (1/alpha)*b*d*O is ALWAYS in b_O's class, for every (m,n).
    #
    # FIXED: previously b_O was certified only through Hecke's Picard
    # discrete log (find_coprime_picard_representative with target -u_B,
    # i.e. mP_O \ I == -u_B). Independent exact-arithmetic verification of a
    # real run (d=21, D=44, q=3; rho_B = 3/86*sqrt(44)+23/43, s1=-53/7,
    # s2=100/21, J=<5, 3/2*sqrt(44)+3>) showed the produced lattice sat in
    # class +[J] instead of [J]^{-1} -- off by the order-2 element of
    # Pic(O)=Z/4, exactly the F_9^x/F_3^x ratio w seen when the criterion
    # was checked externally. So that certification chain cannot be trusted
    # here. b_O is now selected by directly EXHIBITING a generator mu of
    # J*b_O (disc-log-free, maximal-order machinery + HNF only), which is
    # the literal condition [b_O] = [J]^{-1} needs.
    d1_ideal = ideal(O, d_val + 1)
    b_O, mu_Jb = find_b_with_verified_principal_product(O, K, J, [dO_O, d1_ideal])
    verbose && println("  [two-stage] b_O (J*b_O verifiably principal) = ", b_O)
    verbose && println("  [two-stage] certified generator mu of J*b_O = ", mu_Jb)

    # Diagnostic: what does Hecke's Picard disc log CLAIM about this
    # verified-correct b_O? If it prints claimed == +u_B here, the old
    # targeting (mP_O \ I == -u_B) was inverted on this install/order --
    # which also implicates every other place classes are read off mP_O
    # (find_class_representatives, class_of in the elementary-divisors code,
    # make_coprime_representatives). If it prints -u_B, the wrong class in
    # the earlier run came from a different link (e.g. a stale cache or an
    # older b_O = mP_O(-u_B) path), and mP_O is fine.
    claimed = try
        mP_O \ b_O
    catch e
        verbose && println("  [two-stage] WARNING: mP_O \\ b_O threw: ", e)
        nothing
    end
    if claimed !== nothing
        verbose && println("  [two-stage] Hecke disc log claims class(b_O) = ", claimed,
                "   (-u_B = ", -u_B, ", +u_B = ", u_B, ")")
        claimed == -u_B ||
            println("  [two-stage] *** MISMATCH: verified b_O is in [J]^{-1}, but ",
                    "mP_O labels it ", claimed, " -- do NOT trust mP_O-derived ",
                    "classes elsewhere in this pipeline. ***")
    end

    verbose && println("  [two-stage] order(b_O) = ", order(b_O))
    verbose && println("  [two-stage] order(b_O) == O ? ", order(b_O) == O)
    order(b_O) == O || error("b_O is not tagged with order O as expected - got order(b_O) = ",
                              order(b_O), " instead. The invertibility check below would not ",
                              "be testing what we think it's testing.")

    is_inv_b, _ = Hecke.is_invertible(b_O)
    is_inv_b || error("b_O is not invertible - unexpected for a Picard group representative")

    # explicit final verification, rather than trusting the search above
    is_coprime_bO_dO = try
        isone(b_O + dO_O)
    catch
        false
    end
    verbose && println("  [two-stage] b_O coprime to dO_O ? ", is_coprime_bO_dO)
    is_coprime_bO_dO || error("Internal inconsistency: find_coprime_picard_representative ",
                               "returned an ideal b_O that is NOT actually coprime to dO_O ",
                               "(d_val=", d_val, ") - the search should guarantee this, so this ",
                               "points to a bug in that function rather than a genuine ",
                               "mathematical obstruction.")

    bdO_O = b_O * dO_O
    Jb_O  = J * b_O
    NJb   = QQ(norm(Jb_O))
    verbose && println("  [two-stage] norm(Jb_O) = ", NJb)

    places   = real_places(K)
    emb_conj = embedding(places[1])
    emb_main = embedding(places[2])

    # spanning basis of bdO_O itself - alpha must be searched for here, not
    # in all of O (see note below where it's used)
    bd_basis_raw = basis(bdO_O)
    bd1_raw = elem_in_nf(bd_basis_raw[1])
    bd2_raw = elem_in_nf(bd_basis_raw[2])

    # A generator alpha of the equation bd*O = alpha*(Z+rho_B*Z) must itself
    # lie in bdO_O: since 1 is one of rhoB_lattice_O's generators,
    # alpha = alpha*1 is in alpha*rhoB_lattice_O = bdO_O. So alpha should be
    # searched for directly within bdO_O's OWN lattice, not within all of O
    # (bdO_O is generally a much sparser sublattice of O, and most elements
    # of O are simply not in bdO_O at all - scanning O's basis wasted almost
    # all attempts on candidates that could never have worked). We get a
    # spanning basis of bdO_O directly from Hecke (no search - it's ideal
    # linear algebra), then Gauss/Lagrange-reduce it to the shortest
    # possible pair of spanning vectors, so a bounded scan over small integer
    # combinations is far more likely to reach a genuinely small element
    # quickly.
    function embed_pair(x, plcs)
        return (Float64(embedding(plcs[1])(x)), Float64(embedding(plcs[2])(x)))
    end

    function lattice_reduce_2d(v1, v2, plcs)
        norm2(e) = e[1]^2 + e[2]^2
        dotp(ea, eb) = ea[1]*eb[1] + ea[2]*eb[2]

        e1 = embed_pair(v1, plcs)
        e2 = embed_pair(v2, plcs)

        for _ in 1:200
            if norm2(e2) < norm2(e1)
                v1, v2 = v2, v1
                e1, e2 = e2, e1
            end
            mu = round(Int, dotp(e1, e2) / norm2(e1))
            mu == 0 && break
            v2 = v2 - K(mu) * v1
            e2 = embed_pair(v2, plcs)
        end

        return v1, v2
    end

    bd1, bd2 = lattice_reduce_2d(bd1_raw, bd2_raw, places)
    e1 = embed_pair(bd1, places)
    e2 = embed_pair(bd2, places)
    verbose && println("  [two-stage] reduced bdO_O basis: |bd1|^2 = ", e1[1]^2+e1[2]^2,
            "  |bd2|^2 = ", e2[1]^2+e2[2]^2)

    function to_nf_elem(x)
        try
            return elem_in_nf(x)
        catch
        end
        try
            return K(x)
        catch
        end
        return x
    end

    function to_float(q)
        return Float64(BigInt(numerator(q))) / Float64(BigInt(denominator(q)))
    end

    # Both alpha and rho_B come from a SINGLE search over primitive integer
    # pairs (m,n) on bdO_O's own basis (bd1,bd2), rather than two independent
    # searches. Since bdO_O is already an O-ideal (closed under
    # O-multiplication), ANY Z-basis {alpha, beta} of bdO_O automatically
    # satisfies alpha*(O + O*rho_B) = bdO_O for rho_B := beta/alpha - no
    # coprimality-to-conductor condition, no norm filter, no ideal-equality
    # search, and no unit-power sign-twisting needed at all. We only need
    # alpha itself (some m*bd1+n*bd2 with gcd(m,n)=1, so it extends to a
    # genuine basis) to be totally positive, checked directly.
    verbose && println("  [two-stage] searching primitive (m,n) pairs for alpha, rho_B jointly")

    n_mn_tried = 0

    for h in 0:mn_height
        for m in -h:h, n in -h:h
            max(abs(m), abs(n)) == h || continue   # only the new shell at this height
            (m == 0 && n == 0) && continue
            gcd(abs(m), abs(n)) == 1 || continue

            n_mn_tried += 1
            if n_mn_tried % 100_000 == 0
                verbose && println("  [two-stage] ... ", n_mn_tried, " (m,n) pairs checked so far")
            end

            alpha_cand = K(m) * bd1 + K(n) * bd2
            iszero(alpha_cand) && continue

            pos_ok = all(sign(alpha_cand, p) > 0 for p in places)
            neg_ok = all(sign(alpha_cand, p) < 0 for p in places)
            alpha = if pos_ok
                alpha_cand
            elseif neg_ok
                -alpha_cand
            else
                continue   # mixed sign at the two places - this direction can't work
            end

            # complementary basis vector via Bezout: m*q - n*p = 1
            g, x, y = gcdx(m, n)
            g == 1 || continue   # should always hold given the gcd filter above; defensive
            p_, q_ = -y, x
            beta = K(p_) * bd1 + K(q_) * bd2

            # NOTE: the ratio uses alpha_cand (not the possibly-negated
            # alpha) - negating alpha_cand would also negate beta by the same
            # Bezout construction, so the ratio beta/alpha is unchanged
            # either way; using alpha_cand directly here is just simpler.
            rho_B = beta / alpha_cand
            main_val  = Float64(emb_main(rho_B))
            conj_val  = Float64(emb_conj(rho_B))
            # CHANGED: no longer negating or conjugating to force the
            # "larger root" convention. If rho_B fails it (conj(rho_B) >
            # rho_B), this (m,n) pair is rejected outright and the search
            # continues to the next primitive pair, rather than trying to
            # repair this one.
            main_val > conj_val || continue

            (rho_B in exclude) && continue   # skip previously-rejected candidates, used by retry loops

            c1 = coeff(rho_B, 1)
            iszero(c1) && continue   # rho_B must have a nonzero irrational part

            rhoB_lattice_O = try
                fractional_ideal(O, [K(1), rho_B])
            catch
                continue
            end
            order(rhoB_lattice_O) == O || continue

            # explicit verification of the defining equation, rather than
            # trusting the algebraic derivation blindly
            alpha_ideal = try
                fractional_ideal(O, [alpha])
            catch
                continue
            end
            alpha_ideal * rhoB_lattice_O == bdO_O || continue

            verbose && println("  [two-stage] alpha (totally positive) = ", alpha,
                    "  rho_B = ", rho_B, "  from (m,n)=(", m, ",", n, ")",
                    "  ((m,n) tries=", n_mn_tried, ")")

        # ---- STAGE 2: solve for s1, s2 via a generator of Q = Jb_O * (alpha)^{-1}.
        # For each candidate generator we additionally require the MISSING
        # condition: alpha*(s2*rho_B - s1) must actually lie IN the ideal
        # b_O itself, not merely generate an ideal equal to Jb_O. This is a
        # genuinely separate condition from elem_check_ideal == Jb_O (that
        # only confirms the SET (elem_check) equals Jb_O; it does not by
        # itself guarantee elem_check ∈ b_O unless Jb_O ⊆ b_O is known to
        # hold, which isn't assumed here) - so every candidate generator of
        # Q is now checked against BOTH conditions, and only accepted once
        # both hold. This replaces the old "accept the first ideal match"
        # behavior.
        Q = Jb_O * inv(alpha_ideal)
        Q_norm = QQ(norm(Q))
        verbose && println("  [two-stage] norm(Q) (target norm for s2*rho_B - s1) = ", Q_norm)

        Qbasis = basis(Q)
        qb1 = to_nf_elem(Qbasis[1])
        qb2 = to_nf_elem(Qbasis[2])

        found_s = false
        s1 = QQ(0)
        s2 = QQ(0)
        for i in -q_gen_height:q_gen_height, j in -q_gen_height:q_gen_height
            (i == 0 && j == 0) && continue
            cand = K(i) * qb1 + K(j) * qb2
            iszero(cand) && continue
            QQ(abs(norm(K(cand)))) == Q_norm || continue
            cand_ideal = try
                fractional_ideal(O, [cand])
            catch
                continue
            end
            cand_ideal == Q || continue

            # a single global sign flip always suffices, since is_valid only
            # depends on the sign of elem2 at ONE embedding (emb_conj)
            elem2 = Float64(emb_conj(cand)) > 0 ? cand : -cand

            # closed-form solve for s1, s2: elem2 = s2*rho_B - s1, and
            # {1,rho_B} is a full Q-basis of K - exact, no search needed
            qc  = coeff(elem2, 1)
            p0  = coeff(elem2, 0)
            c2b = coeff(rho_B, 1)
            c1b = coeff(rho_B, 0)
            s2_try = QQ(qc / c2b)
            s1_try = QQ(qc * c1b / c2b - p0)

            elem_check = alpha * (K(s2_try) * rho_B - K(s1_try))

            elem_check_ideal = try
                fractional_ideal(O, [elem_check])
            catch
                continue
            end
            elem_check_ideal == Jb_O || continue

            # THE MISSING CONDITION: alpha*(s2*rho_B - s1) must be an
            # element of b_O itself.
            in_bO = try
                elem_check in b_O
            catch
                false
            end
            in_bO || continue

            # EXPLICIT CHECK (requested, on top of the sign-flip above which
            # should already guarantee this): s2*conj(rho_B) - s1 > 0 for
            # this specific (s1_try, s2_try). If it fails, reject this
            # candidate outright and let the search continue to the next
            # lattice point, rather than trusting the sign-flip construction
            # or erroring out.
            rhoB_conj_check = Float64(emb_conj(rho_B))
            s1_f_check = to_float(s1_try)
            s2_f_check = to_float(s2_try)
            (s2_f_check * rhoB_conj_check - s1_f_check) > 0 || continue

            s1, s2 = s1_try, s2_try
            found_s = true
            break
        end
        found_s || continue

        rhoB_conj = Float64(emb_conj(rho_B))
        s1_f = to_float(s1)
        s2_f = to_float(s2)
        is_valid_final = (s2_f * rhoB_conj - s1_f) > 0
        is_valid_final || error("Internal inconsistency: the sign-flip construction should ",
                                 "guarantee is_valid, but it does not (s1=", s1, ", s2=", s2, ")")

        # ---- FINAL INDEPENDENT VERIFICATION (disc-log-free) of the criterion
        # this function must satisfy:
        #     (s2*rho_B - s1) * (Z + Z*rho_B)^{-1}  lies in the Picard class of J.
        # Verified WITHOUT inverses: [L^{-1}] = [conj(L)] and [J^{-1}] = [conj(J)]
        # (since L*conj(L) and J*conj(J) are rational multiples of O, i.e.
        # principal), so the criterion is equivalent to
        #     elem2 * conj(L) * conj(J)   being principal in O,
        # which is tested by exhibiting an explicit generator. Everything here
        # is element arithmetic + HNF + maximal-order principality only.
        #
        # NOTE on exactness beyond the class level: when all identities hold,
        # elem2 * L^{-1} equals J * (1/d_val) exactly as fractional ideals --
        # the class-level statement matches J, while the literal ideal carries
        # the rational factor 1/d_val (and the test element below additionally
        # carries the rational norms N(L), N(J) from using conjugates instead
        # of inverses). Rational factors are invisible to the Picard class but
        # DO matter for any mod-m congruence normalization downstream; lambda
        # is printed so that can be inspected against the intended d-scaling.
        elem2_exact = K(s2) * rho_B - K(s1)
        L_gens = [one(K), rho_B]
        Lconj_gens = [quad_conj(K, g) for g in L_gens]
        Jconj_gens = [quad_conj(K, _nf_elem_of(K, x)) for x in basis(J)]
        crit_gens = [elem2_exact * a for a in lattice_product_gens(Lconj_gens, Jconj_gens)]

        ok_crit, lambda_gen = is_principal_O_verified(K, O, crit_gens)
        if !ok_crit
            error("FINAL VERIFICATION FAILED: (s2*rho_B - s1)*(Z+Z*rho_B)^{-1} is NOT ",
                  "in the Picard class of J (checked independently of picard_group). ",
                  "Since Z+Z*rho_B is in b_O's class for EVERY (m,n), this means b_O ",
                  "is in the wrong class -- the b_O selection above, not the (m,n) ",
                  "search, is where to look. rho_B=", rho_B, " s1=", s1, " s2=", s2)
        end
        verbose && println("  [two-stage] final verification PASSED: elem2*(Z+Z*rho_B)^{-1} is in ",
                "the Picard class of J; certified generator lambda (of ",
                "elem2*conj(L)*conj(J)) = ", lambda_gen)

        verbose && println("  [two-stage] FOUND: rho_B=", rho_B, " alpha=", alpha,
                " s1=", s1, " s2=", s2, "  ((m,n) tries=", n_mn_tried, ")")
        return alpha, rho_B, s1, s2
        end
    end

    error("Could not find alpha, rho_B, s1, s2 within given bounds ",
          "(mn_height=", mn_height, ", q_gen_height=", q_gen_height, ")")
end

function artin_action_on_sqrtm(d::Int, f::Int, m::Int, J, Sigma::Vector{Int}, K)
    nJ = Int(norm(J))
    kron = kronecker_symbol(m, nJ)
    return kron
end

function find_signs(new_ovs, d)
    a = ones(Int, d, d)
    a[1, 1] = 1

    for p1 = 0:d-1
        for p2 = 0:d-1
            (p1 == 0 && p2 == 0) && continue
            mp1 = mod(-p1, d)
            mp2 = mod(-p2, d)
            if (mp1 < p1) || (mp1 == p1 && mp2 < p2)
                continue
            end
            v_p  = new_ovs[p1+1, p2+1]
            v_mp = new_ovs[mp1+1, mp2+1]
            product = v_p * v_mp * (d + 1)
            if abs(imag(product)) > 1e-10
                println("  WARNING: v($p1,$p2)*v($mp1,$mp2)*(d+1) has large imaginary part: ", imag(product))
            end
            if real(product) > 0
                a[p1+1, p2+1]  = 1
                a[mp1+1, mp2+1] = 1
            else
                a[p1+1, p2+1]  = 1
                a[mp1+1, mp2+1] = -1
            end
        end
    end

    max_dev = 0.0
    for p1 = 0:d-1
        for p2 = 0:d-1
            (p1 == 0 && p2 == 0) && continue
            mp1 = mod(-p1, d)
            mp2 = mod(-p2, d)
            v_p  = new_ovs[p1+1, p2+1]
            v_mp = new_ovs[mp1+1, mp2+1]
            product = a[p1+1, p2+1] * v_p * a[mp1+1, mp2+1] * v_mp * (d+1)
            dev = abs(real(product) - 1) + abs(imag(product))
            if dev > 1e-10
                println("  FAIL at ($p1,$p2): product = ", product,
                        " a=", a[p1+1,p2+1], " a(-p)=", a[mp1+1,mp2+1])
            end
            max_dev = max(max_dev, Float64(dev))
        end
    end
    println("  max deviation after sign correction = ", max_dev)
    return a
end

function twisted_chi_general(F::AdmissibleTuple, J, gamma, K, O,
                              is_principal_J::Bool,
                              P_O, mP_O, rho_reps, gQ_reps, F_class,
                              j_cache::Dict)
    d = Int(F.d)
    D = Int(F.D)
    a = gen(K)
    OK = maximal_order(K)

    rho = rho_reps[F_class]

    # shin_rm expects the root as a real BigFloat (confirmed by the actual
    # MethodError: shin_rm(::Matrix, ::BigFloat, ::Vector, ::Int64) is the
    # real signature - not the exact field element). Convert via the same
    # "main embedding" convention used everywhere else in this file for
    # picking/evaluating roots (places[2], e.g. emb_main at line ~713/828).
    places_K_top = real_places(K)
    emb_main_K = embedding(places_K_top[2])
    emb_conj_K = embedding(places_K_top[1])

    Ng = Int(norm(K(gamma)))
    NJ = Int(norm(J))
    println("  Ng = ", Ng)
    println("  NJ = ", NJ)

    if is_principal_J
        chi_raw = kronecker_symbol(d + 1, NJ)
        chi = (chi_raw == 0) ? 1 : chi_raw
        println("  chi = ", chi, " (NJ = ", NJ, ")")
    else
        NJ_Ng = NJ * abs(Ng)
        chi_raw = kronecker_symbol(d + 1, NJ_Ng)
        chi = (chi_raw == 0) ? 1 : chi_raw
        println("  chi = ", chi, " (NJ*|Ng| = ", NJ_Ng, ")")
    end

    ζ = -cispi(BigFloat(1) / d)

    Dr = squarefree_part(D)
    sqrt_Dr = a // isqrt(D ÷ Dr)

    _, f_rho = coredisc(discriminant(F.Q))
    rho_g = (-F.Q.b + Int(f_rho) * sqrt_Dr) / (2 * F.Q.a)
    M_g = multiplication_matrix_mod_d(K(gamma), rho_g, d, D)
    println("  M_g = ", M_g)

    new_ovs = zeros(Complex{BigFloat}, d, d)
    new_ovs[1, 1] = sqrt(BigFloat(d + 1))

    function du_to_int_mod_d(du_int_val, d)
        num = Int(numerator(du_int_val))
        den = Int(denominator(du_int_val))
        g = gcd(abs(num), den)
        num ÷= g
        den ÷= g
        # After reducing to lowest terms, gcd(num, den) == 1 by construction.
        # If du_int_val genuinely has a well-defined residue mod d, den must
        # come out entirely coprime to d at this point - there is no leftover
        # "part shared with d" left to strip out or round away (the previous
        # version's attempt to split den into a coprime part and a
        # d-sharing part was mathematically incoherent given num/den is
        # already in lowest terms: num can never be divisible by any
        # nontrivial factor den shares with d, since gcd(num,den)=1 - so that
        # branch was silently producing a wrong integer whenever it fired,
        # not just in some edge case). Check explicitly rather than repeat
        # that mistake.
        gcd(den, d) == 1 || error("denominator ", den, " of ", du_int_val,
                                   " is not coprime to d=", d,
                                   " - this value has no well-defined residue mod ", d,
                                   " (this points to an upstream bug, not something to round past)")
        return mod(num * invmod(den, d), d)
    end

    function compute_integralnewdu(p1, p2, s1, s2, rhoB_cand, rho, rho_j, M_g_j, d)
        elem1 = K(p2) * rho - K(p1)
        elem2 = K(s2) * rhoB_cand - K(s1)
        product = elem1 * elem2
        u2, u1 = find_rho_j_and_coords(product, rho_j)

        # Same check as the live main loop: u2*conj(rho_j) - u1 must be > 0;
        # use the negated pair if it isn't (see the main loop for the full
        # reasoning - kept in sync here even though this helper is currently
        # unused, so it isn't silently wrong if it's called later).
        rho_j_conj_val_local = Float64(real(emb_conj_K(rho_j)))
        check_val = Float64(u2) * rho_j_conj_val_local - Float64(u1)
        if check_val <= 0
            u1, u2 = -u1, -u2
        end

        du = [to_du_mod_d(u1, d), to_du_mod_d(u2, d)]
        raw_newdu = M_g_j * du
        newdu = [to_du_mod_d(raw_newdu[1], d), to_du_mod_d(raw_newdu[2], d)]
        integralnewdu = Int[]
        for u in newdu
            du_int = d * u
            val = du_to_int_mod_d(du_int, d)
            push!(integralnewdu, val)
        end
        return integralnewdu
    end

    if is_principal_J
        gF = AdmissibleTuple(d, F.r, F.Q)
        println("  gQ = ", F.Q)
        println("  gF = ", gF)

        QQ_form = binary_quadratic_form(gF.A[2,1], gF.A[2,2] - gF.A[1,1], -gF.A[1,2])

        c_g = e(-BigFloat(rademacher(gF.A)) / 24) / (chi * sqrt(BigFloat(d + 1)))
        println("  c_g = ", c_g)

        for p1 = 0:d-1
            for p2 = 0:d-1
                (p1 == 0 && p2 == 0) && continue
                p = [p1, p2]
                gp = [p1, mod(Ng * p2, d)]
                newp = mod.(M_g * p, d)
                QA = BigFloat(-QQ_form(newp...) / (d * (d - 2)))
                s = isodd(d) ? 1 : (1 + newp[1]) * (1 + newp[2])
                new_ovs[gp[1]+1, gp[2]+1] = ζ^QA * (-1)^s * c_g *
                                              shin_of_tuple(gF, newp)
            end
        end

        println("  v(p)*v(-p) for all nonzero p:")
        for p1 = 0:d-1
            for p2 = 0:d-1
                (p1 == 0 && p2 == 0) && continue
                mp1 = mod(-p1, d)
                mp2 = mod(-p2, d)
                product = new_ovs[p1+1, p2+1] * new_ovs[mp1+1, mp2+1]
                println("    p=($p1,$p2): v(p)*v(-p) = ", Float64(real(product)),
                        " + ", Float64(imag(product)), "im",
                        "  |v(p)*v(-p)| = ", Float64(abs(product)),
                        "  1/(d+1) = ", 1.0/(d+1))
            end
        end

    else
        # u_B is computed FIRST, and the cache is keyed on it directly - the
        # cached quantities (rho_B, s1, s2, rho_j, ...) only depend on J's
        # ideal CLASS, not on which specific J object was passed in (as the
        # cache's own log message already said), so keying by class is both
        # the conceptually correct key and avoids objectid(J), which is
        # unstable: Julia can and does reuse memory addresses of
        # garbage-collected objects, so objectid(J) for a later, unrelated J
        # can collide with a stale key from an earlier, different J still
        # sitting in the cache - silently pulling in the WRONG twist's
        # rho_B/s1/s2/rho_j. This was confirmed to be happening (multiple
        # consecutive corrupted twists observed).
        u_B = mP_O \ J
        j_key = u_B

        # d_check depends only on d and F.q (not on gamma or J), so compute
        # it once unconditionally, reused both by the cached J-targeting
        # search below and by the always-fresh new_rho_j search further down.
        # The order O has conductor F.q, and any ideal representing a
        # non-trivial Picard class of a conductor-q order will structurally
        # carry a factor of q in its basis coordinates. When q | d (as here,
        # q=3, d=21), requiring c1's numerator/denominator to be coprime
        # to the *full* d is unsatisfiable by construction, not a sign of a
        # bad candidate. Strip q's prime factors out of d before checking;
        # the joint find_alpha_rhoB_s1_s2_O search remains the real
        # correctness gate and is unaffected by this relaxation.
        q_val = Int(F.q)
        d_check = d
        for (p, _) in factor(q_val)
            while d_check % Int(p) == 0
                d_check ÷= Int(p)
            end
        end
        println("  d = ", d, ", q = ", q_val, ", d_check (coprimality target) = ", d_check)

        # CACHE SCOPE, IMPORTANT: only (j_class, rho_B, s1, s2) are cached
        # here, keyed by u_B (J's class alone). rho_j (via new_rho_j below)
        # is NOT cached, because it is NOT actually gamma-independent:
        # P = J*(gamma)*(Z*rho+Z)^{-1} depends on gamma directly through the
        # principal ideal (gamma) - its CLASS [P]=[J]*[Z*rho+Z]^{-1} doesn't
        # depend on gamma (since (gamma) is always principal), but the
        # SPECIFIC representative new_rho_j the search finds for that class
        # is derived from P_int, an INTEGRAL ideal built by clearing
        # denominators out of P_frac - and that clearing step (scale_P =
        # denominator(P_frac)) genuinely depends on which gamma was used.
        # Caching rho_j under j_key=u_B (gamma-independent) would silently
        # reuse the FIRST gamma's rho_j for every subsequent gamma sharing
        # the same J-class - caught before this went out, not yet exercised
        # against an actual second-gamma run.
        # rho_B_init is needed unconditionally (not just on a cache miss)
        # because the per-point outer retry below may need to search for a
        # DIFFERENT rho_B/s1/s2 if the cached one can't satisfy some point's
        # full stage sequence.
        rho_B_init = rho_reps[u_B]

        if haskey(j_cache, j_key)
            j_class, rho_B, s1, s2 = j_cache[j_key]
            println("  [cache] Reusing rho_B/s1/s2 for this J (class ", u_B, ") - not recomputed")
            println("  [cache] rho_B = ", rho_B, "  s1 = ", s1, "  s2 = ", s2)
        else
            println("  class of J = ", u_B, " rho_B (raw) = ", rho_B_init)

            j_class = F_class + u_B

            alpha, rho_B, s1, s2 = find_alpha_rhoB_s1_s2_O(J, rho_B_init, rho, u_B, P_O, mP_O,
                                                             O, K, d, rho, d_check;
                                                             mn_height = 3000,
                                                             q_gen_height = 3000)
            println("  Using rho_B = ", rho_B, "  alpha = ", alpha)
            println("  s1 = ", s1, " s2 = ", s2)

            j_cache[j_key] = (j_class, rho_B, s1, s2)
        end

        # --- NEW CONSTRUCTION OF rho_j (per project owner) ---
        # ALWAYS recomputed fresh below, for THIS call's specific gamma - see
        # the CACHE SCOPE comment above for why this must not be cached.
        #
        # The previous approach (rho_j_orig = rho_reps[j_class], then
        # HJ-reduce a form built from it, then take THAT form's own root
        # as "rho_j") was already flagged as inconsistent in an earlier
        # debugging session (2026-07-23): reduced_hj_orbit/
        # minimal_hj_stabilizer move to an SL2(Z)-equivalent form, and
        # equivalent forms generally do NOT share a root - so the
        # resulting rho_j was never actually guaranteed consistent with
        # rho_reps[j_class]/the Picard class arithmetic that rho_B, J,
        # alpha are all built around. This was a very plausible source
        # of the det(M_g_j)-composition-map singularity found earlier
        # this session (a spurious factor of 7 for one run, d=21).
        #
        # New approach: derive rho_j directly and consistently from the
        # SAME verified, disc-log-free two-stage search used for rho_B,
        # by targeting the COMBINED class
        #     [P] = [J] * [(gamma)*(Z*rho+Z)^{-1}]
        # (the inverse IS necessary here - reconfirmed per project owner
        # after a brief detour without it)
        # instead of looking anything up via rho_reps/gQ_reps. gamma's
        # own coordinates in the (1,rho) basis give (P2_gamma, P1_gamma)
        # with gamma = P2_gamma*rho - P1_gamma (same convention as
        # elem1 = p2*rho - p1 in the main grid loop below), reusing
        # find_rho_j_and_coords itself for that decomposition (not
        # actually needed for P itself since gamma enters as a whole
        # element, but printed as a useful diagnostic).
        P2_gamma, P1_gamma = find_rho_j_and_coords(K(gamma), rho)
        println("  gamma = P2*rho - P1 with P2 = ", P2_gamma, " P1 = ", P1_gamma)

        # L_rho = Z + Z*rho as a fractional O-ideal. Inverted via the
        # conjugate/norm trick (L_rho^{-1} = conj(L_rho)/N(L_rho)) rather
        # than Hecke's own inv() on O directly - same reasoning as for
        # b_O: O is non-maximal, and is_principal_with_data/picard_group
        # discrete logs were already found unreliable here, so anything
        # built on top of them is avoided in favor of the conjugate
        # trick, which only needs norm() (basic ideal bookkeeping, not
        # principality testing or discrete logs).
        Lrho_gens = [one(K), rho]
        Lrho_ideal = fractional_ideal(O, Lrho_gens)
        N_Lrho = norm(Lrho_ideal)
        Lrho_inv_gens = [quad_conj(K, g) // N_Lrho for g in Lrho_gens]
        Lrho_inv = fractional_ideal(O, Lrho_inv_gens)

        gamma_ideal = fractional_ideal(O, [K(gamma)])
        P_frac = J * gamma_ideal * Lrho_inv
        println("  P = J*(gamma)*(Z+Z*rho)^{-1} = ", P_frac, "  N(P) = ", norm(P_frac))

        # Scaling a fractional ideal by a positive rational never changes
        # its Picard class (the scalar itself generates a principal
        # ideal), so clear denominators to get an INTEGRAL representative
        # of [P] to hand to find_alpha_rhoB_s1_s2_O (which expects an
        # honest AbsSimpleNumFieldOrderIdeal, the same way J is one).
        scale_P = denominator(P_frac)
        P_basis_nf = [_nf_elem_of(K, g) for g in basis(P_frac)]
        P_int_gens = [O(K(scale_P) * g) for g in P_basis_nf]
        P_int = ideal(O, P_int_gens)
        println("  scale_P = ", scale_P, "  P_int = ", P_int, "  N(P_int) = ", norm(P_int))

        # P_int^{-1}, via the conjugate/norm trick (same reasoning as J_inv
        # below: avoid Hecke's inv() on a non-maximal order). Needed for
        # Stage 1's actual target, P_int^{-1}*d*O = alpha*(Z+Z*new_rho_j).
        P_int_gens_nf2 = [_nf_elem_of(K, x) for x in basis(P_int)]
        N_P_int = norm(P_int)
        P_int_inv_gens = [quad_conj(K, g) // N_P_int for g in P_int_gens_nf2]
        P_int_inv = fractional_ideal(O, P_int_inv_gens)

        # J_inv: computed once (doesn't depend on which candidate rho_HJ the
        # retry loop below settles on), via the conjugate/norm trick (same
        # reasoning as everywhere else in this file: avoid Hecke's inv() on
        # a non-maximal order). Still used later in the per-grid-point checks.
        J_gens_nf = [_nf_elem_of(K, x) for x in basis(J)]
        N_J = norm(fractional_ideal(O, J_gens_nf))
        J_inv_gens = [quad_conj(K, g) // N_J for g in J_gens_nf]
        J_inv = fractional_ideal(O, J_inv_gens)

        test1 = true
        max_retries_p = 50
        gQ_j = nothing   # set from the FIRST successfully-processed grid point;
                          # used only for the final gF = AdmissibleTuple(...) return
                          # value below - see printed note when first set.

        for p1 = 0:d-1
            for p2 = 0:d-1
                (p1 == 0 && p2 == 0) && continue

                gp = [p1, mod(NJ * Ng * p2, d)]
                p = [p1, p2]

                elem1 = K(p[2]) * rho - K(p[1])

                # Per project owner: if the per-rho_j search below is fully
                # exhausted for a given rho_B/s1/s2, that doesn't necessarily
                # mean this (p1,p2) is unsatisfiable - it may mean THIS
                # rho_B/s1/s2 choice is wrong for this point. So wrap the
                # whole rho_j search in an OUTER retry over rho_B/s1/s2
                # candidates too: on exhaustion, exclude the current rho_B
                # and search for a different one (via find_alpha_rhoB_s1_s2_O
                # with an updated exclude list), rebuild elem2/product from
                # it, and retry the whole rho_j search fresh. This does NOT
                # touch j_cache - only this point's local search is affected;
                # other points/gammas keep using the originally cached rho_B.
                excluded_rho_B_p = []
                found_overall = false
                max_retries_rho_B_p = 20

                local rho_j_p, gF_j_p, gQ_j_p, QQ_form_p, M_g_j_p, c_g_j_p,
                      rho_j_bf_p, u1_p, u2_p

                for rho_B_attempt in 1:max_retries_rho_B_p
                    rho_B_cur, s1_cur, s2_cur = if rho_B_attempt == 1
                        rho_B, s1, s2
                    else
                        println("    p=($p1,$p2) [rho_B_attempt $rho_B_attempt]: searching for a ",
                                "different rho_B/s1/s2 (previous one exhausted its rho_j search)")
                        _, rho_B_new, s1_new, s2_new = find_alpha_rhoB_s1_s2_O(
                            J, rho_B_init, rho, u_B, P_O, mP_O, O, K, d, rho, d_check;
                            mn_height = 3000, q_gen_height = 3000, exclude = excluded_rho_B_p)
                        (rho_B_new, s1_new, s2_new)
                    end
                    println("    p=($p1,$p2) [rho_B_attempt $rho_B_attempt]: rho_B = ", rho_B_cur,
                            "  s1 = ", s1_cur, "  s2 = ", s2_cur)

                    elem2 = K(s2_cur) * rho_B_cur - K(s1_cur)
                    product = elem1 * elem2

                    # DIAGNOSTIC (narrowing down whether conj(product) is a
                    # real invariant or a bug): compute elem1, elem2, product,
                    # and their conjugate-embedding values DIRECTLY (not via
                    # the u1/u2 decomposition), once per rho_B_attempt.
                    println("    p=($p1,$p2) [rho_B_attempt $rho_B_attempt]: elem1 = ", elem1,
                            "  conj(elem1) = ", Float64(real(emb_conj_K(elem1))))
                    println("    p=($p1,$p2) [rho_B_attempt $rho_B_attempt]: elem2 = ", elem2,
                            "  conj(elem2) = ", Float64(real(emb_conj_K(elem2))))
                    println("    p=($p1,$p2) [rho_B_attempt $rho_B_attempt]: product = ", product,
                            "  conj(product) [direct] = ", Float64(real(emb_conj_K(product))))

                # Per project owner: rho_j may meaningfully depend on p, so
                # Stages 1/2/2.5 (finding new_rho_j, HJ-reducing to rho_HJ,
                # and the P_int^{-1}*(Z+Z*rho_HJ)^{-1} check) now run FRESH
                # for EACH (p1,p2), with their own exclusion list, rather
                # than once for the whole function call. A rejection at any
                # stage (2, 2.5, 4.7, 4.8, 4.9) excludes that new_rho_j
                # candidate and retries Stage 1 - all scoped to THIS
                # (p1,p2); a different grid point starts its search fresh.
                excluded_new_rho_j_p = []
                found_p = false

                for attempt in 1:max_retries_p
                    # STAGE 1: P_int^{-1}*d*O = alpha*(Z+Z*new_rho_j), alpha
                    # totally positive, new_rho_j > conj(new_rho_j).
                    _, new_rho_j_cand = find_alpha_rho_direct(
                        P_int_inv, K, O, d; mn_height = 3000,
                        exclude = excluded_new_rho_j_p)
                    println("    p=($p1,$p2) [attempt $attempt]: new_rho_j = ", new_rho_j_cand)

                    # STAGE 2: Q_new_rho_j has new_rho_j as an actual root
                    # (built from its minimal polynomial); HJ-reduce it and
                    # take the reduced form's own root, rho_HJ.
                    c0_nj = coeff(new_rho_j_cand, 0)
                    c1_nj = coeff(new_rho_j_cand, 1)
                    p0_nj = c0_nj^2 - c1_nj^2 * D
                    p1_nj = -2 * c0_nj
                    p2_nj = 1
                    scale_nj = lcm(Int(denominator(QQ(p0_nj))),
                                   Int(denominator(QQ(p1_nj))), 1)
                    a_int_nj = Int(round(scale_nj * p2_nj))
                    b_int_nj = Int(round(scale_nj * p1_nj))
                    c_int_nj = Int(round(scale_nj * p0_nj))
                    g_int_nj = gcd(a_int_nj, b_int_nj, c_int_nj)
                    a_int_nj ÷= g_int_nj
                    b_int_nj ÷= g_int_nj
                    c_int_nj ÷= g_int_nj
                    Q_new_rho_j_cand = binary_quadratic_form(a_int_nj, b_int_nj, c_int_nj)

                    V_j_cand = reduced_hj_orbit(Q_new_rho_j_cand)
                    gQ_j_cand = minimal_hj_stabilizer(V_j_cand, d)

                    D_gQj = Int(discriminant(gQ_j_cand))
                    quotient_gQj, remainder_gQj = divrem(D_gQj, Dr)
                    remainder_gQj == 0 || error("discriminant(gQ_j) = ", D_gQj,
                                                 " is not an integer multiple of Dr = ", Dr)
                    f_gQj = isqrt(quotient_gQj)
                    f_gQj^2 == quotient_gQj || error("D_gQj / Dr = ", quotient_gQj,
                                                      " is not a perfect square (got isqrt = ", f_gQj, ")")
                    sqrt_gQj = f_gQj * sqrt_Dr
                    root_plus  = (-gQ_j_cand.b + sqrt_gQj) / (2 * gQ_j_cand.a)
                    root_minus = (-gQ_j_cand.b - sqrt_gQj) / (2 * gQ_j_cand.a)
                    rho_HJ_cand = Float64(emb_main_K(root_plus)) >= Float64(emb_main_K(root_minus)) ?
                                  root_plus : root_minus

                    rho_HJ_main_val = Float64(emb_main_K(rho_HJ_cand))
                    rho_HJ_conj_val = Float64(emb_conj_K(rho_HJ_cand))
                    if !(rho_HJ_main_val > rho_HJ_conj_val)
                        println("    p=($p1,$p2) [attempt $attempt]: REJECTED at Stage 2 ",
                                "(rho_HJ=", rho_HJ_main_val, " not > conj=", rho_HJ_conj_val, ")")
                        push!(excluded_new_rho_j_p, new_rho_j_cand)
                        continue
                    end

                    LrhoHJ_gens = [one(K), rho_HJ_cand]
                    LrhoHJ_ideal = fractional_ideal(O, LrhoHJ_gens)
                    N_LrhoHJ = norm(LrhoHJ_ideal)
                    LrhoHJ_inv_gens = [quad_conj(K, g) // N_LrhoHJ for g in LrhoHJ_gens]
                    LrhoHJ_inv = fractional_ideal(O, LrhoHJ_inv_gens)

                    # STAGE 2.5: P_int^{-1}*(Z+Z*rho_HJ)^{-1} principal,
                    # generated by a totally positive alpha_HJ.
                    target_HJ_frac = P_int_inv * LrhoHJ_inv
                    target_HJ_gens_nf = [_nf_elem_of(K, g) for g in basis(target_HJ_frac)]
                    ok_HJ, alpha_HJ = is_principal_totally_positive_verified(K, O, target_HJ_gens_nf)
                    if !ok_HJ
                        println("    p=($p1,$p2) [attempt $attempt]: REJECTED at Stage 2.5 ",
                                "(P_int^{-1}*(Z+Z*rho_HJ)^{-1} not principal+totally-positive)")
                        push!(excluded_new_rho_j_p, new_rho_j_cand)
                        continue
                    end

                    rho_j_cand = rho_HJ_cand
                    gF_j_cand = AdmissibleTuple(d, F.r, gQ_j_cand)
                    QQ_form_cand = binary_quadratic_form(gF_j_cand.A[2,1],
                                                          gF_j_cand.A[2,2] - gF_j_cand.A[1,1],
                                                          -gF_j_cand.A[1,2])
                    M_g_j_cand = multiplication_matrix_mod_d(K(gamma), rho_j_cand, d, D)

                    det_M_g_j = mod(M_g_j_cand[1,1]*M_g_j_cand[2,2] -
                                    M_g_j_cand[1,2]*M_g_j_cand[2,1], d)
                    gcd_det_d = gcd(det_M_g_j, d)
                    if gcd_det_d != 1
                        println("    p=($p1,$p2) [attempt $attempt]: REJECTED at det(M_g_j) check ",
                                "(det mod d = ", det_M_g_j, ", gcd with d = ", gcd_det_d, ")")
                        push!(excluded_new_rho_j_p, new_rho_j_cand)
                        continue
                    end

                    rho_j_bf_cand = BigFloat(real(emb_main_K(rho_j_cand)))

                    LrhoJ_gens = [one(K), rho_j_cand]
                    LrhoJ_ideal_cand = fractional_ideal(O, LrhoJ_gens)
                    N_LrhoJ = norm(LrhoJ_ideal_cand)
                    LrhoJ_inv_gens = [quad_conj(K, g) // N_LrhoJ for g in LrhoJ_gens]
                    LrhoJ_inv_cand = fractional_ideal(O, LrhoJ_inv_gens)

                    target_48_frac = P_int_inv * LrhoJ_inv_cand
                    target_48_gens_nf = [_nf_elem_of(K, g) for g in basis(target_48_frac)]

                    u2_cand, u1_cand = find_rho_j_and_coords(product, rho_j_cand)

                    diag_val = Float64(u2_cand) * Float64(real(emb_conj_K(rho_j_cand))) - Float64(u1_cand)
                    println("    p=($p1,$p2) [attempt $attempt]: u2*conj(rho_j)-u1 = ", diag_val,
                            "  positive? ", diag_val > 0)

                    # STAGE 4.7 (per project owner: reinstated): u2*conj(rho_j) - u1
                    # must be > 0 - reject (no sign-flip fix) if it fails. NOTE:
                    # this condition is mathematically independent of rho_j (it
                    # equals conj(product) > 0 for ANY valid rho_j decomposition),
                    # so retrying with a different rho_j can never fix a failure
                    # here - only the OUTER rho_B/s1/s2 retry (which changes
                    # product itself) can. Kept as a per-rho_j-attempt check
                    # anyway per instruction; it will simply exhaust the rho_j
                    # loop immediately and fall through to the rho_B retry when
                    # conj(product) <= 0 for the current rho_B/s1/s2 choice.
                    if !(diag_val > 0)
                        push!(excluded_new_rho_j_p, new_rho_j_cand)
                        continue
                    end

                    # STAGE 4.8 (per project owner: reinstated): a totally
                    # positive alpha_rho_j must exist generating
                    # P_int^{-1}*(Z+Z*rho_j)^{-1}, such that THAT SAME
                    # alpha_rho_j satisfies alpha_rho_j*(u2*rho_j-u1) is
                    # contained in P_int - searched over the whole family of
                    # totally positive generators (O's fundamental unit has
                    # norm +1, so there's a whole eps^k family of totally
                    # positive generators of the same ideal; this searches
                    # that family directly for one that ALSO satisfies
                    # membership).
                    companion = K(u2_cand) * rho_j_cand - K(u1_cand)
                    membership_target = P_int
                    ok_alpha_p, alpha_rho_j_cand = is_principal_totally_positive_verified_with_membership(
                        K, O, target_48_gens_nf, companion, membership_target)
                    if !ok_alpha_p
                        println("    p=($p1,$p2) [attempt $attempt]: REJECTED at Stage 4.8 ",
                                "(no totally-positive alpha_rho_j satisfying membership found)")
                        push!(excluded_new_rho_j_p, new_rho_j_cand)
                        continue
                    end

                    # STAGE 4.9: alpha_rho_j*(u2*rho_j-u1)*P_int must be in
                    # the same ray-class-group class as J*(p2*rho-p1)*(Z+Z*rho)^{-1}.
                    LHS_elem = K(alpha_rho_j_cand) * (K(u2_cand) * rho_j_cand - K(u1_cand))
                    LHS_frac = fractional_ideal(O, [LHS_elem]) * P_int

                    RHS_frac = J * fractional_ideal(O, [elem1]) * Lrho_inv
                    RHS_gens_nf = [_nf_elem_of(K, g) for g in basis(RHS_frac)]
                    RHS_conj_gens = [quad_conj(K, g) for g in RHS_gens_nf]
                    N_RHS = norm(RHS_frac)

                    LHS_gens_nf = [_nf_elem_of(K, g) for g in basis(LHS_frac)]
                    ratio_gens = [g // N_RHS for g in lattice_product_gens(LHS_gens_nf, RHS_conj_gens)]
                    ok_same_class, _ = is_principal_O_verified(K, O, ratio_gens)
                    if !ok_same_class
                        println("    p=($p1,$p2) [attempt $attempt]: REJECTED at Stage 4.9 ",
                                "(alpha_rho_j*(u2*rho_j-u1)*P_int not in same class as J*elem1*(Z+Z*rho)^{-1})")
                        push!(excluded_new_rho_j_p, new_rho_j_cand)
                        continue
                    end

                    println("    p=($p1,$p2) [attempt $attempt]: ALL STAGES PASSED")

                    # ALL stages passed for this (p1,p2) with this candidate.
                    rho_j_p = rho_j_cand
                    gF_j_p = gF_j_cand
                    gQ_j_p = gQ_j_cand
                    QQ_form_p = QQ_form_cand
                    M_g_j_p = M_g_j_cand
                    c_g_j_p = e(-BigFloat(rademacher(gF_j_cand.A)) / 24) /
                               (chi * sqrt(BigFloat(d + 1)))
                    rho_j_bf_p = rho_j_bf_cand
                    u1_p, u2_p = u1_cand, u2_cand
                    found_p = true
                    break
                end

                if found_p
                    found_overall = true
                    break
                else
                    println("    p=($p1,$p2) [rho_B_attempt $rho_B_attempt]: exhausted ",
                            max_retries_p, " rho_j candidates - excluding rho_B = ", rho_B_cur,
                            " and searching for a different rho_B/s1/s2")
                    push!(excluded_rho_B_p, rho_B_cur)
                end
                end

                found_overall || error("p=($p1,$p2): exhausted ", max_retries_rho_B_p,
                                        " rho_B/s1/s2 candidates, each of which exhausted ",
                                        max_retries_p, " rho_j candidates - could not satisfy ",
                                        "Stages 1/2/2.5/4.7/4.8/4.9 for this point")

                if gQ_j === nothing
                    gQ_j = gQ_j_p
                    println("  [note] gQ_j for the function's final return value taken ",
                            "from the first successfully-processed grid point, p=($p1,$p2) ",
                            "- rho_j now varies by point, so there is no single canonical ",
                            "choice; flagging this assumption explicitly.")
                end

                du = [to_du_mod_d(u1_p, d), to_du_mod_d(u2_p, d)]

                raw_newdu = M_g_j_p * du
                newdu = [to_du_mod_d(raw_newdu[1], d), to_du_mod_d(raw_newdu[2], d)]

                integralnewdu = Int[]
                for u in newdu
                    du_int = d * u
                    val = du_to_int_mod_d(du_int, d)
                    push!(integralnewdu, val)
                end

                println("    p=($p1,$p2): integralnewdu = ", integralnewdu)

                QA = BigFloat(-QQ_form_p(integralnewdu...) / (d * (d - 2)))
                s = isodd(d) ? 1 : (1 + integralnewdu[1]) * (1 + integralnewdu[2])
                new_ovs[gp[1]+1, gp[2]+1] = ζ^QA * (-1)^s * c_g_j_p *
                                              shin_rm(gF_j_p.A, rho_j_bf_p, integralnewdu, d)

                if abs(abs(shin_rm(gF_j_p.A, rho_j_bf_p, integralnewdu, d) *
                           shin_rm(gF_j_p.A, rho_j_bf_p, -integralnewdu, d)) - 1.0) > 1e-10
                    println("    FAIL at p=($p1,$p2)")
                    println("    [fail-debug] elem1=", elem1, " elem2=", elem2, " product=", product)
                    println("    [fail-debug] u1=", u1_p, " u2=", u2_p, " du=", du,
                            " raw_newdu=", raw_newdu, " newdu=", newdu,
                            " integralnewdu=", integralnewdu)
                    test1 = false
                end
            end
        end

        println("  all |vv(p)vv(-p)| correct: ", test1)

        println("  v(p)*v(-p) for all nonzero p (after Galois action):")
        for p1 = 0:d-1
            for p2 = 0:d-1
                (p1 == 0 && p2 == 0) && continue
                mp1 = mod(-p1, d)
                mp2 = mod(-p2, d)
                product = new_ovs[p1+1, p2+1] * new_ovs[mp1+1, mp2+1]
                println("    p=($p1,$p2): v(p)*v(-p) = ", Float64(real(product)),
                        " + ", Float64(imag(product)), "im",
                        "  |v(p)*v(-p)| = ", Float64(abs(product)),
                        "  1/(d+1) = ", 1.0/(d+1))
            end
        end
    end

    max_imag = maximum(abs.(imag.(new_ovs)))
    println("  max imaginary part of new_ovs = ", max_imag)

    dev = ghost_overlap_test_complex(new_ovs, d)
    println("  ghost condition deviation on new_ovs = ", dev)

    println("  Constructing fiducial projector M (unscaled)...")
    Id = Matrix{Complex{BigFloat}}(I, d, d)
    M = Id / BigFloat(d)
    for p1 = 0:d-1
        for p2 = 0:d-1
            (p1 == 0 && p2 == 0) && continue
            M += new_ovs[p1+1, p2+1] / BigFloat(d) * wh(p1, p2, d)
        end
    end

    M_float = Complex{Float64}.(M)
    println("  ||M^2 - M||_F = ", norm(M_float^2 - M_float))
    println("  tr(M) = ", tr(M_float))

    vals, vecs = eigen(M_float)
    println("  All eigenvalues sorted by distance from 1:")
    sorted_idx = sortperm(abs.(vals .- 1))
    for i in 1:min(5, d)
        println("    eigenvalue ", i, ": ", vals[sorted_idx[i]])
    end

    idx = sorted_idx[1]
    psi = Complex{BigFloat}.(vecs[:, idx])
    psi = psi / norm(psi)

    println("  ghost_frame_test = ", ghost_frame_test(psi))
    println("  ghost_overlap_test = ", ghost_overlap_test(psi))
    println("  sic_frame_test = ", sic_frame_test(psi))

    gF = AdmissibleTuple(d, F.r, is_principal_J ? F.Q : gQ_j)
    return new_ovs, gF, psi
end

# LOCAL, more robust version (shadows ray_class_group.jl's if that were also
# loaded - see the include note at the top of this file): searches for O's
# own fundamental unit power t0 and the period j0 of its own subgroup,
# rather than the OK-unit-index formula N1*N3/(N2*N4) used in
# ../01_ray_class_representatives/ray_class_group.jl. That formula-based
# version was found to be wrong by the unit index in some cases (missing the
# -eps^k sign case); this direct-search version fixes both issues.
function num_principal_ray_classes(OK, O, m, Sigma)
    K = number_field(O)
    println("    num_principal: starting")
    GC.gc()

    f = conductor(O)
    println("    num_principal: conductor = ", f)
    GC.gc()

    m_O = ideal(O, minimum(m))
    R_m, _ = quo(O, m_O)
    U_Rm, _ = unit_group(R_m)
    N1 = Int(order(U_Rm)) * 2^length(Sigma)
    println("    num_principal: N1 = ", N1)
    GC.gc()

    U_O, mU_O = unit_group(O)
    R_m_O, proj_m_O = quo(O, m)
    minus_one = O(-1)
    torsion_index = (isone(proj_m_O(minus_one)) &&
                     all(sign(K(elem_in_nf(minus_one)), p) > 0
                         for p in Sigma)) ? 1 : 2
    println("    num_principal: torsion_index = ", torsion_index)

    # Robustly find a generator of infinite order (the fundamental unit),
    # rather than assuming it is always U_OK[2] -- Hecke does not guarantee
    # generator ordering.
    U_OK, mU_OK = unit_group(OK)
    eps = nothing
    for i in 1:ngens(U_OK)
        cand = elem_in_nf(mU_OK(U_OK[i]))
        is_torsion = try
            is_torsion_unit(OK(cand))[1]
        catch
            # Fallback: a torsion unit of a real quadratic field is a root
            # of unity, so it must be +-1 (norm 1, and equal to its own
            # inverse up to sign at every real embedding of absolute value 1).
            all(abs(sign(cand, p)) == 1 && cand == 1 || cand == -1 for p in real_places(K))
        end
        if !is_torsion
            eps = cand
            break
        end
    end
    eps === nothing && error("num_principal_ray_classes: could not find an infinite-order unit generator of OK")
    println("    num_principal: eps (from OK) = ", eps)

    # Search for the minimal k such that EITHER eps^k or -eps^k is congruent
    # to 1 mod m and positive at every place in Sigma. Checking only the
    # pure power eps^k (and never -eps^k) is unsound: since -1 itself is
    # never in the kernel (Sigma is nonempty, so -1 always fails
    # positivity), a subgroup containing the minimal solution u0 cannot also
    # contain -u0. If the true minimal solution is u0 = -eps^{k0}, then the
    # smallest PURE power of eps back in the kernel is eps^{2 k0} = u0^2,
    # i.e. exactly double the true period -- silently corrupting N2 by a
    # factor of 2 whenever this case is hit.
    # First find t0: the minimal positive k such that eps^k is even
    # representable as an element of O. O's own unit group is generated by
    # eps_O := eps^t0 (together with -1), NOT by eps itself whenever
    # t0 > 1 -- O is a proper (non-maximal, conductor > 1) suborder of OK,
    # so most powers of OK's fundamental unit simply do not lie in O.
    t0 = 0
    eps_O_elem = nothing
    for k in 1:100_000
        cand = try O(OK(eps^k)) catch; nothing end
        if cand !== nothing
            t0 = k
            eps_O_elem = cand
            break
        end
    end
    t0 == 0 && error("num_principal_ray_classes: could not find any representable power of eps in O")
    println("    num_principal: t0 (minimal power of eps representable in O) = ", t0)

    # Now search over MULTIPLES of t0 (i.e. powers of O's own fundamental
    # unit eps_O = eps^t0) for the minimal j such that eps_O^j (or its
    # negative) is congruent to 1 mod m and positive at every place in
    # Sigma. Every representable power of eps is automatically a multiple
    # of t0, so this is equivalent to (and replaces) searching over all k.
    j0 = 0
    for j in 1:100_000
        k = j * t0
        epsk_nf = eps^k
        epsk_O = O(OK(epsk_nf))  # guaranteed representable since k is a multiple of t0

        pos_ok = isone(proj_m_O(epsk_O)) && all(sign(K(epsk_nf), p) > 0 for p in Sigma)
        neg_epsk_O = O(OK(-epsk_nf))
        neg_pos_ok = isone(proj_m_O(neg_epsk_O)) && all(sign(-K(epsk_nf), p) > 0 for p in Sigma)

        if pos_ok || neg_pos_ok
            j0 = j
            break
        end
    end
    j0 == 0 && error("Period not found")
    println("    num_principal: j0 (period, in units of O's own fundamental unit eps^t0) = ", j0)
    N2 = torsion_index * j0
    println("    num_principal: N2 = ", N2)
    GC.gc()

    N1_div_N2, r = divrem(N1, N2)
    r != 0 && error("num_principal_ray_classes: N1 = $N1 is not divisible by N2 = $N2 (period/torsion computation is likely still wrong)")
    N = N1_div_N2 - 1
    println("    num_principal: N = ", N)
    return N
end

function ray_class_group_representatives(K, O, m, Sigma)
    OK = maximal_order(K)
    println("  ray_class: OK ready")
    GC.gc()

    P_O, mP_O = picard_group(O)
    println("  ray_class: picard_group done, order = ", order(P_O))
    GC.gc()

    gens_P = gens(P_O)
    divs = elementary_divisors(P_O)
    ranges = [0:Int(d)-1 for d in divs]

    group_elems = []
    for exps in Iterators.product(ranges...)
        g = isempty(gens_P) ? zero(P_O) :
            sum(exps[i] * gens_P[i] for i in 1:length(gens_P))
        push!(group_elems, g)
    end
    println("  ray_class: group_elems built, count = ", length(group_elems))
    GC.gc()

    d_val = Int(minimum(m))
    picard_reps = make_coprime_representatives(O, m, mP_O, group_elems, d_val)
    println("  ray_class: make_coprime_representatives done")
    GC.gc()

    println("  ray_class: picard_reps:")
    for (i, J) in enumerate(picard_reps)
        println("    rep $i: ", J, " group_elem = ", group_elems[i])
    end

    N = num_principal_ray_classes(OK, O, m, Sigma)
    println("  ray_class: num_principal_ray_classes = ", N)
    GC.gc()

    principal_reps = enumerate_principal_ray_classes(K, O, m, Sigma, N)
    println("  ray_class: enumerate done")

    all_reps = []
    all_triples = []
    trivial = ideal(O, 1)
    trivial_elem = zero(P_O)

    for (idx, J) in enumerate(picard_reps)
        J_class = group_elems[idx]
        is_principal_J = (J_class == trivial_elem)

        push!(all_reps, J)
        push!(all_triples, (J, trivial, one(K), is_principal_J))
        for (I, gamma, h) in principal_reps
            I_O = contract(I, O)
            push!(all_reps, J * I_O)
            push!(all_triples, (J, I_O, elem_in_nf(gamma), is_principal_J))
        end
    end

    return all_reps, all_triples
end

# ---------------------------------------------------------------------------
# ray_class_group_data: given an AdmissibleTuple F, builds K, the conductor-
# F.q order O, the modulus m = (dbar) (dbar = F.d if odd, else 2*F.d), and
# Sigma = [second real place] -- exactly the H_{d*infty2}^{O_f} setup used
# throughout the ghost SIC computations -- then uses num_principal_ray_classes
# (via ray_class_group_representatives) to compute:
#   - ray_class_order: the order of H_{d*infty2}^{O_f}
#   - all_reps: a list of representative ideals, one per class
#   - all_triples: (Picard-class-rep J, principal-part ideal I_O, generator
#     gamma, is J itself principal?) for each representative, matching the
#     data used to build twists in test_all_twists
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ray_class_group_elementary_divisors: computes the elementary divisors of
# H = H_{m*infty2}^{O_f} itself, not just its order -- by directly
# multiplying the explicit representative generators the code already
# produces (via ray_class_group_representatives: each ray class is an
# explicit pair (J, gamma) of a Pic(O)-class ideal and a principal-part
# generator) and reducing products back to representative indices.
#
# reduce_index(I'): given ANY fractional O-ideal I' coprime to m, find
# which representative index it matches, by:
#   1. finding its Picard class g = mP_O \ I'
#   2. dividing by that class's base ideal J_g to get a principal ideal,
#      extracting its generator alpha via is_principal
#   3. testing each stored (gamma, idx) for that class: alpha/gamma matches
#      exactly when it is a genuine unit of O, congruent to 1 mod m, and
#      positive at every place in Sigma (the literal definition of
#      U_{m,Sigma}(O)-equivalence)
#
# Once reduce_index gives a black-box multiplication table on the
# (finitely many, small) representative indices, the group's elementary
# divisors are recovered purely from index arithmetic: compute every
# element's order by repeated multiplication, then for each prime p
# dividing |H| use the standard Ulm-invariant formula
#     (#Z/p^j summands) = 2*log_p(a_j) - log_p(a_{j-1}) - log_p(a_{j+1})
# where a_k = #{elements of order dividing p^k}, derived from the
# already-computed element orders.
# ---------------------------------------------------------------------------
function ray_class_group_elementary_divisors(K, O, m, Sigma)
    OK = maximal_order(K)
    P_O, mP_O = picard_group(O)
    all_reps, all_triples = ray_class_group_representatives(K, O, m, Sigma)
    H_order = length(all_triples)

    # Picard-class component: use Hecke's own P_O group directly (abstract
    # group elements, combined via ordinary + in P_O), rather than
    # re-deriving classes by multiplying ideals and taking mP_O \ product.
    class_of = [mP_O \ J for (J, _, _, _) in all_triples]

    reps_by_class = Dict{Any,Vector{Int}}()
    J_of = Dict{Any,Any}()
    for (idx, g) in enumerate(class_of)
        if !haskey(reps_by_class, g)
            reps_by_class[g] = Int[]
            J_of[g] = all_triples[idx][1]
        end
        push!(reps_by_class[g], idx)
    end

    # Principal-class component: reuse the SAME notion of ray equivalence
    # that enumerate_principal_ray_classes uses (L_prime), but avoid its
    # per-comparison cost. Since reduction mod m and sign at a real place
    # are both ring homomorphisms (multiplicative), we can precompute a
    # cheap SIGNATURE (residue in O/m, sign vector at Sigma) for every
    # representative gamma and every element of L_prime ONCE, then combine
    # signatures via plain multiplication -- instead of reconstructing
    # field elements and re-running precision-escalating embeddings on
    # every single comparison (which is what made this O(H^2 * |L_prime|)
    # step prohibitively slow: potentially hundreds of thousands of
    # embedding evaluations for even a group as small as order 32).
    L_prime, sign_at_place = ray_class_unit_reps(K, O, m, Sigma)
    R_m, proj_m_O = quo(O, m)

    function signature(x)
        x_nf = try elem_in_nf(x) catch; K(x) end
        x_O = O(OK(x_nf))
        res = proj_m_O(x_O)
        signs = Tuple(sign_at_place(x_nf, rho) for rho in Sigma)  # each +1 or -1
        return (res, signs)
    end

    combine_sig(s1, s2) = (s1[1] * s2[1], Tuple(s1[2][k] * s2[2][k] for k in 1:length(s1[2])))

    gamma_sig = [signature(all_triples[idx][3]) for idx in 1:H_order]
    L_prime_sig = [signature(u) for u in L_prime]

    function reduce_index(g_combined, mu_sig, i, j)
        base_sig = combine_sig(combine_sig(gamma_sig[i], gamma_sig[j]), mu_sig)
        matches = Int[]
        for idx in reps_by_class[g_combined]
            target = gamma_sig[idx]
            for usig in L_prime_sig
                if combine_sig(base_sig, usig) == target
                    push!(matches, idx)
                    break
                end
            end
        end
        if length(matches) == 0
            error("ray_class_group_elementary_divisors: reduce_index found no matching representative " *
                  "for i=$i, j=$j")
        elseif length(matches) > 1
            println("  WARNING: mul($i,$j) matched MULTIPLE representatives: ", matches,
                     " -- equivalence test may be too permissive")
        end
        return matches[1]
    end

    # mu (the generator of (J_i*J_j)/J_combined) is only defined up to a
    # unit of O -- is_principal_with_data can legitimately return a
    # DIFFERENT unit-multiple on different calls to the SAME ideal, and
    # different unit-multiples generally have different residues mod m /
    # different signs at Sigma. Since (J_i*J_j)/J_combined depends only on
    # the pair of Picard classes (class_of[i], class_of[j]) -- not on the
    # specific i,j within those classes -- cache mu's signature by that
    # pair so it is computed once and reused consistently, rather than
    # silently drifting between calls.
    mu_sig_cache = Dict{Tuple{Any,Any},Any}()
    function mu_signature(gi, gj, g_combined)
        key = (gi, gj)
        haskey(mu_sig_cache, key) && return mu_sig_cache[key]
        Ji, Jj, Jc = J_of[gi], J_of[gj], J_of[g_combined]
        ratio = (Ji * Jj) * inv(Jc)
        fl, mu_elem = is_principal_with_data(ratio)
        fl || error("ray_class_group_elementary_divisors: J_i*J_j / J_combined unexpectedly non-principal")
        mu = elem_in_nf(mu_elem)
        sig = signature(mu)
        mu_sig_cache[key] = sig
        return sig
    end

    function mul(i, j)
        gi, gj = class_of[i], class_of[j]
        g_combined = gi + gj
        mu_sig = mu_signature(gi, gj, g_combined)

        result = reduce_index(g_combined, mu_sig, i, j)
        println("    mul(", i, ",", j, ") = ", result,
                "  [gamma_i=", all_triples[i][3], ", gamma_j=", all_triples[j][3],
                ", gamma_result=", all_triples[result][3], "]")
        return result
    end

    identity_idx = findfirst(idx -> class_of[idx] == zero(P_O) &&
                                    all_triples[idx][3] == one(K),
                              1:H_order)
    identity_idx === nothing && error("ray_class_group_elementary_divisors: could not locate identity element")

    # Order of every element, via repeated multiplication (pure index
    # arithmetic on the black-box table -- no group-object API needed).
    # An element's order must divide H_order, so if this hasn't converged
    # within H_order+1 steps, mul/reduce_index is not implementing the
    # correct group operation (rather than this simply being slow) --
    # fail loudly instead of hanging indefinitely.
    orders = Vector{Int}(undef, H_order)
    for i in 1:H_order
        println("  Computing order of element ", i, " of ", H_order, "...")
        cur = i
        k = 1
        while cur != identity_idx
            cur = mul(cur, i)
            k += 1
            k > H_order + 1 && error("ray_class_group_elementary_divisors: order of element $i " *
                                      "did not converge within $(H_order+1) steps -- mul/reduce_index " *
                                      "is producing an incorrect group operation, not just running slowly")
        end
        orders[i] = k
    end
    println("Element orders: ", sort(orders))

    by_prime = Dict{Int,Vector{Int}}()  # p => descending list of exponents
    for p in Set(first.(collect(factor(H_order))))
        p = Int(p)
        max_k = 0
        while p^(max_k + 1) <= H_order
            max_k += 1
        end
        # a[k+1] = a_k := #{elements of order dividing p^k}, for k=0..max_k+1
        # (one extra level beyond max_k so a_{max_k+1} is available and
        # already saturated, avoiding any boundary special-casing below).
        a = [count(o -> mod(p^k, o) == 0, orders) for k in 0:(max_k + 1)]
        exps = Int[]
        for j in 1:max_k
            exponent = 2 * log(p, a[j+1]) - log(p, a[j]) - log(p, a[j+2])
            mult = round(Int, exponent)
            for _ in 1:mult
                push!(exps, j)
            end
        end
        sort!(exps, rev=true)
        by_prime[p] = exps
    end

    # Convert the classical prime-power (elementary divisor) decomposition
    # into Hecke's own invariant-factor convention d_1 | d_2 | ... | d_n
    # (matching how elementary_divisors(P_O) is reported elsewhere in this
    # file): line up each prime's exponents in descending order, padding
    # shorter lists implicitly with factor 1, and multiply corresponding
    # entries across primes.
    max_len = isempty(by_prime) ? 0 : maximum(length(v) for v in values(by_prime))
    ray_class_divs = Int[]
    for i in 1:max_len
        d = 1
        for (p, exps) in by_prime
            if i <= length(exps)
                d *= p^exps[i]
            end
        end
        d != 1 && push!(ray_class_divs, d)
    end
    sort!(ray_class_divs)

    prod(ray_class_divs; init=1) == H_order ||
        error("ray_class_group_elementary_divisors: product of computed divisors " *
              "$(prod(ray_class_divs; init=1)) does not match |H| = $H_order")

    println("|Pic(O)| = ", order(P_O), ", Pic(O) elementary divisors: ", elementary_divisors(P_O))
    println("H_{m*infty2}^{O_f} elementary divisors: ", ray_class_divs)
    println("|H_{m*infty2}^{O_f}| = ", H_order)

    return ray_class_divs
end

function ray_class_group_elementary_divisors(F::AdmissibleTuple)
    D = Int(F.D)
    q = Int(F.q)

    K, a = quadratic_field(D)
    Dr = squarefree_part(D)
    sqrt_Dr = a // isqrt(D ÷ Dr)
    omega = mod(Dr, 4) == 1 ? (K(1) + sqrt_Dr) // 2 : sqrt_Dr
    O = Order(K, [one(K), q * omega])

    d_val = Int(F.d)
    dbar = mod(d_val, 2) == 1 ? d_val : 2 * d_val
    m = ideal(O, dbar)

    all_places = real_places(K)
    Sigma = [all_places[2]]

    println("d = ", d_val, ", Q = ", F.Q)
    return ray_class_group_elementary_divisors(K, O, m, Sigma)
end

ray_class_group_elementary_divisors(d::Int) = ray_class_group_elementary_divisors(AdmissibleTuple(d))

function ray_class_group_data(F::AdmissibleTuple)
    D = Int(F.D)
    q = Int(F.q)

    K, a = quadratic_field(D)
    Dr = squarefree_part(D)
    sqrt_Dr = a // isqrt(D ÷ Dr)
    omega = mod(Dr, 4) == 1 ? (K(1) + sqrt_Dr) // 2 : sqrt_Dr
    O = Order(K, [one(K), q * omega])

    d_val = Int(F.d)
    dbar = mod(d_val, 2) == 1 ? d_val : 2 * d_val
    m = ideal(O, dbar)

    all_places = real_places(K)
    Sigma = [all_places[2]]

    all_reps, all_triples = ray_class_group_representatives(K, O, m, Sigma)
    ray_class_order = length(all_reps)

    println("d = ", d_val, ", Q = ", F.Q,
            ": |H_{d*infty2}^{O_f}| = ", ray_class_order)

    return ray_class_order, all_reps, all_triples
end

ray_class_group_data(d::Int) = ray_class_group_data(AdmissibleTuple(d))

function test_all_twists(d::Int)
    F = AdmissibleTuple(d)
    @show F

    D = Int(F.D)
    q = Int(F.q)

    Dr = squarefree_part(D)
    K, a = quadratic_field(D)
    OK = maximal_order(K)
    GC.gc()
    GC.gc()

    sqrt_Dr = a // isqrt(D ÷ Dr)
    omega = mod(Dr, 4) == 1 ? (K(1) + sqrt_Dr) // 2 : sqrt_Dr
    O = Order(K, [one(K), q * omega])
    println("conductor of O is: ", conductor(O))
    GC.gc()

    P_O_temp, _ = picard_group(O)
    println("Class group of O has order: ", order(P_O_temp))
    GC.gc()

    println("D = ", D, ", q = ", q)

    d = Int(F.d)
    if mod(d, 2) == 1
        dbar = d
    else
        dbar = 2d
    end

    m = ideal(O, dbar)
    GC.gc()

    all_places = real_places(K)
    Sigma = [all_places[2]]
    println("Using Sigma = second real place only: ", Sigma)
    GC.gc()
    GC.gc()

    all_reps, all_triples = ray_class_group_representatives(K, O, m, Sigma)

    println("Total triples: ", length(all_triples))

    P_O, mP_O, rho_reps, gQ_reps, F_class = find_class_representatives(O, F, K)
    println("Class representatives:")
    for (g, rho) in rho_reps
        println("  class ", g, ": rho = ", rho, " gQ = ", gQ_reps[g])
    end
    println("F.Q is in class: ", F_class)

    principal_triples = [(J, I, gamma) for (J, I, gamma, is_princ)
                         in all_triples if is_princ]
    nonprincipal_triples = [(J, I, gamma) for (J, I, gamma, is_princ)
                             in all_triples if !is_princ]

    println("Principal triples: ", length(principal_triples))
    println("Non-principal triples: ", length(nonprincipal_triples))

    j_cache = Dict{FinGenAbGroupElem, Any}()   # keyed by u_B (J's Picard class); caches rho_B/s1/s2/etc that only depend on J's class, not on gamma or on which specific J object was passed in

    n_principal = min(5, length(principal_triples))
    println("\n=== Testing first ", n_principal, " principal twists ===")
    for (k, (J, I, gamma)) in enumerate(principal_triples[1:n_principal])
        println("\nPrincipal twist ", k, " of ", n_principal, ":")
        println("  J = ", J)
        println("  gamma = ", gamma)
        println("  norm(gamma) = ", norm(K(gamma)))
        new_ovs, gF, psi = twisted_chi_general(F, J, gamma, K, O,
                                                true, P_O, mP_O,
                                                rho_reps, gQ_reps, F_class, j_cache)
    end

    n_nonprincipal = min(20, length(nonprincipal_triples))
    println("\n=== Testing first ", n_nonprincipal, " non-principal twists ===")
    for (k, (J, I, gamma)) in enumerate(nonprincipal_triples[1:n_nonprincipal])
        println("\nNon-principal twist ", k, " of ", n_nonprincipal, ":")
        println("  J = ", J)
        println("  gamma = ", gamma)
        println("  norm(gamma) = ", norm(K(gamma)))
        new_ovs, gF, psi = twisted_chi_general(F, J, gamma, K, O,
                                                false, P_O, mP_O,
                                                rho_reps, gQ_reps, F_class, j_cache)
    end
end

function test_ray_class_group_size(F::AdmissibleTuple)
    @show F

    D = Int(F.D)
    q = Int(F.q)

    K, a = quadratic_field(D)
    OK = maximal_order(K)
    GC.gc()

    Dr = squarefree_part(D)
    sqrt_Dr = a // isqrt(D ÷ Dr)
    omega = mod(Dr, 4) == 1 ? (K(1) + sqrt_Dr) // 2 : sqrt_Dr
    O = Order(K, [one(K), q * omega])
    GC.gc()

    P_O, mP_O = picard_group(O)
    GC.gc()

    d_val = Int(F.d)
    if mod(d_val, 2) == 1
        dbar = d_val
    else
        dbar = 2 * d_val
    end
    m = ideal(O, dbar)
    GC.gc()

    all_places = real_places(K)
    Sigma = [all_places[2]]

    N = num_principal_ray_classes(OK, O, m, Sigma)
    GC.gc()

    ray_class_size = order(P_O) * (N + 1)
    println("|Pic(O)| = ", order(P_O))
    println("N (nontrivial principal ray classes per Picard class) = ", N)
    println("|H_{d*infty2}^{O_f}| = ", ray_class_size)

    ray_class_divs = ray_class_group_elementary_divisors(K, O, m, Sigma)

    return ray_class_size, ray_class_divs
end

test_ray_class_group_size(d::Int) = test_ray_class_group_size(AdmissibleTuple(d))

function test_ray_class_group_size_first_n_tuples(n::Int)
    for k in 1:n
        println("\n=== Tuple ", k, " of ", n, " (dq(", k, ")) ===")
        try
            F = AdmissibleTuple(dq(k))
            test_ray_class_group_size(F)
        catch e
            println("  ERROR on tuple ", k, " (dq(", k, ") = ", dq(k), "): ", e)
        end
        GC.gc()
    end
end

test_all_twists(21)

#test_ray_class_group_size_first_n_tuples(101)

function ray_class_group_data_first_n_tuples(n::Int)
    for k in 1:n
        println("\n=== Tuple ", k, " of ", n, " (dq(", k, ")) ===")
        try
            F = AdmissibleTuple(dq(k))
            ray_class_group_data(F)
        catch e
            println("  ERROR on tuple ", k, " (dq(", k, ") = ", dq(k), "): ", e)
        end
        GC.gc()
    end
end

#ray_class_group_data_first_n_tuples(101)