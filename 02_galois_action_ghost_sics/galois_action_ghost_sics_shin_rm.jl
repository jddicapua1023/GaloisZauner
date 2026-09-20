using Random
#Random.seed!(305817264)
#Random.seed!(672049183)
#Random.seed!(891357246)
#Random.seed!(384726195)
#Random.seed!(927401853)
#Random.seed!(112358013)
#Random.seed!(654890271)
Random.seed!(208475639)




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

"""
    lattice_reduce_2d(v1, v2, K, places)

2D Gauss/Lagrange lattice reduction of a rank-2 Z-basis {v1,v2} of a lattice
in K, using the Euclidean norm on the real-embedding pair (embedding at
places[1], embedding at places[2]) as the reduction quadratic form.

This is the SAME reduction `find_alpha_rho_direct` already performs on its
own (I*dO) search basis, extracted here as a standalone, reusable helper so
`ray_class_monoid_lookup` can apply it too (see that function's docstring
for why this is essential, not optional, for large-norm ideals). Left
`find_alpha_rho_direct`'s own internal copy untouched to avoid any risk to
already-working code.
"""
function lattice_reduce_2d(v1, v2, K, places)
    embed_pair(x) = (Float64(embedding(places[1])(x)), Float64(embedding(places[2])(x)))
    norm2(e) = e[1]^2 + e[2]^2
    dotp(ea, eb) = ea[1]*eb[1] + ea[2]*eb[2]

    e1 = embed_pair(v1)
    e2 = embed_pair(v2)

    for _ in 1:200
        if norm2(e2) < norm2(e1)
            v1, v2 = v2, v1
            e1, e2 = e2, e1
        end
        mu = round(Int, dotp(e1, e2) / norm2(e1))
        mu == 0 && break
        v2 = v2 - K(mu) * v1
        e2 = embed_pair(v2)
    end

    return v1, v2
end

"""
    conjugation_place(K)

Returns the real place of `K` (a real quadratic field) corresponding to
CONJUGATION (the paper's `infty_2`) - the place under which `gen(K) = sqrt(D)`
embeds to a NEGATIVE real number, and hence under which the "larger" root
`rho > rho'` of a quadratic form embeds to the SMALLER value.

BUGFIX: every construction of `Sigma` in this file used to hardcode
`Sigma = [real_places(K)[2]]`, on the unchecked assumption that index 2 is
always conjugation. `diagnose_real_places` showed this is BACKWARDS for
D=44: `real_places(K)[1]` is conjugation and `real_places(K)[2]` is the
identity embedding - meaning the ray class group (`enumerate_principal_ray_classes`,
via `same_sign_on_Sigma`) has been built with Sigma = the IDENTITY place,
while `twisted_chi_general`'s own positivity check
(`emb_conj_K = embedding(places_K_top[1])`) has been correctly using
conjugation - two different sign conventions in the same computation. This
function replaces every hardcoded `all_places[2]` with an explicit,
self-checking detection, so `Sigma` is never again wrong-by-assumption.
"""
function conjugation_place(K)
    places = real_places(K)
    a = gen(K)
    i = findfirst(p -> Float64(real(embedding(p)(a))) < 0, places)
    i === nothing && error("conjugation_place: neither real place of K sends gen(K) negative - ",
                           "cannot identify conjugation")
    return places[i]
end

"""
    multiplication_matrix_mod_d(gamma, rho, d, D)

Matrix (mod d) of multiplication by `gamma` on the lattice Z*rho + Z, in the
POINT convention used by every point vector in this file:

    (a, b)  <->  the element  b*rho - a        (x_p = p2*rho - p1,
                                                 product = u2*rho_j - u1, ...)

BUGFIX (9/13): this used to return the matrix in the (constant, rho-coefficient)
basis - the raw output of express_in_basis, where x = c1 + c2*rho <-> (c1, c2) -
but was then applied directly to point vectors (a, b) = (-c1, c2). Applying a
matrix in one basis to a vector in another is only harmless when the matrix
commutes with the change of basis S = diag(-1, 1), i.e. when gamma is rational.
For gamma = 1 nothing was wrong; for every non-rational gamma the acted-on
point was wrong.

Why it was invisible in the principal path and visible in the non-principal
one: in the principal path newp = M*p is LINEAR in p, so newp(-p) = -newp(p)
holds for ANY matrix and the |v(p)v(-p)| pair check cannot see the error
(the values are simply evaluated at the wrong points). In the non-principal
path du = M*s_bj(p) where s_bj is only <L_j>-orbit-equivariant under p -> -p,
so M must COMMUTE with L_j - which it does in the correct basis and not in the
mixed one. Confirmed on the seed-208475639 log, twist "Non-principal 2 of 20"
(gamma = -3*sqrt(44) - 20): the old convention reproduces the logged
integralnewdu 440/440 and is orbit-equivariant on exactly the 146 passing
pairs; the corrected convention is orbit-equivariant on 440/440.

The fix is the conjugation S*M*S: flip the sign of both off-diagonal entries.
det is unchanged, so the existing det(M) coprimality gate is unaffected.
"""
function multiplication_matrix_mod_d(gamma, rho, d, D)
    col1 = express_in_basis(gamma, rho, D)
    col2 = express_in_basis(gamma * rho, rho, D)
    to_int_mod(x) = Int(mod(ZZ(numerator(x)) * ZZ(invmod(ZZ(denominator(x)), ZZ(d))), ZZ(d)))
    # M in the (constant, rho-coefficient) basis:
    M = [to_int_mod(col1[1]) to_int_mod(col2[1]);
         to_int_mod(col1[2]) to_int_mod(col2[2])]
    # Convert to the point convention (a,b) <-> b*rho - a, i.e. (c1,c2) = (-a,b):
    # M_point = S*M*S with S = diag(-1,1)  =>  negate the off-diagonal entries.
    return [M[1,1]            mod(-M[1,2], d);
            mod(-M[2,1], d)   M[2,2]]
end

# ---------------------------------------------------------------------------
# HJ-reduction of an ARBITRARY (A, x) pair via an explicit R in GL_2(Z).
#
# Convention (per project owner): for M = [a b; c d] in GL_2(Z) and tau a
# number, M*tau means (a*tau+b)/(c*tau+d). For M in SL_2(R), J_M(tau) =
# c*tau+d is the automorphy factor. For r in Q^2, Gamma_r = {A in SL_2(Z) :
# A*r - r in Z^2}.
#
# Theorem (per project owner): for r in Q^2, A in Gamma_r, R in GL_2(Z), and
# beta in the appropriate domain D~_A with A*beta = beta, let
# s_R(beta) = sgn(j_R(beta)) (assumed nonzero). Then
#   shin^{s_R(beta)*R*r}_{R A R^{-1}}(R*beta) = shin^r_A(beta)         if det(R) = 1
#                                              = shin^r_A(conj(beta))  if det(R) = -1
#
# find_hj_reducing_matrix below only solves the "find R" half of applying
# this theorem: given an ARBITRARY A (not necessarily HJ-reduced) and one of
# its two real fixed points x, it finds R in GL_2(Z) such that R*A*R^{-1} is
# HJ-reduced (0 < rho_- < 1 < rho_+), and tells the caller which of R*x or
# R*conj(x) is that reduced form's own (larger) root - i.e. exactly the
# point to feed to shin_rm together with R*A*R^{-1} in order to compute
# shin_A(x, p, d) via the theorem above. Transforming the vector r/p itself
# (via s_R(beta)*R*r) for a full arbitrary-input shin(A,x,p,d) wrapper is a
# separate follow-up step, not yet implemented here.
# ---------------------------------------------------------------------------

"""
    _mobius_apply(M::AbstractMatrix{<:Integer}, z)

`M·z = (a*z+b)/(c*z+d)` for `M = [a b; c d]`, the convention used throughout
this project.
"""
_mobius_apply(M::AbstractMatrix{<:Integer}, z) = (M[1,1]*z + M[1,2]) / (M[2,1]*z + M[2,2])

"""
    hj_roots(A::AbstractMatrix{<:Integer})

The two real fixed points of the Möbius action of `A = [a b; c d]`
(`M·τ = (aτ+b)/(cτ+d)` convention), i.e. the two roots of
`c*z^2 + (d-a)*z - b = 0`, computed to `BigFloat` precision from `A`'s exact
integer entries. Errors if `c = 0` or if `A` is not hyperbolic (fixed-point
discriminant `<= 0`, i.e. no two distinct real fixed points).
"""
function hj_roots(A::AbstractMatrix{<:Integer})
    a, b, c, d = BigFloat(A[1,1]), BigFloat(A[1,2]), BigFloat(A[2,1]), BigFloat(A[2,2])
    c != 0 || error("hj_roots: c = 0 - A is not hyperbolic in the standard (finite fixed points) sense")
    disc = (d - a)^2 + 4*b*c   # = trace(A)^2 - 4*det(A)
    disc > 0 || error("hj_roots: fixed-point discriminant = $disc is not positive - ",
                       "A has no two distinct real fixed points")
    sq = sqrt(disc)
    return ((a - d) + sq) / (2*c), ((a - d) - sq) / (2*c)   # unordered pair
end

"""
    _gl2z_inv(M::AbstractMatrix{<:Integer})

Exact inverse of `M ∈ GL_2(Z)` (`det(M) = ±1`), as an integer matrix.
"""
function _gl2z_inv(M::AbstractMatrix{<:Integer})
    a, b, c, d = M[1,1], M[1,2], M[2,1], M[2,2]
    dt = a*d - b*c
    (dt == 1 || dt == -1) || error("_gl2z_inv: det(M) = $dt, not in GL_2(Z)")
    return dt .* [d -b; -c a]
end

"""
    find_hj_reducing_matrix(A::AbstractMatrix{<:Integer}, x::BigFloat;
                             tol::Real = 1e-20, max_steps::Int = 10_000)

Given `A ∈ GL_2(Z)` and a real number `x` equal to one of the two fixed
points of `A`'s Möbius action (`A·x = x`), finds `R ∈ GL_2(Z)` such that
`R*A*R^{-1}` is HJ-reduced (its two fixed points `ρ₋, ρ₊` satisfy
`0 < ρ₋ < 1 < ρ₊`).

`x`'s Galois conjugate (`A`'s other fixed point) is derived directly from
`A`'s own integer entries via `hj_roots`, and `x` is matched against those
two roots by numeric proximity (within `tol`) as a sanity check that `x`
really is a fixed point of `A`.

The reduction algorithm is the classical translation/inversion algorithm
for real quadratic irrationals: at each step, let `(lo, hi) = extrema` of
the current pair of tracked root-images. If `0 < lo < 1 < hi`, done. Else if
`lo` is not already in `(0,1)`, apply the integer shift
`T_n = [1 -n; 0 1]` (`z ↦ z-n`, `det = 1`) with `n = floor(lo)`. Otherwise
(`lo ∈ (0,1)` but `hi ≤ 1`), apply the inversion `S = [0 1; 1 0]`
(`z ↦ 1/z`, `det = -1`). This always terminates (real quadratic irrationals
have purely periodic continued-fraction-type behavior under this process).

Returns a named tuple `(R, detR, A_reduced, root_used, reduced_root)`:
- `R`             : the `GL_2(Z)` matrix found, as an `Int` matrix.
- `detR`          : `det(R)`, either `1` or `-1`.
- `A_reduced`     : `R*A*R^{-1}` (HJ-reduced), as an `Int` matrix.
- `root_used`     : `:x` if `detR == 1` (meaning `R*x` is `A_reduced`'s own
                    reduced root, `> 1`), or `:conj_x` if `detR == -1`
                    (meaning `R*conj(x)` is instead) - per the theorem
                    above, this is exactly which point to feed `shin_rm`
                    together with `A_reduced` to compute `shin_A(x,p,d)`.
- `reduced_root`  : the `BigFloat` value of that reduced root (`> 1`).
- `x0`            : `x` itself (unmutated), returned for convenience so
                    callers (e.g. `shin` below) don't need to recompute it.
- `conjx0`        : `A`'s other fixed point (`x`'s Galois conjugate),
                    likewise unmutated.
"""
function find_hj_reducing_matrix(A::AbstractMatrix{<:Integer}, x::BigFloat;
                                  tol::Real = 1e-20, max_steps::Int = 10_000)
    r1, r2 = hj_roots(A)

    if abs(x - r1) < tol
        x_val, cx_val = r1, r2
    elseif abs(x - r2) < tol
        x_val, cx_val = r2, r1
    else
        error("find_hj_reducing_matrix: x = $x matches neither fixed point ",
              "of A (roots are $r1, $r2) within tol = $tol")
    end

    x0, cx0 = x_val, cx_val   # preserve originals; x_val/cx_val get mutated by the loop below

    R = BigInt[1 0; 0 1]   # running total; R_new = (latest elementary step) * R_old

    step = 0
    while true
        step += 1
        step > max_steps && error("find_hj_reducing_matrix: did not reach an ",
                                   "HJ-reduced pair after $max_steps steps")

        lo, hi = minmax(x_val, cx_val)
        (lo > 0 && lo < 1 && hi > 1) && break   # 0 < rho_- < 1 < rho_+: done

        if !(lo > 0 && lo < 1)
            n = BigInt(floor(lo))
            T = BigInt[1 -n; 0 1]         # z -> z - n, det = 1
            x_val -= n
            cx_val -= n
            R = T * R
        else
            S = BigInt[0 1; 1 0]          # z -> 1/z, det = -1
            x_val = 1 / x_val
            cx_val = 1 / cx_val
            R = S * R
        end
    end

    Rint = Int.(R)
    detR = Rint[1,1]*Rint[2,2] - Rint[1,2]*Rint[2,1]
    (detR == 1 || detR == -1) || error("find_hj_reducing_matrix: det(R) = $detR, expected ±1")

    Rinv = _gl2z_inv(Rint)
    A_reduced = Rint * Matrix(A) * Rinv

    root_used, reduced_root = detR == 1 ? (:x, x_val) : (:conj_x, cx_val)

    reduced_root > 1 || @warn "find_hj_reducing_matrix: reduced_root = $reduced_root " *
                               "is not > 1 as expected for the HJ-reduced (main) root " *
                               "- check theorem application / algorithm correctness"

    return (R = Rint, detR = detR, A_reduced = A_reduced,
            root_used = root_used, reduced_root = reduced_root,
            x0 = x0, conjx0 = cx0)
end

"""
    shin(A::AbstractMatrix{<:Integer}, x::BigFloat, p::Vector{<:Integer}, d::Int64;
         tol::Real = 1e-20, max_steps::Int = 10_000)

`shin_A^{p/d}(x)` for an ARBITRARY (not necessarily HJ-reduced) `A ∈
GL_2(Z)` and real fixed point `x` of its Möbius action (`A·x = x`), computed
by reducing to `shin_rm` via `find_hj_reducing_matrix` and the
transformation theorem

    shin^{s_R(β)·R·r}_{R A R^{-1}}(R·β) = shin^r_A(β)          if det(R) = 1
                                         = shin^r_A(conj(β))   if det(R) = -1

with `s_R(β) = sgn(J_R(β)) = sgn(c_R·β + d_R)` for `R = [a_R b_R; c_R d_R]`.

Concretely: `find_hj_reducing_matrix(A, x)` gives `R`, `A_reduced =
R*A*R^{-1}`, and tells us which β makes `R*β` equal to `reduced_root` -
that's `β = x` when `det(R) = 1` (`root_used = :x`), or `β = conj(x)` when
`det(R) = -1` (`root_used = :conj_x`), matching the theorem's two cases
exactly so that in BOTH cases the right-hand side above is `shin_A^r(x)` -
the thing we actually want, never `shin_A^r(conj(x))`.

Reading the theorem with that β and `r = p/d`:

    shin_A^{p/d}(x) = shin_{A_reduced}^{s_R(β)·R·(p/d)}(reduced_root)

Since `R` has integer entries, `p` is an integer vector, and `s_R(β) =
±1`, the transformed vector `s_R(β)·R·p` is again an integer vector (then
reduced mod `d`, matching this codebase's convention elsewhere of always
feeding `shin_rm` an integer vector already reduced mod `d`), so the
right-hand side is exactly `shin_rm(A_reduced, reduced_root,
mod.(s_R(β).*(R*p), d), d)`.
"""
function shin(A::AbstractMatrix{<:Integer}, x::BigFloat, p::Vector{<:Integer}, d::Int64;
              tol::Real = 1e-20, max_steps::Int = 10_000)
    red = find_hj_reducing_matrix(A, x; tol = tol, max_steps = max_steps)

    beta = red.root_used == :x ? red.x0 : red.conjx0

    a_R, b_R, c_R, d_R = red.R[1,1], red.R[1,2], red.R[2,1], red.R[2,2]
    j_R_beta = c_R * beta + d_R
    s_R = Int(sign(j_R_beta))
    s_R != 0 || error("shin: s_R(beta) = sgn(J_R(beta)) = 0 (J_R(beta) = $j_R_beta) - ",
                       "theorem's nonzero hypothesis fails for this (A, x, R)")

    p_transformed = mod.(s_R .* (red.R * Int.(p)), d)

    return shin_rm(Matrix(red.A_reduced), red.reduced_root, p_transformed, d)
end

# ---------------------------------------------------------------------------
# Stabilizer, within a congruence subgroup Gamma_s, of a real quadratic
# irrational rho_B under the SL_2(Z) Möbius action - needed to build the
# per-ray-class shin table (see ray_class_shin_table below).
#
# Fact used: the FULL SL_2(Z)-stabilizer of a root of a primitive indefinite
# binary quadratic form (a,b,c) of discriminant Delta = b^2-4ac is
# <-I, L>, where L is the "fundamental automorph"
#     L = [ (t-b*u)/2   -c*u ; a*u   (t+b*u)/2 ]
# built from the minimal positive-integer solution (t,u) of the Pell-type
# equation t^2 - Delta*u^2 = 4 (standard; see e.g. Buell, "Binary Quadratic
# Forms"). Being hyperbolic (infinite order, two distinct real fixed
# points), L cannot swap its own two fixed points, so it fixes EACH of
# rho_B and conj(rho_B) individually - which one is realized as "L" versus
# "L^{-1}" here depends on the sign conventions picked for (a,b,c) and
# (t,u), so this is verified numerically below rather than assumed.
#
# Intersecting <-I, L> with the finite-index congruence subgroup Gamma_s
# gives either <A> or <-I, A> for A = L^k, k the smallest positive power
# with L^k in Gamma_s (since -I is never itself a power of L, having order
# 2 while L has infinite order).
# ---------------------------------------------------------------------------

"""
    minimal_pell_solution_plus4(Delta::Integer; max_u::Integer = 10_000_000)

Minimal pair of positive integers `(t,u)` solving `t^2 - Delta*u^2 = 4`
(`Delta > 0`, not a perfect square), found by direct search over
`u = 1, 2, 3, ...` up to `max_u`.

NOTE: this is a brute-force search, not a continued-fraction-based Pell
solver - simple to verify, but could be slow if `Delta`'s fundamental
solution has very large `u`. A CF-based solver would be the natural
optimization if that becomes a bottleneck in practice.
"""
function minimal_pell_solution_plus4(Delta::Integer; max_u::Integer = 10_000_000)
    Delta > 0 || error("minimal_pell_solution_plus4: Delta = $Delta must be positive")
    r = isqrt(Delta)
    r*r != Delta || error("minimal_pell_solution_plus4: Delta = $Delta is a perfect square")
    for u in 1:max_u
        rhs = 4 + Delta * BigInt(u)^2
        t = isqrt(rhs)
        if t*t == rhs
            return Int(t), u
        end
    end
    error("minimal_pell_solution_plus4: no solution found with u <= $max_u for Delta = $Delta")
end

"""
    hj_form_automorph(a::Integer, b::Integer, c::Integer; max_u::Integer = 10_000_000)

The fundamental automorph `L ∈ SL_2(Z)` of the primitive indefinite binary
quadratic form `(a,b,c)` (`L^T·Q·L = Q`), from the minimal solution `(t,u)`
of `t^2 - Delta*u^2 = 4` (`Delta = b^2-4ac`).
"""
function hj_form_automorph(a::Integer, b::Integer, c::Integer; max_u::Integer = 10_000_000)
    Delta = b^2 - 4*a*c
    t, u = minimal_pell_solution_plus4(Delta; max_u = max_u)
    iseven(t - b*u) || error("hj_form_automorph: t-b*u = $(t - b*u) is odd - ",
                              "unexpected parity failure for a genuine Pell solution")
    L = [ (t - b*u) ÷ 2   -c*u          ;
           a*u             (t + b*u) ÷ 2 ]
    detL = L[1,1]*L[2,2] - L[1,2]*L[2,1]
    detL == 1 || error("hj_form_automorph: det(L) = $detL, expected 1")
    return L
end

"""
    hj_automorph_for_root(rho_B, D::Int; max_u::Integer = 10_000_000, tol::Real = 1e-15)

The part of the stabilizer construction that depends ONLY on `rho_B` (not
on `s`): builds `rho_B`'s minimal-polynomial form `(a,b,c)`, its fundamental
automorph `L` via the (expensive, brute-force) Pell-equation search, and
verifies numerically that the returned matrix genuinely fixes `rho_B`
itself (falling back to `L^{-1}` if not).

Split out from `stabilizer_generator_in_Gamma_s` so that callers processing
many `s` vectors that share the same `rho_B` (e.g. many ray classes in the
same Picard-class fiber, as in `ray_class_shin_table`) can compute this
expensive part ONCE per `rho_B` and reuse it, instead of repeating the
brute-force Pell search for every `s`.
"""
function hj_automorph_for_root(rho_B, D::Int; max_u::Integer = 10_000_000, tol::Real = 1e-15)
    c0 = coeff(rho_B, 0)
    c1 = coeff(rho_B, 1)
    iszero(c1) && error("hj_automorph_for_root: rho_B has zero irrational part")

    p0 = c0^2 - c1^2 * D
    p1 = -2 * c0
    scale = lcm(Int(denominator(QQ(p0))), Int(denominator(QQ(p1))), 1)
    a_int = scale
    b_int = Int(round(scale * p1))
    c_int = Int(round(scale * p0))
    g = gcd(gcd(abs(a_int), abs(b_int)), abs(c_int))
    g == 0 && error("hj_automorph_for_root: degenerate form (a=b=c=0) for rho_B = $rho_B")
    a_int ÷= g; b_int ÷= g; c_int ÷= g
    if a_int < 0
        a_int, b_int, c_int = -a_int, -b_int, -c_int
    end

    L = hj_form_automorph(a_int, b_int, c_int; max_u = max_u)

    rho_B_bf = BigFloat(c0) + BigFloat(c1) * sqrt(BigFloat(D))
    L_rho = _mobius_apply(L, rho_B_bf)
    if abs(L_rho - rho_B_bf) > tol
        Linv = _gl2z_inv(L)
        Linv_rho = _mobius_apply(Linv, rho_B_bf)
        abs(Linv_rho - rho_B_bf) < tol ||
            error("hj_automorph_for_root: neither L nor L^{-1} fixes rho_B numerically ",
                  "(L*rho_B = $L_rho, L^{-1}*rho_B = $Linv_rho, rho_B = $rho_B_bf) - ",
                  "check the (a,b,c)/automorph sign convention")
        L = Linv
    end

    return L
end

"""
    stabilizer_power_in_Gamma_s(L::AbstractMatrix{<:Integer}, s::Vector;
                                 max_power::Int = 10_000)

Given `L ∈ SL_2(Z)` already known to fix some `rho_B` (e.g. from
`hj_automorph_for_root`), finds the smallest positive integer power `k`
with `L^k ∈ Γ_s = {B ∈ SL_2(Z) : B*s - s ∈ Z^2}` (exact rational arithmetic
on `s`, no floating tolerance - this part is cheap, unlike the Pell search
inside `hj_automorph_for_root`). Returns `(A = L^k, has_neg_I)`, `has_neg_I`
recording whether `-I ∈ Γ_s` too (informational only).
"""
function stabilizer_power_in_Gamma_s(L::AbstractMatrix{<:Integer}, s::Vector;
                                      max_power::Int = 10_000)
    s1, s2 = QQ(s[1]), QQ(s[2])
    has_neg_I = isone(denominator(2*s1)) && isone(denominator(2*s2))

    Lk = [1 0; 0 1]
    for k in 1:max_power
        Lk = L * Lk
        new1 = Lk[1,1]*s1 + Lk[1,2]*s2
        new2 = Lk[2,1]*s1 + Lk[2,2]*s2
        if isone(denominator(new1 - s1)) && isone(denominator(new2 - s2))
            return Lk, has_neg_I
        end
    end

    error("stabilizer_power_in_Gamma_s: no power of L up to L^$max_power lies in ",
          "Gamma_s for s = $s - either max_power is too small, or something upstream is wrong")
end

"""
    stabilizer_generator_in_Gamma_s(rho_B, D::Int, s::Vector;
                                     max_power::Int = 10_000, tol::Real = 1e-15,
                                     max_u::Integer = 10_000_000)

Given a real quadratic irrational `rho_B` (an element of `K = Q(sqrt(D))`,
read via `coeff(rho_B,0)`/`coeff(rho_B,1)` in the usual convention used
throughout this file) and `s = [s1,s2] ∈ Q^2`, finds a generator `A` of
`{B ∈ Γ_s : B*rho_B = rho_B}` up to the possible extra factor of `-I`
(`Γ_s = {B ∈ SL_2(Z) : B*s - s ∈ Z^2}`, the linear action of `B` on the
column vector `s`, NOT the Möbius action).

A thin wrapper composing `hj_automorph_for_root` (expensive, depends only
on `rho_B`) and `stabilizer_power_in_Gamma_s` (cheap, depends on `s`) - kept
for standalone use; identical output to before these were split out.
Callers processing many `s` for the same `rho_B` should call
`hj_automorph_for_root` once and reuse it via `stabilizer_power_in_Gamma_s`
directly instead (see `ray_class_shin_table`).
"""
function stabilizer_generator_in_Gamma_s(rho_B, D::Int, s::Vector;
                                          max_power::Int = 10_000,
                                          tol::Real = 1e-15,
                                          max_u::Integer = 10_000_000)
    L = hj_automorph_for_root(rho_B, D; max_u = max_u, tol = tol)
    return stabilizer_power_in_Gamma_s(L, s; max_power = max_power)
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
    D_fund, f_Q = coredisc(D_Q)
    D_fund == D ||
        error("find_class_representatives: coredisc(discriminant(F.Q)) = $D_fund does not ",
              "match the field's own fundamental discriminant D = $D - F.Q may not actually ",
              "live in K")
    # BUGFIX: sqrt_DQ must equal sqrt(D_Q) = sqrt(f_Q^2 * D_fund) = f_Q * sqrt(D_fund).
    # sqrt(D_fund) is exactly K's own generator `a` (a = gen(K) = sqrt(D), and D_fund == D
    # as just checked) - NOT sqrt_Dr, which equals sqrt(Dr) = a/isqrt(D÷Dr), a DIFFERENT
    # (smaller, by a factor of isqrt(D÷Dr)) quantity. Using sqrt_Dr here silently produced
    # a wrong (too-small) rho_F whenever isqrt(D÷Dr) != 1, i.e. whenever D is not itself
    # squarefree - exactly this case (D=44=4*11, isqrt(4)=2).
    sqrt_DQ = f_Q * a
    rho_F = (-F.Q.b + sqrt_DQ) / (2 * F.Q.a)
    println("F.Q = ", F.Q, "  (a,b,c) = (", F.Q.a, ",", F.Q.b, ",", F.Q.c, ")",
            "  disc(F.Q) = ", D_Q, "  coredisc = (", D_fund, ",", f_Q, ")",
            "  sqrt_DQ = ", sqrt_DQ, "  rho_F (fixed) = ", rho_F)

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
    # Do NOT assume the two pivot rows of H sit at fixed positions 1 and 2:
    # for n > 2 (a redundant/over-determined generator list), hnf's row
    # ordering of zero vs. nonzero rows is an implementation detail, not
    # something this code should rely on. Select the pivot rows by CONTENT
    # (the nonzero rows), not by position, so the result is the same
    # canonical form for the same lattice regardless of how many redundant
    # generators were passed in.
    nonzero_rows = [i for i in 1:n if !(iszero(H[i, 1]) && iszero(H[i, 2]))]
    isempty(nonzero_rows) && error("lattice_canonical: zero lattice")
    length(nonzero_rows) < 2 &&
        error("lattice_canonical: lattice not of full rank 2")
    length(nonzero_rows) > 2 &&
        error("lattice_canonical: more than 2 nonzero rows in HNF result ",
              "(unexpected - lattice should be rank 2)")
    r1, r2 = nonzero_rows
    c = gcd(gcd(H[r1, 1], H[r1, 2]), gcd(H[r2, 1], H[r2, 2]))
    c = gcd(c, den)
    H2 = zero_matrix(ZZ, 2, 2)
    for (k, r) in enumerate((r1, r2))
        H2[k, 1] = divexact(H[r, 1], c)
        H2[k, 2] = divexact(H[r, 2], c)
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

    t_b = time()
    for height in 1:max_height
        height >= 3 && println(stderr, "    [b-search] J = ", J, "  height ", height, "/", max_height,
                               "  (", round(time() - t_b, digits = 1), " s so far; cost ~ height^4 per level)")
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

                # Same J_m^*(O) requirement as in enumerate_principal_ray_classes:
                # coprimality to the moduli does not imply invertibility when a
                # conductor prime is not among them (d=15: q=4, moduli 15 and 16;
                # 16 = 2^4 does catch it here, but keep the check explicit - b_O
                # is later inverted, and a non-invertible b would fail silently).
                inv_ok = try
                    first(Hecke.is_invertible(I))
                catch
                    false
                end
                inv_ok || continue

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

# PROJECT OWNER'S THEOREM 2.19 GATE on a candidate rho_j.
#
# Condition (project owner's phrasing): with `a` the monoid class found by
# the lookup and `b` a representative of the Picard class of (J*a)^{-1}, the
# ideal b*(Z*rho_j + Z)^{-1} must be principal AND generated by a totally
# positive alpha; otherwise rho_j must be rejected.
#
# The condition splits into two parts with very different status, and this
# function reports them SEPARATELY because only one of them is actually a
# statement about rho_j:
#
#   PRINCIPALITY is a CLASS INVARIANT. For any other representative
#   b' = gamma*b of the same Picard class, b'*L^{-1} = gamma*(b*L^{-1}), which
#   is principal exactly when b*L^{-1} is. So if this fails, NO choice of b
#   can repair it - it is a genuine, b-independent rejection of rho_j.
#
#   TOTAL POSITIVITY is NOT a class invariant. It is a normalization fixed by
#   the CHOICE of b within the class: replacing b by gamma*b replaces alpha by
#   gamma*alpha. Since O's fundamental unit has norm +1 (so every unit is
#   totally positive and unit-twisting can never repair a mixed-sign
#   generator - see is_principal_totally_positive_verified), the repair must
#   come from a non-unit gamma of NEGATIVE norm, which flips the sign at
#   exactly one real place. Theorem 2.19 additionally requires b coprime to
#   m = dO, and that constraint is what gives this part teeth: gamma must be
#   coprime to dO too.
#
# Returns (status, alpha, gamma) with status one of:
#   :ok             - gate satisfied; alpha is the totally positive generator
#                     for the representative b = gamma*b0.
#   :not_principal  - genuine, b-independent rejection of rho_j.
#   :no_gamma       - principal, but generator has mixed signs and no gamma
#                     coprime to dO was found within gamma_height to fix it.
#
# L^{-1} = conj(L)/N(L) with N(L) a positive rational, so b*conj(L) is tested
# instead of b*L^{-1}: the two differ by a positive rational factor, hence are
# principal together with generators of identical sign at both real places.
# This avoids inv() on a non-maximal-order ideal (unverified in this codebase).
function theorem219_gate(K, O, mP_O, inv_class, rho_j, d; gamma_height::Int = 8,
                          unit_bound::Int = 60)
    b0 = mP_O(inv_class)
    bb1 = _nf_elem_of(K, basis(b0)[1])
    bb2 = _nf_elem_of(K, basis(b0)[2])
    crho = quad_conj(K, rho_j)
    # Z-span of b0*conj(L): b0 = Z*bb1 + Z*bb2 and conj(L) = Z*1 + Z*conj(rho_j),
    # and the product of two Z-modules is the Z-span of the pairwise products.
    gens = [bb1, bb2, bb1 * crho, bb2 * crho]

    prin, g = is_principal_O_verified(K, O, gens; unit_bound = unit_bound)
    prin || return (:not_principal, nothing, nothing)

    places_K = real_places(K)
    sgn = [sign(g, pl) for pl in places_K]
    if all(s -> s > 0, sgn)
        return (:ok, g, one(K))
    elseif all(s -> s < 0, sgn)
        return (:ok, -g, -one(K))
    end

    # Mixed signs: b is only a REPRESENTATIVE, so look for gamma (coprime to
    # dO, per Theorem 2.19's own requirement on b) with gamma*g totally
    # positive, and use b = gamma*b0 instead.
    bO1 = _nf_elem_of(K, basis(O)[1])
    bO2 = _nf_elem_of(K, basis(O)[2])
    dO = ideal(O, d)
    for h in 1:gamma_height
        for c1 in -h:h, c2 in -h:h
            (abs(c1) != h && abs(c2) != h) && continue
            gam = c1 * bO1 + c2 * bO2
            iszero(gam) && continue
            cop = try
                isone(ideal(O, [O(gam)]) + dO)
            catch
                false
            end
            cop || continue
            cand = gam * g
            s2 = [sign(cand, pl) for pl in places_K]
            if all(s -> s > 0, s2)
                return (:ok, cand, gam)
            elseif all(s -> s < 0, s2)
                return (:ok, -cand, -gam)
            end
        end
    end
    return (:no_gamma, nothing, nothing)
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
    t_mn = time()
    for h in 0:mn_height
        (h in (10, 25, 50, 100, 200, 400, 800, 1600, 3000)) &&
            println(stderr, "    [mn-search] height ", h, "/", mn_height,
                    "  (", round(time() - t_mn, digits = 1), " s so far; no (alpha, rho_B) accepted yet)")
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
                                  d1::Int = d_val + 1,             # the (d+1) coprimality ideal; pass d1 = d+1 explicitly when d_val is the modulus d-bar = 2d (even d)
                                  verbose::Bool = false,           # print internal [two-stage] search log (now called many times per outer retry loop, so off by default)
                                  alpha_hint = nothing,            # if alpha_hint, rho_B_hint, b_O_hint are ALL supplied,
                                  rho_B_hint = nothing,            # skip stage 1 (the expensive (m,n) search) entirely and
                                  b_O_hint = nothing)              # go straight to stage 2 for this J - see block below.
    # ADDITIVE FAST PATH (does not alter behavior when hints are omitted):
    # alpha, rho_B, b_O depend only on the Picard class of J (u_B), NOT on
    # J itself - the (m,n) search below that finds them uses only b_O's own
    # ideal bdO_O = b_O*d_val*O, and b_O itself is selected purely from
    # J's CLASS (find_b_with_verified_principal_product's search succeeds
    # or fails identically for any two ideals differing by a principal
    # factor). Only stage 2 (solving for s1,s2 from Jb_O = J*b_O) genuinely
    # needs the literal J. The docstring-comment below states this
    # explicitly: once alpha,rho_B satisfy b*d*O=alpha*(Z+rho_B*Z), a valid
    # (s1,s2) is GUARANTEED to exist - so reusing an (alpha,rho_B,b_O)
    # already found for another J of the SAME Picard class, and re-running
    # only stage 2 here, produces exactly the same result a full search
    # would have produced for this J. This lets callers building a
    # per-ray-class table (e.g. ray_class_shin_table) do the expensive
    # search ONCE per Picard class instead of once per ray class.
    if alpha_hint !== nothing && rho_B_hint !== nothing && b_O_hint !== nothing
        alpha = alpha_hint
        rho_B = rho_B_hint
        b_O = b_O_hint

        dO_O = ideal(O, d_val)
        places   = real_places(K)
        emb_conj = embedding(places[1])
        emb_main = embedding(places[2])

        all(sign(alpha, p) > 0 for p in places) ||
            error("alpha_hint is not totally positive")
        order(b_O) == O || error("b_O_hint is not tagged with order O as expected")
        is_inv_b, _ = Hecke.is_invertible(b_O)
        is_inv_b || error("b_O_hint is not invertible")
        is_coprime_bO_dO = try
            isone(b_O + dO_O)
        catch
            false
        end
        is_coprime_bO_dO || error("b_O_hint is not coprime to dO_O (d_val=", d_val, ")")

        alpha_ideal = fractional_ideal(O, [alpha])
        rhoB_lattice_O = fractional_ideal(O, [K(1), rho_B])
        order(rhoB_lattice_O) == O ||
            error("rho_B_hint does not give order O for Z+Z*rho_B as expected")
        bdO_O = b_O * dO_O
        alpha_ideal * rhoB_lattice_O == bdO_O ||
            error("alpha_hint, rho_B_hint do not satisfy b_O_hint*d_val*O = ",
                  "alpha*(Z+Z*rho_B) - hints are inconsistent with b_O_hint/d_val")
        Float64(emb_main(rho_B)) > Float64(emb_conj(rho_B)) ||
            error("rho_B_hint does not satisfy the larger-root convention (main > conj)")

        function to_nf_elem_fp(x)
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
        function to_float_fp(q)
            return Float64(BigInt(numerator(q))) / Float64(BigInt(denominator(q)))
        end

        t_Q = @elapsed begin
            Jb_O = J * b_O
            Q = Jb_O * inv(alpha_ideal)
            Q_norm = QQ(norm(Q))
        end
        verbose && println("  [fast-path] norm(Q) (target norm for s2*rho_B - s1) = ", Q_norm,
                "   (Jb_O/Q construction: ", round(t_Q, sigdigits=3), " s)")

        Qbasis = basis(Q)
        qb1 = to_nf_elem_fp(Qbasis[1])
        qb2 = to_nf_elem_fp(Qbasis[2])

        # Closed-form quadratic form for norm(i*qb1+j*qb2), via the exact
        # polarization identity N(i*x+j*y) = i^2*N(x) + i*j*(N(x+y)-N(x)-N(y)) + j^2*N(y)
        # (holds for ANY quadratic form, in particular the field norm) -
        # lets the (i,j) loop below reject the overwhelming majority of
        # candidates (those with the wrong norm) via cheap Int/QQ
        # arithmetic, instead of constructing an nf_elem and calling
        # Hecke's generic norm() for every one of up to (2*q_gen_height+1)^2
        # candidates. Mathematically identical to the original per-candidate
        # `QQ(abs(norm(K(cand)))) == Q_norm` check - same accept/reject
        # decision for every (i,j), just reached more cheaply.
        Nqb1 = QQ(norm(qb1))
        Nqb2 = QQ(norm(qb2))
        Ncross = QQ(norm(qb1 + qb2)) - Nqb1 - Nqb2

        # PERFORMANCE FIX (confirmed, not a guess: diagnose_monoid_lookup_v2_timing's
        # internal instrumentation showed this exact loop accounts for
        # essentially 100% of the 8-40s-per-candidate cost - 13-18 million
        # iterations taking ~10s, while Jb_O/Q construction and the final
        # is_principal_O_verified check are both sub-millisecond). Nqb1,
        # Ncross, Nqb2, Q_norm are QQFieldElem (arbitrary-precision rational),
        # so EVERY one of up to (2*q_gen_height+1)^2 ~ 36 million iterations
        # was doing multi-precision GMP arithmetic for what is, for any fixed
        # number field, a comparison against small bounded-denominator
        # rationals. Rescaling all four by their common denominator makes
        # them INTEGERS, so the loop's hot inner comparison can run in native
        # Int128 arithmetic instead - same accept/reject decision for every
        # (i,j), just computed far more cheaply. den is a property of K/O and
        # Q alone (not of i,j), so this is a one-time O(1) cost outside the
        # loop, not a new per-iteration cost.
        den_ij = lcm(lcm(denominator(Nqb1), denominator(Ncross)),
                     lcm(denominator(Nqb2), denominator(Q_norm)))
        Nqb1_i = Int128(numerator(Nqb1 * den_ij))
        Ncross_i = Int128(numerator(Ncross * den_ij))
        Nqb2_i = Int128(numerator(Nqb2 * den_ij))
        Qnorm_i = Int128(numerator(Q_norm * den_ij))

        found_s = false
        s1 = QQ(0)
        s2 = QQ(0)
        n_ij_tried = 0
        n_ij_norm_passed = 0
        t_ijsearch = @elapsed for i in -q_gen_height:q_gen_height, j in -q_gen_height:q_gen_height
            (i == 0 && j == 0) && continue
            n_ij_tried += 1
            nval_i = Int128(i)^2*Nqb1_i + Int128(i)*Int128(j)*Ncross_i + Int128(j)^2*Nqb2_i
            abs(nval_i) == Qnorm_i || continue
            n_ij_norm_passed += 1

            cand = K(i) * qb1 + K(j) * qb2
            iszero(cand) && continue
            cand_ideal = try
                fractional_ideal(O, [cand])
            catch
                continue
            end
            cand_ideal == Q || continue

            elem2 = Float64(emb_conj(cand)) > 0 ? cand : -cand

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

            in_bO = try
                elem_check in b_O
            catch
                false
            end
            in_bO || continue

            rhoB_conj_check = Float64(emb_conj(rho_B))
            s1_f_check = to_float_fp(s1_try)
            s2_f_check = to_float_fp(s2_try)
            (s2_f_check * rhoB_conj_check - s1_f_check) > 0 || continue

            s1, s2 = s1_try, s2_try
            found_s = true
            break
        end
        verbose && println("  [fast-path] (i,j) search: ", n_ij_tried, " tried, ",
                n_ij_norm_passed, " passed the norm filter, found_s=", found_s,
                "   (", round(t_ijsearch, sigdigits=3), " s)")
        found_s || error("Could not find s1, s2 within q_gen_height=", q_gen_height,
                          " using the hinted alpha, rho_B, b_O for this J = ", J)

        rhoB_conj = Float64(emb_conj(rho_B))
        s1_f = to_float_fp(s1)
        s2_f = to_float_fp(s2)
        is_valid_final = (s2_f * rhoB_conj - s1_f) > 0
        is_valid_final || error("Internal inconsistency in fast path: sign-flip ",
                                 "construction should guarantee is_valid, but it does ",
                                 "not (s1=", s1, ", s2=", s2, ")")

        elem2_exact = K(s2) * rho_B - K(s1)
        L_gens = [one(K), rho_B]
        Lconj_gens = [quad_conj(K, g) for g in L_gens]
        Jconj_gens = [quad_conj(K, _nf_elem_of(K, x)) for x in basis(J)]
        crit_gens = [elem2_exact * a for a in lattice_product_gens(Lconj_gens, Jconj_gens)]
        t_verify = @elapsed (ok_crit, lambda_gen) = is_principal_O_verified(K, O, crit_gens)
        verbose && println("  [fast-path] final is_principal_O_verified: ok=", ok_crit,
                "   (", round(t_verify, sigdigits=3), " s)")
        ok_crit || error("FINAL VERIFICATION FAILED (fast path): (s2*rho_B-s1)*",
                          "(Z+Z*rho_B)^{-1} is NOT in the Picard class of J. ",
                          "rho_B=", rho_B, " s1=", s1, " s2=", s2)

        verbose && println("  [fast-path] FOUND: rho_B=", rho_B, " alpha=", alpha,
                " s1=", s1, " s2=", s2, " (stage 1 skipped via hints)")
        return alpha, rho_B, s1, s2
    end

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
    d1_ideal = ideal(O, d1)   # NOT d_val + 1: for even d the modulus d_val = 2d but the coprimality ideal is still (d+1)
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
    # rejection tally per Stage-1 filter, so a fruitless search reports WHICH
    # condition kills every candidate rather than just "nothing found"
    rej = Dict(:mixed_sign => 0, :root_order => 0, :root_order_flipped => 0,
               :excluded => 0, :rational => 0,
               :lattice_err => 0, :order_ne_O => 0, :alpha_ideal_err => 0,
               :ideal_eq => 0, :accepted => 0)
    t_mn = time()
    for h in 0:mn_height
        (h in (10, 25, 50, 100, 200, 400, 800, 1600, 3000)) &&
            println(stderr, "    [mn-search] height ", h, "/", mn_height,
                    "  (", round(time() - t_mn, digits = 1), " s so far; no (alpha, rho_B) accepted yet)",
                    "  Stage-1 rejections: ", sort(collect(rej), by = x -> -x[2]))
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
                rej[:mixed_sign] += 1
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
            # BUGFIX (9/15, d=15 hang): the previous version rejected the
            # (m,n) pair outright when rho_B failed the "larger root"
            # convention (conj(rho_B) > rho_B). That is wrong, and for d=15
            # (q=4, conductor at the prime 2, d+1 = 16) it is fatal: the
            # Stage-1 rejection tally showed :root_order killing ~92% of all
            # candidates and :mixed_sign the other ~8%, with EVERY other
            # filter at exactly 0, at a stable ~11.7:1 ratio across all
            # heights up to 400 - i.e. nothing ever reached Stage 2, at any
            # height, and the mn_height=3000 search was guaranteed to run for
            # ~30 min and then throw.
            #
            # The complementary basis vector is only determined up to the
            # general Bezout solution (p_ + t*m, q_ + t*n), and gcdx returns
            # one arbitrary representative; in particular -beta is an equally
            # valid choice (m*(-q_) - n*(-p_) = -1 is still unimodular), and
            # it sends rho_B -> -rho_B, which SWAPS which root is larger.
            # So a failure here is a fixable orientation, not a bad (m,n).
            # {alpha, -beta} spans the same lattice, so alpha*(Z + Z*(-rho_B))
            # is the same ideal and the downstream bdO_O equality check is
            # unaffected (:ideal_eq stayed 0 throughout, as expected).
            if !(main_val > conj_val)
                beta = -beta
                rho_B = -rho_B
                main_val, conj_val = -main_val, -conj_val
                rej[:root_order_flipped] += 1
            end
            # after the flip this can only still fail if the two embeddings are
            # equal, i.e. rho_B is rational - caught by the iszero(c1) test below
            if !(main_val > conj_val); rej[:root_order] += 1; continue; end

            if rho_B in exclude; rej[:excluded] += 1; continue; end

            c1 = coeff(rho_B, 1)
            if iszero(c1); rej[:rational] += 1; continue; end

            rhoB_lattice_O = try
                fractional_ideal(O, [K(1), rho_B])
            catch
                rej[:lattice_err] += 1
                continue
            end
            if order(rhoB_lattice_O) != O; rej[:order_ne_O] += 1; continue; end

            # explicit verification of the defining equation, rather than
            # trusting the algebraic derivation blindly
            alpha_ideal = try
                fractional_ideal(O, [alpha])
            catch
                rej[:alpha_ideal_err] += 1
                continue
            end
            if alpha_ideal * rhoB_lattice_O != bdO_O; rej[:ideal_eq] += 1; continue; end
            rej[:accepted] += 1

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

        # Closed-form quadratic form for norm(i*qb1+j*qb2) - see the matching
        # comment in the fast-path branch above for the exact identity used;
        # mathematically identical accept/reject decision to the original
        # per-candidate `QQ(abs(norm(K(cand)))) == Q_norm` check, just far
        # cheaper to evaluate for the (2*q_gen_height+1)^2 candidates tried.
        Nqb1 = QQ(norm(qb1))
        Nqb2 = QQ(norm(qb2))
        Ncross = QQ(norm(qb1 + qb2)) - Nqb1 - Nqb2

        # PERFORMANCE FIX (confirmed via diagnose_monoid_lookup_v2_timing's
        # instrumentation of the fast-path branch - see the matching comment
        # there for the full explanation): rescale to integers so the hot
        # (i,j) loop runs in native Int128 arithmetic instead of
        # arbitrary-precision QQFieldElem. Same accept/reject decision, just
        # far cheaper per iteration across up to (2*q_gen_height+1)^2 tries.
        den_ij = lcm(lcm(denominator(Nqb1), denominator(Ncross)),
                     lcm(denominator(Nqb2), denominator(Q_norm)))
        Nqb1_i = Int128(numerator(Nqb1 * den_ij))
        Ncross_i = Int128(numerator(Ncross * den_ij))
        Nqb2_i = Int128(numerator(Nqb2 * den_ij))
        Qnorm_i = Int128(numerator(Q_norm * den_ij))

        found_s = false
        s1 = QQ(0)
        s2 = QQ(0)
        for i in -q_gen_height:q_gen_height, j in -q_gen_height:q_gen_height
            (i == 0 && j == 0) && continue
            nval_i = Int128(i)^2*Nqb1_i + Int128(i)*Int128(j)*Ncross_i + Int128(j)^2*Nqb2_i
            abs(nval_i) == Qnorm_i || continue

            cand = K(i) * qb1 + K(j) * qb2
            iszero(cand) && continue
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

    println(stderr, "    [mn-search] EXHAUSTED: no Stage-1 candidate ever accepted.",
            "  Final rejection tally: ", sort(collect(rej), by = x -> -x[2]))
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
                              j_cache::Dict;
                              ray_class_lookup_table::Union{Nothing,Vector} = nothing,
                              ray_class_lookup_tol::Real = 1e-10,
                              elem1_max_power::Int = 10_000,
                              elem1_max_u::Integer = 10_000_000,
                              monoid_table::Vector = NamedTuple[],
                              candidate_cache::Dict = Dict{Tuple{Int,Int,Int}, Any}(),
                              class_hints::Dict = Dict{Any, NamedTuple}(),
                              full_monoid_table::Union{Nothing,Vector} = nothing,
                              phase_debug_points::Vector{Tuple{Int,Int}} = Tuple{Int,Int}[])
    d = Int(F.d)
    # EVEN d (9/18): ray-class modulus d-bar (2d for even d). All Theorem 2.19
    # point representations, lookups, class products and twisted points below
    # live mod d-bar; the ghost cocycle shin^{p/d} and the phase scale d(d-2)
    # keep d. See the principal branch (which validated this recipe with the
    # twisted point M_g*p mod 2d) and the block comment before the grid loop.
    dm = isodd(d) ? d : 2 * d
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

    # elem1 (used per grid point below) is now recovered via a REVERSE
    # LOOKUP into the ray class shin table, per project owner: for a given
    # (p1,p2), compute shin^{[p1,p2]}_A(rho) for the matrix A that's a power
    # of rho's own fundamental automorph lying in Gamma_{[p1,p2]/d} (built
    # exactly the way ray_class_shin_table_fast builds its own matrices),
    # find the ray class whose table entry has this SAME shin value, and
    # set elem1 = d*(s2*rho_B - s1) from THAT entry's own s1, s2, rho_B -
    # not from rho/p1/p2 directly (the old, incorrect method). Both pieces
    # this needs only depend on rho/F (not on gamma, J, or p1/p2), so they
    # are computed once here rather than per grid point.
    L_rho_base = hj_automorph_for_root(rho, D; max_u = elem1_max_u)
    rho_bf_base = BigFloat(coeff(rho, 0)) + BigFloat(coeff(rho, 1)) * sqrt(BigFloat(D))
    ray_class_lookup_table_local = ray_class_lookup_table === nothing ?
        ray_class_shin_table_fast_quiet(F) : ray_class_lookup_table

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

    # NOTE (currently unused function): elem1 below is the OLD, incorrect
    # construction (elem1 = p2*rho - p1). The live grid loop further down
    # now computes elem1 via a reverse lookup into the ray class shin table
    # instead (see the comment above Ng/NJ) - this helper was NOT updated
    # to match, since it isn't actually called anywhere; don't assume it
    # reflects the current method if it's ever revived.
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

        # EVEN d (9/18): the normalized ghost overlaps nu_p are functions on
        # Z^2 with period d-bar = 2d (paper Def. 1.11 / Lemma 1.42), NOT d: the
        # raw cocycle shin^{p/d}_A(rho) is d-periodic (Lemma 2.14) but the SF
        # phase xi_d^{-Q(p)} (-1)^{(1+p1)(1+p2)} is only 2d-periodic. The
        # identity nu_p * nu_{-p} = 1 (Thm 5.8) holds with -p the honest
        # negative in Z^2, i.e. mod 2d. So for even d we build the twisted
        # overlaps V on the full 2d x 2d grid - the twisted point M_g*p and the
        # index (p1, Ng*p2) both taken mod 2d, with M_g lifted to Z/2d - check
        # V(q) V(-q mod 2d) = 1/(d+1) for ALL q (no sign anywhere), and read the
        # coefficient of D_p, p in [0,d)^2, straight off V. Reading off V is
        # exactly the paper's Pi~ = sum_{p in [0,d)^2} mu~_{Gp} D_p and handles
        # the Lemma 1.42 reduction automatically. For odd d, d-bar = d and this
        # is byte-identical to the old code.
        dbar = isodd(d) ? d : 2 * d
        M_g_bar = isodd(d) ? M_g : multiplication_matrix_mod_d(K(gamma), rho_g, dbar, D)
        isodd(d) || println("  M_g mod d-bar = 2d: ", M_g_bar)
        isodd(d) || isodd(Ng) || error("even d: N(gamma) = ", Ng, " must be coprime to d-bar = 2d")

        V = zeros(Complex{BigFloat}, dbar, dbar)
        for p1 = 0:dbar-1
            for p2 = 0:dbar-1
                (mod(p1, d) == 0 && mod(p2, d) == 0) && continue
                p = [p1, p2]
                gp = mod.([p1, Ng * p2], dbar)          # index, mod d-bar
                newp_bar = mod.(M_g_bar * p, dbar)       # phase argument (2d-periodic)
                newp = mod.(newp_bar, d)                 # cocycle argument (d-periodic)
                QA = BigFloat(-QQ_form(newp_bar...)) / BigFloat(d * (d - 2))
                s = isodd(d) ? 1 : (1 + newp_bar[1]) * (1 + newp_bar[2])
                V[gp[1]+1, gp[2]+1] = chi * ζ^QA * (-1)^s * c_g * shin_of_tuple(gF, newp)
            end
        end
        for p1 = 0:d-1, p2 = 0:d-1
            (p1 == 0 && p2 == 0) && continue
            new_ovs[p1+1, p2+1] = V[p1+1, p2+1]
        end

        println("  v(p)*v(-p) for all p != 0 mod d",
                isodd(d) ? ":" : " (even d: on the 2d x 2d grid, -p mod 2d, expected 1/(d+1) everywhere):")
        n_dev = 0
        for p1 = 0:dbar-1
            for p2 = 0:dbar-1
                (mod(p1, d) == 0 && mod(p2, d) == 0) && continue
                mp1 = mod(-p1, dbar)
                mp2 = mod(-p2, dbar)
                product = V[p1+1, p2+1] * V[mp1+1, mp2+1]
                ok = abs(product - 1 / BigFloat(d + 1)) < 1e-8
                ok || (n_dev += 1)
                println("    p=($p1,$p2): v(p)*v(-p) = ", Float64(real(product)),
                        " + ", Float64(imag(product)), "im",
                        "  |v(p)*v(-p)| = ", Float64(abs(product)),
                        "  1/(d+1) = ", 1.0/(d+1), ok ? "" : "   <-- DEVIATION")
            end
        end
        println("  deviations of v(p)*v(-p) from 1/(d+1): ", n_dev, " of ", dbar^2 - (dbar ÷ d)^2)

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
        d_check = dm
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
                                                             O, K, dm, rho, d_check; d1 = d + 1,
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

        # EVEN d (9/18): as in the principal branch, the twisted overlaps are
        # built on the full d-bar x d-bar grid V_np (index (p1, NJ*Ng*p2) mod
        # d-bar), checked via v(p) v(-p mod d-bar) = 1/(d+1) for all p != 0 mod
        # d, and the coefficient of D_p, p in [0,d)^2, is read straight off V_np.
        # For odd d, d-bar = d and this is the old loop.
        V_np = zeros(Complex{BigFloat}, dm, dm)
        isodd(d) || isodd(NJ * Ng) || error("even d: NJ*Ng = ", NJ * Ng, " must be coprime to d-bar = 2d")

        for p1 = 0:dm-1
            for p2 = 0:dm-1
                (mod(p1, d) == 0 && mod(p2, d) == 0) && continue

                gp = mod.([p1, NJ * Ng * p2], dm)
                p = [p1, p2]

                # elem1 via reverse lookup - see the block comment above
                # (right after Ng/NJ) for the full reasoning.
                A_lookup, _ = stabilizer_power_in_Gamma_s(
                    L_rho_base, [QQ(p1, dm), QQ(p2, dm)]; max_power = elem1_max_power)
                shin_val_lookup = shin(A_lookup, rho_bf_base, [p1, p2], dm)

                match_idx = findfirst(
                    e -> e !== nothing && abs(e.shin_value - shin_val_lookup) < ray_class_lookup_tol,
                    ray_class_lookup_table_local)
                local lookup_entry
                if match_idx === nothing
                    # REVISED PER PROJECT OWNER: the ray class monoid
                    # Jbar_m^flat(O) is FINITE and small (592 classes for
                    # d=21) - build it in FULL, once, up front
                    # (full_monoid_table, via ray_class_monoid_table_v2,
                    # threaded through exactly like monoid_table/
                    # candidate_cache/class_hints), and do a direct value
                    # lookup against it - no incremental norm-bound search at
                    # all. Confirmed empirically (diagnose_full_monoid_
                    # membership) that a genuine shin_val_lookup value DOES
                    # appear in this table; the earlier incremental search
                    # (ray_class_monoid_lookup) was solving a problem that
                    # did not need solving that way.
                    println("  [monoid lookup] shin^{[$p1,$p2]}_A(rho) = ", shin_val_lookup,
                            " not in the group table - looking up in the full monoid table...")
                    full_monoid_table === nothing &&
                        error("twisted_chi_general: a lookup miss against the group table ",
                              "occurred but full_monoid_table was not provided - build it once ",
                              "(ray_class_monoid_table_v2) and pass it through, the same way ",
                              "monoid_table/candidate_cache/class_hints are already threaded")
                    full_match = nothing
                    for e in full_monoid_table
                        if abs(Complex{BigFloat}(e.shin_value) - shin_val_lookup) < ray_class_lookup_tol
                            full_match = e
                            break
                        end
                    end
                    full_match === nothing &&
                        error("p=($p1,$p2): shin^{[$p1,$p2]}_A(rho) = ", shin_val_lookup,
                              " was not found ANYWHERE in the full, exhaustively-built monoid ",
                              "table (", length(full_monoid_table), " classes). Per the project ",
                              "owner: since the monoid is finite and has been built in full, this ",
                              "is definitive - it means shin_val_lookup itself is being computed ",
                              "incorrectly, NOT that a larger search would help. Do not attempt to ",
                              "work around this by searching further.")
                    # full_match gives us WHICH ideal - re-derive (rho_B,s1,s2)
                    # for it via find_alpha_rhoB_s1_s2_O, exactly the same way
                    # ray_class_monoid_lookup already does for its own
                    # candidates (via find_with_class_cache), so lookup_entry
                    # has identical semantics regardless of which path found
                    # it. Uses the SAME class_hints cache, so this pays the
                    # full per-class search cost only once per Picard class,
                    # not once per lookup.
                    u_match = full_match.A0
                    local rho_match, s1_match, s2_match
                    if haskey(class_hints, u_match)
                        hint = class_hints[u_match]
                        _, rho_match, s1_match, s2_match = find_alpha_rhoB_s1_s2_O(
                            full_match.J, rho_reps[u_match], rho, u_match, P_O, mP_O, O, K,
                            dm, rho, d_check; d1 = d + 1,
                            alpha_hint = hint.alpha, rho_B_hint = hint.rho_B, b_O_hint = hint.b_O)
                    else
                        res_match = find_alpha_rhoB_s1_s2_O(
                            full_match.J, rho_reps[u_match], rho, u_match, P_O, mP_O, O, K,
                            dm, rho, d_check; d1 = d + 1)
                        b_O_match, _ = find_b_with_verified_principal_product(
                            O, K, full_match.J, [ideal(O, dm), ideal(O, d + 1)])
                        class_hints[u_match] = (alpha = res_match[1], rho_B = res_match[2],
                                                 b_O = b_O_match)
                        _, rho_match, s1_match, s2_match = res_match
                    end
                    lookup_entry = (J = full_match.J, rho_B = rho_match, s1 = s1_match,
                                    s2 = s2_match, shin_value = full_match.shin_value)
                else
                    lookup_entry = ray_class_lookup_table_local[match_idx]
                end
                elem1 = K(dm) * (K(lookup_entry.s2) * lookup_entry.rho_B - K(lookup_entry.s1))
                println("    p=($p1,$p2): lookup_entry.J (U_elem1, the actual ideal) = ", lookup_entry.J,
                        "   N = ", norm(lookup_entry.J))

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

                local rho_j_p, A_bj_p, gF_j_p, gQ_j_p, QQ_form_p, M_g_j_p, c_g_j_p,
                      rho_j_bf_p, u1_p, u2_p, integralnewdu

                for rho_B_attempt in 1:max_retries_rho_B_p
                    rho_B_cur, s1_cur, s2_cur = if rho_B_attempt == 1
                        rho_B, s1, s2
                    else
                        println("    p=($p1,$p2) [rho_B_attempt $rho_B_attempt]: searching for a ",
                                "different rho_B/s1/s2 (previous one exhausted its rho_j search)")
                        _, rho_B_new, s1_new, s2_new = find_alpha_rhoB_s1_s2_O(
                            J, rho_B_init, rho, u_B, P_O, mP_O, O, K, dm, rho, d_check; d1 = d + 1,
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

                    # NEW rho_j construction (per project owner), replacing
                    # the old Stage 1/2/2.5/4.7/4.8/4.9 search+verify chain
                    # entirely: let U_elem1 be the ray class (full ideal)
                    # used to build THIS point's elem1 (from the reverse
                    # lookup above), b_j := [J*U_elem1] in Pic(O), and find
                    # alpha_bj, rho_j, s1_bj, s2_bj via find_alpha_rhoB_s1_s2_O
                    # so that b_j^{-1}*d*O = alpha_bj*(Z*rho_j+Z) (passing
                    # J*U_elem1 itself as the "J" argument gives exactly
                    # this, since find_alpha_rhoB_s1_s2_O's own internal
                    # b_O is always [input]^{-1}). rho_j is now UNIQUELY
                    # determined here (no candidate search/exclusion at
                    # this level) - the only remaining retry axis is the
                    # outer rho_B_attempt loop above.
                    # rho_j construction (per project owner): lookup_entry
                    # is now ALWAYS populated - either from the static
                    # 288-entry group table, or (on a miss) from the
                    # dynamically-grown Jbar_m^flat(O) monoid search above -
                    # so this single construction is correct for every
                    # point, unit or non-unit. Let U_elem1 be the ray class
                    # ideal used to build THIS point's elem1, b_j :=
                    # [J*U_elem1] in Pic(O), and find alpha_bj, rho_j,
                    # s1_bj, s2_bj via find_alpha_rhoB_s1_s2_O so that
                    # b_j^{-1}*d*O = alpha_bj*(Z*rho_j+Z) (passing J*U_elem1
                    # itself as the "J" argument gives exactly this, since
                    # find_alpha_rhoB_s1_s2_O's own internal b_O is always
                    # [input]^{-1}). rho_j is UNIQUELY determined here (no
                    # candidate search/exclusion at this level) - the only
                    # remaining retry axis is the outer rho_B_attempt loop.
                    local alpha_bj, rho_j_raw, s1_bj, s2_bj
                    U_elem1 = lookup_entry.J
                    J_bj = J * U_elem1
                    u_Bj = mP_O \ J_bj
                    rho_Bj_init = rho_reps[u_Bj]
                    alpha_bj, rho_j_raw, s1_bj, s2_bj = find_alpha_rhoB_s1_s2_O(
                        J_bj, rho_Bj_init, rho_Bj_init, u_Bj, P_O, mP_O, O, K,
                        dm, rho_Bj_init, d_check; d1 = d + 1, mn_height = 3000, q_gen_height = 3000)
                    println("    p=($p1,$p2) [rho_B_attempt $rho_B_attempt]: rho_j = ", rho_j_raw,
                            "  alpha_bj = ", alpha_bj, "  s1_bj = ", s1_bj, "  s2_bj = ", s2_bj)

                    # PROJECT OWNER'S THEOREM 2.19 GATE ON rho_j.
                    #
                    # The three rejection conditions this retry loop already
                    # had (det(M_g_j) coprime to d, diag_val > 0, and
                    # integralnewdu coprimality) never check Theorem 2.19's
                    # actual hypothesis for THIS point's (J, a) pair. So
                    # find_alpha_rhoB_s1_s2_O can hand back a rho_j that is a
                    # perfectly valid representative for its Picard class
                    # while still failing the condition the theorem requires -
                    # and nothing downstream notices. Per the project owner:
                    # with `a` the monoid class found by the lookup
                    # (U_elem1 = lookup_entry.J) and `b` a representative of
                    # the Picard class of (J*a)^{-1}, it must be the case that
                    #     b * (Z*rho_j + Z)^{-1}
                    # is principal AND generated by a TOTALLY POSITIVE alpha;
                    # if not, this rho_j must be rejected and a different one
                    # searched for, exactly like the other three conditions.
                    #
                    # Two reformulations are used, both exact:
                    #
                    # (1) Theorem 2.19 states b*m = alpha*(rho*Z+Z) with
                    #     m = dO. Since m = d*O with d a POSITIVE rational
                    #     integer, including it only rescales alpha by d and
                    #     can change neither principality nor the sign of
                    #     alpha at either real place - so testing
                    #     b*(Z*rho_j+Z)^{-1} (the project owner's phrasing) is
                    #     equivalent, and avoids building m at all.
                    #
                    # (2) L^{-1} = conj(L)/N(L) for an INVERTIBLE fractional
                    #     ideal L of a quadratic field, and N(L) is a positive
                    #     rational - so b*L^{-1} and b*conj(L) differ by a
                    #     positive rational factor, hence are principal
                    #     together and their generators have identical signs
                    #     at both real places. Testing b*conj(L) avoids
                    #     inv() on a non-maximal-order ideal entirely (flagged
                    #     as unverified for this codebase earlier).
                    #
                    # That identity needs L invertible (equivalently: the
                    # Z-lattice Z*rho_j + Z really is O-stable, i.e. a PROPER
                    # O-ideal). That is exactly the condition that rho_j is a
                    # legitimate root for THIS order, so it is checked first
                    # and its failure is itself a rejection - not an error.
                    # It is tested exactly, via lattice_canonical, rather than
                    # by trusting fractional_ideal(O, gens), which returns the
                    # O-module CLOSURE of the generators rather than their
                    # literal Z-span (a confirmed hazard in this codebase and
                    # precisely the thing that would silently paper over a
                    # non-proper L here).
                    b_cls = mP_O(-u_Bj)
                    bO1 = _nf_elem_of(K, basis(O)[1])
                    bO2 = _nf_elem_of(K, basis(O)[2])
                    L_lat = [one(K), rho_j_raw]
                    L_proper = lattice_canonical(K, L_lat) ==
                               lattice_canonical(K, [L_lat[1], L_lat[2],
                                                     bO1 * L_lat[1], bO1 * L_lat[2],
                                                     bO2 * L_lat[1], bO2 * L_lat[2]])
                    if !L_proper
                        println("    p=($p1,$p2) [rho_B_attempt $rho_B_attempt]: REJECTED ",
                                "(Thm 2.19: Z*rho_j+Z is not a proper/invertible O-ideal, so ",
                                "rho_j is not a legitimate root for this order) - trying a ",
                                "different rho_B")
                        push!(excluded_rho_B_p, rho_B_cur)
                        continue
                    end

                    b_b1 = _nf_elem_of(K, basis(b_cls)[1])
                    b_b2 = _nf_elem_of(K, basis(b_cls)[2])
                    crho_j = quad_conj(K, rho_j_raw)
                    t219_status, t219_alpha, t219_gamma =
                        theorem219_gate(K, O, mP_O, -u_Bj, rho_j_raw, dm)
                    if t219_status === :not_principal
                        error("p=($p1,$p2): THEOREM 2.19 GATE FAILED for rho_j = ", rho_j_raw,
                              " - b*(Z*rho_j+Z)^{-1} is NOT PRINCIPAL. Principality here is a ",
                              "CLASS INVARIANT (for b' = gamma*b, b'*L^{-1} = gamma*(b*L^{-1}), ",
                              "principal iff b*L^{-1} is), so no choice of representative b can ",
                              "repair this - rho_j itself is wrong for this point. IMPORTANT: ",
                              "retrying a different rho_B CANNOT help either, because rho_j is ",
                              "computed from J_bj = J*U_elem1 and u_Bj alone, and BOTH are ",
                              "invariant under this loop's rho_B retries (confirmed empirically: ",
                              "rho_B changed between attempts while rho_j did not). Fixing this ",
                              "needs a genuine re-search for a different rho_j, which ",
                              "find_alpha_rhoB_s1_s2_O currently has no mechanism to provide.")
                    elseif t219_status === :no_gamma
                        error("p=($p1,$p2): Theorem 2.19 gate for rho_j = ", rho_j_raw,
                              " - b*(Z*rho_j+Z)^{-1} IS principal, but its generator has MIXED ",
                              "signs and no gamma coprime to dO was found within the search ",
                              "height making gamma*alpha totally positive. Unlike the ",
                              ":not_principal case this is NOT a statement about rho_j - it is ",
                              "about the choice of representative b - so raising gamma_height ",
                              "(or widening how b is chosen) is the right response, not ",
                              "rejecting rho_j.")
                    end
                    println("    p=($p1,$p2) [rho_B_attempt $rho_B_attempt]: Thm 2.19 gate: OK",
                            "   alpha = ", t219_alpha,
                            "   (b = gamma*b0 with gamma = ", t219_gamma, ")")

                    rho_j_bf_raw = BigFloat(coeff(rho_j_raw, 0)) + BigFloat(coeff(rho_j_raw, 1)) * sqrt(BigFloat(D))
                    L_bj = hj_automorph_for_root(rho_j_raw, D; max_u = elem1_max_u)
                    A_bj_cand, _ = stabilizer_power_in_Gamma_s(L_bj, [s1_bj, s2_bj]; max_power = elem1_max_power)

                    # QQ_form_p/gQ_j_p/gF_j_p/c_g_j_p come from the
                    # HJ-REDUCED matrix (per project owner) - obtained from
                    # INSIDE the same reduction machinery the shin wrapper
                    # itself uses (find_hj_reducing_matrix), not from the
                    # old reduced_hj_orbit/minimal_hj_stabilizer route. Only
                    # the reduced MATRIX is needed here (exact, integer -
                    # no floating point involved in A_reduced itself), not
                    # a reduced root value.
                    red_bj = find_hj_reducing_matrix(A_bj_cand, rho_j_bf_raw)
                    gQ_j_cand_raw = binary_quadratic_form(red_bj.A_reduced[2,1],
                                                           red_bj.A_reduced[2,2] - red_bj.A_reduced[1,1],
                                                           -red_bj.A_reduced[1,2])
                    # BUGFIX: A_reduced's raw entries have no reason to be
                    # jointly coprime (it's just some GL_2(Z) matrix), so the
                    # form built directly from them can be a non-primitive
                    # multiple of the true form - inflating its discriminant
                    # by content^2 and its apparent conductor by `content`,
                    # which then fails AdmissibleTuple's "conductor q of Q
                    # must divide f_j" check even though the TRUE (reduced)
                    # form's conductor is correct. Divide out the content
                    # first, matching the same primitivization already done
                    # in find_class_representatives's own candidate-form
                    # construction (g_int = gcd(a,b,c); a_int ÷= g_int; ...).
                    let content_c = gcd(gcd(gQ_j_cand_raw.a, gQ_j_cand_raw.b), gQ_j_cand_raw.c)
                        println("    p=($p1,$p2) [rho_B_attempt $rho_B_attempt]: gQ_j_cand_raw = ",
                                gQ_j_cand_raw, "  content(gcd) = ", content_c)
                    end
                    content_j = gcd(gcd(gQ_j_cand_raw.a, gQ_j_cand_raw.b), gQ_j_cand_raw.c)
                    gQ_j_cand = binary_quadratic_form(divexact(gQ_j_cand_raw.a, content_j),
                                                       divexact(gQ_j_cand_raw.b, content_j),
                                                       divexact(gQ_j_cand_raw.c, content_j))
                    let a_c, b_c, c_c, disc_c, fund_c, cond_c
                        a_c, b_c, c_c = gQ_j_cand.a, gQ_j_cand.b, gQ_j_cand.c
                        disc_c = Int(discriminant(gQ_j_cand))
                        fund_c, cond_c = coredisc(disc_c)
                        println("    p=($p1,$p2) [rho_B_attempt $rho_B_attempt]: gQ_j_cand (primitivized) = ", gQ_j_cand,
                                "  (a,b,c) = (", a_c, ",", b_c, ",", c_c, ")",
                                "  disc = ", disc_c, "  coredisc = (", fund_c, ",", cond_c, ")",
                                "  O's own q = ", q_val, "  F.r = ", F.r)
                    end
                    gF_j_cand = AdmissibleTuple(d, F.r, gQ_j_cand)
                    QQ_form_cand = binary_quadratic_form(gF_j_cand.A[2,1],
                                                          gF_j_cand.A[2,2] - gF_j_cand.A[1,1],
                                                          -gF_j_cand.A[1,2])

                    # Existing chain (per project owner: unchanged other
                    # than using the new rho_j and the new elem1) - M_g_j
                    # and the u1,u2 decomposition both use rho_j_raw (the
                    # exact algebraic element from find_alpha_rhoB_s1_s2_O
                    # above), matching how this chain always used a single
                    # consistent rho_j throughout.
                    M_g_j_cand = multiplication_matrix_mod_d(K(gamma), rho_j_raw, dm, D)

                    det_M_g_j = mod(M_g_j_cand[1,1]*M_g_j_cand[2,2] -
                                    M_g_j_cand[1,2]*M_g_j_cand[2,1], dm)
                    gcd_det_d = gcd(det_M_g_j, dm)
                    if gcd_det_d != 1
                        println("    p=($p1,$p2) [rho_B_attempt $rho_B_attempt]: REJECTED ",
                                "(det(M_g_j) mod d = ", det_M_g_j, ", gcd with d = ", gcd_det_d,
                                ") - trying a different rho_B")
                        push!(excluded_rho_B_p, rho_B_cur)
                        continue
                    end

                    # ---- BUGFIX (9/12, deterministic reproducer J=<5,3/2*sqrt(44)+3>,
                    # gamma=1, 90 failing points identical on every seed) ----
                    # The Galois action on ray classes IS ray-class multiplication:
                    # the acted-on point is the s-vector of the class [J]*[U_elem1],
                    # which find_alpha_rhoB_s1_s2_O(J_bj) above ALREADY computed as
                    # (s1_bj, s2_bj). The previous code discarded it and instead
                    # took du from product = elem1*elem2 - a shortcut that depends
                    # on the particular LIFTS elem1, elem2 and is only valid when
                    # they happen to multiply to a generator of the product ideal.
                    # Analysis of run_sameJ.txt (seed 208475639, all 440 points):
                    #   * (d*s_bj) mod d is <L_j>-orbit-equivariant under p -> -p
                    #     for ALL 440 points;
                    #   * the product-route du agrees with it (up to the <L_j>
                    #     orbit) for 395 points and DISAGREES for 45 = 15 classes
                    #     x 3; the 90 failing pairs are exactly those with a
                    #     disagreeing member, and every pair with du(-p) in the
                    #     orbit of -du(p) passes (orbit shifts k=0,1,2 all pass).
                    # Since shin, zeta^QA (L_j is an automorph of gQ_j) and (-1)^s
                    # are all orbit-invariant, the representative within the orbit
                    # is immaterial. So: use the class-multiplication s-vector.
                    # The product-route coordinates are still computed and printed
                    # below purely as a diagnostic (they no longer feed du).
                    u2_prod, u1_prod = find_rho_j_and_coords(product, rho_j_raw)
                    u2_cand, u1_cand = s2_bj, s1_bj
                    _red_du(u) = begin
                        x = u * dm
                        isone(denominator(x)) ? string(Int(mod(numerator(x), dm))) : string("non-integral(", x, ")")
                    end
                    println("    p=($p1,$p2) [rho_B_attempt $rho_B_attempt]: du from class multiplication (USED) = (",
                            _red_du(u1_cand), ", ", _red_du(u2_cand),
                            ")   du from product elem1*elem2 (diagnostic only) = (",
                            _red_du(u1_prod), ", ", _red_du(u2_prod), ")")

                    diag_val = Float64(u2_cand) * Float64(real(emb_conj_K(rho_j_raw))) - Float64(u1_cand)
                    println("    p=($p1,$p2) [rho_B_attempt $rho_B_attempt]: u2*conj(rho_j)-u1 = ", diag_val,
                            "  positive? ", diag_val > 0)

                    # With u from (s1_bj, s2_bj), conj(s2_bj*rho_j - s1_bj) > 0 is
                    # find_alpha_rhoB_s1_s2_O's own output normalization, so this
                    # should hold by construction; it is kept as a hard check. Note
                    # that rho_B_cur/elem2 no longer influence du, so retrying a
                    # different rho_B cannot change the outcome of this check.
                    if !(diag_val > 0)
                        println("    p=($p1,$p2) [rho_B_attempt $rho_B_attempt]: REJECTED ",
                                "(u2*conj(rho_j)-u1 <= 0 on the class-multiplication s-vector - ",
                                "this should not happen; find_alpha_rhoB_s1_s2_O's normalization ",
                                "is conj-positive) - trying a different rho_B")
                        push!(excluded_rho_B_p, rho_B_cur)
                        continue
                    end

                    c_g_j_cand = e(-BigFloat(rademacher(gF_j_cand.A)) / 24) /
                                  (chi * sqrt(BigFloat(d + 1)))

                    # BUGFIX: the integralnewdu computation (du -> M_g_j*du ->
                    # du_to_int_mod_d) used to happen AFTER this retry loop had
                    # already committed to this rho_B_attempt (found_overall =
                    # true; break). But du_to_int_mod_d has its own hard
                    # requirement - the reduced denominator must be coprime to
                    # d - that is NOT among the two checks (det(M_g_j), diag_val)
                    # this loop actually guards on. A candidate can pass both of
                    # those and still fail the coprimality requirement, and by
                    # the time that happens the loop has already exited, so
                    # there is no longer any way to fall back to a different
                    # rho_B - it just crashes (observed: p=(4,20), attempt 2,
                    # "denominator 9 of 37//9 is not coprime to d=21"), exactly
                    # the failure mode this retry loop exists to route around.
                    # Fix: attempt the SAME du/newdu/integralnewdu computation
                    # here, inside the loop, and treat a coprimality failure
                    # exactly like the other two rejection conditions (exclude
                    # this rho_B, continue to the next attempt) instead of
                    # letting it escape uncaught.
                    du_cand = [to_du_mod_d(u1_cand, dm), to_du_mod_d(u2_cand, dm)]
                    raw_newdu_cand = M_g_j_cand * du_cand
                    newdu_cand = [to_du_mod_d(raw_newdu_cand[1], dm), to_du_mod_d(raw_newdu_cand[2], dm)]
                    integralnewdu_cand = Int[]
                    coprimality_ok = true
                    for u in newdu_cand
                        du_int = dm * u
                        try
                            push!(integralnewdu_cand, du_to_int_mod_d(du_int, dm))
                        catch e
                            println("    p=($p1,$p2) [rho_B_attempt $rho_B_attempt]: REJECTED ",
                                    "(integralnewdu computation failed: ", e, ") - trying a different rho_B")
                            coprimality_ok = false
                            break
                        end
                    end
                    if !coprimality_ok
                        push!(excluded_rho_B_p, rho_B_cur)
                        continue
                    end

                    rho_j_p = rho_j_raw
                    A_bj_p = A_bj_cand
                    gF_j_p = gF_j_cand
                    gQ_j_p = gQ_j_cand
                    QQ_form_p = QQ_form_cand
                    M_g_j_p = M_g_j_cand
                    c_g_j_p = c_g_j_cand
                    rho_j_bf_p = rho_j_bf_raw
                    u1_p, u2_p = u1_cand, u2_cand
                    integralnewdu = integralnewdu_cand
                    found_overall = true
                    break
                end

                found_overall || error("p=($p1,$p2): exhausted ", max_retries_rho_B_p,
                                        " rho_B/s1/s2 candidates - could not satisfy the ",
                                        "det(M_g_j), u2*conj(rho_j)-u1>0, and integralnewdu ",
                                        "coprimality checks for this point")

                if gQ_j === nothing
                    gQ_j = gQ_j_p
                    println("  [note] gQ_j for the function's final return value taken ",
                            "from the first successfully-processed grid point, p=($p1,$p2) ",
                            "- rho_j now varies by point, so there is no single canonical ",
                            "choice; flagging this assumption explicitly.")
                end

                println("    p=($p1,$p2): integralnewdu = ", integralnewdu)

                # ---- PHASE FIX (9/13, revised after review) ----
                # Reference: the PRINCIPAL path gives arg v(p)v(-p) = 0 on all 440
                # points, and evaluates every phase-determining factor with ONE
                # object - the TUPLE's canonical A (gF.A), for every point:
                #     QA = -Q_{gF.A}(newp)/(d(d-2)),  c from rademacher(gF.A),
                #     shin_of_tuple(gF, newp),
                # with newp in the tuple's own coordinates. This path did not:
                #   (a) shin was evaluated via the wrapper with A_bj_p, the
                #       per-point MINIMAL stabilizer power (L^3 for generic
                #       points, L^1 for the two L_j-fixed points) - not the
                #       tuple's A; and
                #   (b) QA used gF_j.A's form on the UNREDUCED du, although the
                #       wrapper evaluates in the HJ-reduced frame (du_red).
                # A first revision of this fix made QA/c consistent with A_red =
                # R*A_bj*R^-1 - i.e. with the minimal stabilizer power - which is
                # the wrong A: Q_{gF.A} is always divisible by d(d-2) (the tuple's
                # A is I mod d(d-2)), which is exactly why the original exact
                # division never failed in 22,000+ evaluations, and Q_{A_red}
                # failed precisely at p=(7,7), du=[14,14], an L_j-FIXED point
                # where A_bj = L^1 != tuple power. Now: keep the reduction to
                # du_red (the coordinate change the principal path never needs)
                # and evaluate QA, c and shin ALL with the tuple's A on du_red -
                # the principal-path recipe applied verbatim to (gF_j, du_red).
                # The wrapper's A_bj-based value is kept for the diagnostic only.
                red_p = find_hj_reducing_matrix(A_bj_p, rho_j_bf_p)
                beta_p = red_p.root_used == :x ? red_p.x0 : red_p.conjx0
                s_R_p = Int(sign(red_p.R[2,1] * beta_p + red_p.R[2,2]))
                # du_red lives mod d-bar (the 2d-periodic phase needs it); the
                # cocycle argument is its reduction mod d (Lemma 2.14).
                du_red = mod.(s_R_p .* (red_p.R * integralnewdu), dm)
                QA_old = BigFloat(-QQ_form_p(integralnewdu...)) / BigFloat(d * (d - 2))
                QA = BigFloat(-QQ_form_p(du_red...)) / BigFloat(d * (d - 2))
                s = isodd(d) ? 1 : (1 + du_red[1]) * (1 + du_red[2])
                shin_tuple_val = shin_of_tuple(gF_j_p, mod.(du_red, d))
                shin_wrapper_val = shin(A_bj_p, rho_j_bf_p, integralnewdu, d)
                # roots of gQ_j = <a,b,c>: a t^2 + b t + c = 0, printed next to the
                # wrapper's reduced_root so a frame mismatch would be visible.
                aQ, bQ, cQ = Int(gQ_j_p.a), Int(gQ_j_p.b), Int(gQ_j_p.c)
                sqQ = sqrt(BigFloat(bQ^2 - 4 * aQ * cQ))
                println("    p=($p1,$p2): [phase-fix] det(R) = ", red_p.detR, "  s_R = ", s_R_p,
                        "  du_red = ", du_red,
                        "  QA(tuple A, du_red) = ", Float64(QA),
                        "  QA(old: tuple A, unreduced du) = ", Float64(QA_old),
                        "  |shin_of_tuple| = ", Float64(abs(shin_tuple_val)),
                        "  |shin wrapper(A_bj)| = ", Float64(abs(shin_wrapper_val)),
                        "  reduced_root = ", Float64(red_p.reduced_root),
                        "  roots of gQ_j = ", Float64((-bQ + sqQ) / (2 * aQ)), ", ", Float64((-bQ - sqQ) / (2 * aQ)))
                V_np[gp[1]+1, gp[2]+1] = chi * ζ^QA * (-1)^s * c_g_j_p * shin_tuple_val

                # DIAGNOSTIC (purely additive, off by default): dump every
                # phase-determining intermediate quantity for hand-picked
                # points, to find where the v(p)*v(-p) phase inconsistency
                # (confirmed: 8 discrete phase clusters, ~360/7-degree
                # multiples plus a separate 60-degree anomaly for points
                # with gcd(p1,p2,d)=7, only 90/440 printed lines at the
                # correct zero phase) actually enters. Since QA/s depend only
                # on integralnewdu and QQ_form_p, and c_g_j_p depends only on
                # chi and rademacher(gF_j_p.A), comparing these directly
                # across points from different phase clusters (rather than
                # re-deriving the whole pipeline in a separate, possibly
                # subtly-different replica) should show which of the four -
                # QA, s, chi, or rademacher(gF_j_p.A) - is not behaving as a
                # class-invariant the way p and -p (or points that ought to
                # agree) require.
                if (p1, p2) in phase_debug_points
                    println("  [phase-debug] p=($p1,$p2)  gp=$gp")
                    println("  [phase-debug]   rho_j_p = ", rho_j_p, "   A_bj_p = ", A_bj_p)
                    println("  [phase-debug]   gQ_j_p = ", gQ_j_p, "   integralnewdu = ", integralnewdu)
                    println("  [phase-debug]   QA = ", QA, "   s = ", s, "   chi = ", chi)
                    println("  [phase-debug]   rademacher(gF_j_p.A) = ", rademacher(gF_j_p.A))
                    println("  [phase-debug]   c_g_j_p = ", c_g_j_p)
                    println("  [phase-debug]   zeta^QA = ", ζ^QA, "   (-1)^s = ", (-1)^s)
                    println("  [phase-debug]   raw shin(...) = ", shin(A_bj_p, rho_j_bf_p, integralnewdu, d))
                    println("  [phase-debug]   V_np[gp] = ", V_np[gp[1]+1, gp[2]+1])
                end

                let sh_plus = shin(A_bj_p, rho_j_bf_p, integralnewdu, d),
                    sh_minus = shin(A_bj_p, rho_j_bf_p, -integralnewdu, d)
                    if abs(abs(sh_plus * sh_minus) - 1.0) > 1e-10
                        println("    FAIL at p=($p1,$p2)  (intra-point check |shin(du)|*|shin(-du)| = 1 violated)")
                        println("    [fail-debug] rho_j_p=", rho_j_p, "  A_bj_p=", A_bj_p, "  gQ_j_p=", gQ_j_p)
                        println("    [fail-debug] u1_p=", u1_p, " u2_p=", u2_p, " integralnewdu=", integralnewdu,
                                "  gcd(du,d)=", gcd(gcd(integralnewdu[1], integralnewdu[2]), d))
                        println("    [fail-debug] |shin(du)|=", abs(sh_plus), "  |shin(-du)|=", abs(sh_minus),
                                "  product=", abs(sh_plus * sh_minus))
                        test1 = false
                    end
                end
            end
        end

        println("  all |vv(p)vv(-p)| correct: ", test1)

        for p1 = 0:d-1, p2 = 0:d-1
            (p1 == 0 && p2 == 0) && continue
            new_ovs[p1+1, p2+1] = V_np[p1+1, p2+1]
        end

        println("  v(p)*v(-p) for all p != 0 mod d (after Galois action",
                isodd(d) ? "):" : "; even d: on the 2d x 2d grid, -p mod 2d, expected 1/(d+1) everywhere):")
        n_dev = 0
        for p1 = 0:dm-1
            for p2 = 0:dm-1
                (mod(p1, d) == 0 && mod(p2, d) == 0) && continue
                mp1 = mod(-p1, dm)
                mp2 = mod(-p2, dm)
                product = V_np[p1+1, p2+1] * V_np[mp1+1, mp2+1]
                ok = abs(product - 1 / BigFloat(d + 1)) < 1e-8
                ok || (n_dev += 1)
                println("    p=($p1,$p2): v(p)*v(-p) = ", Float64(real(product)),
                        " + ", Float64(imag(product)), "im",
                        "  |v(p)*v(-p)| = ", Float64(abs(product)),
                        "  1/(d+1) = ", 1.0/(d+1), ok ? "" : "   <-- DEVIATION")
            end
        end
        println("  deviations of v(p)*v(-p) from 1/(d+1): ", n_dev, " of ", dm^2 - (dm ÷ d)^2)
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
            # I_O MUST be the principal O-ideal (gamma) itself. It was previously
            # rebuilt as contract(OK*(gamma), O), but (gamma*OK) meet O equals
            # gamma*O ONLY when gamma is coprime to the conductor of O. For d=21
            # (conductor prime 3 | d) every generator coprime to m is coprime
            # to the conductor, so the round trip was the identity. For d=15
            # (conductor 4*OK, prime 2 not dividing m=15) even-norm generators
            # are legitimately in J_m^*(O) but the contraction returns a
            # strictly larger, NON-invertible module (e.g. gamma=-2(7+4sqrt3):
            # gamma*O has index 4 in O, contract(gamma*OK,O)=2Z+4sqrt3*Z has
            # index 2). That mismatch is exactly what the fast-fiber check
            # "elem_check does not generate Jfull*b_O" detected on all 48
            # even-norm entries (9/15 run). Build (gamma) directly in O.
            I_O = ideal(O, O(elem_in_nf(gamma)))
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

    all_places = real_places(K)  # kept for any other uses in this function
    Sigma = [conjugation_place(K)]  # BUGFIX: was [all_places[2]] - wrong place, see conjugation_place's docstring

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

    all_places = real_places(K)  # kept for any other uses in this function
    Sigma = [conjugation_place(K)]  # BUGFIX: was [all_places[2]] - wrong place, see conjugation_place's docstring

    all_reps, all_triples = ray_class_group_representatives(K, O, m, Sigma)
    ray_class_order = length(all_reps)

    println("d = ", d_val, ", Q = ", F.Q,
            ": |H_{d*infty2}^{O_f}| = ", ray_class_order)

    return ray_class_order, all_reps, all_triples
end

ray_class_group_data(d::Int) = ray_class_group_data(AdmissibleTuple(d))

# ---------------------------------------------------------------------------
# ray_class_shin_table: for each explicit representative ideal J of a class
# in the ray class group H_{d*infty2}^{O_f}, computes a "shin value" via:
#   - b = a representative of [J]^{-1} in Pic(O); alpha, rho_B from
#     find_alpha_rhoB_s1_s2_O such that b*d*O = alpha*(Z*rho_B+Z), alpha
#     totally positive, rho_B > conj(rho_B).
#   - s1, s2 (also from find_alpha_rhoB_s1_s2_O) with
#     alpha*(s2*rho_B-s1)*b^{-1} in the class of J, s2*conj(rho_B)-s1 > 0,
#     and alpha*(s2*rho_B-s1) in b (all three already enforced by that
#     function's own search).
#   - s = [s1,s2]; A = stabilizer_generator_in_Gamma_s(rho_B, D, s), a
#     generator of {B in Gamma_s : B*rho_B = rho_B} up to the possible
#     extra factor of -I.
#   - shin value for J = shin(A, rho_B, s, d) (the wrapper built earlier),
#     with s1, s2 re-expressed as an integer vector over d (verified exact,
#     not rounded) to match shin's Integer-vector calling convention.
# Prints the full table, and flags (without erroring) any pair of classes
# whose shin values agree to within diff_tol.
# ---------------------------------------------------------------------------

"""
    ray_class_shin_table(F::AdmissibleTuple; mn_height::Int = 3000,
                          q_gen_height::Int = 3000, max_power::Int = 10_000,
                          max_u::Integer = 10_000_000, diff_tol::Real = 1e-10)

Builds and prints a table of shin values, one per class of the ray class
group `H_{d*infty2}^{O_f}` associated to `F` - see the block comment above
for the exact recipe. Returns
`table::Vector{Tuple{Any,Union{Nothing,Vector{Int}},Union{Nothing,Complex{BigFloat}}}}`,
a `(J, d*s, shin_value)` triple for each ray class (`nothing` in place of
`d*s` and the value for any class where the underlying search failed -
printed as a failure rather than aborting the whole table).
"""
# ---------------------------------------------------------------------------
# _ray_class_shin_setup: the O/K/ray-class-representative/Picard-data setup
# shared by ray_class_shin_table and ray_class_shin_table_fast. Factored out
# so compare_ray_class_shin_tables can compute it ONCE and hand the SAME
# all_reps/all_triples (and Picard data) to both table functions - calling
# ray_class_group_representatives independently in each (as both functions
# still do by default, for standalone use) is not guaranteed to reproduce
# the same literal representative ideals or the same class ORDER on every
# call, which otherwise silently misaligns the two tables at the same index.
# ---------------------------------------------------------------------------
function _ray_class_shin_setup(F::AdmissibleTuple)
    D = Int(F.D)
    q = Int(F.q)
    d = Int(F.d)

    K, a = quadratic_field(D)
    Dr = squarefree_part(D)
    sqrt_Dr = a // isqrt(D ÷ Dr)
    omega = mod(Dr, 4) == 1 ? (K(1) + sqrt_Dr) // 2 : sqrt_Dr
    O = Order(K, [one(K), q * omega])

    dbar = mod(d, 2) == 1 ? d : 2 * d
    m = ideal(O, dbar)

    all_places = real_places(K)  # kept for any other uses in this function
    Sigma = [conjugation_place(K)]  # BUGFIX: was [all_places[2]] - wrong place, see conjugation_place's docstring

    all_reps, all_triples = ray_class_group_representatives(K, O, m, Sigma)

    P_O, mP_O, rho_reps, gQ_reps, F_class = find_class_representatives(O, F, K)

    # d_check: strip q's own prime factors out of d before the coprimality
    # target used inside find_alpha_rhoB_s1_s2_O's search (see the identical
    # comment/reasoning in twisted_chi_general - required whenever q | d).
    # EVEN d (9/18): the ray-class modulus is d-bar = 2d (m above), so the
    # coprimality target for the Theorem 2.19 searches is d-bar too.
    d_check = dbar
    for (p, _) in factor(q)
        while d_check % Int(p) == 0
            d_check ÷= Int(p)
        end
    end

    return (D = D, q = q, d = d, dbar = dbar, K = K, O = O,
            all_reps = all_reps, all_triples = all_triples,
            P_O = P_O, mP_O = mP_O, rho_reps = rho_reps,
            gQ_reps = gQ_reps, F_class = F_class, d_check = d_check)
end

"""
    fiber_orientation_sign(K, O, rho_B, rho_ref; unit_bound = 60)

Orientation of the lattice basis (rho_B, 1) relative to (rho_ref, 1), for two
roots whose lattices are in the SAME Picard class (NB: an anchor root rho_B
obtained from b_O*dO = alpha*(Z+Z*rho_B) has its LATTICE in the class of b_O,
i.e. the INVERSE of the anchor J's class - so the reference is rho_reps[-u_B],
not rho_reps[u_B]; the two coincide only when the class has order <= 2, which
is why d=15 (Pic = Z/2) worked and d=21 (Pic of order 4) failed on 9/16): with mu*(Z+Z*rho_B) = Z+Z*rho_ref, returns
`(sign(N(mu)), mu)`. sign(N(mu)) = +1 means the GL_2(Z) element taking rho_ref
to rho_B has det +1 (same NARROW class); -1 means det -1 (opposite narrow
class), in which case the covariance theorem's det = -1 branch applies and a
shin value computed at rho_B is the invariant of the class evaluated at the
CONJUGATE of the reference root. Returns `(nothing, nothing)` if the quotient
lattice could not be verified principal (should not happen for same-class
roots).

Why this matters (9/15, d=15): find_alpha_rhoB_s1_s2_O only enforces
rho_B > rho_B' on each root by itself, which is orientation-blind. For d=21
every anchor root happened to be rho_ref + integer (det +1), so the group and
monoid tables were in the same frame as twisted_chi_general's lookup at
rho_F. For d=15 BOTH anchors came out with N(mu) < 0 (-47 and -13/3), so every
table value was conj(lookup value) (verified: test_ray_class_covariance gave
exact complex conjugates to ~55 digits) and no lookup could ever succeed.
The table builders now evaluate such fibers at conj(rho_B) instead. (If O had
a unit of norm -1 the sign of N(mu) would be ambiguous, but then wide = narrow
class and both frames give the same class invariant.)
"""
function fiber_orientation_sign(K, O, rho_B, rho_ref; unit_bound::Int = 60)
    LB_gens = [K(1), K(rho_B)]
    NB = norm(fractional_ideal(O, LB_gens))
    LB_inv = fractional_ideal(O, [quad_conj(K, g) // NB for g in LB_gens])
    Q = fractional_ideal(O, [K(1), K(rho_ref)]) * LB_inv
    okp, mu = try
        is_principal_O_verified(K, O, [_nf_elem_of(K, x) for x in basis(Q)]; unit_bound = unit_bound)
    catch
        (false, nothing)
    end
    okp || return (nothing, nothing)
    return (norm(K(mu)) > 0 ? 1 : -1, mu)
end

"""
    root_bf_in_frame(rho_B, D, conj_frame::Bool)

BigFloat value of `rho_B` in the main embedding (conj_frame = false) or of its
conjugate (conj_frame = true). NOTE: handing `shin` the conjugate root does NOT
compute shin_A^r(conj(rho_B)) - find_hj_reducing_matrix then ends with the
wrong root on top (its "reduced_root < 1" warning) - so the table builders do
not use the conj_frame = true branch; they use `shin_value_in_frame` instead.
"""
function root_bf_in_frame(rho_B, D, conj_frame::Bool)
    c0 = BigFloat(coeff(rho_B, 0)); c1 = BigFloat(coeff(rho_B, 1)) * sqrt(BigFloat(D))
    return conj_frame ? c0 - c1 : c0 + c1
end

"""
    shin_value_in_frame(A, rho_B, D, p_vec, d, conj_frame::Bool)

The class invariant shin_A^{p/d} for the representative (rho_B, p) expressed
in the REFERENCE root's frame: shin evaluated at rho_B's main embedding when
the fiber has orientation det +1, and its COMPLEX CONJUGATE when det -1
(fiber_orientation_sign). Empirical basis: test_ray_class_covariance(15;0,1)
found the trivial ray class's value computed at the det -1 anchor root
(J = O, b_O = (1+2*sqrt(12))) to be exactly conj(shin^{[0,1]}(rho_F)) to ~55
digits, i.e. the frame flip acts as complex conjugation on the values.
"""
function shin_value_in_frame(A, rho_B, D, p_vec, d, conj_frame::Bool)
    v = shin(A, root_bf_in_frame(rho_B, D, false), p_vec, d)
    return conj_frame ? conj(v) : v
end


function ray_class_shin_table(F::AdmissibleTuple;
                               mn_height::Int = 3000,
                               q_gen_height::Int = 3000,
                               max_power::Int = 10_000,
                               max_u::Integer = 10_000_000,
                               diff_tol::Real = 1e-10,
                               shared_data::Union{Nothing,NamedTuple} = nothing)
    sd = shared_data === nothing ? _ray_class_shin_setup(F) : shared_data
    D, q, d = sd.D, sd.q, sd.d
    # EVEN d (9/18): every Theorem 2.19 point representation runs at the
    # ray-class modulus d-bar (= 2d for even d, = d for odd d): the (s1,s2)
    # vectors have denominator d-bar and the stored value is the d-bar-modulus
    # invariant shin^{p/d-bar}, used only for identification. The ghost
    # cocycle itself (shin^{p/d}) is only ever evaluated in twisted_chi_general.
    dm = hasproperty(sd, :dbar) ? sd.dbar : (isodd(d) ? d : 2 * d)
    K, O = sd.K, sd.O
    all_reps, all_triples = sd.all_reps, sd.all_triples
    P_O, mP_O, rho_reps, gQ_reps, F_class = sd.P_O, sd.mP_O, sd.rho_reps, sd.gQ_reps, sd.F_class
    d_check = sd.d_check
    n_classes = length(all_reps)
    println("ray_class_shin_table: |H_{d*infty2}^{O_f}| = ", n_classes)

    table = Vector{Tuple{Any,Union{Nothing,Vector},Union{Nothing,Complex{BigFloat}}}}(undef, n_classes)

    # Cache of (alpha, rho_B, b_O) keyed by Picard class u_B: these depend
    # only on u_B, not on the literal ray-class ideal J (see the
    # ADDITIVE FAST PATH comment inside find_alpha_rhoB_s1_s2_O) - measured
    # to dominate the per-class runtime (~28s/class), so ray classes sharing
    # a Picard-class fiber reuse the FIRST one's (alpha,rho_B,b_O) via the
    # fast-path hints instead of repeating that search. b_O itself isn't
    # returned by find_alpha_rhoB_s1_s2_O (so its return signature - and
    # every existing caller of it - stays untouched); it's obtained
    # separately, once per Picard class, via the same
    # find_b_with_verified_principal_product call that function uses
    # internally.
    alpha_rhoB_bO_cache = Dict{Any,Tuple{Any,Any,Any}}()
    dO_O_shared = ideal(O, d)
    d1_ideal_shared = ideal(O, d + 1)

    # Timing instrumentation (diagnostic only - does not affect results) to
    # see empirically where time is actually going, rather than guessing.
    t_alpha_rhoB = 0.0
    t_stabilizer = 0.0
    t_shin = 0.0

    for (idx, J) in enumerate(all_reps)
        println("\n--- ray class ", idx, " of ", n_classes, " ---")
        println("  J = ", J)
        try
            u_B = mP_O \ J
            rho_B_init = rho_reps[u_B]

            cached = get(alpha_rhoB_bO_cache, u_B, nothing)
            dt = @elapsed begin
                if cached === nothing
                    alpha, rho_B, s1, s2 = find_alpha_rhoB_s1_s2_O(
                        J, rho_B_init, rho_B_init, u_B, P_O, mP_O, O, K,
                        d, rho_B_init, d_check;
                        mn_height = mn_height, q_gen_height = q_gen_height)
                    b_O, _ = find_b_with_verified_principal_product(O, K, J, [dO_O_shared, d1_ideal_shared])
                    alpha_rhoB_bO_cache[u_B] = (alpha, rho_B, b_O)
                else
                    alpha_hint, rho_B_hint, b_O_hint = cached
                    alpha, rho_B, s1, s2 = find_alpha_rhoB_s1_s2_O(
                        J, rho_B_init, rho_B_init, u_B, P_O, mP_O, O, K,
                        d, rho_B_init, d_check;
                        mn_height = mn_height, q_gen_height = q_gen_height,
                        alpha_hint = alpha_hint, rho_B_hint = rho_B_hint, b_O_hint = b_O_hint)
                end
            end
            t_alpha_rhoB += dt
            println("  alpha = ", alpha, "  rho_B = ", rho_B, "  (", round(dt, digits=3), " s)",
                    cached === nothing ? "  [fresh]" : "  [cached alpha/rho_B/b_O]")
            println("  s1 = ", s1, "  s2 = ", s2)

            s_vec = [s1, s2]
            dt = @elapsed begin
                A, has_neg_I = stabilizer_generator_in_Gamma_s(
                    rho_B, D, s_vec; max_power = max_power, max_u = max_u)
            end
            t_stabilizer += dt
            println("  A = ", A, "  (also has -I in Gamma_s stabilizer? ", has_neg_I,
                    ")  (", round(dt, digits=3), " s)")

            p1_exact = s1 * dm
            p2_exact = s2 * dm
            (isone(denominator(p1_exact)) && isone(denominator(p2_exact))) ||
                error("s = [$s1,$s2] does not have denominator dividing d-bar = $dm ",
                      "(s*d-bar = [$p1_exact, $p2_exact]) - cannot express as an integer ",
                      "vector over d-bar as expected")
            p_vec = [Int(numerator(p1_exact)), Int(numerator(p2_exact))]

            rho_B_bf = BigFloat(coeff(rho_B, 0)) + BigFloat(coeff(rho_B, 1)) * sqrt(BigFloat(D))

            dt = @elapsed begin
                val = shin(A, rho_B_bf, p_vec, d)
            end
            t_shin += dt
            println("  shin value = ", val, "  (", round(dt, digits=3), " s)")
            table[idx] = (J, p_vec, val)
        catch e
            println("  FAILED: ", e)
            table[idx] = (J, nothing, nothing)
        end
    end

    println("\n=== Timing summary (diagnostic only) ===")
    println("  total time in find_alpha_rhoB_s1_s2_O:      ", round(t_alpha_rhoB, digits=1), " s")
    println("  total time in stabilizer_generator_in_Gamma_s: ", round(t_stabilizer, digits=1), " s")
    println("  total time in shin:                          ", round(t_shin, digits=1), " s")

    println("\n=== Checking pairwise shin-value differences (tol = ", diff_tol, ") ===")
    any_failure = false
    for i in 1:n_classes, j in (i+1):n_classes
        vi = table[i][3]
        vj = table[j][3]
        (vi === nothing || vj === nothing) && continue
        if abs(vi - vj) < diff_tol
            any_failure = true
            println("  *** FAILURE: shin values for ray classes ", i, " and ", j,
                    " differ by less than ", diff_tol, " (|diff| = ", abs(vi - vj), ") ***")
        end
    end
    any_failure || println("  No collisions found above tol = ", diff_tol, ".")

    println("\n=== Full shin value table ===")
    for (idx, (J, ds_vec, val)) in enumerate(table)
        println("  class ", idx, ": J = ", J,
                "  d*s = ", ds_vec === nothing ? "FAILED" : ds_vec,
                "  shin = ", val === nothing ? "FAILED" : val)
    end

    return table
end

ray_class_shin_table(d::Int) = ray_class_shin_table(AdmissibleTuple(d))

# ---------------------------------------------------------------------------
# ray_class_shin_table_fast: a SEPARATE, faster implementation of the same
# table as ray_class_shin_table, for cross-checking correctness (per-class
# values should agree with ray_class_shin_table's up to floating tolerance).
#
# Recipe: for each Picard class, do the full (slow) find_alpha_rhoB_s1_s2_O
# search exactly once, on the "anchor" ray class representative (the one
# with trivial principal-part, I_O = O - ray_class_group_representatives
# always includes exactly this entry first for every Picard class). This
# gives alpha, rho_B, b_O, s1, s2, hence a generator
#     elem2_anchor = alpha*(s2*rho_B - s1)
# of J0*b_O (J0 = the anchor's own ideal), already sign-adjusted so that
# emb_conj(elem2_anchor) > 0.
#
# Every OTHER ray class in the same Picard fiber is exactly J0*I_O for some
# principal O-ideal I_O = (gamma) (the same set of (I_O,gamma) pairs is
# reused across every Picard class - see ray_class_group_representatives).
# Since gamma is an actual element of O (not just of K), and b_O is an
# O-ideal (so closed under multiplication by any O-element):
#   - gamma*elem2_anchor automatically lies in b_O (elem2_anchor does, and
#     b_O absorbs multiplication by O-elements);
#   - (gamma*elem2_anchor) = (gamma)*(elem2_anchor) = I_O*(J0*b_O)
#     = (J0*I_O)*b_O, as ideals, purely by elementary ideal-multiplication
#     (elem2_anchor generates J0*b_O by construction).
# So gamma*elem2_anchor (sign-adjusted, since gamma need only be positive
# at the ONE real place in Sigma, not necessarily at its conjugate) is
# EXACTLY a valid generator for this ray class's own version of "elem2" -
# solving for its (s1,s2) via the SAME closed-form linear solve
# find_alpha_rhoB_s1_s2_O uses internally (the "principal class" style
# generator-to-(s1,s2) step) needs no search at all. Both of the algebraic
# facts above are checked explicitly below anyway (never trusted blindly),
# so any flaw in this reasoning surfaces as an error rather than a silently
# wrong value.
# ---------------------------------------------------------------------------

"""
    ray_class_shin_table_fast(F::AdmissibleTuple; mn_height::Int = 3000,
                               q_gen_height::Int = 3000, max_power::Int = 10_000,
                               max_u::Integer = 10_000_000, diff_tol::Real = 1e-10)

Faster alternative construction of the same table as `ray_class_shin_table`
(see the block comment above for the recipe) - a separate implementation
for cross-checking correctness, not a replacement. Same return type and
same printed diagnostics (per-class table, pairwise-collision check) as
`ray_class_shin_table`.
"""
function ray_class_shin_table_fast(F::AdmissibleTuple;
                                    mn_height::Int = 3000,
                                    q_gen_height::Int = 3000,
                                    max_power::Int = 10_000,
                                    max_u::Integer = 10_000_000,
                                    diff_tol::Real = 1e-10,
                                    shared_data::Union{Nothing,NamedTuple} = nothing)
    sd = shared_data === nothing ? _ray_class_shin_setup(F) : shared_data
    D, q, d = sd.D, sd.q, sd.d
    # EVEN d (9/18): every Theorem 2.19 point representation runs at the
    # ray-class modulus d-bar (= 2d for even d, = d for odd d): the (s1,s2)
    # vectors have denominator d-bar and the stored value is the d-bar-modulus
    # invariant shin^{p/d-bar}, used only for identification. The ghost
    # cocycle itself (shin^{p/d}) is only ever evaluated in twisted_chi_general.
    dm = hasproperty(sd, :dbar) ? sd.dbar : (isodd(d) ? d : 2 * d)
    K, O = sd.K, sd.O
    all_reps, all_triples = sd.all_reps, sd.all_triples
    P_O, mP_O, rho_reps, gQ_reps, F_class = sd.P_O, sd.mP_O, sd.rho_reps, sd.gQ_reps, sd.F_class
    d_check = sd.d_check
    n_classes = length(all_reps)
    println("ray_class_shin_table_fast: |H_{d*infty2}^{O_f}| = ", n_classes)

    all_places = real_places(K)
    emb_conj = embedding(all_places[1])
    emb_main = embedding(all_places[2])

    trivial_I_O = ideal(O, 1)

    table = Vector{Tuple{Any,Union{Nothing,Vector},Union{Nothing,Complex{BigFloat}}}}(undef, n_classes)

    # Per-Picard-class cache populated once, at that class's anchor entry.
    fiber_cache = Dict{Any,NamedTuple}()

    t_anchor = 0.0
    t_fast = 0.0

    for (idx, (J, I_O, gamma, is_principal_J)) in enumerate(all_triples)
        Jfull = all_reps[idx]
        println("\n--- ray class ", idx, " of ", n_classes, " ---")
        println("  J (Picard rep) = ", J, "   I_O = ", I_O, "   gamma = ", gamma)
        println("  full ideal = ", Jfull)
        try
            u_B = mP_O \ J
            is_anchor = (I_O == trivial_I_O)

            local alpha, rho_B, s1, s2
            if is_anchor
                dt = @elapsed begin
                    rho_B_init = rho_reps[u_B]
                    alpha, rho_B, s1, s2 = find_alpha_rhoB_s1_s2_O(
                        Jfull, rho_B_init, rho_B_init, u_B, P_O, mP_O, O, K,
                        dm, rho_B_init, d_check; d1 = d + 1,
                        mn_height = mn_height, q_gen_height = q_gen_height)
                    b_O, _ = find_b_with_verified_principal_product(
                        O, K, Jfull, [ideal(O, dm), ideal(O, d + 1)])
                    elem2_anchor = K(s2) * rho_B - K(s1)
                    abs(Float64(emb_conj(elem2_anchor))) > 0 || error("elem2_anchor is zero under emb_conj")
                    Float64(emb_conj(elem2_anchor)) > 0 ||
                        error("elem2_anchor is not sign-adjusted as expected (emb_conj <= 0)")
                    L = hj_automorph_for_root(rho_B, D; max_u = max_u)
                    osign, _ = fiber_orientation_sign(K, O, rho_B, rho_reps[-u_B])   # see docstring
                    if osign === nothing
                        println("  [ANCHOR] WARNING: could not determine orientation of rho_B - assuming det = +1")
                        osign = 1
                    end
                    fiber_cache[u_B] = (alpha = alpha, rho_B = rho_B, b_O = b_O,
                                         elem2_anchor = elem2_anchor, L = L, J0 = Jfull,
                                         conj_frame = (osign == -1))
                end
                t_anchor += dt
                println("  [ANCHOR] alpha = ", alpha, "  rho_B = ", rho_B,
                        "  (", round(dt, digits=3), " s)")
            else
                haskey(fiber_cache, u_B) ||
                    error("no anchor cached yet for this Picard class - expected the trivial-I_O ",
                          "entry to appear first within each Picard-class block of all_triples")
                dt = @elapsed begin
                    fc = fiber_cache[u_B]
                    alpha, rho_B, b_O = fc.alpha, fc.rho_B, fc.b_O

                    w = K(gamma) * fc.elem2_anchor
                    elem2 = Float64(emb_conj(w)) > 0 ? w : -w

                    qc  = coeff(elem2, 1)
                    p0  = coeff(elem2, 0)
                    c2b = coeff(rho_B, 1)
                    c1b = coeff(rho_B, 0)
                    s2 = QQ(qc / c2b)
                    s1 = QQ(qc * c1b / c2b - p0)

                    # Defensive verification - the same two conditions
                    # find_alpha_rhoB_s1_s2_O itself checks for every
                    # candidate, plus the sign condition. Never trust the
                    # algebraic derivation above blindly.
                    rhoB_conj_val = Float64(emb_conj(rho_B))
                    s1_f = Float64(BigInt(numerator(s1))) / Float64(BigInt(denominator(s1)))
                    s2_f = Float64(BigInt(numerator(s2))) / Float64(BigInt(denominator(s2)))
                    (s2_f * rhoB_conj_val - s1_f) > 0 ||
                        error("fast-fiber: sign condition s2*conj(rho_B) - s1 > 0 failed")

                    elem_check = alpha * (K(s2) * rho_B - K(s1))
                    elem_check_ideal = fractional_ideal(O, [elem_check])
                    Jfull_bO = Jfull * b_O
                    elem_check_ideal == Jfull_bO ||
                        error("fast-fiber: elem_check does not generate Jfull*b_O as expected")
                    (elem_check in b_O) ||
                        error("fast-fiber: elem_check is not in b_O as expected")
                end
                t_fast += dt
                println("  [fast-fiber] alpha = ", alpha, "  rho_B = ", rho_B,
                        "  (", round(dt, digits=3), " s)")
            end

            println("  s1 = ", s1, "  s2 = ", s2)

            s_vec = [s1, s2]
            L = fiber_cache[u_B].L
            A, has_neg_I = stabilizer_power_in_Gamma_s(L, s_vec; max_power = max_power)
            println("  A = ", A, "  (also has -I in Gamma_s stabilizer? ", has_neg_I, ")")

            p1_exact = s1 * dm
            p2_exact = s2 * dm
            (isone(denominator(p1_exact)) && isone(denominator(p2_exact))) ||
                error("s = [$s1,$s2] does not have denominator dividing d-bar = $dm ",
                      "(s*d-bar = [$p1_exact, $p2_exact]) - cannot express as an integer ",
                      "vector over d-bar as expected")
            p_vec = [Int(numerator(p1_exact)), Int(numerator(p2_exact))]

            val = shin_value_in_frame(A, rho_B, D, p_vec, dm, fiber_cache[u_B].conj_frame)
            println("  shin value = ", val)
            table[idx] = (Jfull, p_vec, val)
        catch e
            println("  FAILED: ", e)
            table[idx] = (all_reps[idx], nothing, nothing)
        end
    end

    println("\n=== Timing summary (diagnostic only) ===")
    println("  total time in ANCHOR computations:     ", round(t_anchor, digits=1), " s")
    println("  total time in fast-fiber computations: ", round(t_fast, digits=1), " s")

    println("\n=== Checking pairwise shin-value differences (tol = ", diff_tol, ") ===")
    any_failure = false
    for i in 1:n_classes, j in (i+1):n_classes
        vi = table[i][3]
        vj = table[j][3]
        (vi === nothing || vj === nothing) && continue
        if abs(vi - vj) < diff_tol
            any_failure = true
            println("  *** FAILURE: shin values for ray classes ", i, " and ", j,
                    " differ by less than ", diff_tol, " (|diff| = ", abs(vi - vj), ") ***")
        end
    end
    any_failure || println("  No collisions found above tol = ", diff_tol, ".")

    println("\n=== Full shin value table (fast) ===")
    for (idx, (J, ds_vec, val)) in enumerate(table)
        println("  class ", idx, ": J = ", J,
                "  d*s = ", ds_vec === nothing ? "FAILED" : ds_vec,
                "  shin = ", val === nothing ? "FAILED" : val)
    end

    return table
end

ray_class_shin_table_fast(d::Int) = ray_class_shin_table_fast(AdmissibleTuple(d))

# ---------------------------------------------------------------------------
# ray_class_shin_table_fast_quiet: identical computation to
# ray_class_shin_table_fast, but (a) prints nothing at all, and (b) returns
# each entry's s1, s2, rho_B alongside its shin value (not just d*s), since
# a REVERSE lookup by shin value (find which ray class produced a given
# shin value, then read off ITS s1/s2/rho_B) is exactly what's needed to
# fix elem1 in twisted_chi_general - see the comment there. A separate
# function rather than modifying ray_class_shin_table_fast itself, so its
# existing return type/behavior (and compare_ray_class_shin_tables, which
# depends on it) are untouched.
# ---------------------------------------------------------------------------

"""
    ray_class_shin_table_fast_quiet(F::AdmissibleTuple; mn_height::Int = 3000,
                                     q_gen_height::Int = 3000, max_power::Int = 10_000,
                                     max_u::Integer = 10_000_000,
                                     shared_data::Union{Nothing,NamedTuple} = nothing)

Same computation as `ray_class_shin_table_fast`, silenced (no `println` at
all) and returning richer per-entry data for reverse lookup by shin value.

Returns `Vector{Union{Nothing,NamedTuple}}`, one entry per ray class, each
either `nothing` (that class's computation failed) or a named tuple
`(J, s1, s2, rho_B, shin_value)`.
"""
function _ray_class_shin_table_fast_quiet_impl(F::AdmissibleTuple;
                                          mn_height::Int = 3000,
                                          q_gen_height::Int = 3000,
                                          max_power::Int = 10_000,
                                          max_u::Integer = 10_000_000,
                                          shared_data::Union{Nothing,NamedTuple} = nothing)
    sd = shared_data === nothing ? _ray_class_shin_setup(F) : shared_data
    D, q, d = sd.D, sd.q, sd.d
    # EVEN d (9/18): every Theorem 2.19 point representation runs at the
    # ray-class modulus d-bar (= 2d for even d, = d for odd d): the (s1,s2)
    # vectors have denominator d-bar and the stored value is the d-bar-modulus
    # invariant shin^{p/d-bar}, used only for identification. The ghost
    # cocycle itself (shin^{p/d}) is only ever evaluated in twisted_chi_general.
    dm = hasproperty(sd, :dbar) ? sd.dbar : (isodd(d) ? d : 2 * d)
    K, O = sd.K, sd.O
    all_reps, all_triples = sd.all_reps, sd.all_triples
    P_O, mP_O, rho_reps, gQ_reps, F_class = sd.P_O, sd.mP_O, sd.rho_reps, sd.gQ_reps, sd.F_class
    d_check = sd.d_check
    n_classes = length(all_reps)

    all_places = real_places(K)
    emb_conj = embedding(all_places[1])
    emb_main = embedding(all_places[2])

    trivial_I_O = ideal(O, 1)

    table = Vector{Union{Nothing,NamedTuple}}(undef, n_classes)
    fiber_cache = Dict{Any,NamedTuple}()

    for (idx, (J, I_O, gamma, is_principal_J)) in enumerate(all_triples)
        Jfull = all_reps[idx]
        t_triple = time()
        try
            u_B = mP_O \ J
            is_anchor = (I_O == trivial_I_O)
            # progress goes to STDERR: the public wrapper redirects stdout to
            # devnull, so this is the only channel that stays visible. Anchors
            # are the only expensive entries (two nested height-bounded
            # searches); non-anchors reuse the fiber cache and are fast.
            println(stderr, "  [table] triple ", idx, "/", n_classes, "  class ", u_B,
                    is_anchor ? "  ANCHOR (running find_alpha_rhoB_s1_s2_O + b_O search) ..." : "")

            local alpha, rho_B, s1, s2
            if is_anchor
                rho_B_init = rho_reps[u_B]
                alpha, rho_B, s1, s2 = find_alpha_rhoB_s1_s2_O(
                    Jfull, rho_B_init, rho_B_init, u_B, P_O, mP_O, O, K,
                    dm, rho_B_init, d_check; d1 = d + 1,
                    mn_height = mn_height, q_gen_height = q_gen_height)
                b_O, _ = find_b_with_verified_principal_product(
                    O, K, Jfull, [ideal(O, dm), ideal(O, d + 1)])
                elem2_anchor = K(s2) * rho_B - K(s1)
                abs(Float64(emb_conj(elem2_anchor))) > 0 || error("elem2_anchor is zero under emb_conj")
                Float64(emb_conj(elem2_anchor)) > 0 ||
                    error("elem2_anchor is not sign-adjusted as expected (emb_conj <= 0)")
                L = hj_automorph_for_root(rho_B, D; max_u = max_u)
                # Orientation of this anchor root relative to the class's
                # reference root (see fiber_orientation_sign): det -1 means the
                # whole fiber must be evaluated at conj(rho_B).
                # Z+Z*rho_B is in the class of b_O = INVERSE of J's class, so the
                # reference root is rho_reps[-u_B] (== rho_B_init only if 2*u_B = 0).
                rho_orient_ref = rho_reps[-u_B]
                osign, mu_or = fiber_orientation_sign(K, O, rho_B, rho_orient_ref)
                if osign === nothing
                    # Only the trivial fiber's orientation matters for the lookups at
                    # rho_F (points are principal); do not fail the fiber over this.
                    println(stderr, "  [table] anchor class ", u_B, ": WARNING could not determine ",
                            "orientation of rho_B = ", rho_B, " relative to rho_reps[-u_B] = ",
                            rho_orient_ref, " - assuming det = +1")
                    osign = 1
                    conj_frame = false
                else
                    conj_frame = (osign == -1)
                    println(stderr, "  [table] anchor class ", u_B, ": N(mu) = ", norm(K(mu_or)),
                            "  orientation det = ", osign,
                            conj_frame ? "  -> conjugating this fiber's values (frame flip)" : "")
                end
                fiber_cache[u_B] = (alpha = alpha, rho_B = rho_B, b_O = b_O,
                                     elem2_anchor = elem2_anchor, L = L, J0 = Jfull,
                                     conj_frame = conj_frame)
            else
                haskey(fiber_cache, u_B) ||
                    error("no anchor cached yet for this Picard class - expected the trivial-I_O ",
                          "entry to appear first within each Picard-class block of all_triples")
                fc = fiber_cache[u_B]
                alpha, rho_B, b_O = fc.alpha, fc.rho_B, fc.b_O

                w = K(gamma) * fc.elem2_anchor
                elem2 = Float64(emb_conj(w)) > 0 ? w : -w

                qc  = coeff(elem2, 1)
                p0  = coeff(elem2, 0)
                c2b = coeff(rho_B, 1)
                c1b = coeff(rho_B, 0)
                s2 = QQ(qc / c2b)
                s1 = QQ(qc * c1b / c2b - p0)

                rhoB_conj_val = Float64(emb_conj(rho_B))
                s1_f = Float64(BigInt(numerator(s1))) / Float64(BigInt(denominator(s1)))
                s2_f = Float64(BigInt(numerator(s2))) / Float64(BigInt(denominator(s2)))
                (s2_f * rhoB_conj_val - s1_f) > 0 ||
                    error("fast-fiber: sign condition s2*conj(rho_B) - s1 > 0 failed")

                elem_check = alpha * (K(s2) * rho_B - K(s1))
                elem_check_ideal = fractional_ideal(O, [elem_check])
                Jfull_bO = Jfull * b_O
                elem_check_ideal == Jfull_bO ||
                    error("fast-fiber: elem_check does not generate Jfull*b_O as expected")
                (elem_check in b_O) ||
                    error("fast-fiber: elem_check is not in b_O as expected")
            end

            s_vec = [s1, s2]
            L = fiber_cache[u_B].L
            A, has_neg_I = stabilizer_power_in_Gamma_s(L, s_vec; max_power = max_power)

            p1_exact = s1 * dm
            p2_exact = s2 * dm
            (isone(denominator(p1_exact)) && isone(denominator(p2_exact))) ||
                error("s = [$s1,$s2] does not have denominator dividing d-bar = $dm ",
                      "(s*d-bar = [$p1_exact, $p2_exact]) - cannot express as an integer ",
                      "vector over d-bar as expected")
            p_vec = [Int(numerator(p1_exact)), Int(numerator(p2_exact))]

            conj_frame = fiber_cache[u_B].conj_frame
            val = shin_value_in_frame(A, rho_B, D, p_vec, dm, conj_frame)
            table[idx] = (J = Jfull, s1 = s1, s2 = s2, rho_B = rho_B, shin_value = val,
                          conj_frame = conj_frame)
            is_anchor && println(stderr, "  [table] triple ", idx, " done in ",
                                 round(time() - t_triple, digits = 1), " s")
        catch e
            println(stderr, "  [table] triple ", idx, " FAILED after ",
                    round(time() - t_triple, digits = 1), " s: ", sprint(showerror, e))
            table[idx] = nothing
        end
    end

    return table
end

"""
    ray_class_shin_table_fast_quiet(F::AdmissibleTuple; kwargs...)

Genuinely-silent public entry point for `_ray_class_shin_table_fast_quiet_impl`
(same keyword arguments). "Quiet" means quiet: this wraps the call in
`redirect_stdout` so that even output from functions it calls internally
(e.g. `ray_class_group_representatives`, which prints its own diagnostics -
including a full listing of every Picard-class representative - regardless
of any caller's verbosity preference) is suppressed, not just this file's
own `println` calls.
"""
function ray_class_shin_table_fast_quiet(F::AdmissibleTuple; kwargs...)
    result = Ref{Any}(nothing)
    redirect_stdout(devnull) do
        result[] = _ray_class_shin_table_fast_quiet_impl(F; kwargs...)
    end
    return result[]
end

ray_class_shin_table_fast_quiet(d::Int) = ray_class_shin_table_fast_quiet(AdmissibleTuple(d))

"""
    check_table_units(d::Int)

For every table entry, forms the integral representative
`e = d*(s2*rho_B - s1)` (an element of the lattice `I_B = Z+Z*rho_B`) and
tests whether it is a UNIT modulo `d` in `I_B/(d*I_B)`, via the relative
norm `N(e)/N(I_B)` (an integer since `(e) = I_B * C` with `C` integral):
`e` is a unit mod `d` iff `gcd(N(e)/N(I_B), d) == 1`. Ray classes of
modulus `d` correspond exactly to unit residues, so a correctly built table
should report 288/288 units. Grid points `p` whose own element
`x_p = p2*rho - p1` is NOT a unit mod `d` have no modulus-`d` ray class and
therefore cannot appear in this table at all.
"""
function check_table_units(d::Int)
    F = AdmissibleTuple(d)
    sd = _ray_class_shin_setup(F)
    K, O = sd.K, sd.O
    tbl = ray_class_shin_table_fast_quiet(F; shared_data = sd)
    n_unit = 0; n_nonunit = 0
    for (i, e_) in enumerate(tbl)
        e_ === nothing && continue
        I_B = fractional_ideal(O, [K(1), e_.rho_B])
        e = K(d) * (K(e_.s2) * e_.rho_B - K(e_.s1))
        rel = QQ(abs(norm(e))) / QQ(norm(I_B))
        ok = isone(denominator(rel)) && gcd(Int(numerator(rel)), d) == 1
        ok ? (n_unit += 1) : (n_nonunit += 1)
        ok || println("  entry ", i, ": NON-UNIT residue, N(e)/N(I_B) = ", rel,
                      "  J = ", e_.J, "  s1 = ", e_.s1, "  s2 = ", e_.s2)
    end
    println("check_table_units(", d, "): ", n_unit, " unit residues, ", n_nonunit,
            " non-unit residues, out of ", count(x -> x !== nothing, tbl), " entries.")
    return tbl
end

"""
    check_table_conjugation_closure(d::Int; tol::Float64 = 1e-10)

Complex conjugation is an element of Gal(ray class field / K), so it permutes
ray classes; hence the SET of group-table shin values must be closed under
z -> conj(z). For every entry, looks for an entry whose value is conj(its
value) and reports closure per Picard fiber (and per sign of N(gamma) of the
entry's principal part, since orientation is what a mixed-sign generator
changes). A fiber that is NOT closed cannot be a correct set of ray-class
invariants. Added 9/15 after test_ray_class_covariance(15;0,1) showed the
F-fiber anchor value is EXACTLY conj(shin_lookup) with det(gamma) = -1 and
N(mu) = -47 < 0, i.e. the anchor root rho_B is in the right wide Picard class
but the opposite NARROW class (orientation) relative to rho_F.
"""
function check_table_conjugation_closure(d::Int; tol::Float64 = 1e-10)
    F = AdmissibleTuple(d)
    sd = _ray_class_shin_setup(F)
    K, O, mP_O = sd.K, sd.O, sd.mP_O
    tbl = ray_class_shin_table_fast_quiet(F; shared_data = sd)
    vals = [e === nothing ? nothing : Complex{BigFloat}(e.shin_value) for e in tbl]
    # For each entry i, find entries j with |v_j| == |v_i| (modulus match), then
    # classify the ratio conj(v_i)/v_j: +1 (literal closure), -1 (closure up to
    # sign), or something else (printed as a phase in units of 2*pi/(2*d*(d-2)),
    # the natural root-of-unity scale of the ghost phase factors).
    n_plus = 0; n_minus = 0; n_other = 0; n_none = 0; n_total = 0
    per_fiber = Dict{Any,Vector{Int}}()   # cls => [plus, minus, other, none]
    for (i, v) in enumerate(vals)
        v === nothing && continue
        n_total += 1
        cls = mP_O \ tbl[i].J
        st = get!(per_fiber, cls, [0, 0, 0, 0])
        c = conj(v)
        js = [j for (j, w) in enumerate(vals) if w !== nothing && abs(abs(w) - abs(c)) < tol * abs(c)]
        if isempty(js)
            n_none += 1; st[4] += 1
            println("  entry ", i, " (class ", cls, "): no entry with matching modulus |v| = ", Float64(abs(v)))
            continue
        end
        found = false
        for j in js
            r = c / vals[j]
            if abs(r - 1) < tol
                n_plus += 1; st[1] += 1; found = true; break
            elseif abs(r + 1) < tol
                n_minus += 1; st[2] += 1; found = true; break
            end
        end
        if !found
            n_other += 1; st[3] += 1
            j = js[1]
            r = c / vals[j]
            ph = Float64(angle(r)) / (2 * pi) * (2 * d * (d - 2))
            println("  entry ", i, " (class ", cls, "): conj(v_i)/v_", j, " = ", Float64(real(r)), " + ",
                    Float64(imag(r)), "im  (|r| = ", Float64(abs(r)), ", phase = ", ph,
                    " in units of 2pi/(2d(d-2)))")
        end
    end
    println("check_table_conjugation_closure(", d, "): of ", n_total, " entries, conj(v) is in the table:")
    println("  exactly (+1): ", n_plus, "   up to sign (-1): ", n_minus,
            "   other ratio: ", n_other, "   no modulus match at all: ", n_none)
    for (cls, st) in sort(collect(per_fiber), by = x -> string(x[1]))
        println("  fiber ", cls, ": +1: ", st[1], "  -1: ", st[2], "  other: ", st[3], "  none: ", st[4])
    end
    return tbl
end

"""
    diagnose_orientation_flip(d::Int; p1::Int = 0, p2::Int = 1, tol::Float64 = 1e-10)

Tests the orientation hypothesis directly at one point. Takes the F-fiber
ANCHOR entry (rho_B, s1, s2) of the group table and re-evaluates its shin
value with the lattice basis orientation reversed:
    rho_B -> -rho_B,  s2 -> -s2  (so s2*rho_B - s1 is the SAME element),
    L, A recomputed for -rho_B.
Prints the original table value, its conjugate, the lookup value
shin^{[p1,p2]}_{A_lookup}(rho_F) that twisted_chi_general fails to find, and
the flipped value - and whether the flipped value equals the lookup value
and/or conj(table value). Also reports, for EVERY anchor in the table, the
sign of N(mu) for mu*(Z+Z*rho_B) = Z+Z*rho_rep (rho_rep = rho_reps[class])
when that quotient is principal, i.e. whether each fiber's anchor is
oriented consistently (det +1) with that class's reference root.
"""
function diagnose_orientation_flip(d::Int; p1::Int = 0, p2::Int = 1, tol::Float64 = 1e-10)
    F = AdmissibleTuple(d)
    sd = _ray_class_shin_setup(F)
    D, K, O, mP_O, rho_reps, F_class = sd.D, sd.K, sd.O, sd.mP_O, sd.rho_reps, sd.F_class
    tbl = ray_class_shin_table_fast_quiet(F; shared_data = sd)
    trivial_I_O = ideal(O, 1)

    rho = rho_reps[F_class]
    rho_bf = BigFloat(coeff(rho, 0)) + BigFloat(coeff(rho, 1)) * sqrt(BigFloat(D))
    L_rho = hj_automorph_for_root(rho, D)
    A_lookup, _ = stabilizer_power_in_Gamma_s(L_rho, [QQ(p1, d), QQ(p2, d)])
    val_lookup = Complex{BigFloat}(shin(A_lookup, rho_bf, [p1, p2], d))
    println("rho_F = ", rho, "   lookup value shin^[", p1, ",", p2, "](rho_F) = ", val_lookup)

    # anchor orientation survey
    println("\n--- anchor orientation survey (sign N(mu), mu*(Z+Z*rho_B) = Z+Z*rho_rep) ---")
    for (i, e) in enumerate(tbl)
        e === nothing && continue
        sd.all_triples[i][2] == trivial_I_O || continue
        cls = mP_O \ e.J
        rho_rep = rho_reps[-cls]   # class of the LATTICE Z+Z*rho_B is the inverse of J's class
        # inverse via the conjugate/norm trick (Hecke's inv() is avoided on the
        # non-maximal order everywhere else in this file, for the same reason)
        LB_gens = [K(1), e.rho_B]
        NB = norm(fractional_ideal(O, LB_gens))
        LB_inv = fractional_ideal(O, [quad_conj(K, g) // NB for g in LB_gens])
        Q = fractional_ideal(O, [K(1), rho_rep]) * LB_inv
        okp, mu = try
            is_principal_O_verified(K, O, [_nf_elem_of(K, x) for x in basis(Q)])
        catch err
            (false, nothing)
        end
        if okp
            nmu = norm(K(mu))   # rational in general (rho_rep may have denominators)
            println("  anchor ", i, " class ", cls, ": rho_B = ", e.rho_B,
                    "  N(mu) = ", nmu, "  => det = ", nmu > 0 ? "+1" : "-1")
        else
            println("  anchor ", i, " class ", cls, ": rho_B = ", e.rho_B,
                    "  (quotient with rho_rep not verified principal - skipped)")
        end
    end

    # flip experiment on the F-fiber anchor
    idx = findfirst(i -> tbl[i] !== nothing && sd.all_triples[i][2] == trivial_I_O &&
                         (mP_O \ tbl[i].J) == F_class, 1:length(tbl))
    idx === nothing && error("no F-fiber anchor entry found")
    e = tbl[idx]
    val_table = Complex{BigFloat}(e.shin_value)
    println("\nF-fiber anchor entry ", idx, ": J = ", e.J, "  rho_B = ", e.rho_B,
            "  s1 = ", e.s1, "  s2 = ", e.s2)
    println("  table value        = ", val_table)
    println("  conj(table value)  = ", conj(val_table))

    # Where does the lookup value sit in the (frame-corrected) table? The point
    # (p1,p2) = (0,1) at rho_F is the unit rho_F itself, i.e. the TRIVIAL ray
    # class, whose table entry is NOT the anchor (the anchor <47,...> = (1+2*sqrt12)
    # is a different ray class) - so search all entries.
    hits = [i for (i, t) in enumerate(tbl) if t !== nothing &&
            abs(Complex{BigFloat}(t.shin_value) - val_lookup) < tol]
    hits_conj = [i for (i, t) in enumerate(tbl) if t !== nothing &&
                 abs(Complex{BigFloat}(t.shin_value) - conj(val_lookup)) < tol]
    println("\n  entries whose value == lookup value:       ", hits,
            isempty(hits) ? "" : "   J = " * string(tbl[hits[1]].J) *
                                  "  gamma = " * string(sd.all_triples[hits[1]][3]))
    println("  entries whose value == conj(lookup value): ", hits_conj,
            isempty(hits_conj) ? "" : "   J = " * string(tbl[hits_conj[1]].J) *
                                       "  gamma = " * string(sd.all_triples[hits_conj[1]][3]))
    val_flip = val_table
    return (val_lookup = val_lookup, val_table = val_table, val_flip = val_flip)
end

"""
    ray_class_monoid_shin_table(F; shared_data = nothing, group_table = nothing)

The full table of KFA eq. (2.55): for m = dO, the image of Υ_m on the ray
class MONOID Cl^♭ (Kopp–Lagarias, ideals NOT required to be coprime to m) is
GL_2(Z)-orbits on (1/d)Z^2/Z^2 × K_O. Orbits of a point r = p/d under the stabilizer
of ρ_B are orbits under ⟨L_ε⟩ (L_ε = matrix of the totally positive
fundamental unit; -I acts trivially since sgn(j_{-I}) = -1 cancels it), acting
linearly on the column vector p exactly as in stabilizer_power_in_Gamma_s.
For each Picard fiber's ρ_B (read off the group table, which uses one ρ_B per
fiber) and each ⟨L_ε⟩-orbit of Z_d^2 \\ {0}, the shin value is computed exactly
as the group table computes its own entries: A = stabilizer_power_in_Gamma_s(L, p/d),
val = shin(A, ρ_B, p, d). Unit orbits (72 per fiber here) reproduce the group
table; the remaining orbits are the non-unit classes the group table lacks.
Returns a Vector of NamedTuples (rho_B, p_vec, orbit, is_unit, shin_value).
"""
function ray_class_monoid_shin_table(F::AdmissibleTuple; shared_data = nothing, group_table = nothing)
    sd = shared_data === nothing ? _ray_class_shin_setup(F) : shared_data
    D, d, K, O = sd.D, sd.d, sd.K, sd.O
    tbl = group_table === nothing ? ray_class_shin_table_fast_quiet(F; shared_data = sd) : group_table
    rhoBs = unique([e.rho_B for e in tbl if e !== nothing])
    entries = NamedTuple[]
    for rho_B in rhoBs
        # honour the group table's per-fiber frame (see fiber_orientation_sign)
        e_first = tbl[findfirst(e -> e !== nothing && e.rho_B == rho_B, tbl)]
        cf = hasproperty(e_first, :conj_frame) ? e_first.conj_frame : false
        rho_B_bf = root_bf_in_frame(rho_B, D, false)
        L = hj_automorph_for_root(rho_B, D)
        Lpos = (L[1,1] + L[2,2]) > 0 ? L : -L      # matrix of the totally positive unit
        I_B = fractional_ideal(O, [K(1), rho_B])
        NI_B = QQ(norm(I_B))
        seen = falses(d, d)
        for p1 in 0:d-1, p2 in 0:d-1
            (p1 == 0 && p2 == 0) && continue
            seen[p1+1, p2+1] && continue
            orbit = Tuple{Int,Int}[]
            q = [p1, p2]
            for _ in 1:10_000
                push!(orbit, (q[1], q[2]))
                seen[q[1]+1, q[2]+1] = true
                q = mod.(Lpos * q, d)
                (q[1] == p1 && q[2] == p2) && break
            end
            A, _ = stabilizer_power_in_Gamma_s(Lpos, [QQ(p1, d), QQ(p2, d)])
            val = shin(A, rho_B_bf, [p1, p2], d)
            cf && (val = conj(val))                 # frame flip, see shin_value_in_frame
            x = K(p2) * rho_B - K(p1)
            rel = QQ(abs(norm(x))) / NI_B
            is_unit = isone(denominator(rel)) && gcd(Int(numerator(rel)), d) == 1
            push!(entries, (rho_B = rho_B, p_vec = [p1, p2], orbit = orbit,
                            is_unit = is_unit, shin_value = val))
        end
    end
    return entries
end

"""
    diagnose_real_places(d::Int)

Determines which of `real_places(K)[1]`, `real_places(K)[2]` is CONJUGATION
(the paper's infty_2, the one place making up Sigma) and which is the
identity embedding (infty_1).

Test used: `gen(K) = sqrt(D)` is positive under the identity embedding and
negative under conjugation, so the place whose embedding sends sqrt(D) to a
NEGATIVE real is conjugation = infty_2.

Also reports, for every Picard-class root rho currently in use, whether
`sigma_1(rho) > sigma_2(rho)` (the theorem's `rho > rho'` normalization), and
cross-checks the two places actually used in the code:
  - `Sigma = [all_places[2]]`   (ray class group construction)
  - `emb_conj_K = embedding(places_K_top[1])`  (twisted_chi_general's
    `u2*conj(rho_j) - u1 > 0` positivity check)
These two MUST refer to the same place; this prints whether they do.
"""
function diagnose_real_places(d::Int)
    F = AdmissibleTuple(d)
    sd = _ray_class_shin_setup(F)
    K, D = sd.K, sd.D
    a = gen(K)
    places = real_places(K)
    println("K = ", K, "   gen(K) = sqrt(", D, ")")
    i_conj = 0
    for (i, p) in enumerate(places)
        v = Float64(real(embedding(p)(a)))
        isconj = v < 0
        isconj && (i_conj = i)
        println("  real_places(K)[", i, "]:  sqrt(", D, ") -> ", v, "   => ",
                isconj ? "CONJUGATION  = infty_2  (this is Sigma)" :
                         "identity     = infty_1")
    end
    i_conj == 0 && error("diagnose_real_places: neither place sends sqrt(D) to a negative ",
                          "number - cannot identify conjugation")
    i_main = 3 - i_conj
    println("\n=> conjugation (infty_2, Sigma) is real_places(K)[", i_conj, "]")
    println("=> identity    (infty_1)        is real_places(K)[", i_main, "]")

    emb_conj = embedding(places[i_conj])
    emb_main = embedding(places[i_main])

    println("\nChecking rho > rho' (i.e. sigma_1(rho) > sigma_2(rho)) for the class roots:")
    for (g, rho) in sd.rho_reps
        v_main = Float64(real(emb_main(rho)))
        v_conj = Float64(real(emb_conj(rho)))
        println("  class ", g, ": rho = ", rho, "   sigma_1 = ", v_main,
                "  sigma_2 = ", v_conj, "   rho > rho'? ", v_main > v_conj)
    end

    println("\nConsistency of the two places the code currently hard-codes:")
    println("  Sigma       = [all_places[2]]        -> places[2] ", i_conj == 2 ? "IS" : "is NOT",
            " conjugation")
    println("  emb_conj_K  = embedding(places[1])   -> places[1] ", i_conj == 1 ? "IS" : "is NOT",
            " conjugation")
    if i_conj == 2
        println("  ==> MISMATCH: Sigma is right (places[2]), but twisted_chi_general's ",
                "`emb_conj_K = embedding(places_K_top[1])` is picking the IDENTITY place, so ",
                "its `u2*conj(rho_j)-u1 > 0` test is imposing positivity at the WRONG place.")
    else
        println("  ==> MISMATCH: emb_conj_K is right (places[1]), but `Sigma = [all_places[2]]` ",
                "is selecting the IDENTITY place, so the ray class group is being built with ",
                "the WRONG Sigma.")
    end
    println("  (Exactly one of these two is wrong - they cannot both be conjugation.)")

    return (i_conj = i_conj, i_main = i_main)
end

"""
    ray_class_monoid_table_v2(F::AdmissibleTuple; shared_data = nothing, verbose = true)

Builds the ray class monoid shin table DIRECTLY from the definition of
`Cl^flat_{m,Sigma}(O)`, rather than assuming (as `ray_class_monoid_shin_table`
does) that the classes are the `<L_eps>`-orbits on `(Z/d)^2`.

Construction, following Theorem 2.19 with `m = dO`:

  For each `A_0 in Cl(O)`:
    1. pick `b` in the class `A_0^{-1}`, integral and COPRIME to `m`
       (search: `b = x * I^{-1}` for `x in I = mP_O(A_0)` with
       `|N(x)|/N(I)` coprime to `d`);
    2. write `b*m = d*b = alpha*(rho*Z + Z)` with `alpha` TOTALLY POSITIVE
       and `rho > rho'` (search over unimodular changes of the Z-basis of
       `d*b`, then fix `alpha`'s sign class by a unit multiple);
    3. `alpha*(r2*rho - r1) in b  <=>  r in (1/d)Z^2`, so the grid is
       `r = p/d`, `p in (Z/d)^2`; `p = (0,0)` is exactly the zero class
       (`a subset m`), which Theorem 2.19 excludes, so it is skipped;
    4. for each `p`, choose the lift `r in p/d + Z^2` with
       `r2*rho' - r1 > 0` (Theorem 2.19's Sigma-sign normalization).
       NOTE: `ray_class_monoid_shin_table` never does this - it passes
       `[p1,p2]` with `p_i in [0,d)`, i.e. `r in [0,1)^2`, for which
       `r2*rho' - r1 <= 0` at essentially every point since `rho' < 0`.
    5. `gamma_r = alpha*(r2*rho - r1) in b`, and `a_r = gamma_r * b^{-1}`.

  Then group the `p`'s by the equivalence relation ITSELF:

      a_r ~_{m,Sigma} a_r'  <=>  exists c in Jbar^flat_m(O) and
                                 alpha_e, beta_e in O[S_m^{-1}] with
                                 a_r = alpha_e*c, a_r' = beta_e*c,
                                 alpha_e - beta_e in m*O[S_m^{-1}],
                                 sgn(sigma(alpha_e)) = sgn(sigma(beta_e))
                                 for all sigma in Sigma.

  This is implemented via the following reduction, which is proved (not
  assumed) as follows. `c` must lie in the class `[b^{-1}]`, so `c*b` is
  principal, say `c = delta*b^{-1}`. Locally at any `p | m` all of
  `a_r, a_r', c` are principal (invertible modules over a local ring are
  free), say `(A), (B), (C)`; then `alpha_e = A/C`, `beta_e = B/C` up to
  units, and `alpha_e - beta_e in dO_p` gives `(A-B)/C in dO_p`, hence
  `A - B in C*dO_p subset dO_p`. So ANY working `c` implies `c = b^{-1}`
  (i.e. `C = 1`) also works - no valuation theory and no DVR assumption is
  needed, so the singular prime at the conductor `q` causes no trouble.
  With `c = b^{-1}` we get `alpha_e = u_1*gamma_r`, `beta_e = u_2*gamma_r'`
  for units `u_i`, and multiplying through by `u_1^{-1}` leaves

      a_r ~ a_r'  <=>  exists a unit u in O^*  with
                       gamma_r  ==  u*gamma_r'   (mod dO)
                and  sgn_Sigma(gamma_r) == sgn_Sigma(u*gamma_r').

  Since every `gamma_r` here is Sigma-positive by the step-4 normalization,
  the sign condition reduces to "u is Sigma-positive".

  The congruence is tested EXACTLY (not numerically): `x in dO` iff every
  coordinate of `x` w.r.t. the Z-basis of `O` is divisible by `d`.

Returns a vector of NamedTuples, one per monoid class:
`(A0, rho, alpha, J, p_vec, r_num, members, is_unit, shin_value)` where `J`
is the actual ideal a_r = gam[pt]*b_O^{-1} this class represents (an
integral ideal, class A0 - see the b_O computation above for why), suitable
for direct use as `U_elem1` in twisted_chi_general's downstream
`J_bj = J*U_elem1` multiplication; `members` is the full list of grid
points in that class and `r_num` is the integer numerator vector of the
NORMALIZED lift actually fed to `shin`.

Confirmed (2025 session, `diagnose_full_monoid_membership`): every genuine
shin_val_lookup value DOES appear somewhere in this table when the full
592-class (d=21) table is built and searched directly, rather than via
`ray_class_monoid_lookup`'s incremental norm-bound search - the search was
never actually necessary; the monoid is small enough to build in full,
once, up front.
"""
function ray_class_monoid_table_v2(F::AdmissibleTuple; shared_data = nothing,
                                    verbose::Bool = true, basis_height::Int = 8)
    sd = shared_data === nothing ? _ray_class_shin_setup(F) : shared_data
    D, d, K, O = sd.D, sd.d, sd.K, sd.O
    P_O, mP_O = sd.P_O, sd.mP_O
    rho_reps, all_triples = sd.rho_reps, sd.all_triples
    # EVEN d (9/18): the monoid is Jbar^flat_m(O) with m = d-bar*O; every
    # residue, grid point, lift and value below is at modulus d-bar.
    dm = hasproperty(sd, :dbar) ? sd.dbar : (isodd(d) ? d : 2 * d)

    places = real_places(K)
    i_conj = findfirst(p -> Float64(real(embedding(p)(gen(K)))) < 0, places)
    i_conj === nothing && error("ray_class_monoid_table_v2: could not identify conjugation")
    emb_conj = embedding(places[i_conj])          # sigma_2 = Sigma
    emb_main = embedding(places[3 - i_conj])      # sigma_1
    fm(x) = Float64(real(emb_main(x)))
    fc(x) = Float64(real(emb_conj(x)))

    m = ideal(O, dm)

    # x in mO  <=>  all Z-basis coordinates of x divisible by d-bar
    function in_dO(x)
        try
            return all(iszero(mod(c, ZZ(dm))) for c in coordinates(O(x)))
        catch
            return false
        end
    end

    entries = NamedTuple[]

    rho_base = sd.rho_reps[sd.F_class]
    d_check = sd.d_check

    for g in P_O
        # ---- steps 1-2: (alpha, rho) for this Picard class, Theorem 2.19 ----
        # BUGFIX (twice over). This block originally (a) built its own
        # coprime-to-m representative `b` of the class g^{-1} by hand, and
        # (b) derived (alpha, rho) from `b` by an ad-hoc unimodular basis
        # search. Both were wrong:
        #   (a) `numerator(x * inv(I))` does NOT return the ideal x*I^{-1}.
        #       Hecke stores a fractional ideal as num/den, and `numerator`
        #       hands back `num` = den*(x*I^{-1}) - an integer multiple of the
        #       intended ideal whenever den != 1 - silently corrupting `b` and
        #       invalidating the coprimality test performed just above it.
        #       Symptom: find_alpha_rho_direct exhausting its (m,n) search
        #       ("no (alpha, rho_B) found within mn_height=3000") on a lattice
        #       it should have normalized immediately.
        #   (b) "alpha totally positive" + "rho > rho'" do not pin rho down -
        #       every GL2(Z) translate satisfies them too, and almost none are
        #       HJ-reduced, so `shin` failed with "t.Q is not strictly
        #       HJ-reduced".
        # Fix for both: don't construct any of this by hand. Ask
        # find_alpha_rhoB_s1_s2_O - the exact routine the (working) group table
        # uses - for this class's (alpha, rho_B). It internally picks a
        # verified b_O via find_b_with_verified_principal_product and runs
        # find_alpha_rho_direct on it (which lattice-reduces first, hence gives
        # an HJ-reducible rho_B), and it enforces precisely Theorem 2.19's
        # normalization: b*d*O = alpha*(Z*rho_B+Z), alpha totally positive,
        # rho_B > conj(rho_B). Per its own docs alpha/rho_B depend only on the
        # Picard class of J, not on which J in that class is passed.
        Jg = nothing
        for t in all_triples
            if (mP_O \ t[1]) == g
                Jg = t[1]
                break
            end
        end
        Jg === nothing && error("ray_class_monoid_table_v2: no ray class representative ideal ",
                                "found in Picard class ", g)
        alpha, rho, _, _ = find_alpha_rhoB_s1_s2_O(Jg, rho_reps[g], rho_base, g, P_O, mP_O,
                                                    O, K, dm, rho_base, d_check; d1 = d + 1,
                                                    mn_height = 3000, q_gen_height = 3000)
        (fm(alpha) > 0 && fc(alpha) > 0) ||
            error("ray_class_monoid_table_v2: non-totally-positive alpha = ", alpha,
                  " for Picard class ", g)
        fm(rho) > fc(rho) ||
            error("ray_class_monoid_table_v2: rho = ", rho, " has rho <= rho' for Picard class ", g)

        # Same orientation correction as the group table (see fiber_orientation_sign):
        # only the root handed to `shin` changes; the residue/lift bookkeeping
        # below (rho_c, Sigma-positive units, a_r) is ideal-level and stays.
        # lattice Z+Z*rho is in the class of b_O = inverse of J_g's class -> reference rho_reps[-g]
        osign_g, mu_g = fiber_orientation_sign(K, O, rho, rho_reps[-g])
        if osign_g === nothing
            verbose && println("  [monoid v2] class ", g, ": WARNING could not determine orientation - assuming det = +1")
            osign_g = 1
        end
        conj_frame_g = (osign_g == -1)
        verbose && osign_g !== nothing && mu_g !== nothing &&
            println("  [monoid v2] class ", g, ": N(mu) = ", norm(K(mu_g)),
                    "  orientation det = ", osign_g,
                    conj_frame_g ? "  -> conjugating this class's values (frame flip)" : "")
        rho_bf = root_bf_in_frame(rho, D, false)
        rho_c = fc(rho)

        # b_O for this class, needed to construct an actual usable IDEAL
        # a_r = gam[pt]*b_O^{-1} per entry (Theorem 2.19: a_r's class = g,
        # since gam[pt] is principal and b_O has class g^{-1}). This is the
        # same find_b_with_verified_principal_product call
        # find_alpha_rhoB_s1_s2_O already makes internally to get alpha/rho
        # - it does not expose b_O in its return, so this repeats that one
        # search per class (4 times total here), the same established
        # pattern already used for this exact situation in
        # ray_class_monoid_lookup's `find_with_class_cache` and in
        # ray_class_shin_table's `fiber_cache`.
        b_O, _ = find_b_with_verified_principal_product(O, K, Jg, [ideal(O, dm), ideal(O, d + 1)])
        NbO = QQ(norm(b_O))
        bO_gens = [_nf_elem_of(K, x) for x in basis(b_O)]
        bO_inv_gens = [quad_conj(K, bg) // NbO for bg in bO_gens]   # b_O^{-1} = conj(b_O)/N(b_O)

        # ---- units of O, and the Sigma-positive ones, as residues mod dO ----
        # BUGFIX: hj_automorph_for_root can return either L or -L (both
        # represent the same automorphism of rho, since -I acts trivially on
        # rho). ray_class_monoid_shin_table normalizes this to the matrix of
        # the TOTALLY POSITIVE unit before using it anywhere (Lpos = trace>0
        # ? L : -L) - that normalization was missing here, and the wrong
        # sign choice produces a stabilizer matrix that is not strictly
        # HJ-reduced, which is exactly what `shin` errored on. Apply the
        # same normalization here.
        L_raw = hj_automorph_for_root(rho, D)
        L = (L_raw[1,1] + L_raw[2,2]) > 0 ? L_raw : -L_raw
        eps_u = L[2,1] * rho + L[2,2]              # the unit with L*(rho,1)^T = eps*(rho,1)^T
        units = Any[]
        cur = one(K)
        for _ in 1:10_000
            for sgn in (1, -1)
                u = sgn * cur
                fc(u) > 0 && push!(units, u)        # Sigma-positive units only
            end
            cur = cur * eps_u
            in_dO(cur - one(K)) && break            # eps^k == 1 mod dO: full residue cycle
        end
        isempty(units) && error("ray_class_monoid_table_v2: no Sigma-positive units found")

        # ---- steps 3-5: build gamma_r for every grid point, with the
        #      Theorem 2.19 lift normalization r2*rho' - r1 > 0 ----
        gam = Dict{Tuple{Int,Int}, Any}()
        rnum = Dict{Tuple{Int,Int}, Vector{Int}}()
        for p1 in 0:dm-1, p2 in 0:dm-1
            (p1 == 0 && p2 == 0) && continue        # the zero class; excluded by Thm 2.19
            r2 = QQ(p2, dm)
            # pick n1 so that r1 = p1/d-bar + n1 satisfies r2*rho' - r1 > 0
            n1 = Int(floor(Float64(r2) * rho_c - p1 / dm)) - 1
            r1 = QQ(p1, dm) + n1
            @assert Float64(r2) * rho_c - Float64(r1) > 0
            gam[(p1, p2)] = alpha * (r2 * rho - r1)
            rnum[(p1, p2)] = [p1 + dm * n1, p2]
        end

        # ---- group by the equivalence relation ----
        # The Sigma-positive units form a GROUP, so "gamma_p == u*gamma_q mod dO
        # for some Sigma-positive u" is an equivalence whose classes are the
        # orbits of that group acting on O/dO. So instead of testing all
        # O(d^4) pairs, compute each point's orbit as a canonical key (the
        # sorted set of coordinate pairs mod d of u*gamma_p over all u) and
        # group by that key - O(d^2 * #units) instead.
        function coord_key(x)
            c = coordinates(O(x))
            return (Int(mod(c[1], ZZ(dm))), Int(mod(c[2], ZZ(dm))))
        end
        keyof = Dict{Tuple{Int,Int}, Any}()
        for (pt, gx) in gam
            keyof[pt] = sort!([coord_key(u * gx) for u in units])
        end
        buckets = Dict{Any, Vector{Tuple{Int,Int}}}()
        for (pt, k) in keyof
            push!(get!(buckets, k, Tuple{Int,Int}[]), pt)
        end
        classes = [sort!(v) for v in values(buckets)]

        # unit test, using the same convention as ray_class_monoid_shin_table
        # and twisted_chi_general: x_p = p2*rho - p1, normalized by the norm of
        # the lattice (1, rho). (The old form of this used N(b), but `b` is no
        # longer built by hand here - see the BUGFIX note above.)
        I_B = fractional_ideal(O, [K(1), rho])
        NI_B = QQ(norm(I_B))
        for members in classes
            pt = members[1]
            rn = rnum[pt]
            A, _ = stabilizer_power_in_Gamma_s(L, [QQ(rn[1], dm), QQ(rn[2], dm)])
            val = shin(A, rho_bf, rn, dm)
            conj_frame_g && (val = conj(val))       # frame flip, see shin_value_in_frame
            x = K(pt[2]) * rho - K(pt[1])
            rel = QQ(abs(norm(x))) / NI_B
            is_unit = isone(denominator(rel)) && gcd(Int(numerator(rel)), dm) == 1

            # a_r = gam[pt] * b_O^{-1}, the actual ideal this monoid class
            # represents (class = g, per Theorem 2.19 - see the b_O comment
            # above). Guaranteed integral by the theory (r is already in the
            # (1/d)Z^2 normalized form); convert to a genuine integral-typed
            # ideal (not just FractionalIdeal with denominator 1) so it is
            # type-consistent with the group table's own entries - the same
            # fix already applied in ray_class_monoid_lookup, for the exact
            # same downstream reason (mP_O \ (J*U_elem1) has no method for
            # FractionalIdeal).
            a_r_gens = [gam[pt] * bg for bg in bO_inv_gens]
            a_r = fractional_ideal(O, a_r_gens)
            isone(denominator(a_r)) ||
                error("ray_class_monoid_table_v2: a_r for class ", g, ", point ", pt,
                      " is not integral (denominator = ", denominator(a_r), ") - this should be ",
                      "impossible per Theorem 2.19's own normalization; something upstream is wrong")
            J_entry = numerator(a_r)

            push!(entries, (A0 = g, rho = rho, alpha = alpha, J = J_entry, p_vec = [pt[1], pt[2]],
                            r_num = rn, members = members, is_unit = is_unit,
                            shin_value = val))
        end

        if verbose
            println("Picard class ", g, ":  alpha = ", alpha, "  rho = ", rho)
            println("  Sigma-positive unit residues mod dO: ", length(units))
            println("  monoid classes in this fiber: ", length(classes),
                    "   (", count(c -> length(c) > 0, classes), " nonempty)")
        end
    end

    return entries
end

ray_class_monoid_table_v2(d::Int; kwargs...) =
    ray_class_monoid_table_v2(AdmissibleTuple(d); kwargs...)

"""
    compare_monoid_tables(d::Int; tol = 1e-10)

Runs both `ray_class_monoid_shin_table` (the `<L_eps>`-orbit construction)
and `ray_class_monoid_table_v2` (built from the definition of
`Cl^flat_{m,Sigma}(O)`), and reports:
  1. the class counts from each;
  2. whether the v2 classes coincide with the orbit partition (they should,
     if the reduction of the equivalence to "unit multiple mod dO" is right);
  3. the shin values, which will DIFFER wherever the old builder used an
     unnormalized lift (r in [0,1)^2 instead of r2*rho'-r1 > 0).
"""
function compare_monoid_tables(d::Int; tol::Float64 = 1e-10)
    F = AdmissibleTuple(d)
    sd = _ray_class_shin_setup(F)
    old = ray_class_monoid_shin_table(F; shared_data = sd)
    new = ray_class_monoid_table_v2(F; shared_data = sd, verbose = false)
    println("orbit-based table: ", length(old), " rows;  definition-based table: ",
            length(new), " rows")
    n_same_part = 0; n_diff_part = 0
    for e in new
        ms = Set(e.members)
        hit = findfirst(o -> Set(o.orbit) == ms, old)
        hit === nothing ? (n_diff_part += 1) : (n_same_part += 1)
    end
    println("v2 classes matching an orbit exactly: ", n_same_part,
            ";  not matching any orbit: ", n_diff_part)
    if n_diff_part == 0
        println("  => the equivalence relation IS the <L_eps>-orbit relation (partition agrees).")
    else
        println("  => the partitions DIFFER - the orbit construction is not the right ",
                "equivalence.")
    end
    n_val_same = 0; n_val_diff = 0
    for e in new
        hit = findfirst(o -> Set(o.orbit) == Set(e.members), old)
        hit === nothing && continue
        abs(old[hit].shin_value - e.shin_value) < tol ? (n_val_same += 1) : (n_val_diff += 1)
    end
    println("shin values equal on matching classes: ", n_val_same, ";  different: ", n_val_diff)
    n_val_diff > 0 && println("  => as expected if the old builder's unnormalized lift ",
                              "(r in [0,1)^2, r2*rho'-r1 <= 0) was giving the wrong shin argument.")
    return (old = old, new = new)
end

"""
    check_monoid_table(d::Int; tol = 1e-10)

Builds the group table and the (2.55) monoid table, then:
  1. checks every group entry (reduced point (d*s1, d*s2) mod d, same ρ_B)
     lies in some monoid orbit with the SAME shin value;
  2. reports unit / non-unit orbit counts per fiber (expect 72 unit orbits);
  3. computes shin^{[p1,p2]}_{A_lookup}(ρ) for the base root ρ = rho_reps[F_class]
     at the point (0,3) (the point that broke twisted_chi_general) and checks
     it IS present in the monoid table.
"""
function check_monoid_table(d::Int; tol::Float64 = 1e-10, p_test::Vector{Int} = [0, 3])
    F = AdmissibleTuple(d)
    sd = _ray_class_shin_setup(F)
    D, K = sd.D, sd.K
    tbl = ray_class_shin_table_fast_quiet(F; shared_data = sd)
    mono = ray_class_monoid_shin_table(F; shared_data = sd, group_table = tbl)
    println("group table: ", count(x -> x !== nothing, tbl), " entries;  monoid table: ",
            length(mono), " orbits (", count(e -> e.is_unit, mono), " unit, ",
            count(e -> !e.is_unit, mono), " non-unit)")
    for rho_B in unique([e.rho_B for e in mono])
        sub = filter(e -> e.rho_B == rho_B, mono)
        println("  fiber rho_B = ", rho_B, ": ", length(sub), " orbits, ",
                count(e -> e.is_unit, sub), " unit, ", count(e -> !e.is_unit, sub), " non-unit")
    end
    n_ok = 0; n_bad = 0
    for (i, e) in enumerate(tbl)
        e === nothing && continue
        pt = (Int(mod(numerator(e.s1 * d), d)), Int(mod(numerator(e.s2 * d), d)))
        hit = findfirst(m -> m.rho_B == e.rho_B && pt in m.orbit, mono)
        if hit === nothing
            n_bad += 1; println("  group entry ", i, ": point ", pt, " not found in any monoid orbit!")
        elseif abs(mono[hit].shin_value - e.shin_value) < tol
            n_ok += 1
        else
            n_bad += 1
            println("  group entry ", i, ": point ", pt, " in monoid orbit ", mono[hit].orbit,
                    " but shin differs: group ", e.shin_value, " vs monoid ", mono[hit].shin_value)
        end
    end
    println("group entries reproduced by the monoid table: ", n_ok, " ok, ", n_bad, " mismatched")

    rho = sd.rho_reps[sd.F_class]
    rho_bf = BigFloat(coeff(rho, 0)) + BigFloat(coeff(rho, 1)) * sqrt(BigFloat(D))
    L_rho = hj_automorph_for_root(rho, D)
    p1, p2 = p_test
    A_lookup, _ = stabilizer_power_in_Gamma_s(L_rho, [QQ(p1, d), QQ(p2, d)])
    val = shin(A_lookup, rho_bf, [p1, p2], d)
    hits = findall(m -> abs(m.shin_value - val) < tol, mono)
    println("lookup value shin^{[", p1, ",", p2, "]}_A(rho) = ", val)
    println("  found in monoid table? ", !isempty(hits),
            isempty(hits) ? "" : "  (orbit(s): " * join([string(mono[h].orbit, " rho_B=", mono[h].rho_B, " unit=", mono[h].is_unit) for h in hits], "; ") * ")")
    return (group = tbl, monoid = mono)
end

"""
    check_lookup_coverage(d::Int; tol = 1e-10)

For EVERY grid point p in Z_d^2 \\ {0}, computes the lookup value
shin^{p/d}_{A_lookup}(ρ) exactly as twisted_chi_general does (ρ = rho_reps[F_class],
the base root), then:
  1. predicts its trivial-fiber orbit via the translation γ = [1 -t; 0 1] with
     ρ = ρ_B0 + t (ρ_B0 = the trivial fiber's table root): p ↦ (p1 - t*p2, p2)
     mod d, and checks the monoid table's value at that orbit equals the
     lookup value (the covariance theorem, det γ = 1, s_γ = +1);
  2. for unit points, checks whether the lookup value is in the GROUP table.
Then reports: how many distinct (fiber, orbit) pairs the 288 group entries
actually cover (a bijection with the 288 unit orbits needs exactly 288), the
unit orbits with NO group entry (missing classes), and any points where the
theorem check fails. This distinguishes "the group enumeration repeats some
classes and misses others" from "the covariance theorem fails somewhere".
"""
function check_lookup_coverage(d::Int; tol::Float64 = 1e-10)
    F = AdmissibleTuple(d)
    sd = _ray_class_shin_setup(F)
    D, K = sd.D, sd.K
    tbl = ray_class_shin_table_fast_quiet(F; shared_data = sd)
    mono = ray_class_monoid_shin_table(F; shared_data = sd, group_table = tbl)

    # --- distinct orbits covered by the group table ---
    covered = Set{Tuple{Int,Int}}()      # (fiber index, orbit index)
    rhoBs = unique([e.rho_B for e in mono])
    orbit_index(rho_B, pt) = findfirst(m -> m.rho_B == rho_B && pt in m.orbit, mono)
    for e in tbl
        e === nothing && continue
        pt = (Int(mod(numerator(e.s1 * d), d)), Int(mod(numerator(e.s2 * d), d)))
        oi = orbit_index(e.rho_B, pt)
        oi === nothing && continue
        push!(covered, (findfirst(==(e.rho_B), rhoBs), oi))
    end
    println("group table: ", count(x -> x !== nothing, tbl), " entries covering ",
            length(covered), " distinct orbits (bijection with the ",
            count(m -> m.is_unit, mono), " unit orbits needs all of them)")
    missing_unit = [(i, m) for (i, m) in enumerate(mono)
                    if m.is_unit && !((findfirst(==(m.rho_B), rhoBs), i) in covered)]
    println("unit orbits with NO group-table entry: ", length(missing_unit))
    for (i, m) in missing_unit
        println("    fiber rho_B = ", m.rho_B, "  orbit ", m.orbit, "  shin_value = ", m.shin_value)
    end

    # --- per-point covariance + group-membership check with the base root ---
    rho = sd.rho_reps[sd.F_class]
    rho_bf = BigFloat(coeff(rho, 0)) + BigFloat(coeff(rho, 1)) * sqrt(BigFloat(D))
    L_rho = hj_automorph_for_root(rho, D)
    rho_B0 = rhoBs[argmin([abs(Float64(real(embedding(real_places(K)[2])(rho - rb)))) for rb in rhoBs])]
    t_shift = rho - rho_B0
    t_int = iszero(coeff(t_shift, 1)) && isone(denominator(QQ(coeff(t_shift, 0)))) ?
            Int(numerator(QQ(coeff(t_shift, 0)))) : nothing
    println("base root rho = ", rho, ";  trivial-fiber table root rho_B0 = ", rho_B0,
            ";  rho - rho_B0 = ", t_shift, t_int === nothing ? "  (NOT an integer translation - prediction skipped)" : "")
    n_thm_ok = 0; n_thm_bad = 0; n_unit_in_group = 0; n_unit_missing = 0
    missing_pts = Tuple{Int,Int}[]
    for p1 in 0:d-1, p2 in 0:d-1
        (p1 == 0 && p2 == 0) && continue
        A_lookup, _ = stabilizer_power_in_Gamma_s(L_rho, [QQ(p1, d), QQ(p2, d)])
        val = shin(A_lookup, rho_bf, [p1, p2], d)
        if t_int !== nothing
            q = (mod(p1 - t_int * p2, d), p2)
            oi = orbit_index(rho_B0, q)
            if oi !== nothing && abs(mono[oi].shin_value - val) < tol
                n_thm_ok += 1
            else
                n_thm_bad += 1
                println("  [theorem check FAILED] p=(", p1, ",", p2, ") predicted orbit point ", q,
                        oi === nothing ? " not found" : "  monoid value " * string(mono[oi].shin_value),
                        "  lookup value ", val)
            end
        end
        is_unit_p = false
        if t_int !== nothing
            oi = orbit_index(rho_B0, (mod(p1 - t_int * p2, d), p2))
            is_unit_p = oi !== nothing && mono[oi].is_unit
        end
        if is_unit_p
            in_group = any(e -> e !== nothing && abs(e.shin_value - val) < tol, tbl)
            in_group ? (n_unit_in_group += 1) : (n_unit_missing += 1; push!(missing_pts, (p1, p2)))
        end
    end
    println("covariance theorem (lookup with rho vs monoid at gamma-predicted orbit): ",
            n_thm_ok, " ok, ", n_thm_bad, " failed")
    println("unit grid points: ", n_unit_in_group, " found in group table, ", n_unit_missing, " missing")
    isempty(missing_pts) || println("  missing unit points: ", missing_pts)
    return (group = tbl, monoid = mono, missing_unit_orbits = missing_unit, missing_points = missing_pts)
end

"""
    diagnose_group_lookup_ambiguity(d::Int; tol = 1e-10)

Tests the leading hypothesis for the seed-dependent |v(p)v(-p)| failures
(9/12 summary, Section 0 / Section 8 item 1) WITHOUT any new twist run:

  The group-table lookup in twisted_chi_general is
      findfirst(e -> |e.shin_value - shin_val_lookup| < tol, table)
  i.e. a match BY VALUE ONLY, first hit wins. If two group-table entries share
  a shin value (the project owner confirmed nothing forbids this), findfirst
  returns the same entry for BOTH classes' points - correct for one class,
  wrong for the other. Which one is "first" depends on the table's ORDER,
  which depends on Hecke's randomized enumeration => seed-dependent, and a
  wrong entry => wrong elem1 => wrong du => the observed failure.

For every grid point p this computes an INDEPENDENT, value-free
identification of the correct entry - by ORBIT IDENTITY, not by value:
  * p's class lives in the base (trivial) Picard fiber, since x_p = p2*rho-p1
    is in O so (x_p) is principal.
  * With rho = rho_B0 + t (t an integer translation), p corresponds to the
    point q = (p1 - t*p2, p2) mod d in the table root rho_B0's coordinates
    (covariance theorem, det = 1, same convention as check_lookup_coverage).
  * The correct group entry is the one with rho_B == rho_B0 whose reduced
    point (d*s1, d*s2) mod d lies in the <L_eps>-orbit of q
    (orbit action q -> mod.(Lpos*q, d), exactly as ray_class_monoid_shin_table).

It then compares this orbit-identified entry against what findfirst returns,
and reports:
  - the duplicate shin-value groups in the group table (how many entries,
    and whether the duplicates are within one fiber or across fibers),
  - the cross-tab of {value is degenerate} x {findfirst picked the wrong
    entry}, and the list of wrong-pick points with their orbit structure.

PREDICTION being tested: the wrong-pick set has exactly the size of the
failing set from the run (90 for seed 208475639, twist 1), is a union of
whole <±eps>-orbits (15 orbits x 6), and every wrong pick is a degenerate
value. If confirmed, the fix is a value-free tiebreak in the lookup (choose
the entry by orbit identity, use the value only as a consistency check),
which is inherently seed-independent.

Run this with the SAME Random.seed! as the failing run (the seed is set at
the top of this file).
"""
function diagnose_group_lookup_ambiguity(d::Int; tol::Float64 = 1e-10)
    F = AdmissibleTuple(d)
    sd = _ray_class_shin_setup(F)
    D, K = sd.D, sd.K
    tbl = ray_class_shin_table_fast_quiet(F; shared_data = sd)
    idxs = [i for i in eachindex(tbl) if tbl[i] !== nothing]
    println("group table: ", length(idxs), " entries")

    # ---- 1. duplicate shin-value groups in the group table ----
    groups = Vector{Vector{Int}}()
    for i in idxs
        v = tbl[i].shin_value
        g = findfirst(G -> abs(tbl[G[1]].shin_value - v) < tol, groups)
        g === nothing ? push!(groups, [i]) : push!(groups[g], i)
    end
    dups = [G for G in groups if length(G) > 1]
    n_within = count(G -> length(unique([tbl[i].rho_B for i in G])) == 1, dups)
    println("distinct shin values: ", length(groups), "   duplicate groups: ", length(dups),
            "   (within a single fiber: ", n_within, ", spanning fibers: ", length(dups) - n_within, ")")
    println("entries involved in a duplicate group: ", sum(length, dups; init = 0))
    for (k, G) in enumerate(dups)
        fibers = [tbl[i].rho_B for i in G]
        println("  dup group ", k, ": entries ", G, "  size ", length(G),
                length(unique(fibers)) == 1 ? "  (same fiber rho_B = $(fibers[1]))" : "  (fibers: $(unique(fibers)))")
    end

    # ---- 2. orbit identity of each group entry in the base fiber ----
    rho = sd.rho_reps[sd.F_class]
    rho_bf = BigFloat(coeff(rho, 0)) + BigFloat(coeff(rho, 1)) * sqrt(BigFloat(D))
    L_rho = hj_automorph_for_root(rho, D)
    rhoBs = unique([tbl[i].rho_B for i in idxs])
    emb_main = embedding(real_places(K)[2])
    rho_B0 = rhoBs[argmin([abs(Float64(real(emb_main(rho - rb)))) for rb in rhoBs])]
    t_shift = rho - rho_B0
    (iszero(coeff(t_shift, 1)) && isone(denominator(QQ(coeff(t_shift, 0))))) ||
        error("rho - rho_B0 = $t_shift is not an integer translation; orbit identification needs it")
    t_int = Int(numerator(QQ(coeff(t_shift, 0))))
    println("\nbase root rho = ", rho, "   base-fiber table root rho_B0 = ", rho_B0, "   t = ", t_int)

    L0 = hj_automorph_for_root(rho_B0, D)
    Lpos = (L0[1,1] + L0[2,2]) > 0 ? L0 : -L0
    function orbit_of(pt)
        orb = Set{Tuple{Int,Int}}(); q = [pt[1], pt[2]]
        for _ in 1:10_000
            push!(orb, (q[1], q[2])); q = mod.(Lpos * q, d)
            (q[1] == pt[1] && q[2] == pt[2]) && break
        end
        orb
    end
    base_idxs = [i for i in idxs if tbl[i].rho_B == rho_B0]
    entry_orbit = Dict{Int,Set{Tuple{Int,Int}}}()
    for i in base_idxs
        e = tbl[i]
        pt = (Int(mod(numerator(e.s1 * d), d)), Int(mod(numerator(e.s2 * d), d)))
        entry_orbit[i] = orbit_of(pt)
    end
    println("base-fiber entries: ", length(base_idxs))

    # ---- 3. per grid point: findfirst pick vs orbit-identified entry ----
    n_deg = 0; n_wrong = 0; n_deg_wrong = 0; n_nodeg_wrong = 0
    wrong_pts = Tuple{Int,Int}[]; unresolved = Tuple{Int,Int}[]
    orbit_correct_mismatch_value = 0
    for p1 in 0:d-1, p2 in 0:d-1
        (p1 == 0 && p2 == 0) && continue
        A_lookup, _ = stabilizer_power_in_Gamma_s(L_rho, [QQ(p1, d), QQ(p2, d)])
        val = shin(A_lookup, rho_bf, [p1, p2], d)
        matches = [i for i in idxs if abs(tbl[i].shin_value - val) < tol]
        isempty(matches) && continue            # non-unit point: monoid path, not this bug
        ff = matches[1]                         # exactly what findfirst returns
        q = (mod(p1 - t_int * p2, d), p2)
        oc = [i for i in base_idxs if q in entry_orbit[i]]
        if length(oc) != 1
            push!(unresolved, (p1, p2)); continue
        end
        correct = oc[1]
        abs(tbl[correct].shin_value - val) < tol || (orbit_correct_mismatch_value += 1)
        deg = length(matches) > 1
        wrong = (ff != correct)
        deg && (n_deg += 1)
        if wrong
            n_wrong += 1; push!(wrong_pts, (p1, p2))
            deg ? (n_deg_wrong += 1) : (n_nodeg_wrong += 1)
        end
    end

    println("\n=== per-point results (unit points only) ===")
    println("points whose lookup value is DEGENERATE (>=2 group entries match): ", n_deg)
    println("points where findfirst returns the WRONG entry (by orbit identity): ", n_wrong)
    println("    of which degenerate: ", n_deg_wrong, "   non-degenerate: ", n_nodeg_wrong)
    println("orbit-identified entry whose value does NOT match the lookup value: ",
            orbit_correct_mismatch_value, "  (should be 0 - otherwise the covariance step is off)")
    isempty(unresolved) || println("points with no unique orbit-identified base-fiber entry: ", unresolved)

    # orbit structure of the wrong picks, in BASE-point coordinates (the ±<eps>
    # orbits under the translation-conjugated action, closed under negation)
    Tq(p) = (mod(p[1] - t_int * p[2], d), p[2])
    Tinv(q) = (mod(q[1] + t_int * q[2], d), q[2])
    wrong_set = Set(wrong_pts); seen = Set{Tuple{Int,Int}}(); n_orbits = 0; mixed = 0
    for p in wrong_pts
        p in seen && continue
        orb = Set{Tuple{Int,Int}}()
        for qq in orbit_of(Tq(p))
            push!(orb, Tinv(qq)); push!(orb, Tinv((mod(-qq[1], d), mod(-qq[2], d))))
        end
        n_orbits += 1
        all(x -> x in wrong_set, orb) || (mixed += 1)
        union!(seen, orb)
    end
    println("wrong-pick points form ", n_orbits, " ±<eps>-orbits", mixed == 0 ? " (all complete)" : " ($mixed only partially wrong)")
    println("wrong-pick points: ", sort(wrong_pts))
    println("\n=== VERDICT ===")
    if n_wrong > 0 && n_nodeg_wrong == 0
        println("Every wrong pick is a degenerate value: findfirst-by-value is the mechanism.")
        println("Fix: choose the group entry by orbit identity (value-free), value as consistency check.")
    elseif n_wrong == 0
        println("findfirst never picks a wrong entry under this seed - if this seed FAILS in the run,")
        println("the lookup is not the mechanism and the hypothesis is wrong.")
    else
        println("Some wrong picks are NOT degenerate - the orbit identification and the table")
        println("disagree on non-ambiguous points; investigate those before trusting a tiebreak.")
    end
    return (wrong_points = sort(wrong_pts), dup_groups = dups, n_degenerate = n_deg)
end

"""
    diagnose_seed_lift_differences(d::Int, seeds::Vector{Int}; tol = 1e-10)

Follow-up to diagnose_group_lookup_ambiguity, which showed (broken seed
208475639) that the group-table lookup returns the unique CORRECT class for
every unit point - so the failures are downstream of class selection.

What the lookup fixes is each entry's RESIDUE mod d (all the shin value
depends on). What twisted_chi_general actually consumes is a specific LIFT:
    elem1 = d*(s2*rho_B - s1),
and downstream product = elem1*elem2 with elem2 having denominator d, so
du = coords(product) mod d depends on the LIFT, not just the residue
(two lifts elem1, elem1 + d*y give products differing by y*elem2, whose
coordinates are NOT 0 mod d). The lift is canonical only up to units
IF (s2*rho_B - s1) is a genuine generator of the Theorem-2.19 ideal
Q = J*b_O*alpha^{-1}; alpha and b_O come from a randomized search
=> the lift can be SEED-DEPENDENT per class.

This builds the group table under each seed (re-seeding the RNG in-process),
restricts to the BASE fiber (rho_B0 = the F.Q root, where all unit points
live and which is seed-stable), matches classes across seeds by their
<eps>-orbit (seed-independent), and for each class compares the lifts:
    r = elem1[seed_j] / elem1[seed_i]
    unit-equivalent  <=>  r in O and N(r) = +/-1    (then shin/du agree)
    otherwise        =>  a genuinely different lift.
Prints, per seed pair, how many of the 72 base-fiber classes differ
non-trivially, which ones, the two elem1's side by side with their
embeddings, and the predicted failing set (the union of the +-<eps>-orbits of
the differing classes, in base-point coordinates) with its size.

PREDICTION if the lift mechanism is right: good seeds agree with each other
up to units on all 72 classes; a bad seed differs from a good one on exactly
15 classes, whose orbits are exactly the 90 failing points of the run.
Pass e.g. seeds = [good_seed, 208475639].
"""
function diagnose_seed_lift_differences(d::Int, seeds::Vector{Int}; tol::Float64 = 1e-10)
    F = AdmissibleTuple(d)
    per_seed = Dict{Int,Any}()
    for sdn in seeds
        Random.seed!(sdn)
        sd = _ray_class_shin_setup(F)
        D, K, O = sd.D, sd.K, sd.O
        tbl = ray_class_shin_table_fast_quiet(F; shared_data = sd)
        idxs = [i for i in eachindex(tbl) if tbl[i] !== nothing]
        rho = sd.rho_reps[sd.F_class]
        rhoBs = unique([tbl[i].rho_B for i in idxs])
        emb_main = embedding(real_places(K)[2])
        emb_conj = embedding(real_places(K)[1])
        rho_B0 = rhoBs[argmin([abs(Float64(real(emb_main(rho - rb)))) for rb in rhoBs])]
        t_int = Int(numerator(QQ(coeff(rho - rho_B0, 0))))
        L0 = hj_automorph_for_root(rho_B0, D)
        Lpos = (L0[1,1] + L0[2,2]) > 0 ? L0 : -L0
        function orbit_of(pt)
            orb = Set{Tuple{Int,Int}}(); q = [pt[1], pt[2]]
            for _ in 1:10_000
                push!(orb, (q[1], q[2])); q = mod.(Lpos * q, d)
                (q[1] == pt[1] && q[2] == pt[2]) && break
            end
            orb
        end
        classes = Dict{Set{Tuple{Int,Int}},Any}()
        for i in idxs
            e = tbl[i]
            e.rho_B == rho_B0 || continue
            pt = (Int(mod(numerator(e.s1 * d), d)), Int(mod(numerator(e.s2 * d), d)))
            elem1 = K(d) * (K(e.s2) * e.rho_B - K(e.s1))
            classes[orbit_of(pt)] = (idx = i, s1 = e.s1, s2 = e.s2, elem1 = elem1,
                                     main = Float64(real(emb_main(elem1))),
                                     conj = Float64(real(emb_conj(elem1))),
                                     nrm = norm(elem1))
        end
        println("seed ", sdn, ": base fiber rho_B0 = ", rho_B0, ", t = ", t_int,
                ", base-fiber classes = ", length(classes))
        per_seed[sdn] = (K = K, O = O, D = D, classes = classes, t = t_int, orbit_of = orbit_of)
    end

    # r in O with N(r) = +/-1 ?  (coordinates w.r.t. basis(O) must be integers)
    function is_unit_of_O(r, K, O)
        b1 = _nf_elem_of(K, basis(O)[1]); b2 = _nf_elem_of(K, basis(O)[2])
        u0, u1 = QQ(coeff(b1, 0)), QQ(coeff(b1, 1))
        v0, v1 = QQ(coeff(b2, 0)), QQ(coeff(b2, 1))
        c0, c1 = QQ(coeff(r, 0)), QQ(coeff(r, 1))
        det = u0 * v1 - v0 * u1
        iszero(det) && return false
        x = (c0 * v1 - v0 * c1) / det
        y = (u0 * c1 - c0 * u1) / det
        isone(denominator(x)) && isone(denominator(y)) && abs(norm(r)) == 1
    end

    for i in 1:length(seeds), j in i+1:length(seeds)
        si, sj = seeds[i], seeds[j]
        A, B = per_seed[si], per_seed[sj]
        K = A.K
        println("\n===== seed ", si, "  vs  seed ", sj, " =====")
        keys_common = intersect(keys(A.classes), keys(B.classes))
        println("classes matched by orbit: ", length(keys_common),
                "  (unmatched: ", length(A.classes) - length(keys_common), " / ",
                length(B.classes) - length(keys_common), ")")
        differing = Set{Tuple{Int,Int}}[]
        for k in keys_common
            ea, eb = A.classes[k], B.classes[k]
            r = eb.elem1 * inv(ea.elem1)
            if !is_unit_of_O(r, K, A.O)
                push!(differing, k)
                println("  DIFFERENT LIFT for class with orbit ", sort(collect(k)), ":")
                println("      seed ", si, ": s1=", ea.s1, " s2=", ea.s2, "  elem1=", ea.elem1,
                        "  main=", ea.main, " conj=", ea.conj, " N=", ea.nrm)
                println("      seed ", sj, ": s1=", eb.s1, " s2=", eb.s2, "  elem1=", eb.elem1,
                        "  main=", eb.main, " conj=", eb.conj, " N=", eb.nrm)
                println("      ratio = ", r, "   N(ratio) = ", norm(r))
            end
        end
        println("classes with a genuinely different lift: ", length(differing), " of ", length(keys_common))
        # predicted failing set in BASE-point coords for the second seed: union of
        # the +-orbits of the differing classes, pulled back through p = Tinv(q)
        t = B.t
        Tinv(q) = (mod(q[1] + t * q[2], d), q[2])
        pred = Set{Tuple{Int,Int}}()
        for k in differing, q in k
            push!(pred, Tinv(q)); push!(pred, Tinv((mod(-q[1], d), mod(-q[2], d))))
        end
        println("predicted failing points (base coords) if the lift mechanism is right: ",
                length(pred))
        isempty(pred) || println("  ", sort(collect(pred)))
    end
    return per_seed
end

"""
    check_nonunit_shift_invariance(d::Int; p1=nothing, p2=nothing, class_idx::Int=1,
                                     n_shifts_to_test::Int=3, tol::Float64=1e-10)

DIAGNOSTIC - added to directly test an assumption `twisted_chi_general`'s
"direct route" relies on for NON-UNIT grid points, which has never actually
been verified.

Background: for a non-unit point p=(p1,p2) (`x_p = p2*rho-p1` not a unit
mod `d`), the direct route builds
    elem1 := x_p + d*n_shift
for whatever `n_shift >= 0` is the SMALLEST value making `elem1` totally
positive, and proceeds to compute the twisted overlap from `elem1`. Since
`d*n_shift ∈ m`, `elem1 ≡ x_p (mod m)`, so this is a valid "some totally
positive representative of x_p's residue mod m" - but the assumption that
this specific choice of representative doesn't matter (i.e. that any OTHER
totally positive representative `x_p + d*n_shift'` would give the SAME final
answer) is a congruence-plus-sign equivalence that is a PROVEN THEOREM only
for points that are units mod `d` (the group case, Kopp-Lagarias). For
non-unit points (the ray class MONOID, KFA eq. (2.55)) this has never been
checked - `check_lookup_coverage` was written partly to probe exactly this
kind of question and was never run.

This function isolates the question directly: for a genuine non-unit grid
point and a genuine non-principal ray class (J, gamma) (the only branch that
ever exercises the direct route), it replicates `twisted_chi_general`'s
ENTIRE direct-route pipeline (elem1 -> elem2 -> product -> rho_j := rho_B ->
A_bj -> gQ_j (primitivized) -> gF_j -> M_g_j -> u1,u2 -> integralnewdu ->
QA,s,c_g -> final shin-based overlap value) for SEVERAL different valid
choices of `n_shift` (the code's own minimal choice, then +1, +2, ...), and
reports whether the resulting complex overlap values agree.

  - AGREE (within `tol`): the equivalence holds empirically for this case,
    and the direct route's particular choice of `n_shift` is, as intended,
    irrelevant to the final answer.
  - DISAGREE: the direct route is NOT shift-invariant for non-unit points -
    a genuine bug distinct from (and more fundamental than) the retry-loop
    crash fixed separately, and the "rho_j := rho_B, elem1 := shifted x_p"
    construction needs to be reconsidered for non-unit points specifically
    (e.g. by working out how KFA (2.55)'s monoid-orbit representative choice
    actually interacts with shin, rather than assuming the group-case
    congruence theorem extends verbatim).

If `p1`/`p2` are omitted, the first non-unit grid point found (scanning
`(0,0)` excluded, row-major) is used automatically. Raises an error if the
supplied/found point is actually a UNIT point (this test is specifically
about the monoid-only, non-unit case). `class_idx` selects which
non-principal ray class (1-indexed into the non-principal triples, same
ordering `test_all_twists` uses) to twist by; the specific class shouldn't
matter for what's being tested here, but is exposed in case a particular
class needs to be targeted.

Returns a NamedTuple `(p, class_idx, n_shifts, values, agree, max_diff)`.
"""
function check_nonunit_shift_invariance(d::Int; p1::Union{Int,Nothing} = nothing,
                                          p2::Union{Int,Nothing} = nothing,
                                          class_idx::Int = 1,
                                          n_shifts_to_test::Int = 3,
                                          tol::Float64 = 1e-10)
    F = AdmissibleTuple(d)
    sd = _ray_class_shin_setup(F)
    D, K, O = sd.D, sd.K, sd.O
    all_triples = sd.all_triples
    P_O, mP_O, rho_reps, F_class, d_check =
        sd.P_O, sd.mP_O, sd.rho_reps, sd.F_class, sd.d_check

    rho = rho_reps[F_class]

    places_K_top = real_places(K)
    emb_main_K = embedding(places_K_top[2])
    emb_conj_K = embedding(places_K_top[1])

    # --- unit/non-unit test, exactly matching twisted_chi_general's own ---
    function unit_status(pp1, pp2)
        x_p_ = K(pp2) * rho - K(pp1)
        I_rho_p = fractional_ideal(O, [K(1), rho])
        rel_p = QQ(abs(norm(x_p_))) / QQ(norm(I_rho_p))
        is_unit_p = isone(denominator(rel_p)) && gcd(Int(numerator(rel_p)), d) == 1
        return x_p_, is_unit_p
    end

    if p1 === nothing || p2 === nothing
        found = false
        for q1 in 0:d-1, q2 in 0:d-1
            (q1 == 0 && q2 == 0) && continue
            _, is_u = unit_status(q1, q2)
            if !is_u
                p1, p2 = q1, q2
                found = true
                break
            end
        end
        found || error("check_nonunit_shift_invariance: no non-unit grid point found for d=", d,
                        " (every nonzero point is a unit - nothing to test)")
        println("Using first non-unit point found: p = (", p1, ",", p2, ")")
    end

    x_p, is_unit_p = unit_status(p1, p2)
    is_unit_p && error("check_nonunit_shift_invariance: p=(", p1, ",", p2, ") IS a unit point ",
                        "(it has a genuine group ray class) - this test is specifically about the ",
                        "monoid-only direct route for NON-unit points. Pick a different p, or omit ",
                        "p1/p2 to auto-search for a non-unit point.")

    # --- pick a non-principal (J, gamma) triple (the only branch that ever
    # exercises the direct route in twisted_chi_general) ---
    nonprincipal_triples = [(J, gam) for (J, _, gam, is_princ) in all_triples if !is_princ]
    isempty(nonprincipal_triples) &&
        error("check_nonunit_shift_invariance: no non-principal ray classes found for d=", d)
    class_idx in 1:length(nonprincipal_triples) ||
        error("check_nonunit_shift_invariance: class_idx=", class_idx, " out of range 1:",
              length(nonprincipal_triples))
    J, gamma = nonprincipal_triples[class_idx]
    println("Using non-principal class ", class_idx, " of ", length(nonprincipal_triples),
            ":  J = ", J, "  gamma = ", gamma)

    # --- rho_B, s1, s2 for this J (identical call to twisted_chi_general's
    # cache-miss branch) ---
    u_B = mP_O \ J
    rho_B_init = rho_reps[u_B]
    _, rho_B, s1, s2 = find_alpha_rhoB_s1_s2_O(J, rho_B_init, rho, u_B, P_O, mP_O,
                                                 O, K, d, rho, d_check;
                                                 mn_height = 3000, q_gen_height = 3000)
    println("rho_B = ", rho_B, "  s1 = ", s1, "  s2 = ", s2)

    # --- chi (same formula as twisted_chi_general; is_principal_J = false here) ---
    Ng = Int(norm(K(gamma)))
    NJ = Int(norm(J))
    NJ_Ng = NJ * abs(Ng)
    chi_raw = kronecker_symbol(d + 1, NJ_Ng)
    chi = (chi_raw == 0) ? 1 : chi_raw
    ζ = -cispi(BigFloat(1) / d)

    function du_to_int_mod_d_local(du_int_val, dd)
        num = Int(numerator(du_int_val))
        den = Int(denominator(du_int_val))
        g = gcd(abs(num), den)
        num ÷= g
        den ÷= g
        gcd(den, dd) == 1 || error("denominator ", den, " of ", du_int_val,
                                    " is not coprime to d=", dd)
        return mod(num * invmod(den, dd), dd)
    end

    # --- run the direct-route pipeline for a GIVEN elem1, mirroring
    # twisted_chi_general's non-unit branch line for line ---
    function run_direct_route(elem1)
        elem2 = K(s2) * rho_B - K(s1)
        product = elem1 * elem2

        rho_j_raw = rho_B     # the assumption under test: rho_j := rho_B, regardless of n_shift
        s2_bj, s1_bj = find_rho_j_and_coords(product, rho_j_raw)

        rho_j_bf = BigFloat(coeff(rho_j_raw, 0)) + BigFloat(coeff(rho_j_raw, 1)) * sqrt(BigFloat(D))
        L_bj = hj_automorph_for_root(rho_j_raw, D)
        A_bj_cand, _ = stabilizer_power_in_Gamma_s(L_bj, [s1_bj, s2_bj])

        red_bj = find_hj_reducing_matrix(A_bj_cand, rho_j_bf)
        gQ_j_cand_raw = binary_quadratic_form(red_bj.A_reduced[2,1],
                                               red_bj.A_reduced[2,2] - red_bj.A_reduced[1,1],
                                               -red_bj.A_reduced[1,2])
        content_j = gcd(gcd(gQ_j_cand_raw.a, gQ_j_cand_raw.b), gQ_j_cand_raw.c)
        gQ_j_cand = binary_quadratic_form(divexact(gQ_j_cand_raw.a, content_j),
                                           divexact(gQ_j_cand_raw.b, content_j),
                                           divexact(gQ_j_cand_raw.c, content_j))
        gF_j_cand = AdmissibleTuple(d, F.r, gQ_j_cand)
        QQ_form_cand = binary_quadratic_form(gF_j_cand.A[2,1],
                                              gF_j_cand.A[2,2] - gF_j_cand.A[1,1],
                                              -gF_j_cand.A[1,2])

        M_g_j_cand = multiplication_matrix_mod_d(K(gamma), rho_j_raw, d, D)
        det_M_g_j = mod(M_g_j_cand[1,1]*M_g_j_cand[2,2] - M_g_j_cand[1,2]*M_g_j_cand[2,1], d)
        gcd(det_M_g_j, d) == 1 ||
            error("run_direct_route: det(M_g_j) not coprime to d for this elem1 - cannot ",
                  "compare shift invariance for this candidate")

        u2_cand, u1_cand = find_rho_j_and_coords(product, rho_j_raw)
        diag_val = Float64(u2_cand) * Float64(real(emb_conj_K(rho_j_raw))) - Float64(u1_cand)
        diag_val > 0 ||
            error("run_direct_route: u2*conj(rho_j)-u1 <= 0 for this elem1 - cannot ",
                  "compare shift invariance for this candidate")

        du = [to_du_mod_d(u1_cand, d), to_du_mod_d(u2_cand, d)]
        raw_newdu = M_g_j_cand * du
        newdu = [to_du_mod_d(raw_newdu[1], d), to_du_mod_d(raw_newdu[2], d)]
        integralnewdu = [du_to_int_mod_d_local(d * u, d) for u in newdu]

        c_g_j_cand = e(-BigFloat(rademacher(gF_j_cand.A)) / 24) / (chi * sqrt(BigFloat(d + 1)))
        QA = BigFloat(-QQ_form_cand(integralnewdu...)) / BigFloat(d * (d - 2))
        s_par = isodd(d) ? 1 : (1 + integralnewdu[1]) * (1 + integralnewdu[2])

        value = ζ^QA * (-1)^s_par * c_g_j_cand * shin(A_bj_cand, rho_j_bf, integralnewdu, d)
        return value, integralnewdu, gQ_j_cand
    end

    # --- the code's own minimal totally-positive shift, then n_shifts_to_test-1 more ---
    conj_v = Float64(real(emb_conj_K(x_p)))
    main_v = Float64(real(emb_main_K(x_p)))
    n_shift0 = max(0, Int(ceil((-min(conj_v, main_v) + 1) / d)))

    values = ComplexF64[]
    ns_tested = Int[]
    println("\nTesting n_shift = ", n_shift0, " (the code's own minimal choice), then ",
            n_shifts_to_test - 1, " further shifts (n_shift0+1, +2, ...):")
    for k in 0:(n_shifts_to_test - 1)
        n = n_shift0 + k
        elem1 = x_p + K(d * n)
        conj1 = Float64(real(emb_conj_K(elem1)))
        main1 = Float64(real(emb_main_K(elem1)))
        println("  n_shift = ", n, ":  elem1 = ", elem1,
                "  conj(elem1) = ", conj1, "  main(elem1) = ", main1,
                "  totally positive? ", conj1 > 0 && main1 > 0)
        try
            val, integralnewdu, gQ_j_cand = run_direct_route(elem1)
            println("    -> integralnewdu = ", integralnewdu, "  gQ_j = ", gQ_j_cand,
                    "  value = ", ComplexF64(val))
            push!(values, ComplexF64(val))
            push!(ns_tested, n)
        catch err
            println("    -> run_direct_route FAILED for this shift: ", err)
        end
    end

    length(values) < 2 &&
        error("check_nonunit_shift_invariance: fewer than 2 shifts produced a usable value - ",
              "cannot compare. Try a larger n_shifts_to_test, a different p, or a different class_idx.")

    max_diff = maximum(abs(values[i] - values[1]) for i in 2:length(values))
    agree = max_diff < tol
    println("\nShifts tested: ", ns_tested)
    println("Max pairwise difference from the first value: ", max_diff)
    println(agree ?
        "AGREE within tol=$tol - the shift-choice equivalence holds empirically for this case." :
        "DISAGREE (max_diff=$max_diff >= tol=$tol) - the direct route is NOT shift-invariant here; " *
        "this is a genuine bug in the non-unit-point construction, not just a retry-loop issue.")

    return (p = (p1, p2), class_idx = class_idx, n_shifts = ns_tested, values = values,
            agree = agree, max_diff = max_diff)
end

"""
    PROJECT OWNER DEFINITIONS (verbatim - load-bearing, do not paraphrase):

    For an ideal m of O:
        S_m := {alpha in O : alpha*O + m = O}
        J_m^*(O) := {invertible fractional ideals of O coprime to m}
        Jbar_m^flat(O) := {a in J^*_O(O) : a*O[S_m^{-1}] subseteq O[S_m^{-1}]}

    (Note: J^*_O(O) in the last line, as given, is its own object - the
    invertible fractional ideals of O - distinct from J_m^*(O) above; this
    is deliberate notation from the project owner, not a typo.)

    CORRECTION (project owner, after an earlier version of this code only
    ever enumerated INTEGRAL a): the enumeration below must exhaust ALL of
    J^*_O(O) by height, fractional ideals included, not just integral ones.

    Construction used here, and why it is both correct and fully general
    (derived jointly with the project owner):
      a in Jbar_m^flat(O) iff EXISTS s in S_m with s*a subseteq O (unwinding
      the S_m^{-1}-localization definition directly). So:
        - NECESSARY: every a in Jbar_m^flat(O) can be written as
          a = a_int/(beta) with a_int := s*a integral and beta := s in S_m,
          for SOME such s.
        - SUFFICIENT: conversely, for ANY integral a_int and ANY beta in
          S_m, a := a_int/(beta) satisfies beta*a = a_int subseteq O, so
          a in Jbar_m^flat(O) automatically - no further check needed.
      Hence enumerating a := P_i*(alpha)*(beta)^{-1} for alpha in O
      (arbitrary) and beta in S_m (tested via the SAME reliable ideal-sum
      method already used elsewhere in this file for integral-ideal
      coprimality, isone(ideal(O,beta)+m) - NOT via valuation/factorization,
      which is delicate exactly at primes dividing O's own conductor, e.g.
      q=3 here) is BOTH guaranteed to stay inside Jbar_m^flat(O) AND
      exhausts all of it as alpha, beta range over growing height bounds.
"""
function is_beta_in_S_m(beta, O, m_ideal)
    return isone(ideal(O, beta) + m_ideal)
end
# NOTE: no longer called by ray_class_monoid_lookup (see that function's
# docstring, section "WHY THE beta/FRACTIONAL DIMENSION WAS DROPPED"). Left
# defined since it documents part of the Section-1 criterion derivation and
# may still be useful for independent verification.

"""
    ray_class_monoid_lookup(F, target_shin_val; sd, rho_base, p_query,
                             group_table, known_monoid_entries,
                             candidate_cache, tol, initial_norm_bound,
                             max_norm_bound, verbose)

SUPERSEDED, per the project owner: the ray class monoid Jbar_m^flat(O) is
FINITE (592 classes for d=21) and small enough to build in FULL, once, up
front, via `ray_class_monoid_table_v2` - no incremental norm-bound search
needed at all. `twisted_chi_general` no longer calls this function; it does
a direct value lookup against a pre-built `full_monoid_table` instead (see
that function's own comments at the lookup-miss branch). Confirmed
empirically (`diagnose_full_monoid_membership`) that a genuine
shin_val_lookup value DOES appear in the full table - this incremental
search was solving a problem that did not need solving that way. Left
defined (fully working, extensively debugged) since it may still be useful
for comparison/verification, but it is DEAD CODE on the current production
path.

Implements the project owner's algorithm for extending the ray class table
into the monoid Jbar_m^flat(O) on a lookup miss. Matching is by shin VALUE,
per the project owner's original specification - see the retraction note
below for why an exact-grid-point shortcut that was here for one revision
has been removed again. `rho_base` (= rho_reps[F_class] for the caller's
AdmissibleTuple) is REQUIRED, either passed explicitly or derivable from a
full `sd` - see the BUGFIX note below for why. `p_query = (p1,p2)` is
optional and enables the "direct candidate" pre-check described further
down (also new, requested separately from the rho_base fix).

THE ALGORITHM (unchanged in spirit from the project owner's specification):
  1. A lookup miss means the target shin value comes from the monoid part,
     not the (already-verified, 288-entry) group part.
  2. Enumerate elements of Jbar_m^flat(O), ordered by height (norm) of the
     generators.
  3. For each candidate `a` (skipping ones already tried), compute its shin
     value via the SAME Theorem 2.19 machinery used to build the 288-entry
     group table (find_alpha_rhoB_s1_s2_O + shin).
  4. If that value already appears in the table (group entries, or monoid
     entries already found this call or a PRIOR call), discard `a` for good.
     Otherwise, record it as a new table entry (even if it doesn't match the
     target).
  5. Stop as soon as a computed value matches the target, or the norm bound
     is exhausted (error / return not-found).

WHY THE beta / FRACTIONAL DIMENSION WAS DROPPED
  The first implementation (this session) enumerated
      a := P_i * (alpha) * (beta)^{-1},   alpha in O arbitrary, beta in S_m,
  per the project owner's mid-session correction that fractional candidates
  are needed. The timing probe (diagnose_monoid_timing) showed EVERY
  candidate with non-unit beta (i.e. every candidate where `a` was genuinely
  fractional) exhausted the entire inner search bound of
  find_alpha_rhoB_s1_s2_O and threw, at ~2-2.6s each - this is the actual
  mechanism behind the multi-hour hang, not raw candidate volume. The
  reason is structural, not a bug to patch: Theorem 2.19 requires
  alpha(r2*rho-r1) IN b, and since b*m = alpha(rho*Z+Z) with m = d*O
  principal, that condition is EXACTLY r in (1/d)Z^2, which forces
      a_r = alpha(r2*rho-r1)*b^{-1}   to be INTEGRAL, always.
  A genuinely fractional `a` is simply outside the image of 2.19's own
  parametrization, so no amount of raising mn_height/q_gen_height can ever
  make find_alpha_rhoB_s1_s2_O succeed on one - the failures are permanent,
  not merely slow.
  Separately, fractional candidates are also unnecessary: writing the
  Section-1 criterion as "den(a) + m = O", choose (by CRT) s congruent to 1
  mod m with s in den(a); then s in S_m and s*a is INTEGRAL. Applying the
  ~_{m,Sigma} definition with c = a, alpha_e = s, beta_e = 1 gives
  alpha_e - beta_e = s-1 in m*O[S_m^{-1}] automatically, and the Sigma sign
  is fixable by shifting s within m*den(a) (a full-rank lattice, so this
  never runs out of room) - so a ~ s*a, an INTEGRAL representative. Every
  monoid class has one.
  CONSEQUENCE: dropping beta does not just fix the hang, it is also
  necessary - the beta-bearing candidates could never have contributed a
  match in the first place.

  This does NOT mean the earlier "integral-only, a = P_i*(alpha)" attempt
  (which the project owner corrected mid-session) was right as originally
  written - it was incomplete for a DIFFERENT reason: for a fixed integral
  P_i representing class [P_i], "P_i*(alpha), alpha in O" only reaches
  integral ideals b in that class for which b*P_i^{-1} happens to be
  generated by an ALGEBRAIC INTEGER alpha - it silently misses integral b
  for which the connecting element is a genuinely fractional field element
  (P_i^{-1} is a fractional ideal in general, so this is a real gap, not a
  hypothetical one). The fix here is not to reintroduce a fractional `a`,
  but to reparametrize INTEGRAL ideals correctly:
      Q_i  :=  an INTEGRAL ideal representing class [P_i]^{-1},
      a    :=  (alpha) * Q_i^{-1},   alpha ranging over Q_i itself (not O).
  This is exhaustive: for ANY integral b in class [P_i], b*Q_i is principal
  (product of inverse classes) AND integral (product of two integral
  ideals), so b*Q_i = (alpha) for some alpha in O - and since (alpha)
  subseteq Q_i (as b subseteq O), alpha automatically lies in Q_i. So
  restricting the alpha-search to Q_i's own lattice (rather than all of O)
  is exactly the right restriction, not merely an efficiency trick: it is
  what makes the enumeration correct, complete, AND (as a bonus) always
  integral by construction - N(a) = N(alpha)/N(Q_i) is an integer for every
  alpha in Q_i, with no separate coprimality or integrality check needed.
  Q_i^{-1} is computed via the conj(Q_i)/N(Q_i) identity (the same
  avoid-Hecke's-inv() trick already used elsewhere in this file for J_inv),
  not via Hecke's inv(), per this project's standing caution about inv()
  on ideals of a non-maximal order.

TWO BUGS FOUND AFTER Q_i WAS INTRODUCED, BOTH FIXED
  Bug 1 - Q_i via the wrong lookup. The first version of this fix fetched
  Q_i via mP_O(-g), i.e. whatever ideal Hecke happens to store as ITS OWN
  representative of the negated class - with no reason to expect that
  ideal's norm relates to class_ideals[i]'s norm at all. For this project's
  own d=21 example the four class norms are 1, 175, 30625, 5359375; if a
  CHEAP class's inverse happened to be the ~5.36 MILLION-norm one, the
  (then N(Q_i)-scaled) search blew up for that class every round, forever.
  FIXED by building Q_i directly as conj(class_ideals[i]) (the same
  P*conj(P)=N(P)*O identity already used for J_inv), which guarantees
  N(Q_i) = N(class_ideals[i]) exactly - no separate, uncorrelated lookup.
  Bug 2 - searching Q_i's RAW (unreduced) basis. Even after bug 1, the
  search still hung on p=(0,3). Cause: Hecke's own basis(Q_i) is typically
  an HNF-style <N, b+w> pair, where the plain-integer generator N has norm
  N^2 - astronomically larger than Q_i's own minimal elements for a
  several-million-norm ideal, and it dominates any c1*w1+c2*w2 combination
  with c1 != 0. Almost the entire (c1,c2) grid produced useless,
  astronomically-large-normed alpha, and only the 1-dimensional c1=0 slice
  ever reached small N(a). FIXED by Gauss/Lagrange-reducing Q_i's basis
  first (`lattice_reduce_2d`, extracted as a standalone helper from the
  SAME reduction find_alpha_rho_direct already performs on its own search
  basis - which is exactly why THAT search can use a small, fixed
  coordinate bound regardless of the ideal's norm). After reduction, no
  N(Q_i)-dependent scaling of the search bound is needed at all - a fixed
  bound (scaled by norm_bound alone, matching find_alpha_rho_direct's own
  mn_height pattern) suffices for every class.

RETRACTED: EXACT GRID-POINT MATCHING
  A previous revision of this function accepted an optional `target_p`
  argument and matched candidates against the query's own (p1,p2) as exact
  integers, on the reasoning that a matching candidate's native grid
  coordinates should equal the query's. On reflection this was not
  established - there is a separate `gp = [p1, mod(NJ*Ng*p2,d)]` transform
  at the call site suggesting the true correspondence involves more than
  the identity map, and shipping an unverified shortcut risked either a
  false match or silent extra overhead for no benefit. Reverted to pure
  shin-value matching, which is both what the project owner's algorithm
  specifies and what the covariance theorem actually justifies.

CROSS-CALL CANDIDATE CACHE
  Every call used to restart its enumeration from norm 0, recomputing
  find_alpha_rhoB_s1_s2_O + shin for the same cheap, low-norm candidates on
  EVERY lookup miss across a whole run (e.g. test_all_twists makes hundreds
  of calls). `candidate_cache` (keyed by (class_idx, c1, c2), the alpha
  coordinates relative to Q_i's own basis) persists the full computed result
  for a candidate across calls, so only genuinely new candidates ever pay
  the find_alpha_rhoB_s1_s2_O cost. Pass the SAME Dict on every call (e.g.
  built once in test_all_twists alongside monoid_table) to get the reuse;
  defaults to a fresh, call-local Dict if omitted.

DIRECT CANDIDATE PRE-CHECK (new, `p_query`)
  Requested separately from the rho_base fix. Before paying for the full
  enumeration, tests the ONE natural candidate directly implied by the
  query point itself:
      J_direct = (p2*rho_base - p1) * (Z*rho_base + Z)^{-1}.
  This CANNOT be assumed to be where the target value actually comes from -
  that is exactly why the enumeration exists at all - so it is constructed
  and run through the SAME find_alpha_rhoB_s1_s2_O search as every other
  candidate, not special-cased. If it DOES reproduce the target value, a
  "GOT LUCKY" message is printed (this point needed no enumeration at all -
  worth knowing explicitly, not silently accepting), and it is returned as
  the match. If it succeeds but does NOT match, it is still recorded into
  `new_entries` like any other examined candidate. If it fails outright
  (find_alpha_rhoB_s1_s2_O throws, or the resulting s1/s2 don't have
  denominator exactly d), it is silently skipped and the normal enumeration
  proceeds - failure here says nothing about whether the target is
  reachable at all, only that THIS ONE candidate wasn't it.
  Only run when `p_query = (p1,p2)` is supplied; omitted by default.

PERFORMANCE FIX: PER-CLASS find_alpha_rhoB_s1_s2_O CACHING (`class_hints`)
  Three prior A/B timing probes (generator size, Hecke type, rho argument)
  each disproved a hypothesis about find_alpha_rhoB_s1_s2_O's INPUTS, while
  showing a clean PER-CLASS (not per-candidate) timing signature every time:
  9-40s per candidate, identical across different candidates in the same
  class, different across classes. Reading find_alpha_rhoB_s1_s2_O's own
  internals (rather than guessing about its arguments further) found the
  actual cause: its main path calls find_b_with_verified_principal_product,
  an O(height^5)-cost quadruple-nested search whose result depends ONLY on
  J's Picard class (per that function's own docstring comment) - exactly
  matching the observed signature. find_alpha_rhoB_s1_s2_O already has a
  fast path built for precisely this (alpha_hint/rho_B_hint/b_O_hint,
  documented as existing so per-ray-class table builders pay this search
  ONCE per Picard class), which this function was never using - every
  candidate re-ran the full search from scratch even when several shared a
  class. `class_hints` (via the local `find_with_class_cache` helper) now
  caches (alpha, rho_B, b_O) per class: the FIRST candidate in a class pays
  the full cost (plus one extra, deliberately redundant
  find_b_with_verified_principal_product call to expose b_O, which
  find_alpha_rhoB_s1_s2_O computes internally but does not return - this
  mirrors the EXACT pattern already used elsewhere in this file,
  ray_class_shin_table's `fiber_cache`, rather than risking a new subtle
  mismatch by re-deriving alpha/rho_B independently), and every SUBSEQUENT
  candidate in the same class uses the hint fast path.
  find_alpha_rhoB_s1_s2_O's own signature is NOT modified by this fix.

Returns (found::Bool, entry, new_entries) where entry is the NamedTuple
(J, rho_B, s1, s2, shin_value) for the match (or `nothing`), and
new_entries is every non-duplicate monoid entry discovered along the way
(for the caller to persist and reuse across future lookups/calls).
"""
function ray_class_monoid_lookup(F::AdmissibleTuple, target_shin_val;
                                  sd = nothing,
                                  rho_base = nothing,
                                  p_query::Union{Nothing, Tuple{Int,Int}} = nothing,
                                  group_table = nothing,
                                  known_monoid_entries::Vector = NamedTuple[],
                                  candidate_cache::Dict = Dict{Tuple{Int,Int,Int}, Any}(),
                                  class_hints::Dict = Dict{Any, NamedTuple}(),
                                  tol::Float64 = 1e-10,
                                  initial_norm_bound::Int = 50,
                                  max_norm_bound::Int = 10^6,
                                  verbose::Bool = true)
    sdv = sd === nothing ? _ray_class_shin_setup(F) : sd
    D, K, O, d, d_check = sdv.D, sdv.K, sdv.O, sdv.d, sdv.d_check
    P_O, mP_O, rho_reps = sdv.P_O, sdv.mP_O, sdv.rho_reps
    # NOTE ON THE FOLLOWING FIX: kept because it is still the argument-by-
    # argument CORRECT call (matching the real call site exactly), but an
    # A/B timing probe subsequently showed it does NOT explain the observed
    # 9-40s-per-candidate slowness - WRONG (rho_a_init for all 3 roles) and
    # RIGHT (rho_base for the 3rd/10th) timed identically. The actual cause,
    # found by reading find_alpha_rhoB_s1_s2_O's own internals rather than
    # guessing further about its arguments, is documented at the class_hints
    # cache below.
    #
    # find_alpha_rhoB_s1_s2_O's 3rd and 10th positional arguments
    # (both named `rho`/`rho_j` in its own signature) are NOT the per-class
    # rho_B_init - they are `rho_reps[F_class]`, a SINGLE value fixed for
    # the entire twisted_chi_general call, used identically regardless of
    # which Picard class the ideal being processed belongs to (see the real
    # call site: `find_alpha_rhoB_s1_s2_O(J, rho_B_init, rho, u_B, ...,
    # d, rho, d_check)` - rho_B_init varies with J's class, `rho` does not).
    # `rho_base` must be `rho_reps[F_class]` for the CURRENT AdmissibleTuple
    # being twisted, i.e. the SAME value the caller (twisted_chi_general)
    # already has in its own local `rho` variable - pass it through
    # explicitly rather than trying to reconstruct F_class from `sd`, since
    # the ad hoc NamedTuple twisted_chi_general builds for `sd` does not
    # carry F_class.
    rho_base === nothing && (rho_base = try rho_reps[sdv.F_class] catch; nothing end)
    rho_base === nothing &&
        error("ray_class_monoid_lookup: rho_base was not given and could not be derived from `sd` ",
              "(pass rho_base = rho_reps[F_class] explicitly - see this function's BUGFIX note)")
    tblv = group_table === nothing ? ray_class_shin_table_fast_quiet(F; shared_data = sdv) : group_table

    seen_shin = Complex{BigFloat}[Complex{BigFloat}(e.shin_value) for e in tblv if e !== nothing]
    for e in known_monoid_entries
        push!(seen_shin, Complex{BigFloat}(e.shin_value))
    end
    is_dup(val) = any(v -> abs(v - val) < tol, seen_shin)

    new_entries = NamedTuple[]

    # ACTUAL FIX for the 9-40s-per-candidate slowness (found by reading
    # find_alpha_rhoB_s1_s2_O's internals, after the rho-argument hypothesis
    # above was disproven by direct A/B measurement):
    # find_alpha_rhoB_s1_s2_O's main (non-hinted) path calls
    # find_b_with_verified_principal_product, which runs a QUADRUPLE-NESTED
    # coordinate search (O(height^5) cumulative cost) to find a coprime
    # ideal I with J*I verifiably principal. Per that function's own
    # docstring comment, its cost and result depend ONLY on J's Picard
    # class, not on J itself - which exactly matches the observed timing
    # signature (identical cost across different candidates IN the same
    # class, different cost ACROSS classes). find_alpha_rhoB_s1_s2_O
    # already has a fast path for exactly this situation - alpha_hint,
    # rho_B_hint, b_O_hint - built so callers processing many ideals of the
    # same class (e.g. a per-ray-class table) pay this search ONCE per
    # class, then reuse the result. This function was never using that fast
    # path at all - every single candidate re-ran the full O(height^5)
    # search from scratch, even other candidates already known to share the
    # same class. This is the SAME per-class caching pattern already
    # established elsewhere in this file (ray_class_shin_table's
    # `fiber_cache`), applied here via the generic hint mechanism instead of
    # that function's own more specialized (gamma-based) shortcut, since
    # this function's candidates don't share that specific J*(gamma)
    # structure.
    # class_hints is now a PARAMETER (see signature), not created fresh here,
    # so it persists across calls when the caller threads the same Dict
    # through repeatedly (test_all_twists does this alongside monoid_table
    # and candidate_cache) - otherwise each new lookup call would re-pay the
    # full per-class search cost the first time it happens to hit that class,
    # even if an EARLIER call already paid it.

    # Wraps find_alpha_rhoB_s1_s2_O with the class_hints cache above. For
    # the FIRST candidate seen in a given class, runs the full (expensive)
    # search and ALSO separately calls find_b_with_verified_principal_product
    # to fish out b_O (which find_alpha_rhoB_s1_s2_O computes internally but
    # does not expose) - this redundant second call for the anchor candidate
    # ONLY mirrors the exact pattern already used at ray_class_shin_table's
    # `fiber_cache` (see the FIXED comment near line 1520-1530: "find_b_with_
    # verified_principal_product's search succeeds or fails identically for
    # any two ideals differing by a principal factor"), rather than trying to
    # be cleverer and risk a new subtle mismatch. Every SUBSEQUENT candidate
    # in the same class uses the alpha_hint/rho_B_hint/b_O_hint fast path,
    # skipping the expensive search entirely.
    function find_with_class_cache(Jcand, u_cand)
        if haskey(class_hints, u_cand)
            hint = class_hints[u_cand]
            return find_alpha_rhoB_s1_s2_O(
                Jcand, rho_reps[u_cand], rho_base, u_cand, P_O, mP_O, O, K,
                d, rho_base, d_check;
                alpha_hint = hint.alpha, rho_B_hint = hint.rho_B, b_O_hint = hint.b_O)
        else
            result = find_alpha_rhoB_s1_s2_O(
                Jcand, rho_reps[u_cand], rho_base, u_cand, P_O, mP_O, O, K,
                d, rho_base, d_check)
            b_O_found, _ = find_b_with_verified_principal_product(
                O, K, Jcand, [ideal(O, d), ideal(O, d + 1)])
            class_hints[u_cand] = (alpha = result[1], rho_B = result[2], b_O = b_O_found)
            return result
        end
    end

    # BUGFIX: entries returned by this function used to store `a`/`J_direct`
    # directly - both built via `fractional_ideal(O, ...)`, so Hecke types
    # them as AbsSimpleNumFieldOrderFractionalIdeal REGARDLESS of whether
    # their denominator is 1, even though every candidate here IS
    # mathematically integral by construction. twisted_chi_general takes the
    # matched `lookup_entry.J` and later does `mP_O \ (J * U_elem1)` - and
    # Hecke's disc_log_picard_group has NO method for fractional ideals, only
    # genuine integral AbsSimpleNumFieldOrderIdeal objects (confirmed by a
    # real MethodError from a live test_all_twists run, at exactly that
    # call). The group table's own entries never hit this because they are
    # built from genuine integral ideal objects throughout. Convert every
    # `J` stored into an entry via `numerator()` so BOTH sources
    # (group table and this monoid search) are type-consistent for whatever
    # downstream code does with `lookup_entry.J`, rather than requiring every
    # caller to defensively handle both possible types.
    function to_integral_ideal(Jc)
        isone(denominator(Jc)) || error("to_integral_ideal: candidate is not actually integral ",
                                         "(denominator != 1) - this should be impossible by ",
                                         "construction for every ideal this function produces")
        return numerator(Jc)
    end

    picard_classes = collect(P_O)
    n_classes = length(picard_classes)

    # Q[i] represents the INVERSE of picard_classes[i], so that
    # a := (alpha)*Q[i]^{-1}, alpha ranging over Q[i] itself, lands in class
    # picard_classes[i] and is automatically integral - see the docstring
    # section above for why this parametrization (as opposed to alpha
    # ranging over all of O, or a beta denominator) is both correct and
    # complete.
    #
    # BUGFIX: this used to fetch Q[i] via mP_O(-g), i.e. whatever ideal Hecke
    # happens to store as ITS OWN chosen representative of the negated class.
    # There is no reason that ideal's norm has anything to do with class_ideals[i]
    # (= mP_O(g))'s norm - for this project's own d=21 example the four class
    # norms are 1, 175, 30625, 5359375, and if a CHEAP class's inverse happens
    # to be the ~5.36 MILLION-norm one, the coord_bound scaling below (which is
    # keyed to N(Q[i])) blows up for that class on every single round of the
    # norm_bound loop, forever - this is almost certainly what caused the
    # observed multi-minute hang on p=(0,3).
    # FIX: build Q[i] directly from class_ideals[i] itself via the conjugate-
    # ideal identity P*conj(P) = N(P)*O (already used elsewhere in this file
    # for J_inv), which GUARANTEES N(Q[i]) = N(class_ideals[i]) exactly - the
    # same small/large norm the project owner's own timing data already
    # characterizes, with no separate, uncorrelated lookup that can go wrong.
    class_ideals = [mP_O(g) for g in picard_classes]
    Pgens = [[_nf_elem_of(K, x) for x in basis(Pi)] for Pi in class_ideals]
    NQ = [QQ(norm(Pi)) for Pi in class_ideals]                            # = N(Q[i]) = N(P[i])
    Qgens_raw = [[quad_conj(K, g) for g in Pgens[i]] for i in 1:n_classes]  # RAW basis of Q[i] = conj(P[i])
    # Q[i]^{-1} = conj(Q[i])/N(Q[i]) = conj(conj(P[i]))/N(P[i]) = P[i]/N(P[i]),
    # so its generators are just P[i]'s OWN generators divided by N(P[i]) -
    # no double conjugation, and (as before) no Hecke inv() call.
    Qinv_gens = [[g // NQ[i] for g in Pgens[i]] for i in 1:n_classes]

    # BUGFIX (second one, after the N(Q[i]) fix above still hung): Qgens_raw
    # is Hecke's own, UNREDUCED basis for Q[i]. Ideal bases from Hecke are
    # typically HNF-style <N, b+w> pairs, where the plain-integer generator N
    # has norm N^2 - for a several-million norm ideal that is a many-orders-
    # of-magnitude-larger vector than the ideal's own minimal elements, and it
    # dominates any small-coefficient combination c1*w1+c2*w2 with c1 != 0.
    # The result: almost the entire (c1,c2) coordinate grid produces
    # astronomically large, useless alpha, and only the c1=0 axis (a
    # 1-dimensional slice) ever reaches a small N(a) - which is exactly why
    # the earlier N(Q[i])-scaled coord_bound (compensating for the WRONG
    # problem) still hung on p=(0,3): it was scanning billions of doomed
    # candidates for the expensive class(es).
    # FIX: reduce Qgens_raw with lattice_reduce_2d BEFORE searching - the
    # exact same reduction find_alpha_rho_direct already performs on its own
    # search basis, and which is precisely why THAT search can use a small,
    # FIXED coordinate bound regardless of the ideal's norm. After reduction,
    # small integer coefficients correctly correspond to small-norm elements
    # of Q[i], so no N(Q[i])-dependent scaling of the search bound is needed
    # at all.
    places = real_places(K)
    Qgens = [collect(lattice_reduce_2d(Qgens_raw[i][1], Qgens_raw[i][2], K, places))
             for i in 1:n_classes]

    tried = [Set{Tuple{Int,Int}}() for _ in 1:n_classes]

    # ---- EXTRA CHECK, requested by the project owner: before paying for the
    # full enumeration, test the one NATURAL candidate directly implied by
    # the query point itself: J_direct = (p2*rho_base - p1)*(Z*rho_base+Z)^{-1}.
    # This CANNOT be assumed to be where the target value actually comes from
    # (that's exactly why the enumeration search exists at all) - it has to
    # be constructed and run through the SAME find_alpha_rhoB_s1_s2_O search
    # as every other candidate to check. If it DOES hit, that is worth
    # flagging explicitly ("got lucky") rather than silently accepting it,
    # since it means this specific point didn't need the expensive search at
    # all - if this turns out to hit often across many points, it would be
    # a large, well-justified performance win (one find_alpha_rhoB_s1_s2_O
    # call instead of dozens), but it is NOT something to rely on without
    # this kind of explicit, per-point verification.
    if p_query !== nothing
        p1q, p2q = p_query
        x_direct = K(p2q) * rho_base - K(p1q)
        if !iszero(x_direct)
            I_rho = fractional_ideal(O, [K(1), rho_base])
            NI_rho = QQ(norm(I_rho))
            # I_rho^{-1} = conj(I_rho)/N(I_rho) - same avoid-Hecke's-inv()
            # identity used throughout this function for Q[i]^{-1}.
            I_rho_inv_gens = [K(1) // NI_rho, quad_conj(K, rho_base) // NI_rho]
            J_direct_gens = [x_direct * g for g in I_rho_inv_gens]
            J_direct = fractional_ideal(O, J_direct_gens)

            ok_direct = true
            local rho_d, s1_d, s2_d, u_direct, J_direct_int
            try
                # J_direct's class: numerator(J_direct) has the SAME class as
                # J_direct itself (they differ only by the principal ideal
                # generated by the plain rational integer denominator(J_direct),
                # which is trivial in the class group) - so this is a safe way
                # to reach an INTEGRAL ideal for the discrete log, without
                # assuming J_direct itself is integral.
                den_direct = denominator(J_direct)
                J_direct_int = isone(den_direct) ? J_direct : numerator(J_direct)
                u_direct = mP_O \ J_direct_int
                _, rho_d, s1_d, s2_d = find_with_class_cache(J_direct, u_direct)
            catch
                ok_direct = false
            end

            if ok_direct
                rho_d_bf = BigFloat(coeff(rho_d, 0)) + BigFloat(coeff(rho_d, 1)) * sqrt(BigFloat(D))
                L_d = hj_automorph_for_root(rho_d, D)
                A_d, _ = stabilizer_power_in_Gamma_s(L_d, [s1_d, s2_d])
                p1d, p2d = s1_d * d, s2_d * d
                if isone(denominator(p1d)) && isone(denominator(p2d))
                    val_direct = Complex{BigFloat}(
                        shin(A_d, rho_d_bf, [Int(numerator(p1d)), Int(numerator(p2d))], d))
                    if abs(val_direct - target_shin_val) < tol
                        verbose && println("  [monoid lookup] GOT LUCKY: the direct candidate ",
                                            "(p2*rho-p1)(Z*rho+Z)^{-1} for p=", p_query,
                                            " matched the target on the FIRST try (class ",
                                            u_direct, ", s1=", s1_d, ", s2=", s2_d, ") - no ",
                                            "enumeration was needed for this point.")
                        entry = (J = J_direct_int, rho_B = rho_d, s1 = s1_d, s2 = s2_d,
                                shin_value = val_direct)
                        if !is_dup(val_direct)
                            push!(new_entries, entry)
                            push!(seen_shin, val_direct)
                        end
                        return (found = true, entry = entry, new_entries = new_entries)
                    elseif !is_dup(val_direct)
                        # Not a match for THIS query, but it's a genuine,
                        # newly-computed monoid value - record it the same
                        # way any other examined-but-non-matching candidate
                        # is recorded, so a later lookup can benefit from it.
                        push!(new_entries, (J = J_direct_int, rho_B = rho_d, s1 = s1_d, s2 = s2_d,
                                            shin_value = val_direct))
                        push!(seen_shin, val_direct)
                    end
                end
            end
        end
    end

    norm_bound = initial_norm_bound
    while norm_bound <= max_norm_bound
        verbose && println("  [monoid lookup] trying norm bound (on N(a)): ", norm_bound)

        candidates = Tuple{Int,Int,Int,Any}[]   # (class_idx, c1, c2, Na); Na is
                                                  # whatever Hecke rational type
                                                  # norm(alpha)//NQ[i] returns -
                                                  # NOT Julia's Rational{BigInt}.
        # FIXED coordinate bound, NOT scaled by N(Q[i]) - see the BUGFIX note
        # above. After lattice_reduce_2d, this plays exactly the role
        # find_alpha_rho_direct's mn_height plays for its own search.
        coord_bound = max(5, Int(ceil(sqrt(Float64(norm_bound)))) * 3)
        for i in 1:n_classes
            w1, w2 = Qgens[i][1], Qgens[i][2]
            for c1 in 0:coord_bound, c2 in (c1 == 0 ? 1 : -coord_bound):coord_bound
                # c1 > 0, or (c1==0 and c2>0): (c1,c2) and (-c1,-c2) generate
                # the same principal ideal (alpha) = (-alpha), hence the same
                # `a` - skip the redundant mirror instead of computing it twice.
                (c1, c2) in tried[i] && continue
                alpha = K(c1) * w1 + K(c2) * w2
                Nalpha = abs(norm(alpha))
                iszero(Nalpha) && continue
                Na = Nalpha // NQ[i]
                Na > norm_bound && continue
                push!(candidates, (i, c1, c2, Na))
            end
        end
        sort!(candidates, by = x -> x[4])
        verbose && println("    candidates to check: ", length(candidates))

        for (i, c1, c2, Na) in candidates
            push!(tried[i], (c1, c2))

            cached = get(candidate_cache, (i, c1, c2), nothing)
            local a, rho_a, s1_a, s2_a, val
            if cached !== nothing
                a, rho_a, s1_a, s2_a, val = cached
            else
                w1, w2 = Qgens[i][1], Qgens[i][2]
                alpha = K(c1) * w1 + K(c2) * w2
                a_gens = [alpha * g for g in Qinv_gens[i]]
                # RETRACTED (confirmed wrong by an A/B timing probe): reducing
                # a's own generators via lattice_reduce_2d before construction
                # was tried here on the hypothesis that unreduced coefficients
                # in `a`'s representation (not its norm) were behind the
                # observed 9-21s-per-candidate find_alpha_rhoB_s1_s2_O timings.
                # diagnose_monoid_lookup_v2_timing's RAW-vs-RED comparison
                # showed IDENTICAL timing either way across every candidate in
                # every class - the hypothesis was wrong, so the reduction was
                # removed again rather than left as dead-weight "fix" that
                # doesn't do anything. See diagnose_monoid_lookup_v2_timing's
                # current docstring for the hypothesis now being tested
                # instead (FractionalIdeal-typed `a` vs a genuine integral
                # ideal object via numerator(a)).
                a = fractional_ideal(O, a_gens)

                # u_a is known for free: multiplying/dividing by the
                # PRINCIPAL ideal (alpha) never changes ideal class, so a's
                # class is exactly picard_classes[i] - no discrete log
                # needed (mP_O \ a would fail here anyway: Hecke's
                # disc_log_picard_group has no method for fractional ideals,
                # and this construction guarantees `a` is integral, but
                # `a*Q[i]` being the actual verifying step is not repeated
                # per-candidate here since it follows from the construction).
                u_a = picard_classes[i]
                ok = true
                try
                    _, rho_a, s1_a, s2_a = find_with_class_cache(a, u_a)
                catch
                    ok = false
                end
                ok || continue

                rho_a_bf = BigFloat(coeff(rho_a, 0)) + BigFloat(coeff(rho_a, 1)) * sqrt(BigFloat(D))
                L_a = hj_automorph_for_root(rho_a, D)
                A_a, _ = stabilizer_power_in_Gamma_s(L_a, [s1_a, s2_a])
                p1_exact, p2_exact = s1_a * d, s2_a * d
                (isone(denominator(p1_exact)) && isone(denominator(p2_exact))) || continue
                p_vec_a = [Int(numerator(p1_exact)), Int(numerator(p2_exact))]
                val = Complex{BigFloat}(shin(A_a, rho_a_bf, p_vec_a, d))

                candidate_cache[(i, c1, c2)] = (a, rho_a, s1_a, s2_a, val)
            end

            # NOTE: the exact-grid-point-match shortcut (target_p) from the
            # previous revision has been REMOVED. It assumed a matching
            # candidate's own native grid coordinates would equal the query's
            # (p1,p2) directly - on reflection this isn't justified (there is
            # a separate gp = [p1, mod(NJ*Ng*p2,d)] transform nearby at the
            # call site suggesting the real correspondence isn't the
            # identity), and I'd rather match purely on the shin VALUE, which
            # is what the project owner's algorithm actually specifies and
            # what the covariance theorem actually guarantees, than risk a
            # confidently-wrong shortcut.
            is_dup(val) && continue

            entry = (J = to_integral_ideal(a), rho_B = rho_a, s1 = s1_a, s2 = s2_a, shin_value = val)
            push!(new_entries, entry)
            push!(seen_shin, val)

            if abs(val - target_shin_val) < tol
                verbose && println("  [monoid lookup] MATCH found (shin value): J = ", a,
                                    "  class ", i, "  s1 = ", s1_a, "  s2 = ", s2_a)
                return (found = true, entry = entry, new_entries = new_entries)
            end
        end

        norm_bound *= 4
    end

    verbose && println("  [monoid lookup] exhausted max_norm_bound = ", max_norm_bound,
                        " without a match")
    return (found = false, entry = nothing, new_entries = new_entries)
end


"""
    diagnose_monoid_enumeration(d::Int; norm_bound = 6, p1 = 0, p2 = 3,
                                 max_candidates = 400, mn_height = 400,
                                 q_gen_height = 400)

CHEAP PROBE (minutes, not hours) answering one question: does the
`a = P_i*(alpha)*(beta)^{-1}` enumeration inside `ray_class_monoid_lookup`
actually produce genuinely NEW shin values, or does it just re-derive the
same handful of values over and over?

Motivation / the concern being tested: `find_alpha_rhoB_s1_s2_O`'s output
`(alpha, rho_B, s1, s2)` is believed to depend only on the IDEAL CLASS of
its input (this is exactly why the group-table builder caches it per Picard
class in `alpha_rhoB_bO_cache`). But `a = P_i*(alpha)*(beta)^{-1}` always
lies in the SAME class as `P_i`, because (alpha) and (beta) are principal.
If that belief is right, then every candidate in a given class collapses to
ONE shin value, the whole enumeration only ever yields ~4 distinct values
(one per Picard class), all of which are already in the group table - and
`ray_class_monoid_lookup` can never terminate with a match no matter how
long it runs. This probe measures that directly.

Prints, for a small norm bound:
  - how many (alpha,beta,class) candidates were built,
  - how many were actually evaluated (capped by `max_candidates`),
  - the number of DISTINCT (rho_B, s1, s2) triples seen,
  - the number of DISTINCT shin values seen, grouped, with how many
    candidates produced each,
  - for each distinct shin value, whether it is ALREADY in the 288-entry
    group table,
  - and whether the target value (the one p=(p1,p2) actually needs) appeared.

Interpretation:
  * "distinct shin values" ~= number of Picard classes (~4), all already in
    the group table  =>  CONFIRMS the construction is wrong (not merely
    slow): it cannot produce the missing monoid values, so no amount of
    optimization or a bigger bound will help. The Theorem-2.19 entry point
    for the monoid case needs to be something other than a class-only
    `find_alpha_rhoB_s1_s2_O(a, ...)` call.
  * many distinct values, some NOT in the group table  =>  the construction
    is sound and the problem really is just performance/bounds.
"""
function diagnose_monoid_enumeration(d::Int; norm_bound::Int = 6,
                                      p1::Int = 0, p2::Int = 3,
                                      max_candidates::Int = 400,
                                      mn_height::Int = 400,
                                      q_gen_height::Int = 400,
                                      tol::Float64 = 1e-10)
    F = AdmissibleTuple(d)
    sd = _ray_class_shin_setup(F)
    D, K, O, d_check = sd.D, sd.K, sd.O, sd.d_check
    P_O, mP_O, rho_reps, F_class = sd.P_O, sd.mP_O, sd.rho_reps, sd.F_class
    tbl = ray_class_shin_table_fast_quiet(F; shared_data = sd)
    group_vals = Complex{BigFloat}[Complex{BigFloat}(e.shin_value) for e in tbl if e !== nothing]
    in_group(v) = any(g -> abs(g - v) < tol, group_vals)

    # The value p=(p1,p2) actually needs (same construction twisted_chi_general uses).
    rho = rho_reps[F_class]
    rho_bf = BigFloat(coeff(rho, 0)) + BigFloat(coeff(rho, 1)) * sqrt(BigFloat(D))
    L_rho = hj_automorph_for_root(rho, D)
    A_lookup, _ = stabilizer_power_in_Gamma_s(L_rho, [QQ(p1, d), QQ(p2, d)])
    target = Complex{BigFloat}(shin(A_lookup, rho_bf, [p1, p2], d))
    println("target shin^{[", p1, ",", p2, "]}_A(rho) = ", target)
    println("  already in the 288-entry group table? ", in_group(target))
    println("  (if true, there is no monoid problem for this point at all)")

    dbar = mod(d, 2) == 1 ? d : 2 * d
    m_ideal = ideal(O, dbar)
    b1 = _nf_elem_of(K, basis(O)[1])
    b2 = _nf_elem_of(K, basis(O)[2])
    picard_classes = collect(P_O)
    class_ideals = [mP_O(g) for g in picard_classes]
    coord_bound = max(3, Int(ceil(sqrt(Float64(norm_bound)))) * 3)

    betas = Tuple{Int,Int,Int}[]
    for bi in -coord_bound:coord_bound, bj in -coord_bound:coord_bound
        (bi == 0 && bj == 0) && continue
        beta = K(bi) * b1 + K(bj) * b2
        nb = abs(Int(norm(beta)))
        (nb == 0 || nb > norm_bound) && continue
        is_beta_in_S_m(beta, O, m_ideal) || continue
        push!(betas, (bi, bj, nb))
    end
    alphas = Tuple{Int,Int,Int}[]
    for ai in -coord_bound:coord_bound, aj in -coord_bound:coord_bound
        (ai == 0 && aj == 0) && continue
        alpha = K(ai) * b1 + K(aj) * b2
        na = abs(Int(norm(alpha)))
        (na == 0 || na > norm_bound) && continue
        push!(alphas, (ai, aj, na))
    end
    println("\nnorm_bound = ", norm_bound, ": ", length(alphas), " alphas, ",
            length(betas), " betas (in S_m), ", length(class_ideals), " classes",
            "  => ", length(alphas) * length(betas) * length(class_ideals), " candidates")

    triples = Dict{Any,Int}()          # (rho_B, s1, s2) -> count
    valgroups = Tuple{Complex{BigFloat},Int,Any}[]   # (value, count, sample triple)
    n_eval = 0; n_fail = 0
    for ci in 1:length(class_ideals), (ai, aj, na) in alphas, (bi, bj, nb) in betas
        n_eval >= max_candidates && break
        alpha = K(ai) * b1 + K(aj) * b2
        beta  = K(bi) * b1 + K(bj) * b2
        a = class_ideals[ci] * ideal(O, alpha) * inv(ideal(O, beta))
        u_a = picard_classes[ci]
        rho_a_init = rho_reps[u_a]
        local rho_a, s1_a, s2_a
        try
            _, rho_a, s1_a, s2_a = find_alpha_rhoB_s1_s2_O(
                a, rho_a_init, rho_a_init, u_a, P_O, mP_O, O, K,
                d, rho_a_init, d_check; mn_height = mn_height,
                q_gen_height = q_gen_height)
        catch
            n_fail += 1; n_eval += 1; continue
        end
        n_eval += 1
        key = (string(rho_a), string(s1_a), string(s2_a))
        triples[key] = get(triples, key, 0) + 1

        p1e, p2e = s1_a * d, s2_a * d
        (isone(denominator(p1e)) && isone(denominator(p2e))) || continue
        rho_a_bf = BigFloat(coeff(rho_a, 0)) + BigFloat(coeff(rho_a, 1)) * sqrt(BigFloat(D))
        A_a, _ = stabilizer_power_in_Gamma_s(hj_automorph_for_root(rho_a, D), [s1_a, s2_a])
        val = Complex{BigFloat}(shin(A_a, rho_a_bf,
                                     [Int(numerator(p1e)), Int(numerator(p2e))], d))
        hit = findfirst(g -> abs(g[1] - val) < tol, valgroups)
        if hit === nothing
            push!(valgroups, (val, 1, key))
        else
            v, c, k = valgroups[hit]
            valgroups[hit] = (v, c + 1, k)
        end
    end

    println("\ncandidates evaluated: ", n_eval, "  (find_alpha_rhoB_s1_s2_O failures: ", n_fail, ")")
    println("DISTINCT (rho_B,s1,s2) triples seen: ", length(triples))
    println("DISTINCT shin values seen:           ", length(valgroups))
    println("\nper distinct shin value:")
    for (v, c, k) in valgroups
        println("  count=", c, "  in group table? ", in_group(v),
                "  matches target? ", abs(v - target) < tol)
        println("      (rho_B,s1,s2) = ", k)
        println("      value = ", v)
    end
    n_new = count(g -> !in_group(g[1]), valgroups)
    println("\n=== VERDICT ===")
    println("distinct shin values NOT already in the group table: ", n_new)
    if n_new == 0
        println("  => the enumeration produced NO new values. This CONFIRMS the")
        println("     a = P_i*(alpha)*(beta)^{-1} construction cannot reach the")
        println("     missing monoid classes (it only ever varies the ideal")
        println("     CLASS, which find_alpha_rhoB_s1_s2_O already collapses).")
        println("     Optimizing / raising the bound will NOT help; the")
        println("     Theorem-2.19 entry point for the monoid case needs to")
        println("     change so the s-vector depends on `a` itself, not just [a].")
    else
        println("  => the enumeration DOES produce new values, so the construction")
        println("     is sound and the issue is genuinely performance/bounds.")
    end
    return (target = target, valgroups = valgroups, triples = triples)
end


"""
    diagnose_stuck_monoid_lookup(d::Int, p1::Int, p2::Int;
                                  count_rounds::Int = 12,
                                  per_candidate_seconds::Float64 = 0.4,
                                  tol::Float64 = 1e-9)

Written for a lookup that has been running for a long time without a match
(observed: p=(0,18), d=21 - stuck past round 4 (norm bound 3200, 5905
candidates) with no match yet). Answers two separate questions WITHOUT
running the actual (expensive) search further:

  1. Is there a CHEAP, direct match we're simply not trying? The brute-force
     enumeration in `ray_class_monoid_lookup` only ever tries ONE
     "natural" direct candidate up front (`p_query`, using the BASE fiber's
     own root) before falling back to search - it never tries the analogous
     natural candidate at the OTHER 3 Picard classes' own fibers. Per
     Theorem 2.19, evaluating shin at EACH fiber's own root with THIS SAME
     grid point r = (p1,p2)/d is exactly the value that fiber's own
     (alpha, rho_B, b) triple would assign to the monoid class it defines at
     r - a genuine, meaningful candidate for that fiber, not a coincidence.
     This is precisely how EVERY unit-point group-table entry already
     arises (some fiber's own natural point); this checks whether p=(0,18)
     happens to be one of the (rarer, unit-point-style) cases where the SAME
     coincidence holds for a non-unit point too. This is an O(4) check
     (one per Picard class), taking on the order of a few seconds, not
     minutes.

  2. If no direct hit, HOW LONG would the current brute-force search
     actually take? Reproduces `ray_class_monoid_lookup`'s exact candidate-
     COUNTING logic (same Q_i/reduced-basis construction, same coord_bound
     formula) for `count_rounds` successive norm-bound-quadrupling rounds,
     WITHOUT calling the expensive find_alpha_rhoB_s1_s2_O on any of them -
     this reproduces the "candidates to check: N" numbers already visible
     in the stuck run's own log, then projects several rounds further,
     reporting cumulative candidate counts and an estimated wall-clock time
     at `per_candidate_seconds` (default 0.4s, the measured WARM-candidate
     cost after the class_hints and Int128 fixes). This turns "it might
     finish eventually" into a concrete number to decide by.

Does NOT modify or interact with `ray_class_monoid_lookup`'s own state
(candidate_cache, class_hints, monoid_table) - entirely read-only relative
to a run in progress, safe to run in a separate process while the original
search is (or was) still going.
"""
function diagnose_stuck_monoid_lookup(d::Int, p1::Int, p2::Int;
                                       count_rounds::Int = 12,
                                       per_candidate_seconds::Float64 = 0.4,
                                       tol::Float64 = 1e-9)
    F = AdmissibleTuple(d)
    sd = _ray_class_shin_setup(F)
    D, K, O, d_check = sd.D, sd.K, sd.O, sd.d_check
    P_O, mP_O, rho_reps, F_class = sd.P_O, sd.mP_O, sd.rho_reps, sd.F_class
    rho_base = rho_reps[F_class]

    g = gcd(gcd(p1, p2), d)
    println("p = ($p1,$p2)   gcd(p1,p2,d) = ", g,
            g > 1 ? "  -> NON-UNIT point (genuine monoid-only class, no group-table entry can exist)" :
                    "  -> unit point (should already be in the group table - if this is being ",
                    "searched at all, something upstream is wrong)")

    # Target value, computed EXACTLY as twisted_chi_general's own
    # shin_val_lookup - same A_lookup/rho_bf_base construction, same inputs.
    L_rho_base = hj_automorph_for_root(rho_base, D)
    A_lookup, _ = stabilizer_power_in_Gamma_s(L_rho_base, [QQ(p1, d), QQ(p2, d)])
    rho_bf_base = BigFloat(coeff(rho_base, 0)) + BigFloat(coeff(rho_base, 1)) * sqrt(BigFloat(D))
    target_val = Complex{BigFloat}(shin(A_lookup, rho_bf_base, [p1, p2], d))
    println("target shin value = ", target_val)

    # ---- PART 1: the O(4) direct check across ALL Picard classes ----
    println("\n--- PART 1: direct candidate at each Picard class's own fiber ---")
    picard_classes = collect(P_O)
    class_ideals = [mP_O(u) for u in picard_classes]
    any_hit = false
    for (idx, u_i) in enumerate(picard_classes)
        Ji = class_ideals[idx]
        rho_i_init = rho_reps[u_i]
        local alpha_i, rho_i
        ok = true
        try
            alpha_i, rho_i, _, _ = find_alpha_rhoB_s1_s2_O(
                Ji, rho_i_init, rho_base, u_i, P_O, mP_O, O, K, d, rho_base, d_check)
        catch e
            ok = false
        end
        if !ok
            println("  class $idx: could not determine this fiber's own (alpha, rho_B) - skipped")
            continue
        end
        rho_i_bf = BigFloat(coeff(rho_i, 0)) + BigFloat(coeff(rho_i, 1)) * sqrt(BigFloat(D))
        L_i = hj_automorph_for_root(rho_i, D)
        local A_i
        try
            A_i, _ = stabilizer_power_in_Gamma_s(L_i, [QQ(p1, d), QQ(p2, d)])
        catch e
            println("  class $idx: could not build the stabilizer matrix for this point at this ",
                    "fiber - skipped")
            continue
        end
        val_i = Complex{BigFloat}(shin(A_i, rho_i_bf, [p1, p2], d))
        is_match = abs(val_i - target_val) < tol
        any_hit |= is_match
        println("  class $idx (rho_B=", rho_i, "):  shin = ", val_i,
                is_match ? "   <<< MATCH - this is the answer, no brute-force search needed" : "")
    end
    if any_hit
        println("\n=> A direct match was found above. This point does NOT need the brute-force ",
                "search at all - if `ray_class_monoid_lookup` is still running for this point, ",
                "it can be stopped once this diagnostic confirms the value.")
        return (found_direct = true,)
    end
    println("\n=> No direct match at any of the ", length(picard_classes),
            " fibers' own natural point. This is expected/common, not evidence of a bug on its ",
            "own - most monoid classes are not this simple. Proceeding to PART 2.")

    # ---- PART 2: candidate-count projection, WITHOUT the expensive search ----
    println("\n--- PART 2: candidate-count growth projection (counting only, no ",
            "find_alpha_rhoB_s1_s2_O calls) ---")
    places = real_places(K)
    n_classes = length(picard_classes)
    Pgens = [[_nf_elem_of(K, x) for x in basis(Ji)] for Ji in class_ideals]
    NQ = [QQ(norm(Ji)) for Ji in class_ideals]
    Qgens_raw = [[quad_conj(K, gg) for gg in Pgens[i]] for i in 1:n_classes]
    Qgens = [collect(lattice_reduce_2d(Qgens_raw[i][1], Qgens_raw[i][2], K, places))
             for i in 1:n_classes]

    tried = [Set{Tuple{Int,Int}}() for _ in 1:n_classes]
    norm_bound = 50
    cumulative = 0
    println("  (norm bound, this round's candidates, cumulative, est. cumulative time",
            " at ", per_candidate_seconds, "s/candidate)")
    for round_num in 1:count_rounds
        n_this_round = 0
        for i in 1:n_classes
            w1, w2 = Qgens[i][1], Qgens[i][2]
            coord_bound = max(5, Int(ceil(sqrt(Float64(norm_bound)))) * 3)
            for c1 in 0:coord_bound, c2 in (c1 == 0 ? 1 : -coord_bound):coord_bound
                (c1, c2) in tried[i] && continue
                alpha = K(c1) * w1 + K(c2) * w2
                Nalpha = abs(norm(alpha))
                iszero(Nalpha) && continue
                Na = Nalpha // NQ[i]
                Na > norm_bound && continue
                push!(tried[i], (c1, c2))
                n_this_round += 1
            end
        end
        cumulative += n_this_round
        est_hours = cumulative * per_candidate_seconds / 3600
        println("  round $round_num: norm_bound=$norm_bound  this_round=$n_this_round",
                "  cumulative=$cumulative  est_time=", round(est_hours, digits=2), " hours")
        norm_bound *= 4
    end
    println("\nCompare the early rounds above (norm_bound 50, 200, 800, 3200) against the ",
            "candidates-to-check numbers already printed by the actual stuck run - they should ",
            "match exactly (same counting logic); if they don't, that itself is a red flag worth ",
            "reporting back before trusting anything else here.")
    println("Use the projected cumulative/est_time numbers at later rounds to decide whether to ",
            "keep waiting, raise per_candidate_seconds if the real run has been running slower ",
            "than the WARM-candidate benchmark, or abort and treat this point as needing a ",
            "different approach entirely.")
    return (found_direct = false, target_val = target_val)
end


"""
    diagnose_reduced_level_consistency(d::Int, p1::Int, p2::Int; tol::Float64 = 1e-9)

Tests a specific hypothesis about why a non-unit point's monoid lookup might
be searching for a value that no candidate can ever match, raised while
p=(0,18), d=21 was stuck past round 8 (~2 million candidates, norms up to
~800,000, no match - see diagnose_stuck_monoid_lookup) with no direct hit
at any of the 4 Picard classes either.

`shin_val_lookup` in twisted_chi_general is built as:
    A_lookup, _ = stabilizer_power_in_Gamma_s(L_rho_base, [p1/d, p2/d])
    shin_val_lookup = shin(A_lookup, rho_bf_base, [p1, p2], d)
For a NON-UNIT point (gcd(p1,p2,d) = g > 1), the rational number p1/d (resp.
p2/d) has TRUE ORDER d/g in Q/Z, not d - e.g. for p=(0,18), d=21: g=3, and
18/21 = 6/7 in lowest terms, order 7. `stabilizer_power_in_Gamma_s` finds
its result purely from the RATIONAL VALUE of [p1/d,p2/d] (via QQ, which
Hecke reduces to lowest terms internally), so A_lookup is unaffected by
which integer representation was used to express that same rational value -
it is exactly the stabilizer for a point of true order d/g, regardless.

But `shin`'s own internal formula does
    p_transformed = mod.(s_R .* (red.R * Int.(p)), d)
using the RAW INTEGER p and the RAW MODULUS d passed in - NOT the point's
reduced order. So shin(A_lookup, rho, [0,18], 21) and
shin(A_lookup, rho, [0,6], 7) - two ways of asking "what is the value at
the same underlying rational point 6/7" - are NOT obviously required to
agree; they invoke shin's mod-d reduction at two DIFFERENT moduli entirely.
If they disagree, `shin_val_lookup` for non-unit points is not well-defined
as currently computed, which would fully explain a brute-force search never
finding a match at any norm - there would be nothing to find.

Compares these two evaluations directly, with no new field/order setup
needed: `shin` takes `p` and `d` as raw arguments independent of any
AdmissibleTuple, so this only needs the SAME A_lookup and rho_base already
in scope in twisted_chi_general - one comparison, seconds to run, gives a
definitive yes/no rather than more speculation.

If p is already a unit point (gcd(p1,p2,d) = 1), there is no smaller level
to compare against and the function reports that and returns.
"""
function diagnose_reduced_level_consistency(d::Int, p1::Int, p2::Int; tol::Float64 = 1e-9)
    F = AdmissibleTuple(d)
    sd = _ray_class_shin_setup(F)
    D, K, O = sd.D, sd.K, sd.O
    rho_reps, F_class = sd.rho_reps, sd.F_class
    rho_base = rho_reps[F_class]

    g = gcd(gcd(p1, p2), d)
    println("p = ($p1,$p2)   d = $d   gcd(p1,p2,d) = ", g)
    if g == 1
        println("This is already a unit point (primitive at modulus $d) - there is no smaller ",
                "level to compare against, so there is nothing for this diagnostic to test.")
        return nothing
    end

    d_reduced = d ÷ g
    p1_reduced = p1 ÷ g
    p2_reduced = p2 ÷ g
    println("Reduced level: d' = d/g = ", d_reduced, "   (p1',p2') = (", p1_reduced, ",",
            p2_reduced, ")")

    r1 = QQ(p1, d); r1_red = QQ(p1_reduced, d_reduced)
    r2 = QQ(p2, d); r2_red = QQ(p2_reduced, d_reduced)
    println("sanity check: p1/d = ", r1, "   p1'/d' = ", r1_red, "   (must be EQUAL)")
    println("sanity check: p2/d = ", r2, "   p2'/d' = ", r2_red, "   (must be EQUAL)")
    (r1 == r1_red && r2 == r2_red) ||
        error("diagnose_reduced_level_consistency: the reduction arithmetic itself is wrong ",
              "(r1=", r1, " vs r1_red=", r1_red, ", r2=", r2, " vs r2_red=", r2_red,
              ") - stop here, this is a bug in THIS diagnostic, not in the thing being diagnosed")

    L_rho_base = hj_automorph_for_root(rho_base, D)
    A_lookup, _ = stabilizer_power_in_Gamma_s(L_rho_base, [r1, r2])
    rho_bf_base = BigFloat(coeff(rho_base, 0)) + BigFloat(coeff(rho_base, 1)) * sqrt(BigFloat(D))

    val_full = Complex{BigFloat}(shin(A_lookup, rho_bf_base, [p1, p2], d))
    val_reduced = Complex{BigFloat}(shin(A_lookup, rho_bf_base, [p1_reduced, p2_reduced], d_reduced))

    println("\nshin(A, rho, [", p1, ",", p2, "], ", d, ")           = ", val_full)
    println("shin(A, rho, [", p1_reduced, ",", p2_reduced, "], ", d_reduced, ")             = ",
            val_reduced)
    diff = abs(val_full - val_reduced)
    println("|difference| = ", diff, diff < tol ? "   <- SAME (within tol)" : "   <- DIFFERENT")

    if diff < tol
        println("\n=> shin IS consistent between the full and reduced representations of the ",
                "same rational point. shin_val_lookup for this point should be a genuine, ",
                "well-defined target - the stuck search's failure to find it points elsewhere ",
                "(most likely: it genuinely needs a norm beyond what has been tried, or there is ",
                "a separate bug in the enumeration/comparison itself, not in how the target value ",
                "was computed).")
    else
        println("\n=> shin gives a DIFFERENT value depending on whether it is evaluated with the ",
                "point's FULL (unreduced) integers or its TRUE reduced level. shin_val_lookup, as ",
                "currently computed by twisted_chi_general for any non-unit point (gcd(p1,p2,d) > ",
                "1), is ambiguous - two equally natural ways of expressing the SAME target point ",
                "give different answers. This would fully explain why the brute-force search can ",
                "never find a match at any norm: there is no single well-defined value to find. ",
                "This needs to be resolved (almost certainly by pinning down which of the two - ",
                "or a third, correct - convention the theory actually specifies) before trusting ",
                "ANY lookup for a point with gcd(p1,p2,d) > 1.")
    end
    return (val_full = val_full, val_reduced = val_reduced, diff = diff)
end


"""
    diagnose_coord_bound_sufficiency(d::Int; norm_bound::Int = 200,
                                      safety_factors::Vector{Int} = [3, 6, 10, 20, 40])

Tests the ONE remaining unruled-out explanation for why the p=(0,18), d=21
search found nothing across ~2 million candidates (norms up to ~800,000):
that shin_val_lookup is well-defined (confirmed exactly by
diagnose_reduced_level_consistency, difference = 0.0), the enumeration's
underlying MATH is exhaustive (proven earlier: every integral ideal `a` in
a Picard class arises as alpha*Q_i^{-1} for SOME alpha in Q_i), but the
IMPLEMENTATION's coordinate search radius,
    coord_bound = max(5, ceil(sqrt(norm_bound)) * 3)
(the "*3" was chosen by feel when this was written, never verified
sufficient) might be too small for some class's specific lattice geometry
after `lattice_reduce_2d` - which would mean the search silently SKIPS
valid candidates within the norm bound it claims to have covered, not
merely being slow to reach large ones. This is a different, more serious
failure mode than "needs more time" - it means the search could run
FOREVER and still never find a match that exists at a modest, already-
searched norm.

For each Picard class and each safety factor in `safety_factors`, counts
how many DISTINCT Na = N(a) values are found up to `norm_bound`. Prints the
counts side by side.

READ THE OUTPUT AS:
  - If the count is the SAME for every safety factor (per class), the *3
    factor was already sufficient at this norm_bound - no completeness gap
    here, and the p=(0,18) mystery is more likely explained by that class
    genuinely needing a very large norm (surprising for D=44, but not
    something this diagnostic rules out).
  - If the count INCREASES as the safety factor grows (does not plateau),
    that proves the current search implementation is incomplete even at
    small, already-explored norm bounds - candidates within the stated
    bound are being missed, and `ray_class_monoid_lookup`'s coord_bound
    formula needs to be fixed (a provably-sufficient bound derived from the
    reduced basis' own properties, not a fixed multiplicative fudge factor)
    before any further monoid search results can be trusted.
"""
function diagnose_coord_bound_sufficiency(d::Int; norm_bound::Int = 200,
                                           safety_factors::Vector{Int} = [3, 6, 10, 20, 40])
    F = AdmissibleTuple(d)
    sd = _ray_class_shin_setup(F)
    K, O = sd.K, sd.O
    P_O, mP_O = sd.P_O, sd.mP_O

    picard_classes = collect(P_O)
    n_classes = length(picard_classes)
    class_ideals = [mP_O(u) for u in picard_classes]
    Pgens = [[_nf_elem_of(K, x) for x in basis(Ji)] for Ji in class_ideals]
    NQ = [QQ(norm(Ji)) for Ji in class_ideals]
    Qgens_raw = [[quad_conj(K, gg) for gg in Pgens[i]] for i in 1:n_classes]
    places = real_places(K)
    Qgens = [collect(lattice_reduce_2d(Qgens_raw[i][1], Qgens_raw[i][2], K, places))
             for i in 1:n_classes]

    println("norm_bound = ", norm_bound, "   safety_factors tested = ", safety_factors)
    for i in 1:n_classes
        w1, w2 = Qgens[i][1], Qgens[i][2]
        println("\n=== class $i of $n_classes  (N(Q_i) = ", NQ[i], ") ===")
        prev_count = nothing
        for sf in safety_factors
            coord_bound = max(5, Int(ceil(sqrt(Float64(norm_bound)))) * sf)
            found_Nas = Set{Any}()
            for c1 in 0:coord_bound, c2 in (c1 == 0 ? 1 : -coord_bound):coord_bound
                alpha = K(c1) * w1 + K(c2) * w2
                Nalpha = abs(norm(alpha))
                iszero(Nalpha) && continue
                Na = Nalpha // NQ[i]
                Na > norm_bound && continue
                push!(found_Nas, Na)
            end
            n_found = length(found_Nas)
            grew = prev_count !== nothing && n_found > prev_count
            println("  safety_factor=", sf, "  coord_bound=", coord_bound,
                    "  distinct Na values found=", n_found,
                    grew ? "   <-- GREW from previous factor (missing candidates at smaller factor!)" : "")
            prev_count = n_found
        end
    end
    println("\nIf every class's count plateaus (stops growing) well before the largest safety ",
            "factor tested, the *3 default is fine at this norm_bound. If any class is STILL ",
            "growing at the largest factor tested, re-run with larger norm_bound and/or ",
            "safety_factors - an unbounded growth pattern would be strong evidence of a genuine ",
            "completeness bug, not just needing a bigger (but still fixed) fudge factor.")
end


"""
    diagnose_theorem219_cross_check(d::Int, p1::Int, p2::Int; tol::Float64 = 1e-9)

Cross-checks `shin_val_lookup` (the value twisted_chi_general's monoid
search is trying to match) against `ray_class_monoid_table_v2`'s
independent, direct Theorem 2.19 construction - built earlier this session
specifically as a from-the-definition alternative to the search-and-compare
approach, and never previously run against a real stuck case.

This closes a real gap in `diagnose_stuck_monoid_lookup`'s Part 1 check:
that check evaluated shin at each Picard class's OWN root using the RAW,
unshifted (p1,p2) - but Theorem 2.19 requires a SIGN-NORMALIZED LIFT (find
n1 so that r1 = p1/d + n1 satisfies r2*rho' - r1 > 0) before the
construction is valid; `ray_class_monoid_shin_table` (the original, pre-v2
monoid builder) was found earlier this session to have exactly this bug
(using an unnormalized lift), and v2 was written to fix it. Part 1 never
applied that same normalization, so it was testing a DIFFERENT, likely
invalid construction at each class, not the one v2 (or the theorem) actually
specifies.

For each Picard class, replicates ray_class_monoid_table_v2's exact
per-class, per-point logic (find_alpha_rhoB_s1_s2_O for that class's own
(alpha,rho), the trace-sign-normalized L, the sign-normalized lift rn, and
shin(A, rho_bf, rn, d) using the SHIFTED rn - not the raw (p1,p2)) and
compares each class's resulting value against shin_val_lookup.

READ THE OUTPUT AS:
  - A match at any class: that class's Theorem-2.19 construction (with the
    correct lift normalization) IS the answer, and the brute-force search
    was simply exploring the wrong space (or this class's true generator
    genuinely has a large norm, and this bypasses needing to find it by
    search at all, going straight to (alpha, rho, s1, s2) for the class).
  - No match anywhere: strengthens the case that shin_val_lookup itself,
    as twisted_chi_general currently constructs it (raw (p1,p2), no lift
    normalization, evaluated ONLY at the base fiber), is not what Theorem
    2.19's construction actually produces for ANY class at this point - a
    genuine methodology mismatch between how the target is computed and how
    every candidate (whether from the search or from v2) is computed, which
    would explain a search that can never succeed regardless of norm bound.
"""
function diagnose_theorem219_cross_check(d::Int, p1::Int, p2::Int; tol::Float64 = 1e-9)
    F = AdmissibleTuple(d)
    sd = _ray_class_shin_setup(F)
    D, K, O = sd.D, sd.K, sd.O
    P_O, mP_O, rho_reps, F_class, d_check, all_triples =
        sd.P_O, sd.mP_O, sd.rho_reps, sd.F_class, sd.d_check, sd.all_triples
    rho_base = rho_reps[F_class]

    # shin_val_lookup, EXACTLY as twisted_chi_general computes it: raw
    # (p1,p2), no lift normalization, base fiber only.
    L_rho_base = hj_automorph_for_root(rho_base, D)
    A_lookup, _ = stabilizer_power_in_Gamma_s(L_rho_base, [QQ(p1, d), QQ(p2, d)])
    rho_bf_base = BigFloat(coeff(rho_base, 0)) + BigFloat(coeff(rho_base, 1)) * sqrt(BigFloat(D))
    target_val = Complex{BigFloat}(shin(A_lookup, rho_bf_base, [p1, p2], d))
    println("target (shin_val_lookup, twisted_chi_general's own construction) = ", target_val)

    places = real_places(K)
    i_conj = findfirst(pl -> Float64(real(embedding(pl)(gen(K)))) < 0, places)
    emb_conj = embedding(places[i_conj])
    emb_main = embedding(places[3 - i_conj])
    fm(x) = Float64(real(emb_main(x)))
    fc(x) = Float64(real(emb_conj(x)))

    picard_classes = collect(P_O)
    println("\n--- ray_class_monoid_table_v2's own construction, per Picard class ---")
    for g in picard_classes
        Jg = nothing
        for t in all_triples
            if (mP_O \ t[1]) == g
                Jg = t[1]
                break
            end
        end
        if Jg === nothing
            println("  class $g: no ray class representative ideal found - skipped")
            continue
        end

        local alpha, rho
        ok = true
        try
            alpha, rho, _, _ = find_alpha_rhoB_s1_s2_O(Jg, rho_reps[g], rho_base, g, P_O, mP_O,
                                                         O, K, d, rho_base, d_check;
                                                         mn_height = 3000, q_gen_height = 3000)
        catch e
            ok = false
        end
        if !ok || !(fm(alpha) > 0 && fc(alpha) > 0) || !(fm(rho) > fc(rho))
            println("  class $g: could not get a valid Theorem 2.19 (alpha,rho) - skipped")
            continue
        end

        rho_bf = BigFloat(coeff(rho, 0)) + BigFloat(coeff(rho, 1)) * sqrt(BigFloat(D))
        rho_c = fc(rho)

        L_raw = hj_automorph_for_root(rho, D)
        L = (L_raw[1,1] + L_raw[2,2]) > 0 ? L_raw : -L_raw

        # the sign-normalized lift - THIS is what diagnose_stuck_monoid_lookup's
        # Part 1 check omitted
        r2 = QQ(p2, d)
        n1 = Int(floor(Float64(r2) * rho_c - p1 / d)) - 1
        r1 = QQ(p1, d) + n1
        @assert Float64(r2) * rho_c - Float64(r1) > 0
        rn = [p1 + d * n1, p2]

        local A
        try
            A, _ = stabilizer_power_in_Gamma_s(L, [QQ(rn[1], d), QQ(rn[2], d)])
        catch e
            println("  class $g: could not build stabilizer matrix for the shifted point - skipped")
            continue
        end
        val = Complex{BigFloat}(shin(A, rho_bf, rn, d))
        is_match = abs(val - target_val) < tol
        println("  class $g (rho=", rho, ", rn=", rn, "):  shin = ", val,
                is_match ? "   <<< MATCH" : "")
    end
    println("\nIf none of the classes above matched, this is strong evidence that ",
            "shin_val_lookup's own construction (raw point, base fiber only, no lift ",
            "normalization) is not producing a value any Theorem-2.19-correct construction ",
            "would ever assign to this point - a methodology mismatch, not a search-depth ",
            "problem.")
end


"""
    diagnose_full_monoid_membership(d::Int, p1::Int, p2::Int; tol::Float64 = 1e-9)

The decisive test, as opposed to another partial hypothesis: the ray class
monoid is FINITE. Rather than searching an ever-growing norm bound, or
testing one construction against another, build the ENTIRE monoid (every
class in every one of the 4 Picard fibers, via `ray_class_monoid_table_v2`
- the direct, from-the-definition construction: every grid point in every
fiber, correctly grouped by the Sigma-positive-unit equivalence, no search
or norm bound involved at all) and check directly whether `shin_val_lookup`
appears ANYWHERE in it.

Two checks, from most to least specific:

  1. SELF-CONSISTENCY: find the specific monoid class, within the BASE
     fiber (F_class) alone, that (p1,p2) itself belongs to (i.e. the entry
     whose `members` list contains (p1,p2)), and compare ITS shin_value
     against shin_val_lookup - which was ALSO computed at this exact fiber,
     for this exact point. These are two different code paths (v2's
     from-the-definition construction vs twisted_chi_general's direct
     covariance-based formula) evaluating what should be the SAME thing.
     If they disagree, that is a direct, surgical proof of a bug in one of
     the two constructions - not a search-depth or coverage question at
     all.

  2. FULL-TABLE MEMBERSHIP: scan every entry across all 4 fibers for a
     shin_value matching the target, exactly per the project owner's
     proposal - if the finite monoid has been built completely and
     correctly, the target MUST appear somewhere in it if it is a genuine,
     reachable value at all. No match anywhere is definitive: it proves
     shin_val_lookup does not correspond to any actual monoid class, i.e. a
     real bug in how the target value itself is computed - not evidence
     that a larger search or a different construction would eventually
     succeed.

Also reports the total entry/class counts per fiber, for context (how large
the "entire monoid" actually is here).
"""
function diagnose_full_monoid_membership(d::Int, p1::Int, p2::Int; tol::Float64 = 1e-9)
    F = AdmissibleTuple(d)
    sd = _ray_class_shin_setup(F)
    D, K = sd.D, sd.K
    rho_reps, F_class = sd.rho_reps, sd.F_class
    rho_base = rho_reps[F_class]

    L_rho_base = hj_automorph_for_root(rho_base, D)
    A_lookup, _ = stabilizer_power_in_Gamma_s(L_rho_base, [QQ(p1, d), QQ(p2, d)])
    rho_bf_base = BigFloat(coeff(rho_base, 0)) + BigFloat(coeff(rho_base, 1)) * sqrt(BigFloat(D))
    target_val = Complex{BigFloat}(shin(A_lookup, rho_bf_base, [p1, p2], d))
    println("target (shin_val_lookup) = ", target_val)

    println("\nBuilding the ENTIRE monoid (all 4 Picard fibers, every grid point, no search) ",
            "via ray_class_monoid_table_v2 ...")
    entries = ray_class_monoid_table_v2(F; shared_data = sd, verbose = true)
    println("\nTotal monoid classes found across all fibers: ", length(entries))

    # ---- check 1: self-consistency within the base fiber specifically ----
    println("\n--- CHECK 1: self-consistency at the base fiber for (p1,p2)=(", p1, ",", p2, ") ---")
    base_entry = nothing
    for e in entries
        e.A0 == F_class || continue
        if (p1, p2) in e.members
            base_entry = e
            break
        end
    end
    if base_entry === nothing
        println("  (p1,p2) was not found in ANY base-fiber equivalence class - unexpected, ",
                "since ray_class_monoid_table_v2 enumerates every point in 0:d-1 x 0:d-1 for ",
                "every fiber. This itself would be worth reporting back.")
    else
        diff1 = abs(Complex{BigFloat}(base_entry.shin_value) - target_val)
        println("  base-fiber class containing (", p1, ",", p2, "): p_vec=", base_entry.p_vec,
                "  r_num=", base_entry.r_num, "  is_unit=", base_entry.is_unit)
        println("  v2's shin_value for this class = ", base_entry.shin_value)
        println("  |difference from shin_val_lookup| = ", diff1,
                diff1 < tol ? "   <- SAME (self-consistent)" : "   <- DIFFERENT (inconsistent!)")
    end

    # ---- check 2: full-table membership, exactly the project owner's proposal ----
    println("\n--- CHECK 2: does the target value appear ANYWHERE in the full monoid? ---")
    best_idx = 0
    best_diff = nothing
    for (idx, e) in enumerate(entries)
        diff = abs(Complex{BigFloat}(e.shin_value) - target_val)
        if best_diff === nothing || diff < best_diff
            best_diff = diff
            best_idx = idx
        end
    end
    if best_diff !== nothing && best_diff < tol
        e = entries[best_idx]
        println("  MATCH found: fiber A0=", e.A0, "  p_vec=", e.p_vec, "  r_num=", e.r_num,
                "  is_unit=", e.is_unit, "  |diff|=", best_diff)
    else
        println("  NO MATCH anywhere in the full, exhaustively-constructed monoid.")
        println("  closest entry found: fiber A0=", entries[best_idx].A0,
                "  p_vec=", entries[best_idx].p_vec, "  |diff|=", best_diff, " (NOT within tol)")
        println("\n  => Since the monoid is finite and has now been built in full (not merely ",
                "searched up to some norm), this is definitive: shin_val_lookup does not ",
                "correspond to any actual monoid class. This points to a bug in how the target ",
                "value itself is computed (twisted_chi_general's shin_val_lookup construction), ",
                "not in the search or in ray_class_monoid_table_v2's construction.")
    end
    return (target_val = target_val, entries = entries, base_entry = base_entry,
            best_idx = best_idx, best_diff = best_diff)
end


"""
    diagnose_coordinate_convention_mismatch(d::Int, points::Vector{Tuple{Int,Int}};
                                             tol::Float64 = 1e-9)

Follows up on diagnose_full_monoid_membership's finding for p=(0,18), d=21:
shin_val_lookup(0,18) matched ray_class_monoid_table_v2's value for a
DIFFERENT point, (3,0), to 73 decimal digits (diff ~ 1e-73, i.e. as exact
as BigFloat precision allows) - not a coincidence, a genuine match, just
under the "wrong" label. Since 18 = -3 mod 21, (0,18) = (0,-3), and the
matched point (3,0) = (-(-3), 0) - exactly the map
    (p1,p2) -> (-p2, p1)     (the 90-degree rotation matrix [[0,-1],[1,0]])
This is a real, fixable coordinate-CONVENTION mismatch between how
twisted_chi_general labels grid points for shin_val_lookup and how the rest
of the table machinery (v2, and presumably the group table too) labels
them - NOT evidence that the target is unreachable or needs a bigger
search.

This function tests whether that SAME rotation (or one of several other
natural candidate transforms - identity, both 90-degree rotations, both
reflections along the two axes, and negation of both coordinates) explains
the match UNIVERSALLY across several points, not just the one already
checked. For each point in `points` and each candidate transform T, checks
whether shin_val_lookup(p) matches v2's base-fiber value at T(p) (using the
SAME full-monoid membership lookup as diagnose_full_monoid_membership, so
this does not depend on T(p) already being the SAME class in v2's own
grouping - it looks up whichever class actually contains T(p)).

READ THE OUTPUT AS: if exactly ONE candidate transform matches for EVERY
point tested, that confirms a universal, systematic convention mismatch -
almost certainly a swapped/negated (s1,s2) or (p1,p2) coordinate somewhere
in how twisted_chi_general builds A_lookup/rho_bf_base relative to how the
rest of the codebase (v2's rn construction, or the group table's own s1/s2)
builds the equivalent quantities. If different points need different
transforms, or none work consistently, the (0,18) match may be closer to a
coincidence than a systematic bug, and this needs more investigation before
attempting a fix.
"""
function diagnose_coordinate_convention_mismatch(d::Int, points::Vector{Tuple{Int,Int}};
                                                  tol::Float64 = 1e-9)
    F = AdmissibleTuple(d)
    sd = _ray_class_shin_setup(F)
    D, K = sd.D, sd.K
    rho_reps, F_class = sd.rho_reps, sd.F_class
    rho_base = rho_reps[F_class]
    L_rho_base = hj_automorph_for_root(rho_base, D)
    rho_bf_base = BigFloat(coeff(rho_base, 0)) + BigFloat(coeff(rho_base, 1)) * sqrt(BigFloat(D))

    println("Building the full monoid once (shared across all points tested) ...")
    entries = ray_class_monoid_table_v2(F; shared_data = sd, verbose = false)
    println("Total monoid classes: ", length(entries), "\n")

    transforms = [
        ("identity",        (a, b) -> (a, b)),
        ("rotate90 CCW",    (a, b) -> (-b, a)),
        ("rotate90 CW",     (a, b) -> (b, -a)),
        ("negate both",     (a, b) -> (-a, -b)),
        ("swap",            (a, b) -> (b, a)),
        ("reflect p1",      (a, b) -> (-a, b)),
        ("reflect p2",      (a, b) -> (a, -b)),
    ]

    function value_at(p1, p2)
        A_lookup, _ = stabilizer_power_in_Gamma_s(L_rho_base, [QQ(p1, d), QQ(p2, d)])
        return Complex{BigFloat}(shin(A_lookup, rho_bf_base, [p1, p2], d))
    end

    function v2_value_containing(p1, p2)
        pm = (mod(p1, d), mod(p2, d))
        for e in entries
            e.A0 == F_class || continue
            pm in e.members && return Complex{BigFloat}(e.shin_value)
        end
        return nothing
    end

    results = Dict{String, Vector{Bool}}(name => Bool[] for (name, _) in transforms)
    for (p1, p2) in points
        target = value_at(p1, p2)
        println("p=($p1,$p2):  shin_val_lookup = ", target)
        for (name, tfun) in transforms
            tp1, tp2 = tfun(p1, p2)
            v2val = v2_value_containing(tp1, tp2)
            ok = v2val !== nothing && abs(v2val - target) < tol
            push!(results[name], ok)
            println("    T=$name  ->  (", mod(tp1, d), ",", mod(tp2, d), ")  match=", ok)
        end
        println()
    end

    println("--- summary: which transform matched EVERY point tested? ---")
    any_universal = false
    for (name, _) in transforms
        all_ok = all(results[name])
        println("  $name: ", count(results[name]), "/", length(points), " points matched",
                all_ok ? "   <<< UNIVERSAL MATCH" : "")
        any_universal |= all_ok
    end
    any_universal || println("\nNo single candidate transform explained every point tested - ",
                              "the true mismatch (if it is one) may not be one of the 7 simple ",
                              "candidates tried here.")
    return results
end


"""
    diagnose_monoid_lookup_v2_timing(d::Int; n_candidates_per_class = 3,
                                      coord_bound = 5)

Cheap, timed probe for the CURRENT `ray_class_monoid_lookup` enumerator
(Q_i = conj(class_ideal), reduced via `lattice_reduce_2d`, alpha ranging
over Q_i's REDUCED basis). Written after repeated bugs in that function
each cost multiple minutes of a real run to discover. Copies the exact
same construction steps `ray_class_monoid_lookup` uses, capped to a
HANDFUL of small-coordinate candidates per class with timing at every
step, so a further mistake costs seconds, not minutes.

For each Picard class, prints:
  - N(P_i) (the class's own norm) and N(Q_i) (MUST be identical - if not,
    the conj(P_i) construction itself is broken);
  - the RAW and REDUCED Q_i basis' embedded-norm^2 (reduction should bring
    these down close to N(Q_i)'s own scale - a small residual factor, e.g.
    single digits to a few dozen, is normal for 2D lattice reduction on an
    indefinite form and not itself a problem; orders of magnitude off is).

TWO RETRACTED HYPOTHESES (both disproven by earlier versions of this probe):
  1. That the SIZE of `a`'s own generator coefficients (as opposed to
     Q_i's) was driving the observed 9-21s-per-candidate
     find_alpha_rhoB_s1_s2_O timings (vs ~2ms for successful calls during
     the original group-table build). Reduced-vs-unreduced `a` generators,
     same candidate: IDENTICAL timing.
  2. That `a`'s Hecke TYPE (a genuine FractionalIdeal object, even though
     N(a) is always a plain positive integer) was the cause.
     FractionalIdeal-typed `a` vs `numerator(a)`: IDENTICAL timing.
  3. That find_alpha_rhoB_s1_s2_O's 3rd/10th positional arguments
     (`rho`/`rho_j`) needed to be `rho_reps[F_class]` rather than the
     per-class `rho_reps[u_a]` this function was substituting for all three
     roles. This IS the argument-by-argument correct call (matching the
     real call site exactly) and was kept for that reason, but WRONG vs
     RIGHT timed identically - it does not explain the slowness either.
  All three A/B tests showed a clean PER-CLASS timing signature instead -
  different candidates in the same class all took essentially the same
  time, regardless of their own norm, representation, or which call
  arguments were used - pointing at something keyed to the CLASS being
  processed, not to `a` or the call itself.

CONFIRMED CAUSE, found by reading find_alpha_rhoB_s1_s2_O's own internals
rather than continuing to guess about its inputs:
its main (non-hinted) path calls find_b_with_verified_principal_product, an
O(height^5)-cost quadruple-nested coordinate search whose result depends
ONLY on the ideal's Picard class (per that function's own docstring
comment) - exactly matching the observed per-class signature.
find_alpha_rhoB_s1_s2_O already has a fast path built for precisely this
situation (`alpha_hint`/`rho_B_hint`/`b_O_hint`, documented as existing so
per-ray-class table builders pay this search ONCE per Picard class, not
once per ray class) - this probe (and `ray_class_monoid_lookup`, until just
fixed) was simply never using it, so every candidate re-ran the full search
from scratch even when several shared a class.

THIS TEST: mirrors `ray_class_monoid_lookup`'s `class_hints` cache /
`find_with_class_cache` helper exactly. For each class, the FIRST candidate
is COLD (a cache miss - pays the full find_b_with_verified_principal_product
search, plus one extra, deliberately redundant call to expose b_O, which
find_alpha_rhoB_s1_s2_O computes internally but does not return - mirrors
the SAME pattern already used elsewhere in this file, ray_class_shin_table's
`fiber_cache`). Every subsequent candidate in the same class is WARM (a
cache hit - uses the hint fast path).

READ THE OUTPUT AS:
  - If WARM candidates are dramatically faster than COLD (ideally close to
    the ~2ms scale from the very first diagnose_monoid_timing run at the
    start of this whole investigation), the cause is confirmed and the fix
    already applied to `ray_class_monoid_lookup` should carry over to a
    real run - candidates sharing a class (which most will, since there
    are only 4 Picard classes total here) should now be fast after the
    first one per class.
  - If COLD and WARM are similarly slow, this cache-based cause does not
    fully explain the slowness either. WARM calls run with
    find_alpha_rhoB_s1_s2_O's OWN `verbose=true` (a real, non-invasive
    fix added directly to that function, purely additive and gated behind
    the flag it already had - existing callers see no change), so their
    printed output now breaks down the fast path's own three internal
    steps with individual timings: `Jb_O/Q construction` (a single
    ideal multiply plus one Hecke inv() call), the `(i,j) search`
    (bounded by q_gen_height, default 3000 - a (2*3000+1)^2 ~ 36 million
    point grid, with tried/passed counts printed), and the final
    `is_principal_O_verified` check. Whichever of these three is large is
    the next thing to dig into - the (i,j) search or the inv() call being
    slow would point at a genuinely expensive Hecke operation regardless of
    caching; is_principal_O_verified being slow would point back at
    lattice_canonical's HNF cost (flagged as a possibility from the very
    first hypothesis in this investigation, on a candidate `a` had never
    actually been shown to be the cause of - this would finally confirm or
    rule that out directly).
"""
function diagnose_monoid_lookup_v2_timing(d::Int; n_candidates_per_class::Int = 3,
                                            coord_bound::Int = 5)
    F = AdmissibleTuple(d)
    sd = _ray_class_shin_setup(F)
    D, K, O, d_check = sd.D, sd.K, sd.O, sd.d_check
    P_O, mP_O, rho_reps, F_class = sd.P_O, sd.mP_O, sd.rho_reps, sd.F_class
    rho_base = rho_reps[F_class]

    picard_classes = collect(P_O)
    n_classes = length(picard_classes)
    println("Number of Picard classes: ", n_classes)

    # ---- EXACT SAME construction as ray_class_monoid_lookup, steps 1-2 ----
    class_ideals = [mP_O(g) for g in picard_classes]
    Pgens = [[_nf_elem_of(K, x) for x in basis(Pi)] for Pi in class_ideals]
    NQ = [QQ(norm(Pi)) for Pi in class_ideals]
    Qgens_raw = [[quad_conj(K, g) for g in Pgens[i]] for i in 1:n_classes]
    Qinv_gens = [[g // NQ[i] for g in Pgens[i]] for i in 1:n_classes]
    places = real_places(K)

    embnorm2(x) = Float64(embedding(places[1])(x))^2 + Float64(embedding(places[2])(x))^2

    # ---- class_hints cache, mirroring ray_class_monoid_lookup's fix exactly ----
    class_hints = Dict{Any, NamedTuple}()
    function find_with_class_cache(Jcand, u_cand)
        if haskey(class_hints, u_cand)
            hint = class_hints[u_cand]
            # verbose=true here specifically: this call is the hint-based
            # ("WARM") fast path, and the whole point of this probe is to see
            # the NEW internal timing breakdown (Jb_O/Q construction, the
            # (i,j) search, the final is_principal_O_verified) it now prints,
            # to find out which of those three is behind WARM still taking
            # 8-11s instead of dropping to the ~2ms scale.
            return find_alpha_rhoB_s1_s2_O(
                Jcand, rho_reps[u_cand], rho_base, u_cand, P_O, mP_O, O, K,
                d, rho_base, d_check;
                alpha_hint = hint.alpha, rho_B_hint = hint.rho_B, b_O_hint = hint.b_O,
                verbose = true)
        else
            result = find_alpha_rhoB_s1_s2_O(
                Jcand, rho_reps[u_cand], rho_base, u_cand, P_O, mP_O, O, K,
                d, rho_base, d_check)
            b_O_found, _ = find_b_with_verified_principal_product(
                O, K, Jcand, [ideal(O, d), ideal(O, d + 1)])
            class_hints[u_cand] = (alpha = result[1], rho_B = result[2], b_O = b_O_found)
            return result
        end
    end

    for i in 1:n_classes
        println("\n=== class $i of $n_classes ===")
        println("  N(P_i) = ", NQ[i], "   (this should equal N(Q_i) printed below, by construction)")
        println("  RAW Q_i basis, embedded-norm^2: ",
                embnorm2(Qgens_raw[i][1]), ", ", embnorm2(Qgens_raw[i][2]))

        t_reduce = @elapsed reduced = lattice_reduce_2d(Qgens_raw[i][1], Qgens_raw[i][2], K, places)
        w1, w2 = reduced
        println("  REDUCED Q_i basis, embedded-norm^2: ", embnorm2(w1), ", ", embnorm2(w2),
                "   (time: ", t_reduce, " s)")
        println("  |N(w1)| = ", abs(norm(w1)), "   |N(w2)| = ", abs(norm(w2)),
                "   <-- compare these to N(Q_i) = ", NQ[i],
                " above; should be the SAME rough size, not orders of magnitude larger")

        u_a = picard_classes[i]

        tested = 0
        for c1 in 0:coord_bound, c2 in (c1 == 0 ? 1 : -coord_bound):coord_bound
            tested >= n_candidates_per_class && break
            alpha = K(c1) * w1 + K(c2) * w2
            Nalpha = abs(norm(alpha))
            iszero(Nalpha) && continue
            tested += 1
            Na = Nalpha // NQ[i]

            a_gens = [alpha * g for g in Qinv_gens[i]]
            a = fractional_ideal(O, a_gens)

            # RETRACTED HYPOTHESES (all three disproven by earlier probe runs):
            #  1. a's own generator coefficient SIZE - reducing them via
            #     lattice_reduce_2d gave IDENTICAL timing to unreduced.
            #  2. a's Hecke TYPE (FractionalIdeal vs numerator(a)) -
            #     IDENTICAL timing either way.
            #  3. find_alpha_rhoB_s1_s2_O's rho argument (rho_a_init for all
            #     3 roles vs rho_base for the 3rd/10th) - IDENTICAL timing
            #     either way.
            # All three showed the SAME clean PER-CLASS timing signature -
            # same class, different candidates -> same ~10-40s cost -
            # pointing at something keyed to the CLASS itself, not to `a`
            # or to which arguments are passed.
            #
            # CONFIRMED CAUSE, found by reading find_alpha_rhoB_s1_s2_O's own
            # internals rather than guessing about its inputs further: its
            # main path calls find_b_with_verified_principal_product, an
            # O(height^5) search whose cost depends ONLY on the ideal's
            # Picard class (per that function's own docstring comment) -
            # exactly the observed signature. find_alpha_rhoB_s1_s2_O
            # already has a fast path for this (alpha_hint/rho_B_hint/
            # b_O_hint) which was simply never being used. This tests it
            # directly: is candidate #1 in each class (COLD - cache miss,
            # pays full search) dramatically slower than candidates #2, #3
            # (WARM - cache hit, reuses class_hints)?
            is_cold = !haskey(class_hints, u_a)
            local rho_a, s1_a, s2_a
            ok = true
            t_frho = @elapsed begin
                try
                    _, rho_a, s1_a, s2_a = find_with_class_cache(a, u_a)
                catch e
                    ok = false
                end
            end
            label = is_cold ? "COLD (1st in class - full search)" : "WARM (cached - hint fast path)"
            if !ok
                println("  candidate ($c1,$c2): N(alpha)=$Nalpha  N(a)=$Na  $label: FAILED  (",
                        round(t_frho, sigdigits=3), " s)")
                continue
            end
            println("  candidate ($c1,$c2): N(alpha)=$Nalpha  N(a)=$Na  $label: ",
                    round(t_frho, sigdigits=3), "s  s1=$s1_a s2=$s2_a")
        end
    end
    println("\n=== probe done ===")
end

"""
    diagnose_monoid_timing(d::Int; n_candidates = 5, norm_bound = 3,
                            mn_height = 50, q_gen_height = 50)

MINIMAL, TIMED probe: evaluates only `n_candidates` monoid candidates
(default 5), printing elapsed wall-clock time for EACH sub-step of EACH
candidate as it happens (not just at the end), so we get hard data even if
it never finishes. Specifically times, per candidate:
  - constructing a = P_i*(alpha)*(beta)^{-1} (the chained multiply/invert),
  - printing the size of a's resulting basis coefficients (to check whether
    the composed ideal has blown up far beyond what norm_bound=3 suggests -
    the suspected real cost driver, since find_alpha_rhoB_s1_s2_O's own
    search cost is believed to scale with J's basis size, not with our
    nominal alpha/beta height bound),
  - the find_alpha_rhoB_s1_s2_O call itself.
Uses a deliberately tiny norm_bound=3 and search heights=50 so even a
"normal-speed" run should finish in well under a minute; if it does not,
that itself is diagnostic (points at the ideal construction step, not the
search heights, being the actual bottleneck).
"""
function diagnose_monoid_timing(d::Int; n_candidates::Int = 5, norm_bound::Int = 3,
                                 mn_height::Int = 50, q_gen_height::Int = 50)
    F = AdmissibleTuple(d)
    sd = _ray_class_shin_setup(F)
    D, K, O, d_check = sd.D, sd.K, sd.O, sd.d_check
    P_O, mP_O, rho_reps = sd.P_O, sd.mP_O, sd.rho_reps

    dbar = mod(d, 2) == 1 ? d : 2 * d
    m_ideal = ideal(O, dbar)
    b1 = _nf_elem_of(K, basis(O)[1])
    b2 = _nf_elem_of(K, basis(O)[2])
    picard_classes = collect(P_O)
    class_ideals = [mP_O(g) for g in picard_classes]
    coord_bound = max(3, Int(ceil(sqrt(Float64(norm_bound)))) * 3)

    println("Picard class representative norms: ",
            [Int(norm(Pi)) for Pi in class_ideals])

    betas = Tuple{Int,Int,Int}[]
    for bi in -coord_bound:coord_bound, bj in -coord_bound:coord_bound
        (bi == 0 && bj == 0) && continue
        beta = K(bi) * b1 + K(bj) * b2
        nb = abs(Int(norm(beta)))
        (nb == 0 || nb > norm_bound) && continue
        is_beta_in_S_m(beta, O, m_ideal) || continue
        push!(betas, (bi, bj, nb))
    end
    alphas = Tuple{Int,Int,Int}[]
    for ai in -coord_bound:coord_bound, aj in -coord_bound:coord_bound
        (ai == 0 && aj == 0) && continue
        alpha = K(ai) * b1 + K(aj) * b2
        na = abs(Int(norm(alpha)))
        (na == 0 || na > norm_bound) && continue
        push!(alphas, (ai, aj, na))
    end
    println("alphas found: ", length(alphas), "   betas found: ", length(betas))
    isempty(alphas) && error("no alpha candidates at this norm_bound - raise it")
    isempty(betas) && error("no beta (in S_m) candidates at this norm_bound - raise it")

    n_done = 0
    for ci in 1:length(class_ideals), (ai, aj, na) in alphas, (bi, bj, nb) in betas
        n_done >= n_candidates && break
        n_done += 1
        println("\n--- candidate ", n_done, " of ", n_candidates,
                ": class ", ci, "  alpha_coords=(", ai, ",", aj, ") N=", na,
                "  beta_coords=(", bi, ",", bj, ") N=", nb, " ---")
        alpha = K(ai) * b1 + K(aj) * b2
        beta  = K(bi) * b1 + K(bj) * b2

        t1 = @elapsed a = class_ideals[ci] * ideal(O, alpha)
        println("  time to build P_i*(alpha): ", t1, " s")
        t2 = @elapsed a = a * inv(ideal(O, beta))
        println("  time to multiply by inv((beta)): ", t2, " s")

        t3 = @elapsed nrm = norm(a)
        println("  norm(a) = ", nrm, "  (time: ", t3, " s)")
        t4 = @elapsed bas = basis(a)
        coeffs_all = BigInt[]
        for b in bas
            nb_elem = _nf_elem_of(K, b)
            for k in (0, 1)
                c = QQ(coeff(nb_elem, k))
                push!(coeffs_all, abs(BigInt(numerator(c))))
                push!(coeffs_all, abs(BigInt(denominator(c))))
            end
        end
        max_coeff = maximum(coeffs_all)
        println("  basis(a) largest numerator/denominator (coeff size check): ", max_coeff,
                "  (time to fetch basis: ", t4, " s)")

        u_a = picard_classes[ci]
        rho_a_init = rho_reps[u_a]
        t5 = @elapsed begin
            local ok = true
            local rho_a, s1_a, s2_a
            try
                _, rho_a, s1_a, s2_a = find_alpha_rhoB_s1_s2_O(
                    a, rho_a_init, rho_a_init, u_a, P_O, mP_O, O, K,
                    d, rho_a_init, d_check; mn_height = mn_height,
                    q_gen_height = q_gen_height)
            catch e
                ok = false
                println("  find_alpha_rhoB_s1_s2_O THREW: ", e)
            end
            if ok
                println("  find_alpha_rhoB_s1_s2_O succeeded: rho_B=", rho_a,
                        " s1=", s1_a, " s2=", s2_a)
            end
        end
        println("  time in find_alpha_rhoB_s1_s2_O: ", t5, " s")
    end
    println("\n=== done: ", n_done, " candidates timed ===")
end


"""
    check_table_duplicates(d::Int; tol::Float64 = 1e-30)

Scans all entries of the ray class table for modulus `d` and groups them by
`shin_value` (within `tol`). Prints every group with more than one member -
i.e. every case where two or more DIFFERENT ray classes ended up with the
same (or numerically indistinguishable) stored shin value. A single isolated
group could reflect a genuine symmetry in the theory; many such groups would
indicate a systematic non-injectivity bug in how the table is constructed
(e.g. an unresolved sign/orientation ambiguity).
"""
function check_table_duplicates(d::Int; tol::Float64 = 1e-30)
    tbl = ray_class_shin_table_fast_quiet(d)
    valid_idxs = findall(x -> x !== nothing, tbl)
    used = falses(length(valid_idxs))
    n_groups = 0
    n_entries_in_groups = 0
    for (a, ia) in enumerate(valid_idxs)
        used[a] && continue
        group = [ia]
        for (b, ib) in enumerate(valid_idxs)
            b <= a && continue
            used[b] && continue
            if abs(tbl[ia].shin_value - tbl[ib].shin_value) < tol
                push!(group, ib)
                used[b] = true
            end
        end
        if length(group) > 1
            n_groups += 1
            n_entries_in_groups += length(group)
            println("Duplicate group ", n_groups, " (", length(group), " entries, shin_value = ",
                    tbl[ia].shin_value, "):")
            for i in group
                println("    entry ", i, ": J = ", tbl[i].J, "  s1 = ", tbl[i].s1,
                        "  s2 = ", tbl[i].s2, "  rho_B = ", tbl[i].rho_B)
            end
        end
        used[a] = true
    end
    println("\nTotal: ", n_groups, " duplicate group(s) covering ", n_entries_in_groups,
            " of ", length(valid_idxs), " valid entries.")
    return tbl
end

"""
    check_table_denominators(d::Int)

Checks whether the ray class table for modulus `d` contains any entry whose
`s1,s2` reduce to a denominator SMALLER than `d` (i.e. the class structurally
lives at a lower level, e.g. `d/q` for `q` a shared prime between `d` and
`O`'s conductor). Prints every such entry found, or a clear "none found"
message. Used to test whether `ray_class_group_representatives`'s enumeration
of the modulus-`d` ray class group actually includes representatives for the
lower-level part of the group at all, as opposed to only ever producing
classes with full denominator `d`.
"""
function check_table_denominators(d::Int)
    tbl = ray_class_shin_table_fast_quiet(d)
    println("Table has ", length(tbl), " entries (", count(x -> x === nothing, tbl), " missing).")
    n_found = 0
    for (i, e) in enumerate(tbl)
        e === nothing && continue
        d1, d2 = denominator(e.s1), denominator(e.s2)
        if d1 < d || d2 < d
            n_found += 1
            println("  entry ", i, ": J = ", e.J, "  s1 = ", e.s1, "  s2 = ", e.s2,
                    "  (reduced denominators: ", d1, ", ", d2, ")",
                    "  rho_B = ", e.rho_B, "  shin_value = ", e.shin_value)
        end
    end
    if n_found == 0
        println("No entries found with reduced denominator < ", d,
                " - every table entry's (s1,s2) has full denominator ", d,
                ". If some class is structurally supposed to live at a lower ",
                "level (e.g. d/q for q sharing a factor with d), it is NOT ",
                "represented in this table at all.")
    else
        println(n_found, " entries found with reduced denominator < ", d, ".")
    end
    return tbl
end

# ---------------------------------------------------------------------------
# compare_ray_class_shin_tables: runs BOTH ray_class_shin_table (slow) and
# ray_class_shin_table_fast, then compares their shin values class by class.
# Prints true/false per ray class, and true/false overall.
# ---------------------------------------------------------------------------

"""
    compare_ray_class_shin_tables(F::AdmissibleTuple; tol::Real = 1e-10, kwargs...)

Runs `ray_class_shin_table(F; kwargs...)` and `ray_class_shin_table_fast(F; kwargs...)`,
then compares their shin values class by class (`abs(diff) < tol`). Prints
`true`/`false` for each ray class, then prints and returns whether ALL
classes agree.

Both tables are built from the SAME shared setup (`_ray_class_shin_setup`,
computed once and passed to both via the `shared_data` keyword each table
function accepts) - calling `ray_class_group_representatives` independently
in each table is not guaranteed to reproduce the same literal
representative ideals or class order on every call, which otherwise
silently misaligns the two tables at the same index even when both are
individually correct.

A class where either table FAILED for that entry, or where the two tables'
underlying ideal `J` still don't line up at that index (shouldn't happen
now that both use the same shared setup, but checked anyway rather than
assumed), is counted as `false` (never silently skipped or counted as a
match).

Returns `(all_match::Bool, per_class::Vector{Bool})`.
"""
function compare_ray_class_shin_tables(F::AdmissibleTuple; tol::Real = 1e-10, kwargs...)
    println("=== Computing shared setup (O, K, ray class representatives, Picard data) ===")
    sd = _ray_class_shin_setup(F)

    println("\n=== Running ray_class_shin_table (slow) ===")
    table_slow = ray_class_shin_table(F; shared_data = sd, kwargs...)

    println("\n=== Running ray_class_shin_table_fast ===")
    table_fast = ray_class_shin_table_fast(F; shared_data = sd, kwargs...)

    n_slow = length(table_slow)
    n_fast = length(table_fast)
    n_slow == n_fast ||
        error("compare_ray_class_shin_tables: tables have different lengths ",
              "(slow has $n_slow, fast has $n_fast) - cannot compare class by class")

    per_class = Vector{Bool}(undef, n_slow)

    println("\n=== Per-class comparison (tol = ", tol, ") ===")
    for idx in 1:n_slow
        J_slow, _, val_slow = table_slow[idx]
        J_fast, _, val_fast = table_fast[idx]

        same_J = try
            J_slow == J_fast
        catch
            false
        end
        same_J || println("  class ", idx, ": WARNING - underlying ideal J differs between the ",
                "two tables at this index (slow J = ", J_slow, ", fast J = ", J_fast,
                ") - the two tables may not be aligned the same way; comparing by index anyway")

        if val_slow === nothing || val_fast === nothing
            per_class[idx] = false
            println("  class ", idx, ": false  (slow = ",
                    val_slow === nothing ? "FAILED" : val_slow,
                    ", fast = ", val_fast === nothing ? "FAILED" : val_fast, ")")
        else
            ok = same_J && abs(val_slow - val_fast) < tol
            per_class[idx] = ok
            println("  class ", idx, ": ", ok, "  (|diff| = ", abs(val_slow - val_fast), ")")
        end
    end

    all_match = all(per_class)
    println("\n=== Overall result ===")
    println("All shin values match: ", all_match)

    return all_match, per_class
end

compare_ray_class_shin_tables(d::Int; kwargs...) = compare_ray_class_shin_tables(AdmissibleTuple(d); kwargs...)

"""
    diagnose_twist_across_seeds(d, J_gens, gamma_coords, seeds; is_principal = false)

Runs ONE fixed twist (J, gamma) under several RNG seeds and reports, per seed,
how many (p,-p) pairs fail |v(p)v(-p)| = 1/(d+1).

Why: diagnose_group_lookup_ambiguity and diagnose_seed_lift_differences
proved the group table (classes, values, lifts, lookup) is IDENTICAL across
good and broken seeds. And the Picard-class LABELS are permuted between seeds
(class [1] had gQ=(5,-24,9) on one run and (9,-24,5) on another), so
`nonprincipal_triples[1:20]` are NOT the same 20 twists on different seeds.
Hence "half the seeds fail" may really be "certain TWISTS fail, and the seed
decides which twists land in the first 20". This test forces the same J and
gamma on every seed and separates the two:

  * fails on EVERY seed  => the twist (J's class/representative) is the
    problem; seed is a red herring. Dissect that J.
  * passes on some seeds => same J, different seed-dependent twist-level
    data (rho_B_J, alpha, s1_J, s2_J, rho_j, gQ_j); compare the per-seed
    printouts twisted_chi_general emits ("Using rho_B = ...", "rho_j = ...")
    to find the quantity that differs between passing and failing runs.

J_gens: generators as (c0, c1) meaning c0 + c1*sqrt(D), e.g. the failing
run's J = <5, 3//2*sqrt(44) + 3> is [(5, 0), (3, 3//2)]. gamma likewise,
e.g. gamma = 1 is (1, 0). The ideal is built in EACH seed's own O as the sum
of the principal ideals of the generators (so it is the same abstract ideal
every time, regardless of how that seed's enumeration would have labelled it).

twisted_chi_general is very verbose; redirect to a file (`*> run.txt`) and
search for "=== SEED SUMMARY ===" at the end, which collects everything.
"""
function diagnose_twist_across_seeds(d::Int, J_gens::Vector, gamma_coords::Tuple, seeds::Vector{Int};
                                     is_principal::Bool = false, tol::Float64 = 1e-8)
    F = AdmissibleTuple(d)
    results = []
    for sdn in seeds
        println("\n\n################ SEED ", sdn, " ################")
        Random.seed!(sdn)
        sd = _ray_class_shin_setup(F)
        K, O = sd.K, sd.O
        a = gen(K)
        P_O, mP_O, rho_reps, gQ_reps, F_class = sd.P_O, sd.mP_O, sd.rho_reps, sd.gQ_reps, sd.F_class
        mk(c) = K(QQ(c[1])) + K(QQ(c[2])) * a
        J = sum(ideal(O, O(mk(g))) for g in J_gens)
        gamma = mk(gamma_coords)
        uJ = mP_O \ J
        println("J = ", J, "   N(J) = ", norm(J), "   Picard class of J under this seed = ", uJ,
                "   rho_reps[class] = ", rho_reps[uJ], "   gQ = ", gQ_reps[uJ])
        println("gamma = ", gamma, "   N(gamma) = ", norm(gamma))

        ray_class_lookup_table = ray_class_shin_table_fast_quiet(F; shared_data = sd)
        full_monoid_table = ray_class_monoid_table_v2(F; shared_data = sd, verbose = false)
        new_ovs, gF, psi = twisted_chi_general(F, J, gamma, K, O, is_principal, P_O, mP_O,
                                               rho_reps, gQ_reps, F_class,
                                               Dict{FinGenAbGroupElem,Any}();
                                               ray_class_lookup_table = ray_class_lookup_table,
                                               monoid_table = NamedTuple[],
                                               candidate_cache = Dict{Tuple{Int,Int,Int},Any}(),
                                               class_hints = Dict{Any,NamedTuple}(),
                                               full_monoid_table = full_monoid_table)
        fails = Tuple{Int,Int}[]; phase_fails = Tuple{Int,Int}[]
        for p1 in 0:d-1, p2 in 0:d-1
            (p1 == 0 && p2 == 0) && continue
            pr = new_ovs[p1+1, p2+1] * new_ovs[mod(-p1, d)+1, mod(-p2, d)+1]
            abs(Float64(abs(pr)) - 1.0 / (d + 1)) > tol && push!(fails, (p1, p2))
            abs(Float64(angle(pr))) > 1e-8 && push!(phase_fails, (p1, p2))
        end
        push!(results, (seed = sdn, class = uJ, gF = gF, n_fail = length(fails), fails = sort(fails),
                        n_phase_fail = length(phase_fails)))
    end

    println("\n\n=== SEED SUMMARY === (same J and gamma on every seed)")
    for r in results
        println("seed ", r.seed, ":  Picard class label ", r.class, "   gF = ", r.gF,
                "   failing pairs (magnitude): ", r.n_fail, "   pairs with nonzero phase: ", r.n_phase_fail)
        r.n_fail > 0 && println("    failing points: ", r.fails)
    end
    nf = [r.n_fail for r in results]
    if all(>(0), nf)
        println("VERDICT: fails under EVERY seed => the twist itself (this J) is the problem; seed is a red herring.")
    elseif all(==(0), nf)
        println("VERDICT: passes under EVERY seed => this J is fine; the original failure came from a DIFFERENT twist",
                " landing in the first 20 on that seed. Re-run with the other J's from the failing run's list.")
    else
        println("VERDICT: same J passes on some seeds and fails on others => seed-dependent TWIST-LEVEL data;",
                " diff the 'Using rho_B = ... alpha = ... s1 = ... s2 = ...' and 'rho_j = ...' lines between",
                " a passing and a failing seed above.")
    end
    return results
end

function test_all_twists(d::Int; phase_debug_points::Vector{Tuple{Int,Int}} = Tuple{Int,Int}[])
    F = AdmissibleTuple(d)
    @show F

    # BUGFIX: previously this function duplicated _ray_class_shin_setup's own
    # K/O construction inline (quadratic_field(D) + Order(K,...) called a
    # second time here), AND separately called ray_class_shin_table_fast_quiet(F)
    # which calls _ray_class_shin_setup(F) a THIRD time internally. Each call
    # to quadratic_field/Order produces a NEW object even for identical input,
    # so J (built from this function's own O) and lookup_entry.J (built from
    # the lookup table's own separately-constructed O) ended up as ideals of
    # two different (though mathematically identical) order OBJECTS -
    # triggering Hecke's "Ideals do not have the same order" the first time
    # twisted_chi_general tried to multiply J * U_elem1 together.
    # FIX: build the setup ONCE via _ray_class_shin_setup, use its K/O for
    # everything below, and pass it as shared_data so ray_class_shin_table_fast_quiet
    # reuses the SAME K/O instead of constructing its own.
    sd = _ray_class_shin_setup(F)
    D, q, d = sd.D, sd.q, sd.d
    # EVEN d (9/18): every Theorem 2.19 point representation runs at the
    # ray-class modulus d-bar (= 2d for even d, = d for odd d): the (s1,s2)
    # vectors have denominator d-bar and the stored value is the d-bar-modulus
    # invariant shin^{p/d-bar}, used only for identification. The ghost
    # cocycle itself (shin^{p/d}) is only ever evaluated in twisted_chi_general.
    dm = hasproperty(sd, :dbar) ? sd.dbar : (isodd(d) ? d : 2 * d)
    K, O = sd.K, sd.O
    all_reps, all_triples = sd.all_reps, sd.all_triples
    P_O, mP_O, rho_reps, gQ_reps, F_class = sd.P_O, sd.mP_O, sd.rho_reps, sd.gQ_reps, sd.F_class

    println("conductor of O is: ", conductor(O))
    println("Class group of O has order: ", order(P_O))
    println("D = ", D, ", q = ", q)
    println("Total triples: ", length(all_triples))
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

    println("\nComputing ray class shin lookup table once for this run ",
            "(used by every twisted_chi_general call below for elem1) ...")
    ray_class_lookup_table = ray_class_shin_table_fast_quiet(F; shared_data = sd)
    println("  done: ", length(ray_class_lookup_table), " entries.")

    n_missing = count(x -> x === nothing, ray_class_lookup_table)
    println("  missing table entries: ", n_missing, " out of ", length(ray_class_lookup_table))
    if n_missing > 0
        println("  missing indices: ", findall(x -> x === nothing, ray_class_lookup_table))
    end

    # Built once for the whole run: the ENTIRE ray class monoid
    # Jbar_m^flat(O), per the project owner's revised approach - the monoid
    # is finite and small (592 classes for d=21), so build it in full and do
    # a direct value lookup on a group-table miss, rather than an
    # incremental norm-bound search (ray_class_monoid_lookup, now
    # superseded - see its docstring). Threaded through every
    # twisted_chi_general call below exactly like ray_class_lookup_table.
    println("\nComputing the full ray class monoid table once for this run ",
            "(used by every twisted_chi_general call below on a group-table miss) ...")
    full_monoid_table = ray_class_monoid_table_v2(F; shared_data = sd, verbose = true)
    println("  done: ", length(full_monoid_table), " monoid classes across all Picard fibers.")

    # Persistent, growing extension of the ray class table into the monoid
    # Jbar_m^flat(O) (see ray_class_monoid_lookup's docstring for the exact
    # project-owner algorithm and definitions) - shared across every twist
    # below, so a class discovered while processing one twist is reused
    # (not re-searched) by every later lookup that needs it.
    monoid_table = NamedTuple[]

    # Persists (class_idx, alpha_coord1, alpha_coord2) -> the full computed
    # result for that ray_class_monoid_lookup candidate, across EVERY call
    # below (every point of every twist) - see ray_class_monoid_lookup's
    # docstring, "CROSS-CALL CANDIDATE CACHE". Without this, every lookup
    # miss re-pays the find_alpha_rhoB_s1_s2_O cost for the same cheap,
    # low-norm candidates from scratch.
    candidate_cache = Dict{Tuple{Int,Int,Int}, Any}()

    # Persists (alpha, rho_B, b_O) per Picard class across EVERY call below -
    # see ray_class_monoid_lookup's docstring, "PERFORMANCE FIX: PER-CLASS
    # find_alpha_rhoB_s1_s2_O CACHING". Without this, each new lookup call
    # would re-pay the full O(height^5) find_b_with_verified_principal_product
    # search the first time it happens to hit a given class, even if an
    # EARLIER call already paid that cost for the same class.
    class_hints = Dict{Any, NamedTuple}()

    n_principal = min(5, length(principal_triples))
    println("\n=== Testing first ", n_principal, " principal twists ===")
    for (k, (J, I, gamma)) in enumerate(principal_triples[1:n_principal])
        println("\nPrincipal twist ", k, " of ", n_principal, ":")
        println("  J = ", J)
        println("  gamma = ", gamma)
        println("  norm(gamma) = ", norm(K(gamma)))
        new_ovs, gF, psi = twisted_chi_general(F, J, gamma, K, O,
                                                true, P_O, mP_O,
                                                rho_reps, gQ_reps, F_class, j_cache;
                                                ray_class_lookup_table = ray_class_lookup_table,
                                                monoid_table = monoid_table,
                                                candidate_cache = candidate_cache,
                                                class_hints = class_hints,
                                                full_monoid_table = full_monoid_table,
                                                phase_debug_points = phase_debug_points)
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
                                                rho_reps, gQ_reps, F_class, j_cache;
                                                ray_class_lookup_table = ray_class_lookup_table,
                                                monoid_table = monoid_table,
                                                candidate_cache = candidate_cache,
                                                class_hints = class_hints,
                                                full_monoid_table = full_monoid_table,
                                                phase_debug_points = phase_debug_points)
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

    all_places = real_places(K)  # kept for any other uses in this function
    Sigma = [conjugation_place(K)]  # BUGFIX: was [all_places[2]] - wrong place, see conjugation_place's docstring

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

#test_all_twists(15)
#diagnose_twist_across_seeds(21, [(5, 0), (3, 3//2)], (-20, -3), [208475639])

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

# ---------------------------------------------------------------------------
# diagnose_elem1_lookup: inspects A_lookup / shin^{[p1,p2]}_A(rho) directly
# for a given (p1,p2), bypassing the full twisted_chi_general retry
# machinery - for tracking down why a specific point's reverse lookup into
# the ray class shin table fails (e.g. an unreasonably large |shin| value).
# ---------------------------------------------------------------------------

"""
    true_picard_class_of_root(K, O, P_O, mP_O, rho; unit_bound::Int = 300)

Determines the TRUE Picard class of `Z+Z*rho` by testing it against all
`|P_O|` group elements directly, WITHOUT going through `gQ_reps`/
`reduced_hj_orbit`/`minimal_hj_stabilizer`/`form_sign_fix` (suspected of
conflating narrow classes when `O` has no unit of norm -1, since
`form_sign_fix`'s `<a,b,c> -> <-a,b,-c>` is `-Q(x,-y)`, an overall sign flip
composed with a det=-1 substitution - NOT a GL_2(Z) equivalence unless a
norm -1 unit exists to identify it with a genuine class-preserving map).

For each g in P_O, J_g := mP_O(g) (the FORWARD map - i.e. Hecke's own
ideal representative for class g, not the discrete-log direction that was
independently found untrustworthy earlier in this project). Since
`J_g*conj(J_g)` is always principal, `class(Z+Z*rho) = class(J_g)` iff
`(Z+Z*rho)*conj(J_g)` is principal - tested via `is_principal_O_verified`,
exactly the same disc-log-free technique already used elsewhere in this
file (e.g. `find_alpha_rhoB_s1_s2_O`'s final verification).

Returns `(g, lambda)` for the matching class, where `lambda` is the EXACT
verified generator found for `(Z+Z*rho)*conj(J_g)` (only meaningful as a
`Z+Z*rho = lambda*O` statement when `J_g = O` itself, i.e. `g` is the
IDENTITY element of `P_O` - callers wanting that specific relationship
should call this and use `lambda` only when `g == zero(P_O)`; for non-
identity `g`, `lambda` still certifies the class match but is not directly
`Z+Z*rho`'s own generator). Returning `lambda` here (rather than having
callers re-invoke `is_principal_O_verified` separately with differently-
formatted generators for the "same" lattice) avoids a real inconsistency
observed in practice: `is_principal_O_verified` can give different
true/false results for generator sets spanning the identical lattice,
depending on exactly how those generators are packaged - so the safest
policy is to trust and reuse whichever specific call already succeeded,
never re-derive the same fact via a differently-formatted call.

Errors if zero or more than one class matches (the latter would indicate
`is_principal_O_verified`'s unit_bound is too small and is spuriously
failing for the true class as well, not that two classes are both correct).
"""
function true_picard_class_of_root(K, O, P_O, mP_O, rho; unit_bound::Int = 300)
    I_rho_gens = [K(1), rho]
    matches = []
    for g in P_O
        J_g = mP_O(g)
        Jconj_gens = [quad_conj(K, _nf_elem_of(K, x)) for x in basis(J_g)]
        gens = lattice_product_gens(I_rho_gens, Jconj_gens)
        ok, lam = is_principal_O_verified(K, O, gens; unit_bound = unit_bound)
        println("  true_picard_class_of_root: g = ", g, "  (Z+Z*rho)*conj(J_g) principal? ", ok)
        ok && push!(matches, (g, lam))
    end
    length(matches) == 1 ||
        error("true_picard_class_of_root: found ", length(matches), " matching classes ",
              "(expected exactly 1) - matches = ", matches, ". If 0, unit_bound may be too ",
              "small; if >1, something is inconsistent with is_principal_O_verified itself.")
    return matches[1]
end



"""
    diagnose_elem1_lookup(d_val::Int; p1::Int = 0, p2::Int = 1)

Prints, step by step: `rho` and its `BigFloat` value; the full stabilizer
`L_rho_base` of `rho` (and whether it actually fixes `rho`, and its
determinant); `A_lookup` for the given `(p1,p2)` (its entries' size,
determinant, whether it fixes `rho`, and whether it satisfies
`A*r - r ∈ Z^2` for `r = [p1/d, p2/d]`); the `find_hj_reducing_matrix`
result used internally by `shin`; and finally the `shin` value itself and
its magnitude (expected close to 1 for a well-behaved point).
"""
function diagnose_elem1_lookup(d_val::Int; p1::Int = 0, p2::Int = 1)
    F = AdmissibleTuple(d_val)
    D = Int(F.D)
    d = Int(F.d)
    q = Int(F.q)

    K, a = quadratic_field(D)
    Dr = squarefree_part(D)
    sqrt_Dr = a // isqrt(D ÷ Dr)
    omega = mod(Dr, 4) == 1 ? (K(1) + sqrt_Dr) // 2 : sqrt_Dr
    O = Order(K, [one(K), q * omega])

    P_O, mP_O, rho_reps, gQ_reps, F_class = find_class_representatives(O, F, K)
    rho = rho_reps[F_class]
    println("rho = ", rho)

    rho_bf = BigFloat(coeff(rho, 0)) + BigFloat(coeff(rho, 1)) * sqrt(BigFloat(D))
    println("rho_bf = ", rho_bf)

    # The full stabilizer of rho under SL_2(Z) - the expensive Pell-based part.
    L_rho_base = hj_automorph_for_root(rho, D)
    println("\nL_rho_base = ", L_rho_base)
    println("det(L_rho_base) = ", L_rho_base[1,1]*L_rho_base[2,2] - L_rho_base[1,2]*L_rho_base[2,1])
    println("L_rho_base * rho_bf (Mobius) = ", (L_rho_base[1,1]*rho_bf + L_rho_base[1,2]) /
                                                (L_rho_base[2,1]*rho_bf + L_rho_base[2,2]),
            "  (should equal rho_bf)")

    # The specific point requested.
    r_vec = [QQ(p1, d), QQ(p2, d)]
    println("\nr = [p1/d, p2/d] = ", r_vec)

    A_lookup, has_neg_I = stabilizer_power_in_Gamma_s(L_rho_base, r_vec)
    println("\nA_lookup = ", A_lookup)
    det_A = A_lookup[1,1]*A_lookup[2,2] - A_lookup[1,2]*A_lookup[2,1]
    println("det(A_lookup) = ", det_A)
    println("has_neg_I = ", has_neg_I)
    println("A_lookup entries (max abs) = ", maximum(abs.(A_lookup)))

    # Sanity: does A_lookup actually fix rho?
    A_rho = (A_lookup[1,1]*rho_bf + A_lookup[1,2]) / (A_lookup[2,1]*rho_bf + A_lookup[2,2])
    println("A_lookup * rho_bf (Mobius) = ", A_rho, "  (should equal rho_bf)")

    # Sanity: does A_lookup actually satisfy A*r - r in Z^2?
    Ar1 = A_lookup[1,1]*r_vec[1] + A_lookup[1,2]*r_vec[2]
    Ar2 = A_lookup[2,1]*r_vec[1] + A_lookup[2,2]*r_vec[2]
    println("A_lookup*r - r = [", Ar1 - r_vec[1], ", ", Ar2 - r_vec[2], "]  (should be integers)")

    # Now the actual shin value - this is what blows up.
    println("\nComputing shin(A_lookup, rho_bf, [p1,p2], d) ...")
    red = find_hj_reducing_matrix(A_lookup, rho_bf)
    println("find_hj_reducing_matrix result:")
    println("  R = ", red.R, "  det(R) = ", red.detR)
    println("  A_reduced = ", red.A_reduced)
    println("  root_used = ", red.root_used, "  reduced_root = ", red.reduced_root)
    println("  A_reduced entries (max abs) = ", maximum(abs.(red.A_reduced)))

    shin_val = shin(A_lookup, rho_bf, [p1, p2], d)
    println("\nshin_val = ", shin_val)
    println("abs(shin_val) = ", abs(shin_val), "  (should be close to 1 if this is well-behaved)")

    # Precision test: A_lookup/A_reduced have very large entries here, so
    # check whether the blowup is simply the default BigFloat precision
    # (256 bits) being insufficient for this matrix size, rather than a
    # logic error. If abs(shin_val) comes back close to 1 at higher
    # precision, that confirms it - the fix is to raise precision (globally
    # via setprecision, or locally within shin/shin_rm) rather than to
    # change any of the algebra above.
    old_prec = precision(BigFloat)
    # (9/15) 4096/16384 bits took effectively forever at d=15 (A entries ~2700)
    # and 1024 already agreed with 256 to ~55 digits; precision is not the issue.
    for prec in (1024,)
        setprecision(BigFloat, prec) do
            rho_bf_p = BigFloat(coeff(rho, 0)) + BigFloat(coeff(rho, 1)) * sqrt(BigFloat(D))
            sv = shin(A_lookup, rho_bf_p, [p1, p2], d)
            println("  [precision = ", prec, " bits] shin_val = ", sv, "  abs = ", abs(sv))
        end
    end

    return shin_val
end

# ---------------------------------------------------------------------------
# test_ray_class_covariance: checks whether shin^{[p1,p2]}_{A_lookup}(rho)
# (rho = F's OWN base root, NOT the table's rho_B) is related to the ray
# class table's own entry for rho's Picard class via the covariance theorem:
#
#   shin^{s_R(beta)*R*r}_{R A R^{-1}}(R*beta) = shin^r_A(beta)          if det(R)=1
#                                              = shin^r_A(conj(beta))   if det(R)=-1
#
# applied with A = A_lookup, beta = rho, R = gamma (the SL_2(Z)/GL_2(Z)
# element taking rho to the table's rho_B for the SAME Picard class).
#
# gamma is found WITHOUT any search: since Z+Z*rho and Z+Z*rho_B are both
# fractional O-ideals in the SAME class (both represent u_B = F_class), there
# is an explicit mu in K with mu*(Z+Z*rho_B) = Z+Z*rho EXACTLY (found via
# is_principal_O_verified on the ideal quotient (Z+Z*rho)*(Z+Z*rho_B)^{-1}).
# Writing mu = c*rho+d and mu*rho_B = a*rho+b in the {rho,1} basis (both
# forced to have INTEGER coordinates, since mu*(Z+Z*rho_B) = Z+Z*rho exactly,
# not just up to finite index) gives gamma = [a b; c d] in GL_2(Z) satisfying
# gamma . rho = rho_B by construction - no continued-fraction/Pell search
# needed, unlike hj_automorph_for_root.
# ---------------------------------------------------------------------------

"""
    test_ray_class_covariance(d_val::Int; p1::Int = 0, p2::Int = 1)

Tests whether the "missing" value `shin^{[p1,p2]}_{A_lookup}(rho)` (`rho` =
`F`'s own base root, as used in `diagnose_elem1_lookup`) is explained by the
project-owner covariance theorem relating it to the ray class table's own
anchor entry for the SAME Picard class (the entry built from `J = ideal(O,1)`,
i.e. the table's `rho_B` for `u_B = F_class`).

Prints, step by step:
- `rho`, the table's `rho_B` for the same Picard class, and `A_lookup`/
  `A_table` for each;
- `shin_lookup = shin^{[p1,p2]}_{A_lookup}(rho)` and the table's own anchor
  `shin_table_entry = shin^{s_vec*d}_{A_table}(rho_B)`;
- the explicit `gamma` in `GL_2(Z)` with `gamma . rho = rho_B` (found via
  the ideal-quotient generator `mu`, not a search), its determinant, and a
  numeric check that it really does send `rho` to `rho_B`;
- the theorem's transformed vector `p_transformed = s_gamma(rho) * gamma *
  [p1,p2]` (mod `d`), compared against the table anchor's own `p_vec`;
- if `det(gamma) = 1`: an independent re-derivation of `shin_lookup` via
  `shin(gamma*A_lookup*gamma^{-1}, rho_B, p_transformed, d)`, which the
  theorem guarantees must equal `shin_lookup` exactly - this is a pure
  consistency check on the gamma/sign-convention bookkeeping above, not new
  information about completeness;
- finally, whether `p_transformed` (mod `d`) actually matches the table
  anchor's `p_vec`, and whether `shin_lookup` matches `shin_table_entry`.
  A match on `p_transformed` WITHOUT a match on the shin values would point
  to `A_table` being a different power/sign of the stabilizer generator than
  `gamma*A_lookup*gamma^{-1}` (a genuine, checkable bug candidate); no match
  on `p_transformed` at all would mean `[p1,p2]` for `rho` corresponds to a
  DIFFERENT ray class than the anchor entry, and the real comparison should
  be against that other class's table entry instead.

Returns a NamedTuple with all the computed quantities for further inspection.
"""
function test_ray_class_covariance(d_val::Int; p1::Int = 0, p2::Int = 1)
    F = AdmissibleTuple(d_val)
    sd = _ray_class_shin_setup(F)
    D, q, d = sd.D, sd.q, sd.d
    # EVEN d (9/18): every Theorem 2.19 point representation runs at the
    # ray-class modulus d-bar (= 2d for even d, = d for odd d): the (s1,s2)
    # vectors have denominator d-bar and the stored value is the d-bar-modulus
    # invariant shin^{p/d-bar}, used only for identification. The ghost
    # cocycle itself (shin^{p/d}) is only ever evaluated in twisted_chi_general.
    dm = hasproperty(sd, :dbar) ? sd.dbar : (isodd(d) ? d : 2 * d)
    K, O = sd.K, sd.O
    P_O, mP_O, rho_reps, gQ_reps, F_class, d_check =
        sd.P_O, sd.mP_O, sd.rho_reps, sd.gQ_reps, sd.F_class, sd.d_check

    all_places = real_places(K)
    emb_conj = embedding(all_places[1])
    emb_main = embedding(all_places[2])

    # ---- Base point: F's own root rho, and A_lookup for [p1,p2] ----
    rho = rho_reps[F_class]
    rho_bf = BigFloat(coeff(rho, 0)) + BigFloat(coeff(rho, 1)) * sqrt(BigFloat(D))
    println("rho (F's own base root) = ", rho)
    println("rho_bf = ", rho_bf)

    L_rho = hj_automorph_for_root(rho, D)
    r_vec = [QQ(p1, d), QQ(p2, d)]
    A_lookup, _ = stabilizer_power_in_Gamma_s(L_rho, r_vec)
    println("A_lookup = ", A_lookup)

    shin_lookup = shin(A_lookup, rho_bf, [p1, p2], d)
    println("shin^[", p1, ",", p2, "]_A_lookup(rho) = ", shin_lookup)
    println("  |shin_lookup| = ", abs(shin_lookup))

    # ---- Table's anchor entry for rho's TRUE Picard class (NOT F_class,
    # which comes from gQ_reps/reduced_hj_orbit/form_sign_fix - suspected of
    # conflating narrow classes for this order, since O has no unit of norm
    # -1; see true_picard_class_of_root's docstring) ----
    println("\n--- determining rho's TRUE Picard class (bypassing gQ_reps/F_class) ---")
    true_u_B, lambda_rho = true_picard_class_of_root(K, O, P_O, mP_O, rho)
    println("true Picard class of Z+Z*rho = ", true_u_B,
            "   (F_class as reported by find_class_representatives = ", F_class, ")")
    true_u_B == F_class ||
        println("  *** CONFIRMED MISMATCH: F_class is WRONG for this rho - the reduced-form ",
                "comparison (gQ_reps[F_class] == F.Q) does not correctly track rho's actual ",
                "ideal class. Using true_u_B = ", true_u_B, " from here on. ***")
    println("--- end true-class determination ---\n")

    u_B = true_u_B
    J_anchor = mP_O(u_B)
    rho_B_init = rho_reps[u_B]
    alpha, rho_B, s1_tab, s2_tab = find_alpha_rhoB_s1_s2_O(
        J_anchor, rho_B_init, rho_B_init, u_B, P_O, mP_O, O, K,
        d, rho_B_init, d_check)
    println("\nTable anchor (J_anchor = mP_O(true_u_B)): alpha = ", alpha, "  rho_B = ", rho_B)
    println("  s1 = ", s1_tab, "  s2 = ", s2_tab)

    s_vec_tab = [s1_tab, s2_tab]
    L_rhoB = hj_automorph_for_root(rho_B, D)
    A_tab, has_neg_I_tab = stabilizer_power_in_Gamma_s(L_rhoB, s_vec_tab)
    rho_B_bf = BigFloat(coeff(rho_B, 0)) + BigFloat(coeff(rho_B, 1)) * sqrt(BigFloat(D))

    p1_tab_exact = s1_tab * d
    p2_tab_exact = s2_tab * d
    (isone(denominator(p1_tab_exact)) && isone(denominator(p2_tab_exact))) ||
        error("table anchor s = [$s1_tab,$s2_tab] does not have denominator dividing d = $d")
    p_vec_tab = [Int(numerator(p1_tab_exact)), Int(numerator(p2_tab_exact))]

    shin_tab = shin(A_tab, rho_B_bf, p_vec_tab, d)
    println("  A_table = ", A_tab, "  p_vec_table = ", p_vec_tab,
            "  (has_neg_I = ", has_neg_I_tab, ")")
    println("  shin_table_entry = ", shin_tab, "  |shin_table_entry| = ", abs(shin_tab))

    # ---- Verify rho_B's own TRUE class matches rho's TRUE class (true_u_B)
    # before trying to find gamma between them - do NOT assume this just
    # because both trace back to "u_B" internally; find_alpha_rhoB_s1_s2_O's
    # rho_B satisfies bdO_O = alpha*(Z+Z*rho_B) with b_O representing
    # [J_anchor]^{-1}, so Z+Z*rho_B's class is class(b_O) = -true_u_B, which
    # only equals true_u_B itself if true_u_B is an order-2 (self-inverse)
    # element of P_O - check directly rather than assuming. ----
    println("\n--- determining rho_B's TRUE Picard class ---")
    true_rhoB_class, lambda_rhoB = true_picard_class_of_root(K, O, P_O, mP_O, rho_B)
    println("true Picard class of Z+Z*rho_B = ", true_rhoB_class,
            "   (rho's true class = ", true_u_B, ")")
    println("--- end determination ---\n")

    true_rhoB_class == true_u_B ||
        error("rho and rho_B are NOT actually in the same Picard class (true classes ",
              true_u_B, " vs ", true_rhoB_class, ") - no gamma taking rho to rho_B can ",
              "exist. This would mean find_alpha_rhoB_s1_s2_O's rho_B for this J_anchor is ",
              "not in the class this test expected; needs its own separate investigation.")

    # ---- Find mu with mu*(Z+Z*rho_B) = Z+Z*rho.
    # PREFERRED PATH (used whenever the shared class is the IDENTITY element
    # of P_O, as it always is when starting from J_anchor = ideal(O,1)):
    # true_picard_class_of_root's own lambda_rho/lambda_rhoB already satisfy
    # lambda_rho*O = Z+Z*rho and lambda_rhoB*O = Z+Z*rho_B EXACTLY (since
    # J_g = mP_O(zero(P_O)) = O itself there, so the conjugate-product gens
    # reduce to (Z+Z*rho)*O = Z+Z*rho with no extra factor at all) - so
    # mu = lambda_rho/lambda_rhoB works immediately, reusing values already
    # verified successful above, with NO second is_principal_O_verified
    # call on any differently-formatted generator set. This sidesteps a
    # real inconsistency observed in practice: calling
    # is_principal_O_verified(K,O,[K(1),rho]) directly (2 generators) gave a
    # DIFFERENT (false) answer than calling it on the mathematically
    # equivalent 4-generator lattice_product_gens form used inside
    # true_picard_class_of_root, for the exact same lattice Z+Z*rho.
    # FALLBACK (only reached if the shared class is non-identity): the
    # general ideal-quotient approach via inv() - not exercised or verified
    # to be safe from the same generator-formatting sensitivity, but kept
    # in case a future call to this function starts from a non-identity
    # anchor class.
    local mu
    if true_u_B == zero(P_O)
        mu = lambda_rho / lambda_rhoB
        println("\n(identity-class fast path) mu = lambda_rho/lambda_rhoB = ", mu)
    else
        println("\n(non-identity shared class ", true_u_B, " - falling back to the general ",
                "ideal-quotient approach via inv(); this path is less tested)")
        I_rho   = fractional_ideal(O, [K(1), rho])
        I_rho_B = fractional_ideal(O, [K(1), rho_B])
        Q_mu = I_rho * inv(I_rho_B)
        Qb_nf = [_nf_elem_of(K, x) for x in basis(Q_mu)]
        ok_mu, mu_found = is_principal_O_verified(K, O, Qb_nf; unit_bound = 300)
        ok_mu || error("Could not find mu with mu*(Z+Z*rho_B) = Z+Z*rho via the general ",
                       "ideal-quotient fallback, even though both are verified to be in the ",
                       "same true Picard class ", true_u_B, " - this may be exactly the same ",
                       "generator-formatting sensitivity noted above; try a different gens ",
                       "packaging for Q_mu, or increase unit_bound further")
        mu = mu_found
    end
    println("mu (with mu*(Z+Z*rho_B) = Z+Z*rho) = ", mu)

    # Express z (given via coeff(z,0)/coeff(z,1) in the {1,sqrt(D)} basis) in
    # the {rho,1} basis instead: z = c*rho + dc, solved exactly since {rho,1}
    # is a Q-basis of K (rho irrational).
    r0_raw, r1_raw = coeff(rho, 0), coeff(rho, 1)
    function coords_in_rho_basis(z)
        z0, z1 = coeff(z, 0), coeff(z, 1)
        c  = QQ(z1 / r1_raw)
        dc = QQ(z0 - c * r0_raw)
        return c, dc
    end

    c_g, d_g = coords_in_rho_basis(mu)          # mu        = c_g*rho + d_g
    a_g, b_g = coords_in_rho_basis(mu * rho_B)  # mu*rho_B  = a_g*rho + b_g

    for (nm, x) in (("a_g", a_g), ("b_g", b_g), ("c_g", c_g), ("d_g", d_g))
        isone(denominator(x)) ||
            error("gamma entry $nm = $x is not an integer - mu was not the correct ",
                  "ideal-class generator (mu*(Z+Z*rho_B) should equal Z+Z*rho EXACTLY, ",
                  "which forces integer gamma entries; a non-integer here means the ",
                  "unit-twist chosen by is_principal_O_verified doesn't land the lattices ",
                  "on top of each other, only in the same class up to a rational scalar)")
    end
    gamma = [Int(numerator(a_g)) Int(numerator(b_g)); Int(numerator(c_g)) Int(numerator(d_g))]
    det_gamma = gamma[1,1]*gamma[2,2] - gamma[1,2]*gamma[2,1]
    println("gamma = ", gamma, "  det(gamma) = ", det_gamma)
    abs(det_gamma) == 1 || error("det(gamma) = $det_gamma, not +/-1 - gamma is not in ",
                                  "GL_2(Z) as expected")

    gamma_rho_bf = (gamma[1,1]*rho_bf + gamma[1,2]) / (gamma[2,1]*rho_bf + gamma[2,2])
    println("gamma . rho (Mobius, BigFloat check) = ", gamma_rho_bf)
    println("rho_B (BigFloat, for comparison)     = ", rho_B_bf)
    println("  |gamma.rho - rho_B| = ", abs(gamma_rho_bf - rho_B_bf), "  (should be ~0)")

    # ---- Apply the theorem with R = gamma, beta = rho, A = A_lookup ----
    j_gamma_rho = gamma[2,1]*rho_bf + gamma[2,2]   # automorphy factor c_R*beta + d_R
    s_gamma = Int(sign(j_gamma_rho))
    println("\ns_gamma(rho) = sign(c_gamma*rho + d_gamma) = ", s_gamma)

    p_transformed_raw = s_gamma .* (gamma * [p1, p2])
    p_transformed = mod.(p_transformed_raw, d)
    println("p_transformed (before mod d) = ", p_transformed_raw)
    println("p_transformed mod d          = ", p_transformed)
    println("table anchor p_vec mod d     = ", mod.(p_vec_tab, d))
    p_matches = p_transformed == mod.(p_vec_tab, d)
    println("  match? ", p_matches)

    shin_conj = nothing
    if det_gamma == 1
        A_conj = gamma * A_lookup * _gl2z_inv(gamma)
        println("\ndet(gamma) = 1, so the theorem gives:")
        println("  shin^{p_transformed}_{A_conj}(rho_B) should equal shin_lookup exactly")
        shin_conj = shin(A_conj, rho_B_bf, p_transformed, d)
        println("shin(A_conj, rho_B_bf, p_transformed, d) = ", shin_conj)
        println("shin_lookup (original)                    = ", shin_lookup)
        println("  |diff| = ", abs(shin_conj - shin_lookup),
                "  (should be ~0 - this only checks the gamma/sign bookkeeping above, ",
                "NOT completeness)")
    else
        println("\ndet(gamma) = -1: the theorem gives shin_A_lookup(conj(rho)), not ",
                "shin_A_lookup(rho), on the right-hand side. The direct re-derivation ",
                "check above doesn't apply in this branch.")
    end

    println("\n=== Summary ===")
    println("shin_lookup (A_lookup, rho, [", p1, ",", p2, "])      = ", shin_lookup,
            "   |.| = ", abs(shin_lookup))
    println("shin_table_entry (A_table, rho_B, p_vec_table) = ", shin_tab,
            "   |.| = ", abs(shin_tab))
    println("p_transformed matches table anchor's p_vec (mod d)? ", p_matches)
    println("shin_lookup matches shin_table_entry?               ",
            abs(shin_lookup - shin_tab) < 1e-6)

    return (rho = rho, rho_B = rho_B, gamma = gamma, det_gamma = det_gamma,
            A_lookup = A_lookup, A_table = A_tab,
            p_transformed = p_transformed, p_vec_tab = p_vec_tab,
            shin_lookup = shin_lookup, shin_table_entry = shin_tab,
            shin_conj_recheck = shin_conj)
end

#test_all_twists(15)
#diagnose_twist_across_seeds(21, [(5, 0), (3, 3//2)], (-20, -3), [208475639])

# The failing run's twist: J = <5, 3//2*sqrt(44) + 3>, gamma = 1, non-principal.
# Forced onto every seed below (broken seed 208475639 + two others). Redirect
# output to a file and read the "=== SEED SUMMARY ===" block at the end.
test_all_twists(21)
#diagnose_twist_across_seeds(21, [(5, 0), (3, 3//2)], (-20, -3), [208475639])
#diagnose_twist_across_seeds(21, [(5, 0), (3, 3//2)], (1, 0), [208475639])
#diagnose_seed_lift_differences(21, [305817264, 672049183, 208475639])
#diagnose_group_lookup_ambiguity(21)
#test_all_twists(21; phase_debug_points = [(1,15),(20,6),(2,14),(19,7),(0,2),(0,19)])