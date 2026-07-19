using Hecke
using Zauner
import Zauner: _ghost_invariants, _dualize, _approx_complex_intersection, roots, Vandermonde, pow_to_elem_sym_poly
using GenericFFT: fft
using LinearAlgebra
include("ray_unit_quotient.jl")
include("term2.jl")
include("enumeration_with_infinite_places.jl")
include("make_coprime_reps.jl")

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
    rats = [QQ(0)]
    for h in 1:max_h
        for q in 1:h
            for p in -h:h
                r = QQ(p, q)
                gcd(abs(p), q) == 1 || continue
                r in rats && continue
                push!(rats, r)
            end
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
    ALPHA_SEARCH_RANGE = -60:60
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
                              P_O, mP_O, rho_reps, gQ_reps, F_class)
    d = Int(F.d)
    D = Int(F.D)
    a = gen(K)
    OK = maximal_order(K)

    rho = rho_reps[F_class]

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
        temp = den
        while true
            g2 = gcd(temp, d)
            g2 == 1 && break
            while temp % g2 == 0
                temp ÷= g2
            end
        end
        den_coprime_to_d = temp
        den_d_part = den ÷ den_coprime_to_d
        if mod(num, den_d_part) != 0
            return nothing
        end
        num ÷= den_d_part
        if den_coprime_to_d == 1
            return mod(num, d)
        else
            return mod(num * invmod(den_coprime_to_d, d), d)
        end
    end

    function compute_integralnewdu(p1, p2, s1, s2, rhoB_cand, rho, rho_j, M_g_j, d)
        elem1 = K(p2) * rho - K(p1)
        elem2 = K(s2) * rhoB_cand - K(s1)
        product = elem1 * elem2
        u2, u1 = find_rho_j_and_coords(product, rho_j)
        du = [to_du_mod_d(u1, d), to_du_mod_d(u2, d)]
        raw_newdu = M_g_j * du
        newdu = [to_du_mod_d(raw_newdu[1], d), to_du_mod_d(raw_newdu[2], d)]
        integralnewdu = Int[]
        for u in newdu
            du_int = d * u
            val = du_to_int_mod_d(du_int, d)
            val === nothing && return nothing
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
        u_B = mP_O \ J
        rho_B_init = rho_reps[u_B]
        println("  class of J = ", u_B, " rho_B (raw) = ", rho_B_init)

        j_class = F_class + u_B
        rho_j_orig = rho_reps[j_class]

        # compute gQ_j from rho_j_orig and HJ-reduce it
        c0_j = coeff(rho_j_orig, 0)
        c1_j = coeff(rho_j_orig, 1)
        p0_j = c0_j^2 - c1_j^2 * D
        p1_j = -2 * c0_j
        p2_j = 1
        scale_j = lcm(Int(denominator(QQ(p0_j))),
                      Int(denominator(QQ(p1_j))), 1)
        a_int_j = Int(round(scale_j * p2_j))
        b_int_j = Int(round(scale_j * p1_j))
        c_int_j = Int(round(scale_j * p0_j))
        g_int_j = gcd(a_int_j, b_int_j, c_int_j)
        a_int_j ÷= g_int_j
        b_int_j ÷= g_int_j
        c_int_j ÷= g_int_j
        gQ_j_raw = binary_quadratic_form(a_int_j, b_int_j, c_int_j)
        V_j = reduced_hj_orbit(gQ_j_raw)
        gQ_j = minimal_hj_stabilizer(V_j, d)
        gF_j = AdmissibleTuple(d, F.r, gQ_j)

        # rho_j_orig is a root of gQ_j_raw, NOT of the HJ-reduced gQ_j:
        # reduced_hj_orbit / minimal_hj_stabilizer move to an SL2(Z)-equivalent
        # form, and equivalent forms generally do not share a root. Recompute
        # rho_j directly as a root of gQ_j itself (the larger of its two real
        # roots) so that (gQ_j, rho_j) is a matched form/root pair.
        places_K = real_places(K)
        emb_conj_K = embedding(places_K[1])
        emb_main_K = embedding(places_K[2])

        D_gQj = Int(discriminant(gQ_j))
        quotient_gQj, remainder_gQj = divrem(D_gQj, Dr)
        remainder_gQj == 0 || error("discriminant(gQ_j) = ", D_gQj,
                                     " is not an integer multiple of Dr = ", Dr)
        f_gQj = isqrt(quotient_gQj)
        f_gQj^2 == quotient_gQj || error("D_gQj / Dr = ", quotient_gQj,
                                          " is not a perfect square (got isqrt = ", f_gQj, ")")
        sqrt_gQj = f_gQj * sqrt_Dr
        root_plus  = (-gQ_j.b + sqrt_gQj) / (2 * gQ_j.a)
        root_minus = (-gQ_j.b - sqrt_gQj) / (2 * gQ_j.a)

        rho_j = Float64(emb_main_K(root_plus)) >= Float64(emb_main_K(root_minus)) ?
                root_plus : root_minus

        println("  product class = ", j_class, " rho_j_orig (picard rep) = ", rho_j_orig)
        println("  rho_j (larger root of gQ_j) = ", rho_j)
        println("  gQ_j (HJ-reduced) = ", gQ_j)
        println("  gQ_j(rho_j, 1) = ", gQ_j(rho_j, K(1)), " (should be 0)")

        M_g_j = multiplication_matrix_mod_d(K(gamma), rho_j, d, D)
        println("  M_g_j = ", M_g_j)

        QQ_form = binary_quadratic_form(gF_j.A[2,1], gF_j.A[2,2] - gF_j.A[1,1], -gF_j.A[1,2])

        c_g_j = e(-BigFloat(rademacher(gF_j.A)) / 24) /
                 (chi * sqrt(BigFloat(d + 1)))
        println("  c_g_j = ", c_g_j)

        # The order O has conductor F.q, and any ideal representing a
        # non-trivial Picard class of a conductor-q order will structurally
        # carry a factor of q in its basis coordinates. When q | d (as here,
        # q=3, d=21), requiring c1's numerator/denominator to be coprime to
        # the *full* d is unsatisfiable by construction, not a sign of a bad
        # candidate. Strip q's prime factors out of d before checking; the
        # compute_integralnewdu/injectivity test below remains the real
        # correctness gate and is unaffected by this relaxation.
        q_val = Int(F.q)
        d_check = d
        for (p, _) in factor(q_val)
            while d_check % Int(p) == 0
                d_check ÷= Int(p)
            end
        end
        println("  d = ", d, ", q = ", q_val, ", d_check (coprimality target) = ", d_check)

        function find_good_rhoB(rho_B_init, K, d)
            candidates = []
            for n in -20:20
                cand = rho_B_init + K(n)
                c1 = coeff(cand, 1)
                if gcd(Int(abs(numerator(c1))), d_check) == 1 &&
                   gcd(Int(denominator(c1)), d_check) == 1
                    main_val = Float64(emb_main_K(cand))
                    conj_val = Float64(emb_conj_K(cand))
                    if main_val < conj_val
                        c0 = coeff(cand, 0)
                        cand = c0 - c1 * gen(K)
                        println("    swapped to conjugate: ", cand)
                    end
                    push!(candidates, cand)
                end
            end
            try
                cand = -K(1) / rho_B_init
                c1 = coeff(cand, 1)
                if gcd(Int(abs(numerator(c1))), d_check) == 1 &&
                   gcd(Int(denominator(c1)), d_check) == 1
                    main_val = Float64(emb_main_K(cand))
                    conj_val = Float64(emb_conj_K(cand))
                    if main_val < conj_val
                        c0 = coeff(cand, 0)
                        cand = c0 - c1 * gen(K)
                        println("    swapped to conjugate: ", cand)
                    end
                    push!(candidates, cand)
                end
            catch end
            for n in -20:20
                try
                    cand = -K(1) / (rho_B_init + K(n))
                    c1 = coeff(cand, 1)
                    if gcd(Int(abs(numerator(c1))), d_check) == 1 &&
                       gcd(Int(denominator(c1)), d_check) == 1
                        main_val = Float64(emb_main_K(cand))
                        conj_val = Float64(emb_conj_K(cand))
                        if main_val < conj_val
                            c0 = coeff(cand, 0)
                            cand = c0 - c1 * gen(K)
                            println("    swapped to conjugate: ", cand)
                        end
                        push!(candidates, cand)
                    end
                catch end
            end
            println("  Found ", length(candidates), " rho_B candidates with c1 num and den coprime to d_check=", d_check)
            for cand in candidates
                println("    rho_B = ", cand, " c1 = ", coeff(cand, 1),
                        " main = ", Float64(emb_main_K(cand)),
                        " conj = ", Float64(emb_conj_K(cand)))
            end
            return candidates
        end

        good_rhoB_candidates = find_good_rhoB(rho_B_init, K, d)

        if isempty(good_rhoB_candidates)
            error("Could not find rho_B with c1 coprime to d_check=", d_check, " (d=", d, ")")
        end

        s1, s2, rho_B = nothing, nothing, nothing
        for rhoB_cand in good_rhoB_candidates
            println("  Trying rho_B candidate = ", rhoB_cand)
            try
                s1_try, s2_try = find_s1_s2(J, rhoB_cand, rho, u_B, P_O, mP_O, O, K, OK, d, rho_j)

                # test injectivity and valid denominators over all p
                seen = Set{Vector{Int}}()
                valid = true
                for p1 = 0:d-1
                    valid || break
                    for p2 = 0:d-1
                        (p1 == 0 && p2 == 0) && continue
                        indu = compute_integralnewdu(p1, p2, s1_try, s2_try, rhoB_cand, rho, rho_j, M_g_j, d)
                        if indu === nothing
                            println("  Candidate ", rhoB_cand, " failed du_to_int_mod_d at p=($p1,$p2)")
                            valid = false
                            break
                        end
                        if indu == [0, 0]
                            println("  Candidate ", rhoB_cand, " gives [0,0] at p=($p1,$p2)")
                            valid = false
                            break
                        end
                        if indu in seen
                            println("  Candidate ", rhoB_cand, " not injective at p=($p1,$p2): ", indu, " already seen")
                            valid = false
                            break
                        end
                        push!(seen, indu)
                    end
                end

                if valid
                    s1, s2 = s1_try, s2_try
                    rho_B = rhoB_cand
                    println("  Success with rho_B = ", rhoB_cand)
                    break
                else
                    println("  Candidate ", rhoB_cand, " failed injectivity/validity check")
                end
            catch e
                println("  Failed with rho_B = ", rhoB_cand, ": ", e)
                continue
            end
        end

        if s1 === nothing
            error("Could not find valid s1, s2 for any rho_B transform")
        end

        println("  Using rho_B = ", rho_B)
        println("  s1 = ", s1, " s2 = ", s2)

        test1 = true
        for p1 = 0:d-1
            for p2 = 0:d-1
                (p1 == 0 && p2 == 0) && continue

                gp = [p1, mod(NJ * Ng * p2, d)]
                p = [p1, p2]

                elem1 = K(p[2]) * rho - K(p[1])
                elem2 = K(s2) * rho_B - K(s1)
                product = elem1 * elem2

                u2, u1 = find_rho_j_and_coords(product, rho_j)
                println("    p=($p1,$p2): u1=", u1, " u2=", u2)

                du = [to_du_mod_d(u1, d), to_du_mod_d(u2, d)]

                raw_newdu = M_g_j * du
                newdu = [to_du_mod_d(raw_newdu[1], d), to_du_mod_d(raw_newdu[2], d)]

                integralnewdu = Int[]
                for u in newdu
                    du_int = d * u
                    val = du_to_int_mod_d(du_int, d)
                    push!(integralnewdu, val)
                end

                if p1 == 7 && p2 == 7
                    println("  p=(7,7): integralnewdu = ", integralnewdu)
                end

                QA = BigFloat(-QQ_form(integralnewdu...) / (d * (d - 2)))
                s = isodd(d) ? 1 : (1 + integralnewdu[1]) * (1 + integralnewdu[2])
                new_ovs[gp[1]+1, gp[2]+1] = ζ^QA * (-1)^s * c_g_j *
                                              shin_of_tuple(gF_j, integralnewdu)

                if abs(abs(shin_of_tuple(gF_j, integralnewdu) * shin_of_tuple(gF_j, -integralnewdu)) - 1.0) > 1e-10
                    println("    FAIL at p=($p1,$p2)")
                    test1 = false
                end
            end
        end
        println("  all |vv(p)vv(-p)| correct: ", test1)
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

    U_OK, mU_OK = unit_group(OK)
    eps = elem_in_nf(mU_OK(U_OK[2]))
    println("    num_principal: eps (from OK) = ", eps)

    period = 0
    for k in 1:100_000
        epsk_nf = eps^k
        epsk_O = try O(OK(epsk_nf)) catch; nothing end
        epsk_O === nothing && continue

        if isone(proj_m_O(epsk_O)) &&
           all(sign(K(epsk_nf), p) > 0 for p in Sigma)
            period = k
            break
        end
    end
    period == 0 && error("Period not found")
    println("    num_principal: period = ", period)
    N2 = torsion_index * period
    println("    num_principal: N2 = ", N2)
    GC.gc()

    N = Int(N1 / N2) - 1
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

    n_principal = min(5, length(principal_triples))
    println("\n=== Testing first ", n_principal, " principal twists ===")
    for (k, (J, I, gamma)) in enumerate(principal_triples[1:n_principal])
        println("\nPrincipal twist ", k, " of ", n_principal, ":")
        println("  J = ", J)
        println("  gamma = ", gamma)
        println("  norm(gamma) = ", norm(K(gamma)))
        new_ovs, gF, psi = twisted_chi_general(F, J, gamma, K, O,
                                                true, P_O, mP_O,
                                                rho_reps, gQ_reps, F_class)
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
                                                rho_reps, gQ_reps, F_class)
    end
end

test_all_twists(21)