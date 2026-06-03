using Hecke
using Zauner
using GenericFFT: fft
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

function act_on_quadratic_form(J, F, K, O)
    D = Int(F.D)
    a = gen(K)
    OK = maximal_order(K)

    _, f = coredisc(discriminant(F.Q))
    rho = (-F.Q.b + Int(f) * a) / (2 * F.Q.a)

    t_rho = fractional_ideal(OK, [K(1), rho])
    g_t_rho = J * t_rho
    b = basis(g_t_rho)

    rational_shift = coeff(rho, 0)
    rho_2 = b[2] + rational_shift

    c1 = coeff(rho_2, 0)
    c2 = coeff(rho_2, 1)

    p0 = c1^2 - c2^2 * D
    p1 = -2 * c1
    p2 = 1

    scale = lcm(Int(denominator(p0)), Int(denominator(p1)), Int(denominator(p2)))
    a_int = Int(scale * p2)
    b_int = Int(scale * p1)
    c_int = Int(scale * p0)

    g_int = gcd(a_int, b_int, c_int)
    a_int, b_int, c_int = a_int ÷ g_int, b_int ÷ g_int, c_int ÷ g_int

    gQ = binary_quadratic_form(a_int, b_int, c_int)

    d = Int(F.d)
    V = reduced_hj_orbit(gQ)
    gQ_reduced = minimal_hj_stabilizer(V, d)

    return gQ_reduced
end

function artin_action_on_sqrtm(d::Int, f::Int, m::Int, J, Sigma::Vector{Int}, K)
    nJ = Int(norm(J))
    kron = kronecker_symbol(m, nJ)
    return kron
end

function twisted_chi(F::AdmissibleTuple, J, gamma, K, O)
    d = Int(F.d)
    D = Int(F.D)
    a = gen(K)

    gQ = act_on_quadratic_form(J, F, K, O)
    println("  gQ = ", gQ)
    gF = AdmissibleTuple(d, F.r, gQ)
    println("  gF = ", gF)

    Ng = Int(norm(K(gamma)))
    println("  Ng = ", Ng)

    chi = artin_action_on_sqrtm(D, Int(F.q), d + 1, J, [1, 2], K)
    println("  chi = ", chi)

    _, f = coredisc(discriminant(F.Q))
    rho = (-F.Q.b + Int(f) * a) / (2 * F.Q.a)

    _, f_g = coredisc(discriminant(gF.Q))
    rho_g = (-gF.Q.b + Int(f_g) * a) / (2 * gF.Q.a)

    M_g = multiplication_matrix_mod_d(K(gamma), rho_g, d, D)
    println("  M_g = ", M_g)

    ζ = -cispi(BigFloat(1) / d)
    QQ = binary_quadratic_form(F.A[2,1], F.A[2,2] - F.A[1,1], -F.A[1,2])
    c_g = e(-BigFloat(rademacher(F.A)) / 24) / (chi * sqrt(BigFloat(d + 1)))
    println("  c_g = ", c_g)

    if Ng == 1
        println("  Testing unit invariance of shin_of_tuple:")
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

        units_norm1 = []
        if eps !== nothing
            current = one(K)
            for k in 1:1000
                current = current * eps
                n = norm(current)
                if mod(Int(abs(n)), d) == 1
                    push!(units_norm1, current)
                end
                length(units_norm1) >= 20 && break
            end
        end

        println("  Found ", length(units_norm1), " units with norm ≡ 1 mod d")

        p_test = [1, 1]
        println("  shin_of_tuple at p=$p_test = ", shin_of_tuple(gF, p_test))

        for (k, u) in enumerate(units_norm1)
            u_gamma = u * K(gamma)
            M_u = multiplication_matrix_mod_d(u_gamma, rho_g, d, D)
            newp = mod.(M_u * p_test, d)
            val = shin_of_tuple(gF, newp)
            println("  k=$k norm(u)=$(norm(u)) M_u=$M_u newp=$newp shin=$val")
        end
    end

    new_ovs = zeros(Complex{BigFloat}, d, d)
    new_ovs[1, 1] = sqrt(BigFloat(d + 1))
    for p1 = 0:d-1
        for p2 = 0:d-1
            (p1 == 0 && p2 == 0) && continue
            p = [p1, p2]

            gp = [p1, mod(Ng * p2, d)]

            newp = mod.(M_g * p, d)

            newQA = BigFloat(-QQ(newp...) / (d * (d - 2)))
            news = (isodd(d) ? 1 : (1 + newp[1]) * (1 + newp[2]))

            # corrected: (-1)^(Ng*s) instead of (-1)^s
            new_ovs[gp[1]+1, gp[2]+1] = ζ^(newQA) * (-1)^(news) * c_g * shin_of_tuple(gF, newp)
        end
    end

    max_imag = maximum(abs.(imag.(new_ovs)))
    println("  max imaginary part of new_ovs = ", max_imag)

    dev = ghost_overlap_test_complex(new_ovs, d)
    println("  ghost condition deviation on new_ovs = ", dev)

    println("  Checking sign of v(p)*v(-p):")
    pos_count = 0
    neg_count = 0
    for p1 = 0:d-1
        for p2 = 0:d-1
            (p1 == 0 && p2 == 0) && continue
            ov_p = new_ovs[p1+1, p2+1]
            ov_mp = new_ovs[mod(-p1,d)+1, mod(-p2,d)+1]
            product = ov_p * ov_mp
            normed = real(product) * (d+1)
            if abs(imag(product)) > 1e-10
                println("  WARNING: v($p1,$p2)*v($(mod(-p1,d)),$(mod(-p2,d))) has large imaginary part: ", imag(product) * (d+1))
            end
            if normed > 0
                pos_count += 1
            else
                neg_count += 1
            end
        end
    end
    println("    v(p)*v(-p) = +1/(d+1) count: ", pos_count)
    println("    v(p)*v(-p) = -1/(d+1) count: ", neg_count)

    return new_ovs
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

            # diagnostic: check imaginary part of product
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

    # verify
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
                println("  FAIL at ($p1,$p2): product = ", product, " a=", a[p1+1,p2+1], " a(-p)=", a[mp1+1,mp2+1])
            end
            max_dev = max(max_dev, Float64(dev))
        end
    end
    println("  max deviation after sign correction = ", max_dev)

    return a
end

function num_principal_ray_classes(OK, O, m, Sigma)
    K = number_field(O)
    Sigma_prime = InfPlc[]
    mp = ideal(OK, 1)
    println("    num_principal: starting")
    GC.gc()

    f = conductor(O)
    println("    num_principal: conductor = ", f)
    GC.gc()

    delta = ideal(OK, minimum(f) * minimum(m))
    println("    num_principal: delta constructed")
    GC.gc()

    _, structure = residue_unit_quotient(OK, O, mp, m, delta, Sigma_prime, Sigma)
    println("    num_principal: residue_unit_quotient done, structure = ", structure)
    N1 = prod(structure)
    GC.gc()

    Q_ray, _, _ = ray_unit_quotient(OK, O, mp, m, Sigma_prime, Sigma)
    println("    num_principal: ray_unit_quotient done")
    N2 = order(Q_ray)
    GC.gc()

    P_OK, _ = picard_group(OK)
    println("    num_principal: picard OK done")
    N3 = order(P_OK)
    GC.gc()

    P_O, _ = picard_group(O)
    println("    num_principal: picard O done")
    N4 = order(P_O)

    N = Int(N1 * N3 / (N2 * N4)) - 1
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
        g = isempty(gens_P) ? zero(P_O) : sum(exps[i] * gens_P[i] for i in 1:length(gens_P))
        push!(group_elems, g)
    end
    println("  ray_class: group_elems built, count = ", length(group_elems))
    GC.gc()

    picard_reps = make_coprime_representatives(O, m, mP_O, group_elems)
    println("  ray_class: make_coprime_representatives done")
    GC.gc()

    N = num_principal_ray_classes(OK, O, m, Sigma)
    println("  ray_class: num_principal_ray_classes = ", N)
    GC.gc()

    principal_reps = enumerate_principal_ray_classes(K, O, m, Sigma, N)
    println("  ray_class: enumerate done")

    all_reps = []
    all_triples = []
    trivial = ideal(OK, 1)

    for J in picard_reps
        push!(all_reps, J)
        push!(all_triples, (J, trivial, one(K)))
        for (I, gamma, h) in principal_reps
            push!(all_reps, J * I)
            push!(all_triples, (J, I, elem_in_nf(gamma)))
        end
    end

    return all_reps, all_triples
end

function test_principal_only_twists(d::Int)
    F = AdmissibleTuple(d)
    @show F

    D = Int(F.D)
    q = Int(F.q)
    K, a = quadratic_field(D)
    OK = maximal_order(K)
    GC.gc()
    GC.gc()
    O = order(K, [K(1), K(q * OK(a))])
    GC.gc()

    println("D = ", D, ", q = ", q)

    d = Int(F.d)
    m = ideal(O, d)
    GC.gc()

    # only the second infinite place
    all_places = real_places(K)
    Sigma = [all_places[2]]
    println("Using Sigma = second real place only: ", Sigma)
    GC.gc()
    GC.gc()

    all_reps, all_triples = ray_class_group_representatives(K, O, m, Sigma)

    trivial_ideal = ideal(OK, 1)

    principal_triples = [(J, I, gamma) for (J, I, gamma) in all_triples if J == trivial_ideal]
    println("Number of purely principal triples: ", length(principal_triples))

    # square each class
    squared_triples = [(J, I * I, K(gamma) * K(gamma)) for (J, I, gamma) in principal_triples]
    println("Number of squared triples: ", length(squared_triples))

    n = min(10, length(squared_triples))
    println("Testing first ", n, " squared principal twists:")

    for (k, (J, I, gamma)) in enumerate(squared_triples[1:n])
        println("\nSquared principal twist ", k, " of ", n, ":")
        println("  gamma = ", gamma)
        println("  norm(gamma) = ", norm(K(gamma)))
        new_ovs = twisted_chi(F, J, gamma, K, O)

        println("  Finding sign corrections...")
        a = find_signs(new_ovs, d)
        println("  Sign matrix a(p):")
        for p1 = 0:d-1
            println("    p1=", p1, ": ", [a[p1+1, p2+1] for p2 = 0:d-1])
        end
    end
end

test_principal_only_twists(13)