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

# Complex version of ghost overlap test - checks equiangularity without
# requiring overlaps to be real
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

# Express x = p + q*sqrt(D) in the basis {1, rho}
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

# Compute gQ: the quadratic form obtained by acting on F.Q with ideal J,
# then HJ-reduce to get a valid form for AdmissibleTuple
function act_on_quadratic_form(J, F, K, O)
    D = Int(F.D)
    a = gen(K)

    _, f = coredisc(discriminant(F.Q))
    rho = (-F.Q.b + Int(f) * a) / (2 * F.Q.a)

    t_rho = fractional_ideal(O, [K(1), rho])
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

# Compute the action of the Artin symbol σ_J on √m
function artin_action_on_sqrtm(d::Int, f::Int, m::Int, J, Sigma::Vector{Int}, K)
    nJ = Int(norm(J))
    kron = kronecker_symbol(m, nJ)
    return kron
end

# Compute the twisted overlap matrix for a single class group element
function twisted_chi(F::AdmissibleTuple, J, gamma, K, O)
    d = Int(F.d)
    D = Int(F.D)
    a = gen(K)

    # compute gQ and gF
    gQ = act_on_quadratic_form(J, F, K, O)
    println("  gQ = ", gQ)
    gF = AdmissibleTuple(d, F.r, gQ)
    println("  gF = ", gF)

    # norm of the principal part
    Ng = Int(abs(norm(K(gamma))))
    println("  Ng = ", Ng)

    # compute Artin action of J on sqrt(d+1)
    chi = artin_action_on_sqrtm(D, Int(F.q), d + 1, J, [1, 2], K)
    println("  chi = ", chi)

    # positive root of F.Q for the multiplication matrix
    _, f = coredisc(discriminant(F.Q))
    rho = (-F.Q.b + Int(f) * a) / (2 * F.Q.a)

    # matrix of multiplication by gamma on {1, rho} mod d
    M_g = multiplication_matrix_mod_d(K(gamma), rho, d, D)
    println("  M_g = ", M_g)

    # phase factors
    ζ = -cispi(BigFloat(1) / d)
    QQ = binary_quadratic_form(F.A[2,1], F.A[2,2] - F.A[1,1], -F.A[1,2])
    c_g = e(-BigFloat(Ng) * BigFloat(rademacher(F.A)) / 24) / (chi * sqrt(BigFloat(d + 1)))
    println("  c_g = ", c_g)

    # unit invariance test for trivial twist
    if Ng == 1
        println("  Testing unit invariance of shin_of_tuple:")
        U, mU = unit_group(O)

        # find fundamental unit of O
        eps = nothing
        for i in 1:ngens(U)
            u = elem_in_nf(mU(U[i]))
            if !(u == one(K) || u == -one(K))
                eps = u
                break
            end
        end

        # collect 20 units with norm ≡ 1 mod d
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
            M_u = multiplication_matrix_mod_d(u_gamma, rho, d, D)
            newp = mod.(M_u * p_test, d)
            val = shin_of_tuple(gF, newp)
            println("  k=$k norm(u)=$(norm(u)) M_u=$M_u newp=$newp shin=$val")
        end
    end

    # compute new overlaps directly using the twisted formula
    new_ovs = zeros(Complex{BigFloat}, d, d)
    new_ovs[1, 1] = sqrt(BigFloat(d + 1))  # (0,0) overlap
    for p1 = 0:d-1
        for p2 = 0:d-1
            (p1 == 0 && p2 == 0) && continue
            p = [p1, p2]

            # new position after Galois action
            gp = [p1, mod(Ng * p2, d)]

            # what we feed to shin_of_tuple
            newp = mod.(M_g * p, d)

            QA = BigFloat(-QQ(p...) / (d * (d - 2)))
            s = (isodd(d) ? 1 : (1 + p[1]) * (1 + p[2]))

            # new overlap computed directly
            new_ovs[gp[1]+1, gp[2]+1] = ζ^(Ng * QA) * (-1)^s * c_g * shin_of_tuple(gF, newp)
        end
    end

    # check imaginary parts
    max_imag = maximum(abs.(imag.(new_ovs)))
    println("  max imaginary part of new_ovs = ", max_imag)

    # check ghost condition
    dev = ghost_overlap_test_complex(new_ovs, d)
    println("  ghost condition deviation on new_ovs = ", dev)

    # check sign distribution
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

function num_principal_ray_classes(OK, O, m, Sigma)
    K = number_field(O)
    Sigma_prime = InfPlc[]
    mp = ideal(OK, 1)

    f = conductor(O)
    delta = ideal(OK, minimum(f) * minimum(m))

    _, structure = residue_unit_quotient(OK, O, mp, m, delta, Sigma_prime, Sigma)
    N1 = prod(structure)

    Q_ray, _, _ = ray_unit_quotient(OK, O, mp, m, Sigma_prime, Sigma)
    N2 = order(Q_ray)

    P_OK, _ = picard_group(OK)
    N3 = order(P_OK)

    P_O, _ = picard_group(O)
    N4 = order(P_O)

    N = Int(N1 * N3 / (N2 * N4)) - 1
    return N
end

function ray_class_group_representatives(K, O, m, Sigma)
    OK = maximal_order(K)

    P_O, mP_O = picard_group(O)
    gens_P = gens(P_O)
    divs = elementary_divisors(P_O)
    ranges = [0:Int(d)-1 for d in divs]

    group_elems = []
    for exps in Iterators.product(ranges...)
        g = isempty(gens_P) ? zero(P_O) : sum(exps[i] * gens_P[i] for i in 1:length(gens_P))
        push!(group_elems, g)
    end

    picard_reps = make_coprime_representatives(O, m, mP_O, group_elems)

    N = num_principal_ray_classes(OK, O, m, Sigma)
    println("Number of principal ray classes: ", N)

    principal_reps = enumerate_principal_ray_classes(K, O, m, Sigma, N)

    all_reps = []
    all_triples = []

    trivial = ideal(O, 1)

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

function test_principal_only_twists()
    F = AdmissibleTuple(7)
    @show F

    # construct K and O from F
    D = Int(F.D)
    q = Int(F.q)
    K, a = quadratic_field(D)
    OK = maximal_order(K)
    O = order(K, [K(1), K(q * OK(a))])

    println("D = ", D, ", q = ", q)

    d = Int(F.d)
    m = ideal(O, d)
    Sigma = real_places(K)

    all_reps, all_triples = ray_class_group_representatives(K, O, m, Sigma)

    trivial_ideal = ideal(O, 1)
    principal_triples = [(J, I, gamma) for (J, I, gamma) in all_triples if J == trivial_ideal]
    println("Number of purely principal triples: ", length(principal_triples))

    n = min(10, length(principal_triples))
    println("Testing first ", n, " purely principal twists:")

    for (k, (J, I, gamma)) in enumerate(principal_triples[1:n])
        println("\nPrincipal twist ", k, " of ", n, ":")
        println("  gamma = ", gamma)
        println("  norm(gamma) = ", abs(norm(K(gamma))))
        new_ovs = twisted_chi(F, J, gamma, K, O)
    end
end

test_principal_only_twists()

function test_shin_values(F::AdmissibleTuple)
    d = Int(F.d)
    println("Testing shin_of_tuple for all p vectors with F = ", F)
    for p1 = 0:d-1
        for p2 = 0:d-1
            p = [p1, p2]
            val = shin_of_tuple(F, p)
            println("  p=($p1,$p2): shin = ", val)
        end
    end
end

F = AdmissibleTuple(7)
test_shin_values(F)