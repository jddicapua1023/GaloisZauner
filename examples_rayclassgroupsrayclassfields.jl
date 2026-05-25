using Hecke
include("ray_unit_quotient.jl")
include("term2.jl")
include("enumeration_with_infinite_places.jl")

# Example 1 conductor 1 m is 7
K, a = quadratic_field(2)
O  = equation_order(K)           # maximal order
OK = maximal_order(K)            # 

P, mP = picard_group(OK)
println("h(OK) for K = Q(sqrt(2))): ", order(P))
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

m = ideal(O, 7)
mp = ideal(OK, 1)
delta = ideal(OK, 7)

plcs = real_places(K)
Sigma = [plcs[2]]
Sigma_prime = InfPlc[]

println("Sigma is: ", Sigma)

Q, structure = residue_unit_quotient(OK, O, mp, m, delta, Sigma_prime, Sigma)

N1 = prod(structure)

Q, reps, phi = ray_unit_quotient(OK, O, mp, m, Sigma_prime, Sigma)

P_O2, mP_O2 = picard_group(O)
N4 = order(P_O2)
println("N4 is (order of ring class group) ", N4)

N2 = order(Q)

N3 = order(P)

N = N1*N3/(N2*N4) - 1   # number of classes to find

println("N is ", N)

result = enumerate_principal_ray_classes(K, O, m, Sigma, N)

for (I, gamma, h) in result
    println("Generator: ", gamma, "  Norm: ", h, "  Ideal: ", I)
end

# Example 2 conductor 2 m is 7
K, a = quadratic_field(2)
O = Order(K, [K(1), 2*a])      # order of conductor 2
OK = maximal_order(K)            # 

P, mP = picard_group(OK)
println("h(OK) for K = Q(sqrt(2))): ", order(P))
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

m = ideal(O, 7)
mp = ideal(OK, 1)
delta = ideal(OK, 14)

plcs = real_places(K)
Sigma = [plcs[2]]
Sigma_prime = InfPlc[]

println("Sigma is: ", Sigma)

Q, structure = residue_unit_quotient(OK, O, mp, m, delta, Sigma_prime, Sigma)

N1 = prod(structure)

Q, reps, phi = ray_unit_quotient(OK, O, mp, m, Sigma_prime, Sigma)

P_O2, mP_O2 = picard_group(O)
N4 = order(P_O2)
println("N4 is (order of ring class group) ", N4)

N2 = order(Q)

N3 = order(P)

N = N1*N3/(N2*N4) - 1   # number of classes to find

println("N is ", N)

result = enumerate_principal_ray_classes(K, O, m, Sigma, N)

for (I, gamma, h) in result
    println("Generator: ", gamma, "  Norm: ", h, "  Ideal: ", I)
end

# Example 3 conductor 5 m is 7
K, a = quadratic_field(2)
O = Order(K, [K(1), 5*a])      # order of conductor 5
OK = maximal_order(K)            # 

P, mP = picard_group(OK)
println("h(OK) for K = Q(sqrt(2))): ", order(P))
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

m = ideal(O, 7)
mp = ideal(OK, 1)
delta = ideal(OK, 35)

plcs = real_places(K)
Sigma = [plcs[2]]
Sigma_prime = InfPlc[]

println("Sigma is: ", Sigma)

Q, structure = residue_unit_quotient(OK, O, mp, m, delta, Sigma_prime, Sigma)

N1 = prod(structure)

Q, reps, phi = ray_unit_quotient(OK, O, mp, m, Sigma_prime, Sigma)

P_O2, mP_O2 = picard_group(O)
N4 = order(P_O2)
println("N4 is (order of ring class group) ", N4)

N2 = order(Q)

N3 = order(P)

N = N1*N3/(N2*N4) - 1   # number of classes to find

println("N is ", N)

result = enumerate_principal_ray_classes(K, O, m, Sigma, N)

for (I, gamma, h) in result
    println("Generator: ", gamma, "  Norm: ", h, "  Ideal: ", I)
end



