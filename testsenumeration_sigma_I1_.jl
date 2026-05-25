using Hecke
include("ray_unit_quotient.jl")
include("term2.jl")
include("enumeration_with_infinite_places.jl")

# Example 1 conductor 2 m is 3
K, a = quadratic_field(5)
O2  = equation_order(K)           # Z[√5], conductor 2
OK = maximal_order(K)            # Z[(1+√5)/2]

P, mP = picard_group(OK)
println("h(OK) for K = Q(sqrt(5))): ", order(P))
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

m = ideal(O2, 3)
mp = ideal(OK, 1)
delta = ideal(OK, 6)

plcs = real_places(K)
Sigma = [plcs[1]]
Sigma_prime = InfPlc[]

println("Sigma is: ", Sigma)

Q, structure = residue_unit_quotient(OK, O2, mp, m, delta, Sigma_prime, Sigma)

N1 = prod(structure)

Q, reps, phi = ray_unit_quotient(OK, O2, mp, m, Sigma_prime, Sigma)

P_O2, mP_O2 = picard_group(O2)
N4 = order(P_O2)

N2 = order(Q)

N3 = order(P)

N = N1*N3/(N2*N4) - 1   # number of classes to find

println("N is ", N)

result = enumerate_principal_ray_classes(K, O2, m, Sigma, N)

for (I, gamma, h) in result
    println("Generator: ", gamma, "  Norm: ", h, "  Ideal: ", I)
end

# Example 2 conductor 4 m is 3
K, a = quadratic_field(5)
O4 = Order(K, [K(1), 2*a]) 
OK = maximal_order(K)            # Z[(1+√5)/2]

P, mP = picard_group(OK)
println("h(OK) for Q(sqrt(5): ", order(P))
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

m = ideal(O4, 3)
mp = ideal(OK, 1)
delta = ideal(OK, 12)

plcs = real_places(K)
Sigma = [plcs[1]]
Sigma_prime = InfPlc[]

println("Sigma is: ", Sigma)

Q, structure = residue_unit_quotient(OK, O4, mp, m, delta, Sigma_prime, Sigma)

N1 = prod(structure)

Q, reps, phi = ray_unit_quotient(OK, O4, mp, m, Sigma_prime, Sigma)

P_O4, mP_O4 = picard_group(O4)
N4 = order(P_O4)

N2 = order(Q)

N3 = order(P)

N = N1*N3/(N2*N4) - 1  # number of classes to find

println("N is ", N)

result = enumerate_principal_ray_classes(K, O4, m, Sigma, N)

for (I, gamma, h) in result
    println("Generator: ", gamma, "  Norm: ", h, "  Ideal: ", I)
end


# Example 3
K, a = quadratic_field(5)
O5 = Order(K, [K(1), 5*(1+a)//2]) 
OK = maximal_order(K)            # Z[(1+√5)/2]

P, mP = picard_group(OK)
println("h(OK) for Q(sqrt(5): ", order(P))
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

m = ideal(O5, 3)
mp = ideal(OK, 3)
delta = ideal(OK, 15)

plcs = real_places(K)
Sigma = [plcs[1]]
Sigma_prime = InfPlc[]

println("Sigma is: ", Sigma)

Q, structure = residue_unit_quotient(OK, O5, mp, m, delta, Sigma_prime, Sigma)

N1 = prod(structure)

Q, reps, phi = ray_unit_quotient(OK, O5, mp, m, Sigma_prime, Sigma)

P_O5, mP_O5 = picard_group(O5)
N4 = order(P_O5)

N2 = order(Q)

N3 = order(P)

N = N1*N3/(N2*N4) - 1  # number of classes to find

println("N is ", N)

result = enumerate_principal_ray_classes(K, O5, m, Sigma, N)

for (I, gamma, h) in result
    println("Generator: ", gamma, "  Norm: ", h, "  Ideal: ", I)
end

# Example 4
K, a = quadratic_field(5)
O5 = Order(K, [K(1), 5*(1+a)//2]) 
OK = maximal_order(K)            # Z[(1+√5)/2]

P, mP = picard_group(OK)
println("h(OK) for Q(sqrt(5): ", order(P))
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

m = ideal(O5, 5)
mp = ideal(OK, 5)
delta = ideal(OK, 25)

plcs = real_places(K)
Sigma = [plcs[1]]
Sigma_prime = InfPlc[]

println("Sigma is: ", Sigma)

Q, structure = residue_unit_quotient(OK, O5, mp, m, delta, Sigma_prime, Sigma)

N1 = prod(structure)

Q, reps, phi = ray_unit_quotient(OK, O5, mp, m, Sigma_prime, Sigma)

P_O5, mP_O5 = picard_group(O5)
N4 = order(P_O5)


N2 = order(Q)

N3 = order(P)

N = N1*N3/(N2*N4) - 1  # number of classes to find

println("N is ", N)

result = enumerate_principal_ray_classes(K, O5, m, Sigma, N)

for (I, gamma, h) in result
    println("Generator: ", gamma, "  Norm: ", h, "  Ideal: ", I)
end


# Example 5
K, a = quadratic_field(5)
O4 = Order(K, [K(1), 2*a])          
OK = maximal_order(K)            

P, mP = picard_group(OK)
println("h(OK) for Q(sqrt(5): ", order(P))
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

m = ideal(O4, 4)
mp = ideal(OK, 1)
delta = ideal(OK, 16)

plcs = real_places(K)
Sigma = [plcs[1]]
Sigma_prime = InfPlc[]

println("Sigma is: ", Sigma)

Q, structure = residue_unit_quotient(OK, O4, mp, m, delta, Sigma_prime, Sigma)

N1 = prod(structure)

Q, reps, phi = ray_unit_quotient(OK, O4, mp, m, Sigma_prime, Sigma)

P_O4, mP_O4 = picard_group(O4)
N4 = order(P_O4)

N2 = order(Q)

N3 = order(P)

N = N1*N3/(N2*N4) - 1  # number of classes to find

println("N is ", N)

#= Diagnostics for Example 5
println("Norm of m: ", norm(m))
println("Discriminant of K: ", discriminant(OK))

# Check which small elements are coprime to m
B7 = basis(O7)
for i in -5:5, j in -5:5
    gamma = i * B7[1] + j * B7[2]
    iszero(gamma) && continue
    g = elem_in_nf(gamma)
    I = try ideal(O7, O7(g)) catch; continue end
    if isone(I + m)
        println("Coprime to m: ", gamma, "  norm=", norm(gamma))
    end
end


println("Unit reps mod m computation...")
U7, mU7 = unit_group(O7)
println("Unit group generators: ", [elem_in_nf(mU7(U7[i])) for i in 1:ngens(U7)])
=#

result = enumerate_principal_ray_classes(K, O4, m, Sigma, N)

for (I, gamma, h) in result
    println("Generator: ", gamma, "  Norm: ", h, "  Ideal: ", I)
end

# Example 6
K, a = quadratic_field(3)
O7 = Order(K, [K(1), 7*a])          
OK = maximal_order(K)            

P, mP = picard_group(OK)
println("h(OK) for Q(sqrt(3): ", order(P))
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

m = ideal(O7, 3)
mp = ideal(OK, 1)
delta = ideal(OK, 21)

plcs = real_places(K)
Sigma = [plcs[1]]
Sigma_prime = InfPlc[]

println("Sigma is: ", Sigma)

Q, structure = residue_unit_quotient(OK, O7, mp, m, delta, Sigma_prime, Sigma)

N1 = prod(structure)

Q, reps, phi = ray_unit_quotient(OK, O7, mp, m, Sigma_prime, Sigma)

P_O7, mP_O7 = picard_group(O7)
N4 = order(P_O7)


N2 = order(Q)

N3 = order(P)

N = N1*N3/(N2*N4) - 1  # number of classes to find

println("N is ", N)

# Diagnostics for Example 5
println("Norm of m: ", norm(m))
println("Discriminant of K: ", discriminant(OK))

# Check which small elements are coprime to m
B7 = basis(O7)
for i in -5:5, j in -5:5
    gamma = i * B7[1] + j * B7[2]
    iszero(gamma) && continue
    g = elem_in_nf(gamma)
    I = try ideal(O7, O7(g)) catch; continue end
    if isone(I + m)
        println("Coprime to m: ", gamma, "  norm=", norm(gamma))
    end
end

println("Unit reps mod m computation...")
U7, mU7 = unit_group(O7)
println("Unit group generators: ", [elem_in_nf(mU7(U7[i])) for i in 1:ngens(U7)])

result = enumerate_principal_ray_classes(K, O7, m, Sigma, N)

for (I, gamma, h) in result
    println("Generator: ", gamma, "  Norm: ", h, "  Ideal: ", I)
end

# Example 7 conductor 2 m is 7
println("Example 7")
K, a = quadratic_field(5)
O2 = Order(K, [K(1), a])          
OK = maximal_order(K)            

P, mP = picard_group(OK)
println("h(OK) for Q(sqrt(5): ", order(P))
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

m = ideal(O2, 7)
mp = ideal(OK, 1)
delta = ideal(OK, 14)

plcs = real_places(K)
Sigma = [plcs[1]]
Sigma_prime = InfPlc[]

println("Sigma is: ", Sigma)

Q, structure = residue_unit_quotient(OK, O2, mp, m, delta, Sigma_prime, Sigma)

N1 = prod(structure)

Q, reps, phi = ray_unit_quotient(OK, O2, mp, m, Sigma_prime, Sigma)

P_O2, mP_O2 = picard_group(O2)
N4 = order(P_O2)


N2 = order(Q)

N3 = order(P)

N = N1*N3/(N2*N4) - 1  # number of classes to find

println("N is ", N)

#= Diagnostics for Example 5
println("Norm of m: ", norm(m))
println("Discriminant of K: ", discriminant(OK))

# Check which small elements are coprime to m
B7 = basis(O2)
for i in -5:5, j in -5:5
    gamma = i * B7[1] + j * B7[2]
    iszero(gamma) && continue
    g = elem_in_nf(gamma)
    I = try ideal(O7, O7(g)) catch; continue end
    if isone(I + m)
        println("Coprime to m: ", gamma, "  norm=", norm(gamma))
    end
end

println("Unit reps mod m computation...")
U7, mU7 = unit_group(O7)
println("Unit group generators: ", [elem_in_nf(mU7(U7[i])) for i in 1:ngens(U7)])
=#

result = enumerate_principal_ray_classes(K, O2, m, Sigma, N)

for (I, gamma, h) in result
    println("Generator: ", gamma, "  Norm: ", h, "  Ideal: ", I)
end

# Example 8 conductor 4 m is 7
println("Example 8")
K, a = quadratic_field(5)
O4 = Order(K, [K(1), 2*a])          
OK = maximal_order(K)            

P, mP = picard_group(OK)
println("h(OK) for Q(sqrt(5): ", order(P))
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

m = ideal(O4, 7)
mp = ideal(OK, 1)
delta = ideal(OK, 28)

plcs = real_places(K)
Sigma = [plcs[1]]
Sigma_prime = InfPlc[]

println("Sigma is: ", Sigma)

Q, structure = residue_unit_quotient(OK, O4, mp, m, delta, Sigma_prime, Sigma)

N1 = prod(structure)

Q, reps, phi = ray_unit_quotient(OK, O4, mp, m, Sigma_prime, Sigma)

P_O4, mP_O4 = picard_group(O4)
N4 = order(P_O4)


N2 = order(Q)

N3 = order(P)

N = N1*N3/(N2*N4) - 1  # number of classes to find

println("N is ", N)

#= Diagnostics for Example 5
println("Norm of m: ", norm(m))
println("Discriminant of K: ", discriminant(OK))

# Check which small elements are coprime to m
B7 = basis(O7)
for i in -5:5, j in -5:5
    gamma = i * B7[1] + j * B7[2]
    iszero(gamma) && continue
    g = elem_in_nf(gamma)
    I = try ideal(O7, O7(g)) catch; continue end
    if isone(I + m)
        println("Coprime to m: ", gamma, "  norm=", norm(gamma))
    end
end

println("Unit reps mod m computation...")
U7, mU7 = unit_group(O7)
println("Unit group generators: ", [elem_in_nf(mU7(U7[i])) for i in 1:ngens(U7)])
=#

result = enumerate_principal_ray_classes(K, O4, m, Sigma, N)

for (I, gamma, h) in result
    println("Generator: ", gamma, "  Norm: ", h, "  Ideal: ", I)
end

# Example 9 conductor 5 modulus is 11
println("Example 9")
K, a = quadratic_field(5)
O5 = Order(K, [K(1), 5*(1+a)//2]) 
OK = maximal_order(K)            # Z[(1+√5)/2]

P, mP = picard_group(OK)
println("h(OK) for Q(sqrt(5): ", order(P))
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

m = ideal(O5, 11)
mp = ideal(OK, 1)
delta = ideal(OK, 55)

plcs = real_places(K)
Sigma = [plcs[1]]
Sigma_prime = InfPlc[]

println("Sigma is: ", Sigma)

Q, structure = residue_unit_quotient(OK, O5, mp, m, delta, Sigma_prime, Sigma)

N1 = prod(structure)

Q, reps, phi = ray_unit_quotient(OK, O5, mp, m, Sigma_prime, Sigma)

P_O5, mP_O5 = picard_group(O5)
N4 = order(P_O5)


N2 = order(Q)

N3 = order(P)

N = N1*N3/(N2*N4) - 1  # number of classes to find

println("N is ", N)

result = enumerate_principal_ray_classes(K, O5, m, Sigma, N)

for (I, gamma, h) in result
    println("Generator: ", gamma, "  Norm: ", h, "  Ideal: ", I)
end


# Example 10 conductor 4 m is 11
println("Example 10")
K, a = quadratic_field(5)
O4 = Order(K, [K(1), 2*a])          
OK = maximal_order(K)            

P, mP = picard_group(OK)
println("h(OK) for Q(sqrt(5): ", order(P))
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

m = ideal(O4, 11)
mp = ideal(OK, 1)
delta = ideal(OK, 44)

plcs = real_places(K)
Sigma = [plcs[1]]
Sigma_prime = InfPlc[]

println("Sigma is: ", Sigma)

Q, structure = residue_unit_quotient(OK, O4, mp, m, delta, Sigma_prime, Sigma)

N1 = prod(structure)

Q, reps, phi = ray_unit_quotient(OK, O4, mp, m, Sigma_prime, Sigma)

P_O4, mP_O4 = picard_group(O4)
N4 = order(P_O4)


N2 = order(Q)

N3 = order(P)

N = N1*N3/(N2*N4) - 1  # number of classes to find

println("N is ", N)

#= Diagnostics for Example 5
println("Norm of m: ", norm(m))
println("Discriminant of K: ", discriminant(OK))

# Check which small elements are coprime to m
B7 = basis(O7)
for i in -5:5, j in -5:5
    gamma = i * B7[1] + j * B7[2]
    iszero(gamma) && continue
    g = elem_in_nf(gamma)
    I = try ideal(O7, O7(g)) catch; continue end
    if isone(I + m)
        println("Coprime to m: ", gamma, "  norm=", norm(gamma))
    end
end

println("Unit reps mod m computation...")
U7, mU7 = unit_group(O7)
println("Unit group generators: ", [elem_in_nf(mU7(U7[i])) for i in 1:ngens(U7)])
=#

result = enumerate_principal_ray_classes(K, O4, m, Sigma, N)

for (I, gamma, h) in result
    println("Generator: ", gamma, "  Norm: ", h, "  Ideal: ", I)
end

# Example 11 conductor 5 m is 13
println("Example 11")
K, a = quadratic_field(5)
O5 = Order(K, [K(1), 5*(1+a)//2])         
OK = maximal_order(K)            

P, mP = picard_group(OK)
println("h(OK) for Q(sqrt(5): ", order(P))
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

m = ideal(O5, 13)
mp = ideal(OK, 1)
delta = ideal(OK, 65)

plcs = real_places(K)
Sigma = [plcs[1]]
Sigma_prime = InfPlc[]

println("Sigma is: ", Sigma)

Q, structure = residue_unit_quotient(OK, O5, mp, m, delta, Sigma_prime, Sigma)

N1 = prod(structure)

Q, reps, phi = ray_unit_quotient(OK, O5, mp, m, Sigma_prime, Sigma)

P_O5, mP_O5 = picard_group(O5)
N4 = order(P_O5)


N2 = order(Q)

N3 = order(P)

N = N1*N3/(N2*N4) - 1  # number of classes to find

println("N is ", N)

#= Diagnostics for Example 5
println("Norm of m: ", norm(m))
println("Discriminant of K: ", discriminant(OK))

# Check which small elements are coprime to m
B7 = basis(O7)
for i in -5:5, j in -5:5
    gamma = i * B7[1] + j * B7[2]
    iszero(gamma) && continue
    g = elem_in_nf(gamma)
    I = try ideal(O7, O7(g)) catch; continue end
    if isone(I + m)
        println("Coprime to m: ", gamma, "  norm=", norm(gamma))
    end
end

println("Unit reps mod m computation...")
U7, mU7 = unit_group(O7)
println("Unit group generators: ", [elem_in_nf(mU7(U7[i])) for i in 1:ngens(U7)])
=#

result = enumerate_principal_ray_classes(K, O5, m, Sigma, N)

for (I, gamma, h) in result
    println("Generator: ", gamma, "  Norm: ", h, "  Ideal: ", I)
end

# Example 12 conductor 4 m is 13
println("Example 12")
K, a = quadratic_field(5)
O4 = Order(K, [K(1), 2*a])          
OK = maximal_order(K)            

P, mP = picard_group(OK)
println("h(OK) for Q(sqrt(5): ", order(P))
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

m = ideal(O4, 13)
mp = ideal(OK, 1)
delta = ideal(OK, 52)

plcs = real_places(K)
Sigma = [plcs[1]]
Sigma_prime = InfPlc[]

println("Sigma is: ", Sigma)

Q, structure = residue_unit_quotient(OK, O4, mp, m, delta, Sigma_prime, Sigma)

N1 = prod(structure)

Q, reps, phi = ray_unit_quotient(OK, O4, mp, m, Sigma_prime, Sigma)

P_O4, mP_O4 = picard_group(O4)
N4 = order(P_O4)


N2 = order(Q)

N3 = order(P)

N = N1*N3/(N2*N4) - 1  # number of classes to find

println("N is ", N)

#= Diagnostics for Example 5
println("Norm of m: ", norm(m))
println("Discriminant of K: ", discriminant(OK))

# Check which small elements are coprime to m
B7 = basis(O7)
for i in -5:5, j in -5:5
    gamma = i * B7[1] + j * B7[2]
    iszero(gamma) && continue
    g = elem_in_nf(gamma)
    I = try ideal(O7, O7(g)) catch; continue end
    if isone(I + m)
        println("Coprime to m: ", gamma, "  norm=", norm(gamma))
    end
end

println("Unit reps mod m computation...")
U7, mU7 = unit_group(O7)
println("Unit group generators: ", [elem_in_nf(mU7(U7[i])) for i in 1:ngens(U7)])
=#

result = enumerate_principal_ray_classes(K, O4, m, Sigma, N)

for (I, gamma, h) in result
    println("Generator: ", gamma, "  Norm: ", h, "  Ideal: ", I)
end

# Example 13 conductor 5 m is 15
println("Example 13")
K, a = quadratic_field(5)
O5 = Order(K, [K(1), 5*(1+a)//2])         
OK = maximal_order(K)            

P, mP = picard_group(OK)
println("h(OK) for Q(sqrt(5): ", order(P))
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

m = ideal(O5, 15)
mp = ideal(OK, 1)
delta = ideal(OK, 75)

plcs = real_places(K)
Sigma = [plcs[1]]
Sigma_prime = InfPlc[]

println("Sigma is: ", Sigma)

Q, structure = residue_unit_quotient(OK, O5, mp, m, delta, Sigma_prime, Sigma)

N1 = prod(structure)

Q, reps, phi = ray_unit_quotient(OK, O5, mp, m, Sigma_prime, Sigma)

P_O5, mP_O5 = picard_group(O5)
N4 = order(P_O5)


N2 = order(Q)

N3 = order(P)

N = N1*N3/(N2*N4) - 1  # number of classes to find

println("N is ", N)

#= Diagnostics for Example 5
println("Norm of m: ", norm(m))
println("Discriminant of K: ", discriminant(OK))

# Check which small elements are coprime to m
B7 = basis(O7)
for i in -5:5, j in -5:5
    gamma = i * B7[1] + j * B7[2]
    iszero(gamma) && continue
    g = elem_in_nf(gamma)
    I = try ideal(O7, O7(g)) catch; continue end
    if isone(I + m)
        println("Coprime to m: ", gamma, "  norm=", norm(gamma))
    end
end

println("Unit reps mod m computation...")
U7, mU7 = unit_group(O7)
println("Unit group generators: ", [elem_in_nf(mU7(U7[i])) for i in 1:ngens(U7)])
=#

result = enumerate_principal_ray_classes(K, O5, m, Sigma, N)

for (I, gamma, h) in result
    println("Generator: ", gamma, "  Norm: ", h, "  Ideal: ", I)
end

# Example 14 conductor 4 m is 15
println("Example 14")
K, a = quadratic_field(5)
O4 = Order(K, [K(1), 2*a])          
OK = maximal_order(K)            

P, mP = picard_group(OK)
println("h(OK) for Q(sqrt(5): ", order(P))
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

m = ideal(O4, 15)
mp = ideal(OK, 1)
delta = ideal(OK, 60)

plcs = real_places(K)
Sigma = [plcs[1]]
Sigma_prime = InfPlc[]

println("Sigma is: ", Sigma)

Q, structure = residue_unit_quotient(OK, O4, mp, m, delta, Sigma_prime, Sigma)

N1 = prod(structure)

Q, reps, phi = ray_unit_quotient(OK, O4, mp, m, Sigma_prime, Sigma)

P_O4, mP_O4 = picard_group(O4)
N4 = order(P_O4)


N2 = order(Q)

N3 = order(P)

N = N1*N3/(N2*N4) - 1  # number of classes to find

println("N is ", N)

#= Diagnostics for Example 5
println("Norm of m: ", norm(m))
println("Discriminant of K: ", discriminant(OK))

# Check which small elements are coprime to m
B7 = basis(O7)
for i in -5:5, j in -5:5
    gamma = i * B7[1] + j * B7[2]
    iszero(gamma) && continue
    g = elem_in_nf(gamma)
    I = try ideal(O7, O7(g)) catch; continue end
    if isone(I + m)
        println("Coprime to m: ", gamma, "  norm=", norm(gamma))
    end
end

println("Unit reps mod m computation...")
U7, mU7 = unit_group(O7)
println("Unit group generators: ", [elem_in_nf(mU7(U7[i])) for i in 1:ngens(U7)])
=#

result = enumerate_principal_ray_classes(K, O4, m, Sigma, N)

for (I, gamma, h) in result
    println("Generator: ", gamma, "  Norm: ", h, "  Ideal: ", I)
end

