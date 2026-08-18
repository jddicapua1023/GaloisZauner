# test_unit_group_quotients.jl
#
# Worked examples/manual tests for ray_unit_quotient and
# residue_unit_quotient (unit_group_quotients.jl).
#
# (Merged from tests_ray_unit_quotient.jl and teststerm2.jl.)

using Hecke
include("../01_ray_class_representatives/unit_group_quotients.jl")

# ============================================================
# Part 1: ray_unit_quotient examples (from tests_ray_unit_quotient.jl)
# ============================================================



# Example usage:
K, a = quadratic_field(5)
O  = equation_order(K)           # Z[√5], conductor 2
OK = maximal_order(K)            # Z[(1+√5)/2]

m  = ideal(O,  3)                # ideal (3) in O
mp = ideal(OK, 3)                # ideal (3) in OK  (m' = (3)OK ⊇ m·OK)

plcs = real_places(K)
Sigma       = plcs               # both real places
Sigma_prime = InfPlc[]           # empty (no sign conditions for upper group)

Q, reps, phi = ray_unit_quotient(OK, O, mp, m, Sigma_prime, Sigma)
# Q gives the abstract group, e.g. Z/2 x Z/4
# reps gives explicit unit representatives, one per coset
# phi maps any unit in U_{m',Σ'}(OK) to its coset in Q

println(elementary_divisors(Q))
println(reps)

# the code below checks the order of the above computation is correct
#=
let
exponen = 1
x = (1+a)//2
while mod(2*coeff(x^exponen,1),2) != 0 || (sign(x^exponen,plcs[1])<0 || sign(x^exponen,plcs[2])<0 )
    if mod(2*coeff(x^exponen,1),1) == 0
        println("exponent and coefficients are: ", exponen, ", ", coeff(x^exponen,0), ", ", coeff(x^exponen,1))
    end
    exponen += 1
end 
println("exponent is: ", exponen)
end
=#



# Example usage:
K, a = quadratic_field(5)
O2  = equation_order(K)           # Z[√5], conductor 2
O4 = Order(K, [K(1), 2*a])            # Z[(1+√5)/2]

m  = ideal(O4,  3)                # ideal (3) in O
mp = ideal(O2, 3)                # ideal (3) in OK  (m' = (3)OK ⊇ m·OK)

plcs = real_places(K)
Sigma       = plcs               # both real places
Sigma_prime = InfPlc[]           # empty (no sign conditions for upper group)

Q, reps, phi = ray_unit_quotient(O2, O4, mp, m, Sigma_prime, Sigma)
# Q gives the abstract group, e.g. Z/2 x Z/4
# reps gives explicit unit representatives, one per coset
# phi maps any unit in U_{m',Σ'}(OK) to its coset in Q


println("second example:", elementary_divisors(Q))
println(reps)
println("for m = 3, f = 2,4, and d=5")

# Example usage:
K, a = quadratic_field(5)
O2  = equation_order(K)           # Z[√5], conductor 2
O4 = Order(K, [K(1), 2*a])            # Z[(1+√5)/2]

m  = ideal(O4,  7)                # ideal (3) in O
mp = ideal(O2, 7)                # ideal (3) in OK  (m' = (3)OK ⊇ m·OK)

plcs = real_places(K)
Sigma       = plcs               # both real places
Sigma_prime = InfPlc[]           # empty (no sign conditions for upper group)

Q, reps, phi = ray_unit_quotient(O2, O4, mp, m, Sigma_prime, Sigma)
# Q gives the abstract group, e.g. Z/2 x Z/4
# reps gives explicit unit representatives, one per coset
# phi maps any unit in U_{m',Σ'}(OK) to its coset in Q


println("third example:", elementary_divisors(Q))
println(reps)
println("for m = 7, f = 2,4, and d=5")

# Example usage:
K, a = quadratic_field(5)
O2  = equation_order(K)           # Z[√5], conductor 2
O4 = Order(K, [K(1), 2*a])            # Z[(1+√5)/2]

m  = ideal(O4,  7)                # ideal (3) in O
mp = ideal(O2, 7)                # ideal (3) in OK  (m' = (3)OK ⊇ m·OK)

plcs = real_places(K)
Sigma       = InfPlc[]              # both real places
Sigma_prime = InfPlc[]           # empty (no sign conditions for upper group)

Q, reps, phi = ray_unit_quotient(O2, O4, mp, m, Sigma_prime, Sigma)
# Q gives the abstract group, e.g. Z/2 x Z/4
# reps gives explicit unit representatives, one per coset
# phi maps any unit in U_{m',Σ'}(OK) to its coset in Q


println("fourth example:", elementary_divisors(Q))
println(reps)
println("for m = 7, f = 2,4, and d=5")

# Example usage:
K, a = quadratic_field(5)
O3 = Order(K, [K(1), 3*(1+a)//2])          # Z[√5], conductor 3
O6 = Order(K, [K(1), 3*a])            # Z[(1+√5)/2]

m  = ideal(O6,  2)                # ideal (3) in O
mp = ideal(O3, 2)                # ideal (3) in OK  (m' = (3)OK ⊇ m·OK)

plcs = real_places(K)
Sigma       = InfPlc[]              # both real places
Sigma_prime = InfPlc[]           # empty (no sign conditions for upper group)

Q, reps, phi = ray_unit_quotient(O3, O6, mp, m, Sigma_prime, Sigma)
# Q gives the abstract group, e.g. Z/2 x Z/4
# reps gives explicit unit representatives, one per coset
# phi maps any unit in U_{m',Σ'}(OK) to its coset in Q


println("fifth example:", elementary_divisors(Q))
println(reps)
println("for m = 2, f = 3,6, and d=5")

# Example usage:
K, a = quadratic_field(5)
O3 = Order(K, [K(1), 3*(1+a)//2])          # Z[√5], conductor 3
O6 = Order(K, [K(1), 3*a])            # Z[(1+√5)/2]

m  = ideal(O6,  3)                # ideal (3) in O
mp = ideal(O3, 3)                # ideal (3) in OK  (m' = (3)OK ⊇ m·OK)

plcs = real_places(K)
Sigma       = InfPlc[]              # both real places
Sigma_prime = InfPlc[]           # empty (no sign conditions for upper group)

Q, reps, phi = ray_unit_quotient(O3, O6, mp, m, Sigma_prime, Sigma)
# Q gives the abstract group, e.g. Z/2 x Z/4
# reps gives explicit unit representatives, one per coset
# phi maps any unit in U_{m',Σ'}(OK) to its coset in Q


println("sixth example:", elementary_divisors(Q))
println(reps)
println("for m = 3, f = 3,6, and d=5")

# Example usage:
K, a = quadratic_field(3)
O7 = Order(K, [K(1), 7*a])          # Z[√5], conductor 3
O49 = Order(K, [K(1), 14*a])            # Z[(1+√5)/2]

m  = ideal(O49,  5)                # ideal (3) in O
mp = ideal(O7, 5)                # ideal (3) in OK  (m' = (3)OK ⊇ m·OK)

plcs = real_places(K)
Sigma       = InfPlc[]              # both real places
Sigma_prime = InfPlc[]           # empty (no sign conditions for upper group)

Q, reps, phi = ray_unit_quotient(O7, O49, mp, m, Sigma_prime, Sigma)
# Q gives the abstract group, e.g. Z/2 x Z/4
# reps gives explicit unit representatives, one per coset
# phi maps any unit in U_{m',Σ'}(OK) to its coset in Q


println("seventh example:", elementary_divisors(Q))
println(reps)
println("for m = 5, f = 7, 14, and d=3")

# Example usage:
K, a = quadratic_field(3)
O7 = Order(K, [K(1), 7*a])          # Z[√5], conductor 3
O49 = Order(K, [K(1), 49*a])            # Z[(1+√5)/2]

m  = ideal(O49,  5)                # ideal (3) in O
mp = ideal(O7, 5)                # ideal (3) in OK  (m' = (3)OK ⊇ m·OK)

plcs = real_places(K)
Sigma       = InfPlc[]              # both real places
Sigma_prime = InfPlc[]           # empty (no sign conditions for upper group)

Q, reps, phi = ray_unit_quotient(O7, O49, mp, m, Sigma_prime, Sigma)
# Q gives the abstract group, e.g. Z/2 x Z/4
# reps gives explicit unit representatives, one per coset
# phi maps any unit in U_{m',Σ'}(OK) to its coset in Q


println("eigth example:", elementary_divisors(Q))
println(reps)
println("for m = 5, f = 7, 49, and d=3")
# the code below checks order of the above computation is correct
#=
let
exponen = 1
x = 2-a
while mod(coeff(x^exponen,1),49) != 0
    if mod(coeff(x^exponen,1),7) == 0
        println("exponent and coefficients are: ", exponen, ", ", coeff(x^exponen,0), ", ", coeff(x^exponen,1))
    end
    exponen += 1
end 
println("exponent is: ", exponen)
end
=#

# Example usage:
K, a = quadratic_field(3)
O7 = Order(K, [K(1), 7*a])          # Z[√5], conductor 3
O49 = Order(K, [K(1), 49*a])            # Z[(1+√5)/2]

m  = ideal(O49,  3)                # ideal (3) in O
mp = ideal(O7, 3)                # ideal (3) in OK  (m' = (3)OK ⊇ m·OK)

plcs = real_places(K)
Sigma       = InfPlc[]              # both real places
Sigma_prime = InfPlc[]           # empty (no sign conditions for upper group)

Q, reps, phi = ray_unit_quotient(O7, O49, mp, m, Sigma_prime, Sigma)
# Q gives the abstract group, e.g. Z/2 x Z/4
# reps gives explicit unit representatives, one per coset
# phi maps any unit in U_{m',Σ'}(OK) to its coset in Q


println("ninth example:", elementary_divisors(Q))
println(reps)
println("for m = 3, f = 7, 49, and d=3")

# the code below checks the order of the above computation is correct
#=
let
exponen = 1
x = 2-a
while mod(coeff(x^exponen,1),49) != 0
    if mod(coeff(x^exponen,1),7) == 0
        println("exponent and coefficients are: ", exponen, ", ", coeff(x^exponen,0), ", ", coeff(x^exponen,1))
    end
    exponen += 1
end 
println("exponent is: ", exponen)
end
=#

# Example usage:
K, a = quadratic_field(3)
O7 = Order(K, [K(1), 7*a])          # Z[√5], conductor 3
O14 = Order(K, [K(1), 14*a])            # Z[(1+√5)/2]

m  = ideal(O14,  3)                # ideal (3) in O
mp = ideal(O7, 3)                # ideal (3) in OK  (m' = (3)OK ⊇ m·OK)

plcs = real_places(K)
Sigma       = InfPlc[]              # both real places
Sigma_prime = InfPlc[]           # empty (no sign conditions for upper group)

Q, reps, phi = ray_unit_quotient(O7, O14, mp, m, Sigma_prime, Sigma)
# Q gives the abstract group, e.g. Z/2 x Z/4
# reps gives explicit unit representatives, one per coset
# phi maps any unit in U_{m',Σ'}(OK) to its coset in Q


println("tenth example:", elementary_divisors(Q))
println(reps)
println("for m = 3, f = 7, 14, and d=3")

# the code below checks the order of the above computation is correct
#=
let
exponen = 1
x = 2-a
while mod(coeff(x^exponen,1),14) != 0
    if mod(coeff(x^exponen,1),7) == 0
        println("exponent and coefficients are: ", exponen, ", ", coeff(x^exponen,0), ", ", coeff(x^exponen,1))
    end
    exponen += 1
end 
println("exponent is: ", exponen)
end
=#

#beginning tests where Sigma might be nonempty (Sigma has at least one infinite place)
# Example usage:
K, a = quadratic_field(5)
OK = maximal_order(K)            # Z[(1+√5)/2]
O2 = Order(K, [K(1), a])            # Z[(1+√5)/2]

m  = ideal(O2,  3)                # ideal (3) in O
mp = ideal(OK, 1)                # ideal (3) in OK  (m' = (3)OK ⊇ m·OK)

plcs = real_places(K)
Sigma       = [plcs[1]]              
Sigma_prime = InfPlc[]           # empty (no sign conditions for upper group)

Q, reps, phi = ray_unit_quotient(OK, O2, mp, m, Sigma_prime, Sigma)
# Q gives the abstract group, e.g. Z/2 x Z/4
# reps gives explicit unit representatives, one per coset
# phi maps any unit in U_{m',Σ'}(OK) to its coset in Q


println("eleventh example:", elementary_divisors(Q))
println(reps)
println("for m = 3, f = 1, 2, and d=5")

# the code below checks the order of the above computation is correct
#=
let
exponen = 1
x = 2-a
while mod(coeff(x^exponen,1),14) != 0
    if mod(coeff(x^exponen,1),7) == 0
        println("exponent and coefficients are: ", exponen, ", ", coeff(x^exponen,0), ", ", coeff(x^exponen,1))
    end
    exponen += 1
end 
println("exponent is: ", exponen)
end
=#
# ============================================================
# Part 2: residue_unit_quotient examples (from teststerm2.jl)
# ============================================================


K, a = quadratic_field(5)
O  = equation_order(K)       # conductor 2
OK = maximal_order(K)

m   = ideal(O,  3)
mp  = ideal(OK, 3)

# delta must satisfy delta ⊆ (m : OK)
# for this example take delta = m itself extended to OK
delta = ideal(OK, 3)

plcs        = real_places(K)
Sigma       = plcs       # both places
Sigma_prime = InfPlc[]   # empty

Q, structure = residue_unit_quotient(OK, O, mp, m, delta, Sigma_prime, Sigma)
println("Structure: ", structure)

# Example usage:
K, a = quadratic_field(5)
O2  = equation_order(K)           # Z[√5], conductor 2
O4 = Order(K, [K(1), 2*a])            # Z[(1+√5)/2]

m  = ideal(O4,  3)                # ideal (3) in O
mp = ideal(O2, 3)                # ideal (3) in OK  (m' = (3)OK ⊇ m·OK)

delta = ideal(O2, 3)

plcs = real_places(K)
Sigma       = plcs               # both real places
Sigma_prime = InfPlc[]           # empty (no sign conditions for upper group)

Q, structure = residue_unit_quotient(O2, O4, mp, m, delta, Sigma_prime, Sigma)
println("Structure: ", structure)

# Example usage:
K, a = quadratic_field(5)
O2  = equation_order(K)           # Z[√5], conductor 2
O4 = Order(K, [K(1), 2*a])            # Z[(1+√5)/2]

m  = ideal(O4,  4)                # ideal (3) in O
mp = ideal(O2, 4)                # ideal (3) in OK  (m' = (3)OK ⊇ m·OK)

delta = ideal(O2, 2)

plcs = real_places(K)
Sigma       = plcs               # both real places
Sigma_prime = InfPlc[]           # empty (no sign conditions for upper group)

Q, structure = residue_unit_quotient(O2, O4, mp, m, delta, Sigma_prime, Sigma)
println("Structure for example two: ", structure)

# Example usage:
K, a = quadratic_field(5)
O2  = equation_order(K)           # Z[√5], conductor 2
O4 = Order(K, [K(1), 2*a])            # Z[(1+√5)/2]

m  = ideal(O4,  7)                # ideal (3) in O
mp = ideal(O2, 7)                # ideal (3) in OK  (m' = (3)OK ⊇ m·OK)

delta = ideal(O2, 7)

plcs = real_places(K)
Sigma       = plcs               # both real places
Sigma_prime = InfPlc[]           # empty (no sign conditions for upper group)

Q, structure = residue_unit_quotient(O2, O4, mp, m, delta, Sigma_prime, Sigma)
println("Structure for example three: ", structure)

# Example usage:
K, a = quadratic_field(5)
O2  = equation_order(K)           # Z[√5], conductor 2
O4 = Order(K, [K(1), 2*a])            # Z[(1+√5)/2]

m  = ideal(O4,  7)                # ideal (3) in O
mp = ideal(O2, 7)                # ideal (3) in OK  (m' = (3)OK ⊇ m·OK)

plcs = real_places(K)
Sigma       = InfPlc[]              # both real places
Sigma_prime = InfPlc[]           # empty (no sign conditions for upper group)

delta = ideal(O2, 7)

Q, structure = residue_unit_quotient(O2, O4, mp, m, delta, Sigma_prime, Sigma)
println("Structure for example four: ", structure)

# Example usage:
K, a = quadratic_field(5)
O3 = Order(K, [K(1), 3*(1+a)//2])          # Z[√5], conductor 3
O6 = Order(K, [K(1), 3*a])            # Z[(1+√5)/2]

m  = ideal(O6,  2)                # ideal (3) in O
mp = ideal(O3, 2)                # ideal (3) in OK  (m' = (3)OK ⊇ m·OK)

plcs = real_places(K)
Sigma       = InfPlc[]              # both real places
Sigma_prime = InfPlc[]           # empty (no sign conditions for upper group)

delta = ideal(O3, 2)

Q, structure = residue_unit_quotient(O3, O6, mp, m, delta, Sigma_prime, Sigma)
println("Structure for example five: ", structure)

# Example usage:
K, a = quadratic_field(5)
O3 = Order(K, [K(1), 3*(1+a)//2])          # Z[√5], conductor 3
O6 = Order(K, [K(1), 3*a])            # Z[(1+√5)/2]

m  = ideal(O6,  3)                # ideal (3) in O
mp = ideal(O3, 3)                # ideal (3) in OK  (m' = (3)OK ⊇ m·OK)

plcs = real_places(K)
Sigma       = InfPlc[]              # both real places
Sigma_prime = InfPlc[]           # empty (no sign conditions for upper group)

delta = ideal(O3,3)

Q, structure = residue_unit_quotient(O3, O6, mp, m, delta, Sigma_prime, Sigma)
println("Structure for example six: ", structure)

K, a = quadratic_field(3)
O7 = Order(K, [K(1), 7*a])          # Z[√5], conductor 3
O49 = Order(K, [K(1), 14*a])            # Z[(1+√5)/2]

m  = ideal(O49,  5)                # ideal (3) in O
mp = ideal(O7, 5)                # ideal (3) in OK  (m' = (3)OK ⊇ m·OK)

plcs = real_places(K)
Sigma       = InfPlc[]              # both real places
Sigma_prime = InfPlc[]           # empty (no sign conditions for upper group)

delta = ideal(O7, 5)

Q, structure = residue_unit_quotient(O7, O49, mp, m, delta, Sigma_prime, Sigma)
println("Structure for example seven: ", structure)

# Example usage:
K, a = quadratic_field(3)
O7 = Order(K, [K(1), 7*a])          # Z[√5], conductor 3
O49 = Order(K, [K(1), 49*a])            # Z[(1+√5)/2]

m  = ideal(O49,  5)                # ideal (3) in O
mp = ideal(O7, 5)                # ideal (3) in OK  (m' = (3)OK ⊇ m·OK)

plcs = real_places(K)
Sigma       = InfPlc[]              # both real places
Sigma_prime = InfPlc[]           # empty (no sign conditions for upper group)

delta = ideal(O7, 35)

Q, structure = residue_unit_quotient(O7, O49, mp, m, delta, Sigma_prime, Sigma)
println("Structure for example eight: ", structure)

# Example usage:
K, a = quadratic_field(3)
O7 = Order(K, [K(1), 7*a])          # Z[√5], conductor 3
O49 = Order(K, [K(1), 49*a])            # Z[(1+√5)/2]

m  = ideal(O49,  3)                # ideal (3) in O
mp = ideal(O7, 3)                # ideal (3) in OK  (m' = (3)OK ⊇ m·OK)

plcs = real_places(K)
Sigma       = InfPlc[]              # both real places
Sigma_prime = InfPlc[]           # empty (no sign conditions for upper group)

delta = ideal(O7, 21)

Q, structure = residue_unit_quotient(O7, O49, mp, m, delta, Sigma_prime, Sigma)
println("Structure for example nine: ", structure)

# Example usage:
K, a = quadratic_field(3)
O7 = Order(K, [K(1), 7*a])          # Z[√5], conductor 3
O14 = Order(K, [K(1), 14*a])            # Z[(1+√5)/2]

m  = ideal(O14,  3)                # ideal (3) in O
mp = ideal(O7, 3)                # ideal (3) in OK  (m' = (3)OK ⊇ m·OK)

plcs = real_places(K)
Sigma       = InfPlc[]              # both real places
Sigma_prime = InfPlc[]           # empty (no sign conditions for upper group)

delta = ideal(O7, 6)

Q, structure = residue_unit_quotient(O7, O14, mp, m, delta, Sigma_prime, Sigma)
println("Structure for example ten: ", structure)
