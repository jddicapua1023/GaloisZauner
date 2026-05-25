using Hecke
include("ray_unit_quotient.jl")
include("term2.jl")

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
