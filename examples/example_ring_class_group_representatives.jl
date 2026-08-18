# example_ring_class_group_representatives.jl
#
# Minimal worked example: Picard group (ring class group) of a non-maximal
# order in Q(sqrt(-5)), with explicit ideal representatives for each class.
#
# (Renamed from ring_class_grp_reps_example.jl.)

using Hecke

# Q(sqrt(-5)) has class number 2 for the maximal order.
# The non-maximal order of conductor 2 will have a larger Picard group.
Qx, x = polynomial_ring(QQ, :x)
K, a = number_field(x^2 + 5, :a)

OK = maximal_order(K)

# Build a non-maximal order of conductor 2: Z-basis [1, 2a]
O = Order(K, [OK(1), OK(2*a)])

# Compute the Picard group
P, mP = picard_group(O)
println("Picard group structure: ", P)
println("Number of classes: ", order(P))

# Enumerate all group elements and their ideal representatives
println()
for g in P
    I = mP(g)                      # ideal representative for this class
    b = basis(I)                   # Z-basis of I (always available)
    println("Class: ", g)
    println("  Ideal: ", I)
    println("  Z-basis (generators): ", b)
    println("  Norm: ", norm(I))
    println()
end