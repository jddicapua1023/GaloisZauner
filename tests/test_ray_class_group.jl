# test_ray_class_group.jl
#
# Worked examples/manual tests for num_principal_ray_classes and
# ray_class_group_representatives (ray_class_group.jl), including the two
# examples from the Kopp-Lagarias paper.
#
# (Renamed from full_rayclassgrp_tests.jl.)

using Hecke
include("../01_ray_class_representatives/ray_class_group.jl")





# Example 1
K, a = quadratic_field(5)
O2 = equation_order(K)
OK = maximal_order(K)
m = ideal(O2, 3)
plcs = real_places(K)
Sigma = [plcs[2]]

N = num_principal_ray_classes(OK, O2, m, Sigma)
println("Number of principal ray classes: ", N)


# Example 2
println("Example 2")
K, a = quadratic_field(5)
O2 = equation_order(K)
OK = maximal_order(K)
m = ideal(O2, 3)
plcs = real_places(K)
Sigma = [plcs[2]]

reps, pairs = ray_class_group_representatives(K, O2, m, Sigma)
println("\nRay class group representatives:")
for r in reps
    println("  ", r)
end
println("\nAs (Picard rep, principal ray class rep) pairs:")
for (J, I) in pairs
    println("  Picard: ", J, "  Principal: ", I)
end
println("Total number of classes: ", length(reps))

# Example 3
println("Example 3")
# Q(sqrt(10)), maximal order, m = 3
K, a = quadratic_field(10)
OK = maximal_order(K)
m = ideal(OK, 3)
plcs = real_places(K)
Sigma = [plcs[1]]  # one real place

reps, pairs = ray_class_group_representatives(K, OK, m, Sigma)
println("\nRay class group representatives:")
for r in reps
    println("  ", r)
end
println("\nAs (Picard rep, principal ray class rep) pairs:")
for (J, I) in pairs
    println("  Picard: ", J, "  Principal: ", I)
end
println("Total number of classes: ", length(reps))

# Example 4 - example 1 from Kopp Lagarias paper
println("Example 4")
# Q(sqrt(2))
K, a = quadratic_field(2)
OK = maximal_order(K)
m = ideal(OK, 7)
plcs = real_places(K)
Sigma = [plcs[2]]  # one real place

reps, pairs = ray_class_group_representatives(K, OK, m, Sigma)
println("\nRay class group representatives:")
for r in reps
    println("  ", r)
end
println("\nAs (Picard rep, principal ray class rep) pairs:")
for (J, I) in pairs
    println("  Picard: ", J, "  Principal: ", I)
end
println("Total number of classes: ", length(reps))

# Example 5 - example 2 from Kopp Lagarias
println("Example 5")
K, a = quadratic_field(2)
O2 = Order(K, [K(1), 2*a])
m = ideal(O2, 7)
plcs = real_places(K)
Sigma = [plcs[2]]  # one real place

reps, pairs = ray_class_group_representatives(K, O2, m, Sigma)
println("\nRay class group representatives:")
for r in reps
    println("  ", r)
end
println("\nAs (Picard rep, principal ray class rep) pairs:")
for (J, I) in pairs
    println("  Picard: ", J, "  Principal: ", I)
end
println("Total number of classes: ", length(reps))

# Example 6 - example 4 from Kopp Lagarias
println("Example 6")
K, a = quadratic_field(2)
O5 = Order(K, [K(1), 5*a])
m = ideal(O5, 7)
plcs = real_places(K)
Sigma = [plcs[2]]  # one real place

reps, pairs = ray_class_group_representatives(K, O5, m, Sigma)
println("\nRay class group representatives:")
for r in reps
    println("  ", r)
end
println("\nAs (Picard rep, principal ray class rep) pairs:")
for (J, I) in pairs
    println("  Picard: ", J, "  Principal: ", I)
end
println("Total number of classes: ", length(reps))