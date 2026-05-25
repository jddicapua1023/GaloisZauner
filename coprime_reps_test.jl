using Hecke
include("ray_unit_quotient.jl")
include("term2.jl")
include("enumeration_with_infinite_places.jl")
include("make_coprime_reps.jl")

# Example 1: Q(sqrt(5)), non-maximal order, m = 3
println("Example 1")
K, a = quadratic_field(5)
O2 = equation_order(K)
m = ideal(O2, 3)

P, mP = picard_group(O2)
gens_P = gens(P)
divs = elementary_divisors(P)
ranges = [0:Int(d)-1 for d in divs]

group_elems = []
for exps in Iterators.product(ranges...)
    g = isempty(gens_P) ? zero(P) : sum(exps[i] * gens_P[i] for i in 1:length(gens_P))
    push!(group_elems, g)
end

reps = make_coprime_representatives(O2, m, mP, group_elems)
println("After correction:")
for r in reps
    println("  rep: ", r)
end

# Example 2: Q(sqrt(-5)), maximal order, m = 2
println("\nExample 2")
K, a = quadratic_field(-5)
OK = maximal_order(K)
m = ideal(OK, 2)

P, mP = picard_group(OK)
println("Class number: ", order(P))
println("Elementary divisors: ", elementary_divisors(P))

gens_P = gens(P)
divs = elementary_divisors(P)
ranges = [0:Int(d)-1 for d in divs]

group_elems = []
for exps in Iterators.product(ranges...)
    g = isempty(gens_P) ? zero(P) : sum(exps[i] * gens_P[i] for i in 1:length(gens_P))
    push!(group_elems, g)
end

println("Before correction:")
for g in group_elems
    I = mP(g)
    println("  rep: ", I)
end

reps = make_coprime_representatives(OK, m, mP, group_elems)
println("After correction:")
for r in reps
    println("  rep: ", r)
end

# Diagnostic 1.1
println("Diagnostic 1.1")
K, a = quadratic_field(5)
O2 = equation_order(K)
OK = maximal_order(K)

println(conductor(O2))
println(typeof(conductor(O2)))
println(minimum(ideal(O2, 3)))

# Diagnostic 1.2
println("Diagnostic 1.2")
K, a = quadratic_field(5)
O2 = equation_order(K)
OK = maximal_order(K)

f = conductor(O2)
println("conductor: ", f)
println("minimum(f): ", minimum(f))
println("norm(f): ", norm(f))

# Then try forming delta
n = minimum(ideal(O2, 3))
delta = ideal(OK, minimum(f) * n)
println("delta: ", delta)

# Diagnostic 1.3
println("Diagnostic 1.3")
K, a = quadratic_field(-5)
O2 = equation_order(K)
OK = maximal_order(K)

I_OK = ideal(OK, 3)  # ideal in OK
I_O2 = ideal(O2, 3)  # ideal in O2

# Try multiplying
try
    J = I_OK * I_O2
    println("product: ", J)
    println("order: ", order(J))
catch e
    println("Error: ", e)
end

#=
K, a = quadratic_field(5)
O2 = equation_order(K)
m = ideal(O2, 3)

P, mP = picard_group(O2)
gens_P = gens(P)
divs = elementary_divisors(P)
ranges = [0:Int(d)-1 for d in divs]

group_elems = []
for exps in Iterators.product(ranges...)
    g = isempty(gens_P) ? zero(P) : sum(exps[i] * gens_P[i] for i in 1:length(gens_P))
    push!(group_elems, g)
end

reps = make_coprime_representatives(O2, m, mP, group_elems)
# Check both ways

OK = maximal_order(number_field(O2))
m_OK = OK * m

for r in reps
    println("rep: ", r, 
            "  coprime to m_OK: ", isone(r + m_OK),
            "  coprime to m: ", isone(r + m))
end



# Q(sqrt(-5)) has class number 2, so picard_group(OK) will be nontrivial
println("Example 2")
K, a = quadratic_field(-5)
OK = maximal_order(K)
m = ideal(OK, 2)  # 2 ramifies in Q(sqrt(-5)), so representatives may not be coprime to m

P, mP = picard_group(OK)
println("Class number: ", order(P))
println("Elementary divisors: ", elementary_divisors(P))

gens_P = gens(P)
divs = elementary_divisors(P)
ranges = [0:Int(d)-1 for d in divs]

group_elems = []
for exps in Iterators.product(ranges...)
    g = isempty(gens_P) ? zero(P) : sum(exps[i] * gens_P[i] for i in 1:length(gens_P))
    push!(group_elems, g)
end

println("\nBefore correction:")
for g in group_elems
    I = mP(g)
    println("  rep: ", I, "  coprime to m: ", isone(I + m))
end


reps = make_coprime_representatives(OK, m, mP, group_elems)
println("\nAfter correction:")
# Check both ways

OK = maximal_order(number_field(O2))
m_OK = OK * m

for r in reps
    println("rep: ", r, 
            "  coprime to m_OK: ", isone(r + m_OK),
            "  coprime to m: ", isone(r + m))
end
=#

#=diagnostic 1
println("Diagnostic 1")
K, a = quadratic_field(-5)
OK = maximal_order(K)
m_OK = ideal(OK, 2)
fac = factor(m_OK)
p = first(keys(fac))

I_OK = ideal(OK, 2)  # just a test ideal
pi_p = elem_in_nf(uniformizer(p))
J_OK = pi_p^(-1) * I_OK  # this will be fractional

println("J_OK: ", J_OK)
println("typeof: ", typeof(J_OK))

# Try these conversions:
println(numerator(J_OK))
println(typeof(numerator(J_OK)))

# diagnostic 2
println("Diagnostic 2")
K, a = quadratic_field(-5)
OK = maximal_order(K)
O2 = equation_order(K)

m_OK = ideal(OK, 2)
fac = factor(m_OK)
p = first(keys(fac))

I_OK = ideal(OK, 2)
pi_p = elem_in_nf(uniformizer(p))
J_OK = pi_p^(-1) * I_OK
J_num = numerator(J_OK)

println("J_num: ", J_num)
println("order(J_num): ", order(J_num))

# Try contracting back to O2
J_O2 = ideal(O2, basis_matrix(J_num))
println("J_O2: ", J_O2)
println("typeof(J_O2): ", typeof(J_O2))
println("order(J_O2): ", order(J_O2))

#= diagnostic 3
println("Diagnostic 3")
K, a = quadratic_field(-5)
OK = maximal_order(K)
O2 = equation_order(K)

P, mP = picard_group(O2)
gens_P = gens(P)
g = gens_P[1]
I = mP(g)  # the nontrivial class rep in O2
println("I: ", I, "  order(I): ", order(I))

m_OK = OK * ideal(O2, 2)
fac = factor(m_OK)
p = first(keys(fac))
pi_p = elem_in_nf(uniformizer(p))
alpha = pi_p^(-valuation(OK * I, p))

println("alpha: ", alpha)

# Try forming the ideal of O2 generated by alpha
alpha_O2 = O2(alpha)
println("alpha_O2: ", alpha_O2)
J = alpha_O2 * I
println("J: ", J, "  order(J): ", order(J))
=#

# diagnostic 4
println("Diagnostic 4")
K, a = quadratic_field(-5)
OK = maximal_order(K)
O2 = equation_order(K)

P, mP = picard_group(O2)
g = gens(P)[1]
I = mP(g)
println("I: ", I, " order(I): ", order(I))

m_OK = OK * ideal(O2, 2)
fac = factor(m_OK)
p = first(keys(fac))
pi_p = elem_in_nf(uniformizer(p))
alpha = pi_p^(-valuation(I, p))

J_frac = alpha * I
println("J_frac: ", J_frac)
println("typeof: ", typeof(J_frac))

J = numerator(J_frac)
println("J: ", J, "  order(J): ", order(J))
println("coprime to m_OK: ", isone(J + m_OK))

# diagnostic 5
println("Diagnostic 5")
J_frac = alpha * I
println("J_frac: ", J_frac)

# Try simplifying / reducing
J = numerator(J_frac)
println("J before simplify: ", J)
println("norm(J): ", norm(J))

d = denominator(J_frac)
println("denominator: ", d)

# Try dividing out the denominator from the norm
J2 = J_frac * ideal(OK, d)  # multiply by (d) to clear
println("J2: ", J2)

# Or try simplify/reduce if it exists
println(methods(simplify))

#diagnostic 6
println("Diagnostic 6")
J_frac = alpha * I
J_simplified = simplify(J_frac)
println("J_simplified: ", J_simplified)
println("typeof: ", typeof(J_simplified))

J = numerator(J_simplified)
println("J: ", J, "  order(J): ", order(J))
println("coprime to m_OK: ", isone(J + m_OK))
=#

