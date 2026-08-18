# example_picard_group_principality.jl
#
# For the order O associated to AdmissibleTuple(15) (D, q from the ghost-SIC
# admissible tuple), computes the Picard group of O and checks, for each
# class, whether the corresponding ideal becomes principal after extension
# to the maximal order OK.
#
# (Renamed from checkingN.jl.)

using Hecke
using Zauner

F = AdmissibleTuple(15)
D = Int(F.D)
q = Int(F.q)

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

Dr = squarefree_part(D)
K, a = quadratic_field(D)
OK = maximal_order(K)

sqrt_Dr = a // isqrt(D ÷ Dr)

if mod(Dr, 4) == 1
    omega = (K(1) + sqrt_Dr) // 2
else
    omega = sqrt_Dr
end

O = Order(K, [one(K), q * omega])
println("O = ", O)
println("conductor of O: ", conductor(O))
println("D = ", D, ", q = ", q)

P_O, mP_O = picard_group(O)
println("Picard group order: ", order(P_O))
println("elementary_divisors: ", elementary_divisors(P_O))

for g in gens(P_O)
    I = mP_O(g)
    println("generator maps to ideal: ", I)
    println("order of element: ", order(g))
    println("is_principal in OK? ", is_principal(extend(I, OK)))
end

# also check the zero element
I0 = mP_O(zero(P_O))
println("zero element maps to: ", I0)

# check all elements
println("\nAll Picard group elements:")
for g in P_O
    I = mP_O(g)
    println("  g=", g, " -> I=", I, " principal in OK? ", is_principal(extend(I, OK)))
end