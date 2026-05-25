using Pkg
Pkg.activate("C:\\Users\\jddic\\OneDrive\\Desktop\\Zauner.jl")
using Zauner
include("ray_unit_quotient.jl")
include("term2.jl")
include("enumeration_with_infinite_places.jl")
include("make_coprime_reps.jl")

D = 24
K, a = quadratic_field(D)
OK = maximal_order(K)
q = 2
O = order(K, [K(1), K(q * OK(a))])
F = AdmissibleTuple(11, binary_quadratic_form(3, -12, 4))
d = Int(F.d)
m = ideal(O, d)
Sigma = real_places(K)

all_reps, all_triples = ray_class_group_representatives(K, O, m, Sigma)
J, I, gamma = all_triples[1]

@show J
@show basis(J)
@show norm(J)
@show J == ideal(O, 1)