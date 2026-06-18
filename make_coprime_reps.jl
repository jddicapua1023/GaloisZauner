using Hecke

function make_coprime_representatives(O, m, mP, group_elems, d)
    K = number_field(O)
    OK = maximal_order(K)
    P_O = domain(mP)
    trivial_elem = zero(P_O)

    n_needed = length(group_elems)

    # basis of O as number field elements
    basis_O = basis(O)
    b1 = elem_in_nf(basis_O[1])
    b2 = elem_in_nf(basis_O[2])

    # need coprimality with both m and ideal(O, d+1)
    m_d1 = ideal(O, d + 1)

    println("  make_coprime: searching for ", n_needed, " Picard reps coprime to m and d+1=", d+1)

    found_classes = Dict{FinGenAbGroupElem, Any}()

    max_height = 100
    for height in 1:max_height
        length(found_classes) == n_needed && break

        for c1 in -height:height
            length(found_classes) == n_needed && break
            for c2 in -height:height
                length(found_classes) == n_needed && break
                for d1 in -height:height
                    length(found_classes) == n_needed && break
                    for d2 in -height:height
                        length(found_classes) == n_needed && break

                        # skip if not on boundary
                        (abs(c1) != height && abs(c2) != height &&
                         abs(d1) != height && abs(d2) != height) && continue

                        alpha1 = c1 * b1 + c2 * b2
                        alpha2 = d1 * b1 + d2 * b2

                        iszero(alpha1) && iszero(alpha2) && continue

                        try
                            gens = AbsSimpleNumFieldOrderElem[]
                            if !iszero(alpha1)
                                push!(gens, O(alpha1))
                            end
                            if !iszero(alpha2)
                                push!(gens, O(alpha2))
                            end
                            isempty(gens) && continue

                            I = ideal(O, gens)

                            # check coprimality with m and d+1
                            if !isone(I + m)
                                continue
                            end
                            if !isone(I + m_d1)
                                continue
                            end

                            # find Picard class
                            class = mP \ I

                            if class ∉ keys(found_classes)
                                found_classes[class] = I
                                println("  make_coprime: found class ", class,
                                        " with ideal ", I,
                                        " from alpha1=", alpha1, " alpha2=", alpha2)
                            end
                        catch
                            continue
                        end
                    end
                end
            end
        end
    end

    if length(found_classes) < n_needed
        error("Could not find coprime representatives for all Picard classes within height bound")
    end

    # return reps in the same order as group_elems
    result = []
    for g in group_elems
        push!(result, found_classes[g])
    end

    return result
end