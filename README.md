# Project layout

This project constructs explicit representatives for ray class groups of
maximal (and non-maximal) orders, and uses them to describe the Galois
action on ghost SICs. It's organized into two functional components, plus
tests and examples for each.

```
01_ray_class_representatives/   Component 1: explicit ray class group reps
02_galois_action_ghost_sics/    Component 2: Galois action on ghost SICs
tests/                          Worked-example test scripts for both components
examples/                       Small standalone demo scripts
archive/                        Superseded/broken files, kept for reference
```

Every file below was checked line-for-line against the corresponding
original file(s): all code content is unchanged except (a) `using`/`include`
lines updated for the new file layout, (b) dead/commented-out draft code
removed where noted, and (c) some function/file renames for clarity. No
algorithm, computation, or logic was altered.

## 01_ray_class_representatives/

- **unit_group_quotients.jl** - `ray_unit_quotient`, `residue_unit_quotient`.
  (Merged from `ray_unit_quotient.jl` + `term2.jl`.)
- **ray_class_enumeration.jl** - `ray_class_group_order`,
  `enumerate_principal_ray_classes`. (From `enumeration_with_infinite_places.jl`,
  which had accumulated 8 dated draft implementations of
  `enumerate_principal_ray_classes` as block comments during development;
  only the one live/working version is kept here. See "What was removed"
  below.) Also carries the standalone, reusable `ray_class_unit_reps` /
  `ray_class_same_mod_m` / `ray_class_same_sign_on_Sigma` /
  `ray_class_are_equivalent` / `ray_class_is_trivial_class` functions from
  the same source file - these were missed in an earlier pass of this
  reorganization even though `ray_class_unit_reps` is a live dependency of
  `02_galois_action_ghost_sics/galois_action_ghost_sics.jl`'s
  `ray_class_group_elementary_divisors`; added back here.
- **coprime_representatives.jl** - `make_coprime_representatives`.
  (Renamed from `make_coprime_reps.jl`.)
- **ray_class_group.jl** - `num_principal_ray_classes`,
  `ray_class_group_representatives`. (Renamed from `full_rayclassgrp.jl`.)
  NOTE: `02_galois_action_ghost_sics/galois_action_ghost_sics.jl` does NOT
  include this file - it defines its own, more robust local versions of
  both functions instead (see that file's header). This file's versions
  are used only by `tests/test_ray_class_group.jl` and anything else that
  includes it directly.

## 02_galois_action_ghost_sics/

- **galois_action_ghost_sics.jl** - the current/canonical version, as of
  this session (previously `galois_action_naive_search_OLD.jl`, see
  "Changelog" below for why it changed). Computes the twisted ghost
  character/overlaps (`twisted_chi_general`) for each ray class
  representative; entry point `test_all_twists(d)`. Uses a two-stage joint
  search (`find_alpha_rhoB_s1_s2_O`) plus disc-log-free Picard-class
  verification for selecting `b_O`. (Renamed from
  `galois_twist_vector_lattice_search.jl`.)
- **archive/galois_action_naive_search_OLD.jl** - the PREVIOUS canonical
  version: `find_alpha_rhoB_s1_s2_O` here uses three separate bounded
  searches instead of the joint two-stage approach, and its `b_O` selection
  (`mP_O(-u_B)`, trusted without independent verification) was found in
  this session to sometimes deliver a representative of the wrong Picard
  class. Superseded as of this session. (Originally `galois_twist_naive_search.jl`.)
- **archive/galois_action_lattice_based_OLD.jl** - superseded intermediate
  version using explicit 2D lattice reduction instead of the current
  brute-force search. (Originally `galois_twist.jl`.)
- **archive/galois_action_early_draft_OLD.jl** - earlier, simpler draft
  predating the order-refinement work. (Originally `galois_twist2.jl`.)

These archived files are kept exactly as uploaded (not rewired to the new
include paths) since they're historical reference only and aren't part of
the maintained code path.

## tests/

- **test_unit_group_quotients.jl** - merged from `tests_ray_unit_quotient.jl`
  + `teststerm2.jl`.
- **test_ray_class_enumeration.jl** - merged from
  `testsenumeration_sigma_I1_.jl`, `testsenumeration_sigma_I2_.jl`, and
  `testsenumeration2.jl` (three sections, each kept verbatim - see file
  header). `testsenumeration.jl` was left out (see archive/ below).
- **test_coprime_representatives.jl** - renamed from `coprime_reps_test.jl`.
- **test_ray_class_group.jl** - renamed from `full_rayclassgrp_tests.jl`.

## examples/

- **example_picard_group_principality.jl** - from `checkingN.jl`.
- **example_ring_class_group_representatives.jl** - from
  `ring_class_grp_reps_example.jl`.
- **example_ray_class_groups_and_fields.jl** - from
  `examples_rayclassgroupsrayclassfields.jl`.

## archive/

- **testsenumeration_OLD_broken_include.jl** - from `testsenumeration.jl`,
  which does `include("enumeration.jl")`, a file that doesn't exist in the
  project (likely renamed to `enumeration_with_infinite_places.jl` at some
  point). Doesn't run as uploaded; superseded by
  `tests/test_ray_class_enumeration.jl`.

## What was removed (dead code only, per your confirmation)

`enumeration_with_infinite_places.jl` contained 9 definitions of
`enumerate_principal_ray_classes(K, ord, m, Sigma, N)`. On closer look, 8 of
these were already wrapped in `#= ... =#` block comments (dated drafts like
"5/3 using lists L and L' for infinite places", "revised version 4/29
attempt to fix congruence checks", "Hybrid version 1", etc.) - only one
(the one now in `ray_class_enumeration.jl`) was ever live code. These 8
inert drafts, plus some unrelated commented-out scratch/example code near
the top of the same file, were dropped from the reorganized version. The
original upload is untouched if you ever want to look back at that history.

`full_rayclassgrp.jl` similarly had one block-commented draft of
`ray_class_group_representatives` after the live one; dropped for the same
reason.

## Changelog

**This session:**

- **Component 2's canonical source changed** from `galois_twist_naive_search.jl`
  to `galois_twist_vector_lattice_search.jl` (per project owner). The old
  canonical file is now `archive/galois_action_naive_search_OLD.jl`.
- **Bug fix, ported in from the new canonical source:** in
  `find_alpha_rhoB_s1_s2_O`, the representative `b_O` of Picard class
  `[J]^{-1}` was previously selected as `mP_O(-u_B)` (or a coprime variant
  thereof) and trusted based on Hecke's `picard_group` discrete log alone.
  Independent exact-arithmetic verification (K=Q(√44), conductor-3 order,
  d=21) showed this delivered a representative of class `[J]` instead of
  `[J]^{-1}` for at least one case - since the function's output lattice is
  always in `b_O`'s class, this silently corrupted the resulting ray class
  representative by the discrepancy between `[J]` and `[J]^{-1}` (the
  order-2 element of `Pic(O) = Z/4` there). Fixed by selecting `b_O` via
  disc-log-free verification: a generator of `J * b_O` is directly exhibited
  (maximal-order `is_principal_with_data` + a bounded unit twist + HNF
  lattice-equality, no `picard_group`/`inv()` dependence), and the full
  output criterion is independently re-checked before returning. New helper
  functions in `galois_action_ghost_sics.jl`: `is_principal_O_verified`,
  `find_b_with_verified_principal_product`, `lattice_canonical`,
  `lattices_equal`, `lattice_product_gens`, `quad_conj`, `_nf_elem_of`. A
  diagnostic print on every run compares the verified-correct class against
  Hecke's `picard_group` discrete log claim, to surface directly whether
  that discrete log is inverted for a given order (if so, other
  `picard_group`-dependent code - e.g. `ray_class_group.jl`'s functions -
  may need the same treatment; not yet investigated).
- Added `ray_class_unit_reps` and its four helper functions to
  `01_ray_class_representatives/ray_class_enumeration.jl` (see that
  section above) - present in the original source file but missed in the
  prior reorganization pass.
- Added `archive/testsenumeration_OLD_broken_include.jl` at the top level
  (referenced by the "archive/" section below but not actually present in
  the previous pass).

## Known pre-existing issues (not touched, flagged for you)

- `ray_class_group.jl` and `test_coprime_representatives.jl` both call
  `make_coprime_representatives(O, m, mP_O, group_elems)` with 4 arguments,
  but `coprime_representatives.jl` defines it requiring a 5th argument `d`
  (no default). These calls will error as written in the original code -
  this is not something the reorganization introduced.
