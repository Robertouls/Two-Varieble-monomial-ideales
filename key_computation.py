"""
Key computation: Demazure root structure of monomial algebras.

CORRECTED FINDING (October 2026):
  R_inn(m^d) = 0 in ALL dimensions, not just dimension 2.
  The structural difference between dim 2 and dim >= 3 for the Jacobian
  Conjecture comes from the Jung-van der Kulk theorem (all automorphisms
  of k[x,y] are tame), NOT from inner root structure of m^d.

  However, inner roots DO appear for GENERAL monomial ideals (e.g. (x^3,y^2))
  even in dimension 2. This distinction matters for the broader theory.
"""


def outer_roots(d: int, n: int) -> list:
    """Outer Demazure roots of m^d in k[x_1,...,x_n]."""
    roots = []
    for j in range(n):
        for beta_tuple in _nonneg_tuples(n, d - 1):
            if beta_tuple[j] == 0:
                alpha = list(beta_tuple)
                alpha[j] = -1
                roots.append(tuple(alpha))
    return roots


def inner_roots(d: int, n: int) -> list:
    """
    Inner Demazure roots of m^d in k[x_1,...,x_n].

    alpha is an inner root if for EXACTLY ONE j in {0,...,n-1}:
      alpha + e_j >= 0  AND  |alpha + e_j| < d
    AND alpha is not an outer root.
    """
    outer_set = set(outer_roots(d, n))
    roots = []

    for alpha_tuple in _all_tuples(n, -1, d - 1):
        valid_j = []
        for j in range(n):
            shifted = list(alpha_tuple)
            shifted[j] += 1
            if all(s >= 0 for s in shifted) and sum(shifted) < d:
                valid_j.append(j)

        if len(valid_j) == 1 and alpha_tuple not in outer_set:
            roots.append(alpha_tuple)

    return roots


def inner_roots_general(generators: list, n: int) -> list:
    """
    Inner Demazure roots for a general monomial ideal I = (x^a1, x^a2, ...).

    generators: list of exponent tuples, e.g. [(3,0), (0,2)] for (x^3, y^2)
    n: number of variables

    A monomial x^beta is in I iff beta >= some generator componentwise.
    alpha is an outer root if alpha + e_j >= 0 and x^{alpha+e_j} not in I
    for some j with alpha_j = -1.
    alpha is an inner root if for exactly one j: alpha + e_j >= 0 and
    x^{alpha+e_j} not in I, and alpha is not outer.
    """
    def in_ideal(beta):
        for gen in generators:
            if all(beta[i] >= gen[i] for i in range(n)):
                return True
        return False

    max_deg = max(sum(g) for g in generators)

    outer_set = set()
    for j in range(n):
        for beta_tuple in _nonneg_tuples(n, max_deg):
            if beta_tuple[j] == 0 and not in_ideal(beta_tuple):
                alpha = list(beta_tuple)
                alpha[j] = -1
                outer_set.add(tuple(alpha))

    roots = []
    for alpha_tuple in _all_tuples(n, -1, max_deg):
        valid_j = []
        for j in range(n):
            shifted = list(alpha_tuple)
            shifted[j] += 1
            if all(s >= 0 for s in shifted) and not in_ideal(tuple(shifted)):
                valid_j.append(j)

        if len(valid_j) == 1 and tuple(alpha_tuple) not in outer_set:
            roots.append(tuple(alpha_tuple))

    return roots


def _nonneg_tuples(n, max_sum):
    """Generate all n-tuples of nonneg ints with sum <= max_sum."""
    if n == 1:
        for i in range(max_sum + 1):
            yield (i,)
    else:
        for i in range(max_sum + 1):
            for rest in _nonneg_tuples(n - 1, max_sum - i):
                yield (i,) + rest


def _all_tuples(n, lo, hi):
    """Generate all n-tuples with entries in [lo, hi]."""
    if n == 1:
        for i in range(lo, hi + 1):
            yield (i,)
    else:
        for i in range(lo, hi + 1):
            for rest in _all_tuples(n - 1, lo, hi):
                yield (i,) + rest


def main():
    print("=" * 70)
    print("DEMAZURE ROOT STRUCTURE OF MONOMIAL ALGEBRAS")
    print("=" * 70)
    print()

    # Part 1: R_inn(m^d) = 0 in ALL dimensions
    print("PART 1: INNER ROOTS OF m^d")
    print("-" * 40)
    print()
    print("CLAIM: R_inn(m^d) = 0 for ALL d >= 2 and ALL n >= 2.")
    print()

    for n in [2, 3, 4]:
        print(f"  n = {n}:")
        for d in range(2, 6):
            out = outer_roots(d, n)
            inn = inner_roots(d, n)
            print(f"    d = {d}:  outer = {len(out):3d},  inner = {len(inn):3d}")
        print()

    print("  PROOF SKETCH:")
    print("  An inner root alpha needs EXACTLY ONE j with alpha+e_j >= 0")
    print("  and |alpha+e_j| < d. For m^d, the condition |alpha+e_j| < d")
    print("  means sum(alpha) < d-1. But if alpha+e_j >= 0 for some j,")
    print("  then for any other k != j with alpha_k >= 0, we also have")
    print("  alpha+e_k >= 0 and sum(alpha+e_k) = sum(alpha)+1 < d.")
    print("  So if alpha has two or more non-negative entries (besides j),")
    print("  there are multiple valid j's. The only way to get exactly one")
    print("  valid j is if alpha_k = -1 for all k != j, making it outer.")
    print("  Contradiction with alpha not being outer. QED")
    print()

    # Part 2: Inner roots for GENERAL monomial ideals
    print("=" * 70)
    print("PART 2: INNER ROOTS FOR GENERAL MONOMIAL IDEALS")
    print("-" * 40)
    print()
    print("For non-maximal-power ideals, inner roots CAN appear.")
    print()

    test_ideals = [
        ("(x^3, y^2)", [(3, 0), (0, 2)]),
        ("(x^4, y^3)", [(4, 0), (0, 3)]),
        ("(x^2, xy, y^3)", [(2, 0), (1, 1), (0, 3)]),
        ("(x^3, x^2*y, y^2)", [(3, 0), (2, 1), (0, 2)]),
        ("(x^2, y^2)", [(2, 0), (0, 2)]),
    ]

    for name, gens in test_ideals:
        inn = inner_roots_general(gens, 2)
        out_set = set()
        def in_ideal(beta, gs=gens):
            return any(all(beta[i] >= g[i] for i in range(2)) for g in gs)
        for j in range(2):
            max_deg = max(sum(g) for g in gens)
            for beta_tuple in _nonneg_tuples(2, max_deg):
                if beta_tuple[j] == 0 and not in_ideal(beta_tuple):
                    alpha = list(beta_tuple)
                    alpha[j] = -1
                    out_set.add(tuple(alpha))
        print(f"  I = {name}:  outer = {len(out_set)},  inner = {len(inn)}")
        if inn:
            print(f"    inner roots: {inn}")
    print()

    # Part 3: The TRUE structural difference
    print("=" * 70)
    print("PART 3: THE TRUE STRUCTURAL DIFFERENCE (dim 2 vs dim >= 3)")
    print("-" * 40)
    print()
    print("The absence of inner roots for m^d does NOT distinguish dim 2")
    print("from dim >= 3 (inner roots vanish for m^d in ALL dimensions).")
    print()
    print("The REAL structural differences are:")
    print()
    print("  1. JUNG-VAN DER KULK THEOREM (dim 2 only):")
    print("     Every automorphism of k[x,y] is tame (= composition of")
    print("     affine and triangular automorphisms).")
    print("     In dim >= 3: wild automorphisms exist (Nagata, Shestakov-Umirbaev).")
    print()
    print("  2. ABHYANKAR-MOH-SUZUKI THEOREM (dim 2 only):")
    print("     Every embedding of the line in the plane is rectifiable.")
    print("     No analog in dim >= 3.")
    print()
    print("  3. RENTSCHLER'S THEOREM (dim 2 only):")
    print("     Every LND of k[x,y] is conjugate to u(x) d/dy.")
    print("     In dim >= 3: LND structure is much richer.")
    print()
    print("  4. ANICK'S APPROXIMATION (all dims, but decisive in dim 2):")
    print("     Tame automorphisms are m-adically dense in {det JF in k*}.")
    print("     Combined with Jung-vdK (tame = ALL in dim 2), this means:")
    print("     Aut(k[x,y]) is dense in {det JF in k*} for the m-adic topology.")
    print()

    # Part 4: Consequences for JC_2
    print("=" * 70)
    print("PART 4: CONSEQUENCES FOR THE JACOBIAN CONJECTURE IN DIM 2")
    print("-" * 40)
    print()
    print("For the truncated algebras A_d = k[x,y]/m^d:")
    print()
    print("  G^Jac(A_d) = <T, outer root subgroups>  [Prop 5.4, Paper 3]")
    print("             = pi_d(Tame(k[x,y]))         [since R_inn = 0]")
    print("             = pi_d(Aut(k[x,y]))           [by Jung-van der Kulk]")
    print()
    print("This means: at EVERY finite truncation level, the image of")
    print("Aut(k[x,y]) coincides with G^Jac(A_d).")
    print()
    print("TARGET THEOREM (reduces JC_2 to a degree bound):")
    print("  If F: k^2 -> k^2 polynomial with det(JF) = 1,")
    print("  and if deg(F_d^{-1}) is bounded uniformly in d")
    print("  (where F_d = pi_d(F) and F_d^{-1} is its inverse in A_d),")
    print("  then F is a polynomial automorphism.")
    print()

    # Part 5: The equivalence chain
    print("=" * 70)
    print("PART 5: THE EQUIVALENCE CHAIN (post-2026)")
    print("-" * 40)
    print()
    print("  JC_n  = Jacobian Conjecture in n variables")
    print("  DC_m  = Dixmier Conjecture for Weyl algebra W_m")
    print("  PC_m  = Poisson Conjecture in 2m variables")
    print()
    print("Equivalences (Belov-Kanel-Kontsevich, Tsuchimoto):")
    print("  JC_{2m} <=> DC_m <=> PC_m")
    print()
    print("Current status (October 2026):")
    print("  JC_1: TRUE (trivial)")
    print("  JC_2 <=> DC_1 <=> PC_1: *** OPEN ***")
    print("    (claimed proof of DC_1 by Zheglov, arXiv:2410.06959)")
    print("  JC_3: FALSE (Alpoge, July 2026)")
    print("  JC_n (n>=3): FALSE (Gallagher, arXiv:2608.00222)")
    print("  DC_n (n>=2): FALSE (from JC_{2n} false)")
    print("  PC_n (n>=2): FALSE")
    print()
    print("Key insight: JC_2 is the LAST OPEN CASE. The tools that make")
    print("dim 2 special (Jung-vdK, Abhyankar-Moh-Suzuki, Rentschler)")
    print("are precisely the ones absent in dim >= 3 where JC fails.")


if __name__ == "__main__":
    main()
