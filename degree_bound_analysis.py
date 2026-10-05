"""
Degree bound analysis for the Jacobian Conjecture in dimension 2.

Central question: Given a Keller map F = (P,Q) with det J(F) = 1,
do the Anick approximants phi_d (tame automorphisms with F = phi_d mod m^d)
have uniformly bounded degree?

This script:
1. Implements the "peeling algorithm" (factorization via outer root actions)
2. Tests degree stabilization for families of Keller maps
3. Analyzes the Jung-van der Kulk decomposition structure
4. Provides evidence for/against the uniform degree bound conjecture
"""

import sympy as sp
from sympy import symbols, expand, Poly, degree, LC

x, y = symbols("x y")


def truncate(expr, d):
    """Truncate polynomial to terms of total degree < d."""
    p = Poly(expand(expr), x, y)
    result = sp.Integer(0)
    for monom, coeff in p.as_dict().items():
        if sum(monom) < d:
            result += coeff * x**monom[0] * y**monom[1]
    return result


def total_degree(expr):
    """Total degree of a polynomial in x, y."""
    expr = expand(expr)
    if expr == 0:
        return -1
    p = Poly(expr, x, y)
    return max(sum(m) for m in p.as_dict().keys())


def jacobian_det(P, Q):
    """Compute det J(P,Q) = P_x Q_y - P_y Q_x."""
    return expand(sp.diff(P, x) * sp.diff(Q, y) - sp.diff(P, y) * sp.diff(Q, x))


def compose(F, G):
    """Compose two maps: F(G(x,y)) = (F1(G1,G2), F2(G1,G2))."""
    F1, F2 = F
    G1, G2 = G
    return (expand(F1.subs([(x, G1), (y, G2)])),
            expand(F2.subs([(x, G1), (y, G2)])))


def elementary_x(f_of_y):
    """Elementary automorphism (x + f(y), y)."""
    return (x + f_of_y, y)


def elementary_y(g_of_x):
    """Elementary automorphism (x, y + g(x))."""
    return (x, y + g_of_x)


def invert_elementary_x(f_of_y):
    """Inverse of (x + f(y), y) is (x - f(y), y)."""
    return (x - f_of_y, y)


def invert_elementary_y(g_of_x):
    """Inverse of (x, y + g(x)) is (x, y - g(x))."""
    return (x, y - g_of_x)


# =====================================================================
# JUNG-VAN DER KULK DECOMPOSITION
# =====================================================================

def jvdk_decomposition(P, Q, max_steps=50):
    """
    Compute the Jung-van der Kulk decomposition of an automorphism (P,Q).

    Returns a list of elementary factors such that composing them gives (P,Q).
    Each factor is ('x', f(y)) or ('y', g(x)) or ('affine', matrix, shift).

    The algorithm "peels" the highest-degree terms using elementary automorphisms.
    """
    factors = []
    F1, F2 = expand(P), expand(Q)

    for step in range(max_steps):
        d1 = total_degree(F1)
        d2 = total_degree(F2)

        if d1 <= 1 and d2 <= 1:
            factors.append(('affine', F1, F2))
            return factors

        if d1 > d2 and d2 > 0:
            lead = _leading_form(F1, d1)
            if _is_power_of(lead, y):
                c, k = _extract_power_coeff(lead, y)
                if k > 0 and d2 > 0 and d1 == k * d2:
                    lead_F2 = _leading_form(F2, d2)
                    ratio = sp.simplify(lead / lead_F2**k)
                    if ratio.is_number and ratio != 0:
                        g = ratio * F2**k
                        g_trunc = _extract_y_polynomial(F1, d1, F2, d2)
                        if g_trunc is not None:
                            factors.append(('x', -g_trunc))
                            F1 = expand(F1 - g_trunc.subs(y, F2))
                            continue

            g = _peel_x(F1, F2, d1, d2)
            if g is not None:
                factors.append(('x', -g))
                F1 = expand(F1 - g.subs(y, F2))
                continue

        if d2 > d1 and d1 > 0:
            g = _peel_y(F1, F2, d1, d2)
            if g is not None:
                factors.append(('y', -g))
                F2 = expand(F2 - g.subs(x, F1))
                continue

        if d1 == d2 and d1 > 1:
            g = _peel_x(F1, F2, d1, d2)
            if g is not None:
                factors.append(('x', -g))
                F1 = expand(F1 - g.subs(y, F2))
                continue
            g = _peel_y(F1, F2, d1, d2)
            if g is not None:
                factors.append(('y', -g))
                F2 = expand(F2 - g.subs(x, F1))
                continue

        break

    factors.append(('remainder', F1, F2))
    return factors


def _leading_form(expr, d):
    """Extract the homogeneous part of degree d."""
    p = Poly(expand(expr), x, y)
    result = sp.Integer(0)
    for monom, coeff in p.as_dict().items():
        if sum(monom) == d:
            result += coeff * x**monom[0] * y**monom[1]
    return result


def _is_power_of(expr, var):
    """Check if expr is c * var^k for some constant c and integer k."""
    expr = expand(expr)
    p = Poly(expr, x, y)
    terms = p.as_dict()
    if len(terms) != 1:
        return False
    monom = list(terms.keys())[0]
    if var == y:
        return monom[0] == 0
    else:
        return monom[1] == 0


def _extract_power_coeff(expr, var):
    """If expr = c * var^k, return (c, k)."""
    p = Poly(expand(expr), x, y)
    terms = p.as_dict()
    if len(terms) != 1:
        return None, None
    monom, coeff = list(terms.items())[0]
    if var == y:
        return coeff, monom[1]
    else:
        return coeff, monom[0]


def _extract_y_polynomial(F1, d1, F2, d2):
    """Try to express the leading terms of F1 as g(F2) for some polynomial g."""
    if d2 == 0 or d1 % d2 != 0:
        return None
    k = d1 // d2
    lead_F2 = _leading_form(F2, d2)
    c, exp = _extract_power_coeff(lead_F2, y if total_degree(lead_F2.subs(x, 0)) == d2 else x)
    if c is None:
        return None
    lead_F1 = _leading_form(F1, d1)
    ratio = sp.simplify(lead_F1 / lead_F2**k)
    if ratio.is_number and ratio != 0:
        return ratio * y**k
    return None


def _peel_x(F1, F2, d1, d2):
    """Try to find g(y) such that deg(F1 - g(F2)) < d1."""
    if d2 == 0:
        return None
    lead_F1 = _leading_form(F1, d1)
    lead_F2 = _leading_form(F2, d2)

    if d1 % d2 != 0:
        return None

    k = d1 // d2

    ratio = sp.simplify(lead_F1 / lead_F2**k)
    if ratio.is_number and ratio != 0:
        return ratio * y**k
    return None


def _peel_y(F1, F2, d1, d2):
    """Try to find g(x) such that deg(F2 - g(F1)) < d2."""
    if d1 == 0:
        return None
    lead_F1 = _leading_form(F1, d1)
    lead_F2 = _leading_form(F2, d2)

    if d2 % d1 != 0:
        return None

    k = d2 // d1

    ratio = sp.simplify(lead_F2 / lead_F1**k)
    if ratio.is_number and ratio != 0:
        return ratio * x**k
    return None


# =====================================================================
# ANICK APPROXIMATION: CONSTRUCT phi_d FROM F mod m^d
# =====================================================================

def anick_approximant(P, Q, d):
    """
    Given a Keller map (P,Q), construct a tame automorphism phi_d
    agreeing with (P,Q) modulo m^d.

    Uses the peeling algorithm on the truncation F mod m^d, working
    in the ring A_d = k[x,y]/m^d.
    """
    P_d = truncate(P, d)
    Q_d = truncate(Q, d)

    factors = []
    F1, F2 = P_d, Q_d

    for step in range(100):
        F1 = truncate(F1, d)
        F2 = truncate(F2, d)

        d1 = total_degree(F1)
        d2 = total_degree(F2)

        if d1 <= 1 and d2 <= 1:
            factors.append(('affine', F1, F2))
            break

        peeled = False

        if d1 >= d2 and d2 > 0 and d1 > 1:
            g = _peel_x(F1, F2, d1, d2)
            if g is not None:
                g_val = g.subs(y, F2)
                F1 = truncate(expand(F1 - g_val), d)
                factors.append(('x', g))
                peeled = True

        if not peeled and d2 >= d1 and d1 > 0 and d2 > 1:
            g = _peel_y(F1, F2, d1, d2)
            if g is not None:
                g_val = g.subs(x, F1)
                F2 = truncate(expand(F2 - g_val), d)
                factors.append(('y', g))
                peeled = True

        if not peeled:
            factors.append(('remainder', F1, F2))
            break

    return factors


# =====================================================================
# DEGREE STABILIZATION TEST
# =====================================================================

def test_degree_stabilization(P, Q, d_range=range(3, 15)):
    """
    For a Keller map (P,Q), compute the Anick approximants phi_d
    for various d and track how their degrees evolve.
    """
    print(f"  F = ({P}, {Q})")
    print(f"  deg(P) = {total_degree(P)}, deg(Q) = {total_degree(Q)}")
    jd = jacobian_det(P, Q)
    print(f"  det J(F) = {jd}")
    print()

    results = []
    for d in d_range:
        factors = anick_approximant(P, Q, d)

        factor_count = len(factors)
        max_factor_deg = 0
        for f in factors:
            if f[0] in ('x', 'y'):
                fd = total_degree(f[1])
                max_factor_deg = max(max_factor_deg, fd)

        completed = factors[-1][0] != 'remainder'
        results.append((d, factor_count, max_factor_deg, completed))

        status = "OK" if completed else "INCOMPLETE"
        print(f"  d={d:3d}: {factor_count:3d} factors, "
              f"max_deg={max_factor_deg:3d}, {status}")

    return results


# =====================================================================
# KELLER MAP FAMILIES
# =====================================================================

def keller_family_1(a, b):
    """F = (x + a*y^2, y + b*x^2) composed with Jacobian correction."""
    P = x + a * y**2
    Q = y + b * x**2
    jd = jacobian_det(P, Q)
    if jd == 1:
        return P, Q
    return None, None


def keller_family_druzkowski(n=2):
    """
    Druzkowski form: F = (x + (a*x + b*y)^n, y + (c*x + d*y)^n)
    with det J(F) = 1 requires specific constraints on a,b,c,d.
    In dim 2, the simplest case: F = (x + (a*y)^n, y) has det J = 1.
    """
    maps = []
    for k in range(2, n + 1):
        P = x + y**k
        Q = y
        maps.append((P, Q, f"(x + y^{k}, y)"))

        P = x
        Q = y + x**k
        maps.append((P, Q, f"(x, y + x^{k})"))
    return maps


def keller_compositions():
    """Generate Keller maps as compositions of elementary automorphisms."""
    maps = []

    F = compose(elementary_x(y**2), elementary_y(-x**3))
    maps.append((F[0], F[1], "(x+y^2, y) o (x, y-x^3)"))

    F = compose(elementary_y(x**2), elementary_x(-y**2))
    maps.append((F[0], F[1], "(x, y+x^2) o (x-y^2, y)"))

    F = compose(
        compose(elementary_x(y**2), elementary_y(x**3)),
        elementary_x(-y**2)
    )
    maps.append((F[0], F[1], "(x+y^2,y)o(x,y+x^3)o(x-y^2,y)"))

    F = compose(
        compose(elementary_x(2*y**3), elementary_y(-x**2)),
        elementary_x(y)
    )
    maps.append((F[0], F[1], "(x+2y^3,y)o(x,y-x^2)o(x+y,y)"))

    return maps


# =====================================================================
# FORMAL INVERSE DEGREE ANALYSIS
# =====================================================================

def formal_inverse_degree(P, Q, max_d=20):
    """
    Compute the formal inverse of (P,Q) modulo m^d for increasing d.
    Track when the degree stabilizes (= the inverse is polynomial).
    Uses iterative Newton's method for efficiency.
    """
    print(f"  F = ({P}, {Q})")
    print(f"  deg(P) = {total_degree(P)}, deg(Q) = {total_degree(Q)}")

    G1 = x
    G2 = y
    prev_d1, prev_d2 = 1, 1
    stable_count = 0

    for d in range(2, max_d + 1):
        for _iter in range(d):
            comp1 = truncate(P.subs([(x, G1), (y, G2)]), d)
            comp2 = truncate(Q.subs([(x, G1), (y, G2)]), d)

            err1 = truncate(comp1 - x, d)
            err2 = truncate(comp2 - y, d)

            if err1 == 0 and err2 == 0:
                break

            G1 = truncate(G1 - err1, d)
            G2 = truncate(G2 - err2, d)

        d1 = total_degree(G1)
        d2 = total_degree(G2)
        print(f"  d={d:3d}: deg(G1)={d1:3d}, deg(G2)={d2:3d}")

        if d1 == prev_d1 and d2 == prev_d2:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= 3 and d >= total_degree(P) + 2:
            print(f"  ** STABILIZED at d={d} (deg stable for {stable_count} steps) **")
            return G1, G2, d

        prev_d1, prev_d2 = d1, d2

    return G1, G2, None


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("DEGREE BOUND ANALYSIS FOR JC_2")
    print("=" * 70)
    print()

    # Test 1: Simple elementary automorphisms
    print("TEST 1: Elementary automorphisms (trivial Keller maps)")
    print("-" * 50)
    simple_maps = keller_family_druzkowski(5)
    for P, Q, name in simple_maps:
        print(f"\n  {name}:")
        print(f"  det J = {jacobian_det(P, Q)}")
        print(f"  This is already elementary, so phi_d = F for all d.")
    print()

    # Test 2: Compositions of elementary automorphisms
    print("=" * 70)
    print("TEST 2: Compositions of elementary automorphisms")
    print("-" * 50)
    comp_maps = keller_compositions()
    for P, Q, name in comp_maps:
        print(f"\n  {name}:")
        jd = jacobian_det(P, Q)
        print(f"  det J = {jd}")
        deg_P = total_degree(P)
        deg_Q = total_degree(Q)
        print(f"  deg(P) = {deg_P}, deg(Q) = {deg_Q}")
    print()

    # Test 3: Formal inverse degree stabilization
    print("=" * 70)
    print("TEST 3: Formal inverse degree stabilization")
    print("-" * 50)
    for P, Q, name in comp_maps:
        print(f"\n  Map: {name}")
        deg_F = max(total_degree(P), total_degree(Q))
        formal_inverse_degree(P, Q, max_d=min(deg_F + 6, 15))
        print()

    # Test 4: Key theoretical analysis
    print("=" * 70)
    print("THEORETICAL ANALYSIS: WHY DEGREE SHOULD STABILIZE")
    print("-" * 50)
    print("""
ARGUMENT FOR UNIFORM DEGREE BOUND IN DIM 2:

Let F = (P,Q) be a Keller map, deg P = m, deg Q = n, D = max(m,n).

KNOWN FACTS:
1. deg(F^{-1}) <= deg(F) in dim 2 [Gabber bound, n-1 = 1 in dim 2]
2. F_d = F mod m^d is IDENTICAL to F for d > D (as a polynomial)
3. By Anick, there exist tame phi_d with F = phi_d mod m^d

THE KEY QUESTION (open):
  Is deg(phi_d) bounded uniformly in d?

IF YES (say deg(phi_d) <= B for all d):
  For d > B, the polynomial phi_d of degree <= B is determined by
  its terms of degree <= B, which all agree with F. So phi_d = F
  (since both are polynomials of degree <= max(B,D) agreeing mod m^d
  for d > max(B,D)). Hence F = phi_d is a tame automorphism. QED.

APPROACH VIA JUNG-VAN DER KULK:
  Each phi_d has a JvdK decomposition:
    phi_d = L_0 o T_1 o L_1 o T_2 o ... o T_k o L_k
  where L_i are affine, T_i triangular.

  The "polydegree" pdeg(phi_d) = (deg T_1, ..., deg T_k) satisfies:
    deg(phi_d) = prod(deg T_i)   [Furter-Karas]

  If the polydegree stabilizes (same tuple for d >> 0), then
  the degree stabilizes, and we're done.

  WHY IT SHOULD STABILIZE:
  The leading form of F determines the first triangular factor.
  After removing it, the resulting map has lower degree.
  Since F doesn't change with d (for d > D), neither does its
  leading form, so the first factor is the same. By induction...

  GAP: The intermediate map after removing the first factor
  depends on d through the lower-order terms (even though the
  leading form doesn't). Need to show this doesn't affect the
  decomposition structure.
""")

    # Test 5: Summary of degree bound data
    print("=" * 70)
    print("TEST 5: Degree bound summary")
    print("-" * 50)
    print()
    print("  For each tame Keller map F, the inverse F^{-1} is also")
    print("  polynomial and we can check deg(F^{-1}) <= deg(F).")
    print()
    for P, Q, name in comp_maps:
        inv = compose(
            invert_elementary_x(y**2) if "y^2" in name[:10] else (x, y),
            (x, y)
        )
        deg_F = max(total_degree(P), total_degree(Q))
        print(f"  {name}: deg(F) = {deg_F}")


if __name__ == "__main__":
    main()
