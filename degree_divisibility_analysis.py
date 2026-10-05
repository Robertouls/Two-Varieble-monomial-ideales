"""
Analysis of degree divisibility for Keller maps in dimension 2.

CENTRAL QUESTION: If F = (P,Q) is a Keller map (det J(F) = c != 0),
does deg P | deg Q or deg Q | deg P?

This is KNOWN for automorphisms (by Jung-van der Kulk).
If true for all Keller maps, it implies JC_2 via the peeling algorithm.

We analyze what the Jacobian condition constrains about the leading forms
and the degree relationship.
"""

import sympy as sp
from sympy import symbols, expand, Poly, gcd as sp_gcd, resultant

x, y = symbols("x y")
t = symbols("t")


def total_degree(expr):
    expr = expand(expr)
    if expr == 0:
        return -1
    p = Poly(expr, x, y)
    return max(sum(m) for m in p.as_dict().keys())


def leading_form(expr, d=None):
    expr = expand(expr)
    p = Poly(expr, x, y)
    if d is None:
        d = total_degree(expr)
    result = sp.Integer(0)
    for monom, coeff in p.as_dict().items():
        if sum(monom) == d:
            result += coeff * x**monom[0] * y**monom[1]
    return result


def jacobian_det(P, Q):
    return expand(sp.diff(P, x) * sp.diff(Q, y) - sp.diff(P, y) * sp.diff(Q, x))


def analyze_jacobian_leading_terms(m, n):
    """
    Analyze the constraint det J(P,Q) = c imposes on leading forms
    of P (degree m) and Q (degree n).

    The key identity: if P = p_m + (lower) and Q = q_n + (lower),
    then det J(P,Q) has leading terms of degree m + n - 2 given by:
      det J(p_m, q_n) = (p_m)_x (q_n)_y - (p_m)_y (q_n)_x

    For det J(P,Q) = c (constant, degree 0), we need:
      det J(p_m, q_n) = 0  (if m + n - 2 > 0, i.e., m + n >= 3)

    This means: p_m and q_n are algebraically dependent as
    homogeneous polynomials.
    """
    print(f"  Degrees: deg P = {m}, deg Q = {n}")
    print(f"  Leading term degrees: m+n-2 = {m+n-2}")
    print()

    if m + n < 3:
        print("  Trivial case: m + n < 3, leading terms don't constrain.")
        return

    print("  CONSTRAINT: det J(p_m, q_n) = 0")
    print("  where p_m, q_n are the leading (homogeneous) forms.")
    print()

    # For homogeneous polynomials in 2 variables, det J = 0
    # iff they are algebraically dependent iff one is a power of
    # a linear form times the other (up to constants).

    # More precisely: homogeneous p_m(x,y) and q_n(x,y) satisfy
    # J(p_m, q_n) = 0 iff there exists a linear form L = ax + by
    # such that p_m = c_1 * L^m and q_n = c_2 * L^n (up to constants).

    print("  THEOREM (homogeneous Jacobian zero):")
    print("  If p_m(x,y) and q_n(x,y) are homogeneous of degrees m, n")
    print("  with det J(p_m, q_n) = 0, then there exists a linear")
    print("  form L = ax + by such that:")
    print(f"    p_{m} = c_1 * L^{m}")
    print(f"    q_{n} = c_2 * L^{n}")
    print()
    print("  PROOF: For homogeneous f, g in k[x,y], the Jacobian")
    print("  J(f,g) = f_x g_y - f_y g_x = 0 means f and g have")
    print("  the same tangent directions at the origin. By Euler's")
    print("  identity for homogeneous polynomials:")
    print("    m * f = x * f_x + y * f_y")
    print("    n * g = x * g_x + y * g_y")
    print("  Combined with J(f,g) = 0, this gives:")
    print("    n * f * g_x = m * g * f_x  (and similarly for _y)")
    print("  So f^n / g^m = const (where both are nonzero), meaning")
    print("  f = c * g^{m/n} if n | m (as homogeneous polys).")
    print("  For this to be polynomial: n | m or m | n.")
    print()

    print("  CONSEQUENCE FOR KELLER MAPS:")
    print(f"  If det J(P,Q) = c with deg P = {m}, deg Q = {n}:")
    print(f"  The leading forms p_{m} and q_{n} satisfy J(p_m, q_n) = 0,")
    print(f"  so p_{m} = c_1 * L^{m} and q_{n} = c_2 * L^{n}")
    print(f"  for some linear form L.")
    print()

    if m % n == 0 or n % m == 0:
        r = m // n if m >= n else n // m
        print(f"  Since {max(m,n)} / {min(m,n)} = {r} (integer), this is consistent.")
        print(f"  The leading form of the higher-degree component is the")
        print(f"  {r}-th power of the leading form of the lower-degree one.")
    else:
        g = sp_gcd(m, n)
        print(f"  gcd({m},{n}) = {g}")
        print(f"  For p_m = c1 * L^m and q_n = c2 * L^n to hold, we need")
        print(f"  L to be a LINEAR form (degree 1). This is always possible!")
        print(f"  Both p_m and q_n are pure powers of the SAME linear form L.")
        print()
        print(f"  BUT: does this force {m} | {n} or {n} | {m}?")
        print(f"  NO! Both can be powers of the same L independently.")
        print(f"  p_m = c1 * L^{m}, q_n = c2 * L^{n} is perfectly consistent")
        print(f"  with gcd(m,n) = {g} < min(m,n).")
        print()
        print(f"  However, the NEXT-TO-LEADING term constraint from")
        print(f"  det J(P,Q) = c may impose additional restrictions...")


def degree_constraint_from_subleading():
    """
    Analyze the sub-leading term constraint.

    If P = c1 * L^m + P_{m-1} + ... and Q = c2 * L^n + Q_{n-1} + ...,
    then det J(P,Q) at degree m + n - 3 gives:
      det J(c1*L^m, Q_{n-1}) + det J(P_{m-1}, c2*L^n) = 0

    This constrains P_{m-1} and Q_{n-1}.
    """
    print("SUBLEADING TERM ANALYSIS")
    print("-" * 40)
    print()
    print("Write P = c1*L^m + P_{m-1} + ..., Q = c2*L^n + Q_{n-1} + ...")
    print("where L = ax + by (WLOG L = y after a linear change).")
    print()
    print("Then: P = c1*y^m + P_{m-1} + ..., Q = c2*y^n + Q_{n-1} + ...")
    print()
    print("det J(P,Q) at degree m+n-3:")
    print("  J(c1*y^m, Q_{n-1}) + J(P_{m-1}, c2*y^n) = 0")
    print()
    print("Since J(c1*y^m, f) = c1*m*y^{m-1} * f_x  (for any f),")
    print("we get:")
    print("  c1*m*y^{m-1} * (Q_{n-1})_x + c2*n*y^{n-1} * (P_{m-1})_x = 0")
    print()
    print("Wait, that's not right. Let me recompute:")
    print("  J(c1*y^m, f) = (c1*y^m)_x * f_y - (c1*y^m)_y * f_x")
    print("               = 0 * f_y - c1*m*y^{m-1} * f_x")
    print("               = -c1*m*y^{m-1} * f_x")
    print()
    print("Similarly: J(f, c2*y^n) = f_x * c2*n*y^{n-1} - f_y * 0")
    print("                        = c2*n*y^{n-1} * f_x")
    print()
    print("So the constraint at degree m+n-3 is:")
    print("  -c1*m*y^{m-1} * (Q_{n-1})_x + c2*n*y^{n-1} * (P_{m-1})_x = 0")
    print()
    print("If m > n: y^{n-1} divides first term, so (Q_{n-1})_x has y^{n-1} factor.")
    print("  But Q_{n-1} is homogeneous of degree n-1, so (Q_{n-1})_x is")
    print("  homogeneous of degree n-2. For y^{n-1} to divide it, we need n-1 <= n-2,")
    print("  contradiction unless (Q_{n-1})_x = 0, meaning Q_{n-1} = c*y^{n-1}.")
    print()
    print("  Then: c2*n*y^{n-1} * (P_{m-1})_x = 0, so (P_{m-1})_x = 0,")
    print("  meaning P_{m-1} = c'*y^{m-1}.")
    print()
    print("  So: P = c1*y^m + c'*y^{m-1} + ..., Q = c2*y^n + c*y^{n-1} + ...")
    print("  The SUBLEADING terms are also pure powers of y!")
    print()
    print("  By induction on the degree of the sub-leading terms,")
    print("  ALL terms of P and Q are of the form a_k * y^k,")
    print("  meaning P = p(y), Q = q(y) (polynomials in y alone).")
    print()
    print("  But then det J(P,Q) = P_x * Q_y - P_y * Q_x = 0 * Q_y - P_y * 0 = 0,")
    print("  contradicting det J(P,Q) = c != 0.")
    print()
    print("  WAIT: this is only if m > n and m/n is not an integer,")
    print("  because in that case the factoring constraint is too strong.")
    print()
    print("  Let me reconsider for m > n with m = k*n...")


def careful_subleading_analysis():
    """
    More careful analysis of what happens when deg P = m, deg Q = n,
    with m = k*n (divisibility holds) vs m not divisible by n.
    """
    print()
    print("=" * 70)
    print("CAREFUL ANALYSIS: DIVISIBILITY vs NON-DIVISIBILITY")
    print("=" * 70)
    print()

    # Case 1: m = k*n (divisibility holds)
    print("CASE 1: m = k*n (degree divisibility holds)")
    print("-" * 40)
    print()
    print("WLOG (after linear change): leading forms are")
    print("  p_m = c1 * y^m = c1 * y^{kn}")
    print("  q_n = c2 * y^n")
    print()
    print("The peeling step: set tau = (x - (c1/c2^k) * y^k, y)")
    print("Then P' = P - (c1/c2^k) * Q^k has deg P' < m.")
    print("And det J(P', Q) = det J(P,Q) = c.")
    print("So (P', Q) is a new Keller map with lower degree.")
    print("This is exactly the peeling algorithm, and it works.")
    print()

    # Case 2: m not divisible by n (hypothetical)
    print("CASE 2: m not divisible by n (hypothetical)")
    print("-" * 40)
    print()
    print("Leading forms: p_m = c1 * L^m, q_n = c2 * L^n")
    print("WLOG L = y, so p_m = c1*y^m, q_n = c2*y^n.")
    print()
    print("The Jacobian condition forces ALL sub-leading terms to")
    print("be pure powers of y (as shown above). But then")
    print("P = P(y) and Q = Q(y), giving det J = 0. Contradiction!")
    print()
    print("THEREFORE: m | n or n | m MUST hold for any Keller map!")
    print()
    print("BUT WAIT: is the induction step correct?")
    print("Let's verify with an explicit computation...")
    print()

    # Verification
    a, b, c1_coeff, c2_coeff = symbols("a b c1 c2", nonzero=True)

    # Try P = c1*y^m + a*x*y^{m-2} + ..., Q = c2*y^n + b*x*y^{n-2} + ...
    # with m = 5, n = 3 (not divisible)
    m_val, n_val = 5, 3

    print(f"  Test: m = {m_val}, n = {n_val} (gcd = {sp_gcd(m_val, n_val)})")
    print(f"  5 does not divide 3, and 3 does not divide 5.")
    print()

    # Most general P of degree 5 with leading form y^5:
    a0, a1, a2, a3, a4, a5 = symbols("a0 a1 a2 a3 a4 a5")
    b0, b1, b2, b3 = symbols("b0 b1 b2 b3")

    P5 = y**5 + a0*x*y**3 + a1*x**2*y  # homogeneous deg 5 sub-leading
    Q3 = y**3 + b0*x*y  # homogeneous deg 3 sub-leading

    # The degree m+n-2 = 6 term of J(P,Q) must be 0:
    # J(y^5, y^3) = (0)(3y^2) - (5y^4)(0) = 0 ✓ (trivially)

    # The degree m+n-3 = 5 term:
    # J(y^5, Q_2) + J(P_4, y^3) where P_4 is the deg-4 part of P, Q_2 is deg-2 part of Q
    # But we parameterized P_5 with sub-leading in the same degree.
    # Let me be more careful.

    # Write P = y^5 + P_4 + P_3 + ... Q = y^3 + Q_2 + Q_1 + ...
    # where P_k, Q_k are homogeneous of degree k.

    P_4 = a0*x*y**3 + a1*x**2*y  # general homogeneous of degree 4 with no y^4 term
    # Actually, y^4 term IS allowed in P_4
    P_4_full = a0*x*y**3 + a1*x**2*y + a5*y**4
    Q_2 = b0*x*y + b1*x**2 + b3*y**2

    # J(y^5, Q_2) = (0)*Q_2_y - (5y^4)*Q_2_x = -5*y^4 * Q_2_x
    Q_2_x = sp.diff(Q_2, x)
    term1 = expand(-5*y**4 * Q_2_x)
    print(f"  J(y^5, Q_2) = {term1}")

    # J(P_4, y^3) = P_4_x * (3y^2) - P_4_y * 0 = 3*y^2 * P_4_x
    P_4_x = sp.diff(P_4_full, x)
    term2 = expand(3*y**2 * P_4_x)
    print(f"  J(P_4, y^3) = {term2}")

    constraint = expand(term1 + term2)
    print(f"  Sum (must be 0): {constraint}")
    print()

    # Collect by monomials
    p = Poly(constraint, x, y)
    for monom, coeff in sorted(p.as_dict().items()):
        print(f"    x^{monom[0]}*y^{monom[1]}: {coeff} = 0")

    print()
    print("  Solving these constraints...")
    from sympy import solve
    eqs = [coeff for monom, coeff in p.as_dict().items()]
    sol = solve(eqs, [a0, a1, a5, b0, b1, b3])
    print(f"  Solution: {sol}")
    print()

    if sol:
        print("  Constraints on subleading coefficients:")
        print(f"    a0 = 5*b0/3 (x*y^3 term in P_4 is proportional to x*y in Q_2)")
        print(f"    a1 = 0 (no x^2*y term in P_4)")
        print(f"    b1 = 0 (no x^2 term in Q_2)")
        print(f"    a5, b0, b3 are FREE")
        print()
        print("  CRITICAL: x-terms survive (b0*x*y in Q_2, (5b0/3)*x*y^3 in P_4)")
        print("  The cascade does NOT force P = P(y), Q = Q(y)!")
        print()
        print("  GAP IN THE ARGUMENT: The subleading terms are constrained but")
        print("  not forced to be univariate. The degree divisibility claim")
        print("  is NOT proved by this cascade argument alone.")
        print()
        print("  WHAT IS PROVED:")
        print("    - Leading forms are powers of the same linear form L")
        print("    - Certain x-monomials are killed (x^2 in Q, x^2*y in P)")
        print("    - Remaining x-terms satisfy algebraic relations")
        print()
        print("  WHAT IS NEEDED:")
        print("    - Either: show the full cascade (all degrees) kills all x-terms")
        print("      (seems unlikely given the above)")
        print("    - Or: use a different approach to degree divisibility")
        print("    - Or: bypass degree divisibility entirely")


def main():
    print("=" * 70)
    print("DEGREE DIVISIBILITY FOR KELLER MAPS IN DIMENSION 2")
    print("=" * 70)
    print()
    print("CENTRAL QUESTION: If F = (P,Q) with det J(F) = c (const != 0),")
    print("must we have deg P | deg Q or deg Q | deg P?")
    print()
    print("If YES: JC_2 follows from the peeling algorithm.")
    print("If NO: Need another approach.")
    print()

    analyze_jacobian_leading_terms(6, 3)
    print()
    analyze_jacobian_leading_terms(5, 3)
    print()

    degree_constraint_from_subleading()
    careful_subleading_analysis()

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("The analysis shows that the Jacobian condition det J(P,Q) = c")
    print("combined with two-variable homogeneous polynomial theory")
    print("FORCES the leading forms to be powers of the same linear form.")
    print()
    print("The sub-leading term analysis then shows that if deg P and deg Q")
    print("are NOT in a divisibility relation, the Jacobian condition forces")
    print("P = P(y), Q = Q(y) (both univariate in the same variable),")
    print("which gives det J = 0, contradicting det J = c != 0.")
    print()
    print("HOWEVER: The explicit computation for (m,n)=(5,3) shows that")
    print("the subleading x-terms are NOT killed -- they satisfy relations")
    print("but remain nonzero. The cascade argument has a GAP.")
    print()
    print("STATE OF THE ART:")
    print("  1. Degree divisibility is KNOWN for automorphisms (Jung-vdK)")
    print("  2. Degree divisibility is NOT KNOWN for arbitrary Keller maps")
    print("  3. Known: gcd(deg P, deg Q) >= 16 for counterexamples")
    print("  4. Known: gcd(deg P, deg Q) != 2p for any prime p")
    print("  5. The peeling algorithm WORKS when degree divisibility holds")
    print()
    print("PROMISING DIRECTIONS:")
    print("  A. Prove degree divisibility using algebraic geometry of")
    print("     the fibers (Abhyankar-Moh for non-proper maps)")
    print("  B. Use the Dixmier equivalence (Zheglov's DC_1 proof)")
    print("  C. Use Lee-Li Conjecture E (Newton polygon constraints)")
    print("  D. Use the inverse limit + Anick directly (avoid divisibility)")


if __name__ == "__main__":
    main()
