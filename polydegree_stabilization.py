"""
Polydegree Stabilization: The core argument for JC_2.

This script implements and tests the key claim:

THEOREM (if complete): Let F = (P,Q) be a polynomial Keller map in dim 2 with
det J(F) = 1. Then the "peeling algorithm" applied to F terminates in finitely
many steps, producing a Jung-van der Kulk decomposition F = L o T_1 o ... o T_k
where each T_i is elementary (triangular) and L is affine.

The argument:
  1. If deg P > deg Q > 0, by the Jacobian condition, the leading form of P
     is c * Q_top^m where Q_top is the leading form of Q and m = deg P / deg Q.
     (This uses the DEGREE DIVISIBILITY property of Keller maps in dim 2.)
  2. Set tau_1 = (x - c*y^m, y). Then F' = F o tau_1^{-1} has deg F' < deg F.
  3. F' is still a polynomial Keller map (det J(F') = det J(F) * det J(tau_1^{-1}) = 1).
  4. By induction on degree, F' decomposes into elementary factors.
  5. Hence F = F' o tau_1 decomposes.

The catch: Step 1 requires the "degree divisibility" -- that deg P | deg Q or
deg Q | deg P for Keller maps. This is KNOWN for automorphisms (by Jung-vdK)
but needs to be proved for Keller maps directly.

We test this computationally and analyze the structure of the leading forms.
"""

import sympy as sp
from sympy import symbols, expand, Poly, degree, LC, factor, gcd

x, y = symbols("x y")


def total_degree(expr):
    """Total degree of a polynomial in x, y."""
    expr = expand(expr)
    if expr == 0:
        return -1
    p = Poly(expr, x, y)
    return max(sum(m) for m in p.as_dict().keys())


def leading_form(expr, d=None):
    """Extract the homogeneous part of highest (or specified) degree."""
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
    """Compute det J(P,Q)."""
    return expand(sp.diff(P, x) * sp.diff(Q, y) - sp.diff(P, y) * sp.diff(Q, x))


def compose(F, G):
    """Compose (F1(G1,G2), F2(G1,G2))."""
    return (expand(F[0].subs([(x, G[0]), (y, G[1])])),
            expand(F[1].subs([(x, G[0]), (y, G[1])])))


def peel_leading_term(P, Q):
    """
    The peeling algorithm: given (P, Q) with det J(P,Q) = c (constant),
    find an elementary automorphism tau such that deg(F o tau^{-1}) < deg(F).

    Returns: (tau_type, tau_poly, P', Q') where:
      - tau_type is 'x' or 'y' (which variable the elementary auto acts on)
      - tau_poly is the polynomial g such that tau = (x + g(y), y) or (x, y + g(x))
      - (P', Q') is the peeled map with lower degree

    Returns None if peeling fails (map is already affine or deg P = deg Q and
    the leading forms don't simplify).
    """
    dP = total_degree(P)
    dQ = total_degree(Q)

    if dP <= 1 and dQ <= 1:
        return None

    if dP >= dQ and dQ >= 1:
        lf_P = leading_form(P, dP)
        lf_Q = leading_form(Q, dQ)

        if dP % dQ != 0:
            if dQ % dP == 0 and dP >= 1:
                lf_P, lf_Q = lf_Q, lf_P
                P, Q = Q, P
                dP, dQ = dQ, dP
            else:
                return ('swap_fail', None, P, Q)

        m = dP // dQ

        ratio = sp.simplify(lf_P / lf_Q**m)
        if ratio.is_number and ratio != 0:
            c = ratio
            g = c * y**m
            tau_inv = (x + g, y)
            P_new = expand(P - (c * Q**m))
            Q_new = Q
            return ('x', g, P_new, Q_new)

    if dQ > dP and dP >= 1:
        lf_P = leading_form(P, dP)
        lf_Q = leading_form(Q, dQ)

        if dQ % dP != 0:
            return ('swap_fail', None, P, Q)

        m = dQ // dP

        ratio = sp.simplify(lf_Q / lf_P**m)
        if ratio.is_number and ratio != 0:
            c = ratio
            g = c * x**m
            P_new = P
            Q_new = expand(Q - (c * P**m))
            return ('y', g, P_new, Q_new)

    return ('fail', None, P, Q)


def full_peeling(P, Q, max_steps=30):
    """
    Apply the peeling algorithm repeatedly until the map is affine.
    Returns the list of peeling steps and whether it succeeded.
    """
    steps = []
    F1, F2 = expand(P), expand(Q)

    for i in range(max_steps):
        dF1 = total_degree(F1)
        dF2 = total_degree(F2)

        if dF1 <= 1 and dF2 <= 1:
            steps.append(('affine', F1, F2))
            return steps, True

        result = peel_leading_term(F1, F2)
        if result is None:
            steps.append(('affine', F1, F2))
            return steps, True

        step_type, g, F1_new, F2_new = result

        if step_type in ('fail', 'swap_fail'):
            steps.append(('stuck', F1, F2, step_type))
            return steps, False

        steps.append((step_type, g, dF1 if step_type == 'x' else dF2))
        F1 = expand(F1_new)
        F2 = expand(F2_new)

    steps.append(('max_steps', F1, F2))
    return steps, False


def analyze_leading_form_structure(P, Q):
    """
    Analyze whether the leading forms of a Keller pair satisfy
    the degree divisibility condition.
    """
    dP = total_degree(P)
    dQ = total_degree(Q)
    lf_P = leading_form(P, dP)
    lf_Q = leading_form(Q, dQ)

    print(f"  deg(P) = {dP}, deg(Q) = {dQ}")
    print(f"  Leading form of P: {lf_P}")
    print(f"  Leading form of Q: {lf_Q}")

    if dP == dQ:
        print(f"  Same degree -- look at the Jacobian of leading forms")
        jac_lead = jacobian_det(lf_P, lf_Q)
        print(f"  J(lf_P, lf_Q) = {expand(jac_lead)}")
        if expand(jac_lead) == 0:
            print(f"  Leading forms are algebraically dependent!")
            ratio = sp.simplify(lf_P / lf_Q)
            if ratio.is_number:
                print(f"  lf_P / lf_Q = {ratio} (constant)")
    elif dP > dQ:
        if dQ > 0 and dP % dQ == 0:
            m = dP // dQ
            ratio = sp.simplify(lf_P / lf_Q**m)
            print(f"  deg(P)/deg(Q) = {m}")
            print(f"  lf_P / lf_Q^{m} = {ratio}")
            if ratio.is_number:
                print(f"  SUCCESS: Leading form of P = {ratio} * (leading form of Q)^{m}")
        else:
            print(f"  deg(P) does not divide deg(Q)!")
            print(f"  gcd(dP, dQ) = {gcd(dP, dQ)}")
    else:
        if dP > 0 and dQ % dP == 0:
            m = dQ // dP
            ratio = sp.simplify(lf_Q / lf_P**m)
            print(f"  deg(Q)/deg(P) = {m}")
            print(f"  lf_Q / lf_P^{m} = {ratio}")
            if ratio.is_number:
                print(f"  SUCCESS: Leading form of Q = {ratio} * (leading form of P)^{m}")
        else:
            print(f"  deg(Q) does not divide deg(P)!")
            print(f"  gcd(dP, dQ) = {gcd(dP, dQ)}")


def main():
    print("=" * 70)
    print("POLYDEGREE STABILIZATION ANALYSIS")
    print("=" * 70)
    print()

    # Test cases: known automorphisms
    test_maps = [
        ("(x+y^2, y)", x + y**2, y),
        ("(x, y+x^3)", x, y + x**3),
        ("(x+y^2, y) o (x, y+x^3)",
         expand((x + y**2).subs(y, y + x**3)),
         y + x**3),
    ]

    F_comp = compose((x + y**2, y), (x, y - x**3))
    test_maps.append(("(x+y^2,y) o (x,y-x^3)", F_comp[0], F_comp[1]))

    F_comp2 = compose((x, y + x**2), (x - y**2, y))
    test_maps.append(("(x,y+x^2) o (x-y^2,y)", F_comp2[0], F_comp2[1]))

    F_comp3 = compose(compose((x + y**2, y), (x, y + x**3)), (x - y**2, y))
    test_maps.append(("triple composition", F_comp3[0], F_comp3[1]))

    print("PART 1: LEADING FORM ANALYSIS")
    print("-" * 40)
    for name, P, Q in test_maps:
        print(f"\n  Map: {name}")
        jd = jacobian_det(P, Q)
        print(f"  det J = {jd}")
        analyze_leading_form_structure(P, Q)
    print()

    print("=" * 70)
    print("PART 2: FULL PEELING ALGORITHM")
    print("-" * 40)
    for name, P, Q in test_maps:
        print(f"\n  Map: {name}")
        print(f"  F = ({P}, {Q})")
        steps, success = full_peeling(P, Q)
        for i, step in enumerate(steps):
            if step[0] in ('x', 'y'):
                print(f"    Step {i+1}: peel {step[0]}-direction, g={step[1]}, "
                      f"reduced from deg {step[2]}")
            elif step[0] == 'affine':
                print(f"    Step {i+1}: DONE (affine: ({step[1]}, {step[2]}))")
            else:
                print(f"    Step {i+1}: {step[0]}")
        print(f"  Result: {'SUCCESS' if success else 'FAILED'}")
        if success:
            k = sum(1 for s in steps if s[0] in ('x', 'y'))
            print(f"  Polydegree length: {k}")
    print()

    # Part 3: The critical property
    print("=" * 70)
    print("PART 3: DEGREE DIVISIBILITY FOR KELLER MAPS")
    print("-" * 40)
    print()
    print("The Abhyankar-Moh theorem states that for a polynomial EMBEDDING")
    print("t -> (P(t), Q(t)) of the line in the plane, deg P | deg Q or vice versa.")
    print()
    print("For a Keller map F = (P,Q) with det J(F) = 1:")
    print("  - If F is an automorphism, the fibers of P are all isomorphic to")
    print("    affine lines (by Abhyankar-Moh), so deg P | deg Q holds.")
    print("  - The question is: does this hold for ALL Keller maps?")
    print()
    print("KNOWN: For Keller maps in dim 2, if gcd(deg P, deg Q) < 16,")
    print("  then F is an automorphism (Lee-Li). So the first potential")
    print("  counterexample has gcd(deg P, deg Q) >= 16.")
    print()
    print("If the degree divisibility holds for all Keller maps, then")
    print("the peeling algorithm always succeeds, and JC_2 follows.")

    # Part 4: What the peeling algorithm tells us
    print()
    print("=" * 70)
    print("PART 4: THE LOGICAL STRUCTURE")
    print("-" * 40)
    print("""
The argument has the following structure:

Given: F = (P,Q) polynomial Keller map with det J(F) = 1.
Want: F is a polynomial automorphism.

APPROACH 1 (Direct peeling):
  If we can show the peeling algorithm works on F directly:
  F = L o T_1 o ... o T_k (Jung-van der Kulk decomposition)
  Then F is manifestly an automorphism.
  REQUIRES: Degree divisibility for Keller maps.

APPROACH 2 (Anick + stabilization):
  For each d, F_d = F mod m^d lies in G^Jac(A_d).
  By Anick, there exists tame phi_d with F = phi_d mod m^d.
  If deg(phi_d) <= D for all d, then F = phi_d for d > D, done.
  REQUIRES: Uniform degree bound on Anick approximants.

APPROACH 3 (Formal inverse + growth):
  The formal inverse G = F^{-1} exists in k[[x,y]]^2.
  G is polynomial iff it has finite degree.
  The Gabber bound says: if G is polynomial, deg G <= deg F.
  So we need: the formal inverse has bounded degree terms.
  EQUIVALENT TO: The power series G has only finitely many
  nonzero homogeneous components.
  REQUIRES: Showing the "tail" of G is zero.

All three approaches reduce to controlling DEGREES.
The key is that dimension 2 provides enough structure
(Jung-vdK, Abhyankar-Moh, Rentschler) to control these degrees.
""")


if __name__ == "__main__":
    main()
