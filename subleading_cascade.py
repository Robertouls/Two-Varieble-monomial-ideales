"""
Full subleading cascade analysis for the degree divisibility question.

We test whether the Jacobian condition det J(P,Q) = c, applied at ALL
degree levels simultaneously, forces P and Q to satisfy deg P | deg Q
or deg Q | deg P.

Strategy: Parameterize P and Q by their homogeneous components,
impose det J(P,Q) = c degree by degree, and solve the resulting system.
If the only solutions have P = P(y), Q = Q(y) (univariate), this proves
degree divisibility. If nontrivial solutions exist, degree divisibility
may fail.

We focus on the critical case (m,n) = (5,3) with gcd = 1.
"""

import sympy as sp
from sympy import symbols, expand, Poly, solve, Rational

x, y = symbols("x y")


def homogeneous_poly(deg, prefix="a"):
    """Create a general homogeneous polynomial of given degree."""
    terms = []
    coeffs = []
    for i in range(deg + 1):
        j = deg - i
        c = symbols(f"{prefix}_{i}_{j}")
        coeffs.append(c)
        terms.append(c * x**i * y**j)
    return sum(terms), coeffs


def jacobian_det(P, Q):
    return expand(sp.diff(P, x) * sp.diff(Q, y) - sp.diff(P, y) * sp.diff(Q, x))


def extract_homogeneous(expr, deg):
    """Extract homogeneous part of given degree."""
    expr = expand(expr)
    if expr == 0:
        return sp.Integer(0)
    p = Poly(expr, x, y)
    result = sp.Integer(0)
    for monom, coeff in p.as_dict().items():
        if sum(monom) == deg:
            result += coeff * x**monom[0] * y**monom[1]
    return result


def full_cascade(m, n, c_val=1):
    """
    Full cascade analysis for Keller maps with deg P = m, deg Q = n.

    Construct P = sum P_k (k = 1..m), Q = sum Q_k (k = 1..n)
    with leading forms P_m = y^m, Q_n = y^n (WLOG after linear change).
    Impose det J(P,Q) = c at each degree level.
    """
    print(f"FULL CASCADE: deg P = {m}, deg Q = {n}, gcd = {sp.gcd(m,n)}")
    print("=" * 60)
    print()

    # Build P and Q with symbolic coefficients
    # P = y^m + P_{m-1} + ... + P_1 + P_0
    # Q = y^n + Q_{n-1} + ... + Q_1 + Q_0
    # With normalization: P_m = y^m, Q_n = y^n (leading forms)
    # Also: P_0 = 0, Q_0 = 0, P_1 = x, Q_1 = y (identity at linear order)
    # Actually, for generality, let P_1 = x + a*y, Q_1 = b*x + y

    all_coeffs = []
    P = y**m
    for k in range(m - 1, 1, -1):
        pk, ck = homogeneous_poly(k, f"p{k}")
        P = P + pk
        all_coeffs.extend(ck)

    # Linear and constant parts: P = ... + x, Q = ... + y
    P = P + x
    Q = y**n
    for k in range(n - 1, 1, -1):
        qk, ck = homogeneous_poly(k, f"q{k}")
        Q = Q + qk
        all_coeffs.extend(ck)
    Q = Q + y

    P = expand(P)
    Q = expand(Q)

    print(f"P = {P}")
    print(f"Q = {Q}")
    print(f"Number of free coefficients: {len(all_coeffs)}")
    print()

    # Compute Jacobian
    J = jacobian_det(P, Q)
    J = expand(J)

    # The Jacobian should be c (constant). So all non-constant terms = 0.
    # And the constant term = c.
    equations = []
    max_deg = m + n - 2

    for d in range(max_deg, 0, -1):
        hom_part = extract_homogeneous(J, d)
        if hom_part != 0:
            p = Poly(hom_part, x, y)
            for monom, coeff in p.as_dict().items():
                if coeff != 0:
                    equations.append(coeff)
                    print(f"  deg {d}, x^{monom[0]}*y^{monom[1]}: {coeff} = 0")

    # Check constant term
    const_term = J
    for c_var in all_coeffs:
        const_term = const_term.subs(c_var, 0)
    # Actually, easier: extract degree 0
    J_const = extract_homogeneous(J, 0)
    print(f"\n  Constant term of det J: {J_const} = {c_val}")
    if J_const != c_val:
        equations.append(J_const - c_val)

    print(f"\nTotal equations: {len(equations)}")
    print(f"Total unknowns: {len(all_coeffs)}")
    print()

    # Solve
    print("Solving system...")
    try:
        sol = solve(equations, all_coeffs, dict=True)
        print(f"Number of solution branches: {len(sol)}")
        for i, s in enumerate(sol):
            print(f"\n  Solution {i+1}:")
            free_params = [c for c in all_coeffs if c not in s]
            determined = {k: v for k, v in s.items() if v != 0}
            zero_vars = [k for k, v in s.items() if v == 0]

            print(f"    Free parameters: {free_params}")
            print(f"    Forced to zero: {zero_vars}")
            print(f"    Determined (nonzero): {determined}")

            # Check if all x-containing coefficients are zero
            all_x_zero = True
            for c_var in all_coeffs:
                name = str(c_var)
                # Coefficients with nonzero x-power (i > 0 in p{k}_{i}_{j})
                parts = name.split("_")
                if len(parts) == 3:
                    i_val = int(parts[1])
                    if i_val > 0:
                        val = s.get(c_var, c_var)
                        if val != 0 and not (isinstance(val, sp.Basic) and val.is_zero):
                            all_x_zero = False

            if all_x_zero:
                print("    ALL x-coefficients are zero -> P = P(y), Q = Q(y)")
                print("    -> det J = 0, CONTRADICTION with det J = c")
            else:
                print("    Some x-coefficients survive -> degree divisibility")
                print("    NOT forced by Jacobian condition alone")

    except Exception as e:
        print(f"  Solve failed: {e}")
        print("  System may be too complex for symbolic solver.")
        print("  Trying to solve degree by degree instead...")
        solve_incrementally(equations, all_coeffs, m, n)


def solve_incrementally(equations, all_coeffs, m, n):
    """Solve the system one degree at a time, top down."""
    print()
    print("INCREMENTAL SOLVE (degree by degree)")
    print("-" * 40)

    solved = {}
    remaining_coeffs = list(all_coeffs)

    # Group equations by degree (they come in order)
    max_deg = m + n - 2
    eq_idx = 0

    for d in range(max_deg, 0, -1):
        degree_eqs = []
        while eq_idx < len(equations):
            eq = equations[eq_idx]
            # Substitute already-solved values
            eq_sub = eq
            for var, val in solved.items():
                eq_sub = eq_sub.subs(var, val)
            eq_sub = expand(eq_sub)

            if eq_sub == 0:
                eq_idx += 1
                continue

            # Check if this equation is at the current degree level
            degree_eqs.append(eq_sub)
            eq_idx += 1

            # Simple heuristic: stop when we've processed enough equations
            if len(degree_eqs) >= d + 1:
                break

        if not degree_eqs:
            continue

        # Find variables in these equations
        eq_vars = set()
        for eq in degree_eqs:
            eq_vars.update(eq.free_symbols & set(remaining_coeffs))

        if not eq_vars:
            continue

        print(f"\n  Degree {d}: {len(degree_eqs)} equations, {len(eq_vars)} unknowns")
        for eq in degree_eqs:
            print(f"    {eq} = 0")

        try:
            sol = solve(degree_eqs, list(eq_vars), dict=True)
            if sol:
                s = sol[0]
                for var, val in s.items():
                    solved[var] = val
                    if val == 0:
                        print(f"    -> {var} = 0")
                    else:
                        print(f"    -> {var} = {val}")
        except:
            print("    (could not solve this batch)")


def main():
    print("=" * 70)
    print("SUBLEADING CASCADE ANALYSIS")
    print("Testing degree divisibility via full Jacobian expansion")
    print("=" * 70)
    print()

    # Case 1: (m,n) = (3,2) -- gcd = 1, small enough to fully solve
    full_cascade(3, 2)
    print()

    # Case 2: (m,n) = (5,3) -- gcd = 1, the critical test
    print()
    full_cascade(5, 3)
    print()

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("The full cascade analysis determines whether the Jacobian")
    print("condition det J(P,Q) = c, with deg P not dividing deg Q,")
    print("has nontrivial polynomial solutions or only the trivial")
    print("P = P(y), Q = Q(y) solution (which gives det J = 0).")
    print()
    print("If ONLY trivial: degree divisibility proved -> JC_2 proved.")
    print("If nontrivial: need another approach.")


if __name__ == "__main__":
    main()
