from collections import defaultdict
import math

def brute_force_set(N):
    """Return set of all M with 2+ square roots and trace < N."""
    seen = defaultdict(set)
    for p in range(1, N):
        if p*p + 3 >= N:
            break
        for s in range(1, N):
            ps_sq = p*p + s*s
            if ps_sq + 2 >= N:
                break
            max_qr = (N - 1 - ps_sq) // 2
            t = p + s
            for q in range(1, max_qr + 1):
                max_r = max_qr // q
                for r in range(1, max_r + 1):
                    T = ps_sq + 2*q*r
                    if T >= N:
                        break
                    M = (p*p + q*r, q*t, r*t, s*s + q*r)
                    seen[M].add((p,q,r,s))
    return {M for M, roots in seen.items() if len(roots) >= 2}

def formula_set(N):
    """Return set of all M from the analytic formula."""
    max_n = N // 10 + 2
    result = set()
    limit = 2 * N

    for alpha in range(2, int(math.isqrt(limit)) + 2):
        a2 = alpha * alpha
        if a2 + 1 >= limit:
            break
        for beta in range(1, alpha):
            b2 = beta * beta
            s = a2 + b2
            if s >= limit:
                break
            if math.gcd(alpha, beta) != 1:
                continue
            both_odd = (alpha & 1) and (beta & 1)
            diff = alpha - beta
            max_g = int(math.isqrt((limit - 1) // s))
            while max_g >= 1 and max_g * max_g * s >= limit:
                max_g -= 1
            if max_g < 1:
                continue
            for g in range(1, max_g + 1):
                if (g & 1) and not both_odd:
                    continue
                num = g * diff + 2
                denom = 2 * alpha
                m_min = (num + denom - 1) // denom
                m_max = g >> 1
                for m in range(m_min, m_max + 1):
                    n_val = m * (g - m)
                    if n_val <= 0:
                        continue
                    e = g - 2 * m
                    # Compute M entries
                    T = g*g * s // 2
                    ab_e = alpha * beta * e
                    a_entry = g * (g * s + 2 * ab_e) // 4
                    d_entry = g * (g * s - 2 * ab_e) // 4
                    # b = g*alpha*beta*B, c = g*alpha*beta*C
                    # BC = m*(g-m)
                    gab = g * alpha * beta
                    for B in range(1, n_val + 1):
                        if n_val % B == 0:
                            C = n_val // B
                            b_entry = gab * B
                            c_entry = gab * C
                            M = (a_entry, b_entry, c_entry, d_entry)
                            result.add(M)
    return result

N = 200
bf = brute_force_set(N)
fm = formula_set(N)

print(f"Brute force: {len(bf)}")
print(f"Formula: {len(fm)}")
print(f"In BF but not formula: {len(bf - fm)}")
print(f"In formula but not BF: {len(fm - bf)}")

missing = sorted(bf - fm, key=lambda M: M[0]+M[3])
for M in missing[:20]:
    print(f"  Missing: M={M}, trace={M[0]+M[3]}")
