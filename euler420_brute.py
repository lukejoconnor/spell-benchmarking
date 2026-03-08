from collections import defaultdict
import math

def brute_force(N):
    """Brute force: enumerate all P, compute M=P^2, count M with 2+ roots."""
    seen = defaultdict(set)

    for p in range(1, N):
        if p*p + 3 >= N:  # min trace when s=1,q=r=1: p^2+1+2
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

    count = 0
    examples = []
    for M, roots in seen.items():
        if len(roots) >= 2:
            count += 1
            if M[0] + M[3] < 50:
                examples.append((M, roots))

    examples.sort()
    for M, roots in examples:
        print(f"  M={M}, trace={M[0]+M[3]}, roots={roots}")

    return count

print(f"Brute F(50) = {brute_force(50)}")
print(f"Brute F(1000) = {brute_force(1000)}")
