from collections import defaultdict

def brute_force_detail(N):
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
    return seen

seen = brute_force_detail(200)
target = (33, 96, 12, 57)
if target in seen:
    print(f"M={target}, roots={seen[target]}")

target2 = (28, 72, 24, 76)
if target2 in seen:
    print(f"M={target2}, roots={seen[target2]}")
