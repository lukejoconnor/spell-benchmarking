from decimal import Decimal, getcontext

def solve(N):
    MOD = 10**9 + 7

    r = []
    power = 1
    for n in range(3 * N):
        r.append((power % 101) + 50)
        power = (power * 5) % MOD

    trolls = []
    H_total = 0
    for n in range(N):
        h = r[3*n]
        l = r[3*n + 1]
        q = r[3*n + 2]
        trolls.append((h, l, q))
        H_total += h

    getcontext().prec = 50
    sqrt2 = Decimal(2).sqrt()
    T = Decimal(H_total) * (1 - 1/sqrt2)
    F = int(T)

    trolls.sort(key=lambda x: x[0] + x[1])

    max_s = H_total
    dp = [-1] * (max_s + 200)
    dp[0] = 0

    for h, l, q in trolls:
        cap = F + l
        for s in range(min(cap, max_s), -1, -1):
            if dp[s] >= 0:
                ns = s + h
                val = dp[s] + q
                if dp[ns] < val:
                    dp[ns] = val

    return max(dp)

print(f"Q(5) = {solve(5)}")
print(f"Q(15) = {solve(15)}")
print(f"Q(1000) = {solve(1000)}")
