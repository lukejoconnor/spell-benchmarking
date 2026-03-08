import math

MOD = 10**9 + 7

def solve(N):
    trolls = []
    H0 = 0
    for n in range(N):
        h = (pow(5, 3*n, MOD) % 101) + 50
        l = (pow(5, 3*n+1, MOD) % 101) + 50
        q = (pow(5, 3*n+2, MOD) % 101) + 50
        trolls.append((h, l, q))
        H0 += h

    # D_N = H0 / sqrt(2)
    # Condition for troll to escape: cumH_before <= H0 - D_N + l = H0*(1-1/sqrt(2)) + l
    C = H0 * (1.0 - 1.0/math.sqrt(2))
    # Since cumH is integer and C is irrational, cumH <= C + l iff cumH <= floor(C) + l
    F = int(math.floor(C))

    # Sort by l + h (non-decreasing) - optimal escape ordering via exchange argument
    trolls.sort(key=lambda t: t[1] + t[0])

    # 0/1 knapsack DP
    # dp[cumH] = max total IQ achievable with cumulative escaped height = cumH
    max_lh = max(t[1] + t[0] for t in trolls)
    max_cumH = F + max_lh

    dp = [-1] * (max_cumH + 2)
    dp[0] = 0

    for h, l, q in trolls:
        deadline = F + l  # cumH must be <= deadline to include this troll
        upper = min(deadline, max_cumH - h)
        if upper < 0:
            continue
        for cumH in range(upper, -1, -1):
            if dp[cumH] >= 0:
                new_cumH = cumH + h
                new_q = dp[cumH] + q
                if dp[new_cumH] < new_q:
                    dp[new_cumH] = new_q

    return max(v for v in dp if v >= 0)

print(f"Q(5) = {solve(5)}")
print(f"Q(15) = {solve(15)}")
print(f"Q(1000) = {solve(1000)}")
