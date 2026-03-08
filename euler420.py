import math

def solve(N):
    max_n = N // 10 + 2
    num_div = [0] * (max_n + 1)
    for i in range(1, max_n + 1):
        for j in range(i, max_n + 1, i):
            num_div[j] += 1

    total = 0
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
                    n = m * (g - m)
                    if n > 0:
                        if n <= max_n:
                            dn = num_div[n]
                        else:
                            dn = 0
                            sq = int(math.isqrt(n))
                            for i in range(1, sq + 1):
                                if n % i == 0:
                                    dn += 2 if i * i != n else 1
                        # e = g - 2m; if e > 0 (m < g/2), mirror matrix exists
                        if m < m_max or (g & 1):
                            # e > 0: count matrix and its mirror
                            total += 2 * dn
                        else:
                            # e = 0 (m = g/2, g even): a = d, no mirror needed
                            total += dn

    return total

print(f"F(50) = {solve(50)}")
print(f"F(1000) = {solve(1000)}")
print(f"F(10^7) = {solve(10**7)}")
