"""
Project Euler 311: Biclinic Integral Quadrilaterals

Place O at origin, B=(-d,0), D=(d,0) where BD=2d (must be even).
A=(ax,ay) above x-axis, C=(cx,cy) below, both at distance r=AO=CO from O.

Side lengths:
  AB^2 = r^2 + d^2 + 2*d*ax
  AD^2 = r^2 + d^2 - 2*d*ax
  BC^2 = r^2 + d^2 + 2*d*cx
  CD^2 = r^2 + d^2 - 2*d*cx

So AB^2+AD^2 = BC^2+CD^2 = 2(r^2+d^2), and AB^2+BC^2+CD^2+AD^2 = 4(r^2+d^2).

Let S = r^2 + d^2. Constraint: 4S <= N, i.e., S <= N/4.

Representations of S as a^2+b^2 with a>b>=1 correspond (via bijection) to
representations of 2S as p^2+q^2 with 1<=p<q (the side lengths).
Bijection: p=a-b, q=a+b.

For each S:
  h(S) = #{(a,b): a^2+b^2=S, a>b>=1}  (number of "side" representation pairs)
  g(S) = #{(r,d): r^2+d^2=S, 1<=r<=d} = h(S) + [S=2k^2]  (number of (r,d) pairs)

Each pair of side representations with the (r,d) choice gives a distinct quadrilateral.
Contribution of S: g(S) * C(h(S), 2) = g(S) * h(S) * (h(S)-1) / 2

Total B(N) = sum over S of g(S) * C(h(S), 2).
"""

import math
from collections import Counter

def solve_small(N):
    """Brute force for small N - verification."""
    N4 = N // 4
    counts = Counter()
    max_a = math.isqrt(N4)
    for a in range(2, max_a + 1):
        for b in range(1, a):
            S = a * a + b * b
            if S > N4:
                break
            counts[S] += 1
    total = 0
    for S, h in counts.items():
        if h < 2:
            continue
        eps2 = 0
        if S % 2 == 0:
            half = S // 2
            k = math.isqrt(half)
            if k * k == half:
                eps2 = 1
        g = h + eps2
        total += g * h * (h - 1) // 2
    return total

print(f"B(10000) = {solve_small(10000)}")
print(f"B(1000000) = {solve_small(1000000)}")
