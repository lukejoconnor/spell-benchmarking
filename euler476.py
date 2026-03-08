import numpy as np
import math
import time

def R_greedy(a, b, c):
    """Compute R(a,b,c) using greedy approach: incircle + 2 largest gap circles."""
    s = (a + b + c) / 2.0
    area = math.sqrt(s * (s-a) * (s-b) * (s-c))
    r = area / s

    # Angles
    cos_A = (b**2 + c**2 - a**2) / (2*b*c)
    cos_B = (a**2 + c**2 - b**2) / (2*a*c)
    cos_C = (a**2 + b**2 - c**2) / (2*a*b)

    sin_half_A = math.sqrt(max((1 - cos_A) / 2, 0))
    sin_half_B = math.sqrt(max((1 - cos_B) / 2, 0))
    sin_half_C = math.sqrt(max((1 - cos_C) / 2, 0))

    rho_A = r * (1 - sin_half_A) / (1 + sin_half_A)
    rho_B = r * (1 - sin_half_B) / (1 + sin_half_B)
    rho_C = r * (1 - sin_half_C) / (1 + sin_half_C)

    # Sort descending, take top 2
    rhos = sorted([rho_A, rho_B, rho_C], reverse=True)

    return math.pi * (r**2 + rhos[0]**2 + rhos[1]**2)

def R_greedy_all3(a, b, c):
    """Greedy: pick incircle + best 2 of 3 gap circles (checking all combos)."""
    s = (a + b + c) / 2.0
    area = math.sqrt(s * (s-a) * (s-b) * (s-c))
    r = area / s

    cos_A = (b**2 + c**2 - a**2) / (2*b*c)
    cos_B = (a**2 + c**2 - b**2) / (2*a*c)
    cos_C = (a**2 + b**2 - c**2) / (2*a*b)

    sin_half_A = math.sqrt(max((1 - cos_A) / 2, 0))
    sin_half_B = math.sqrt(max((1 - cos_B) / 2, 0))
    sin_half_C = math.sqrt(max((1 - cos_C) / 2, 0))

    rho_A = r * (1 - sin_half_A) / (1 + sin_half_A)
    rho_B = r * (1 - sin_half_B) / (1 + sin_half_B)
    rho_C = r * (1 - sin_half_C) / (1 + sin_half_C)

    # Also consider: 3 gap circles (no incircle) - Malfatti-like
    # And: incircle + 1 gap circle + sub-gap circle
    # For now, try all combos of incircle + 2 gap circles
    best = 0
    for r1, r2 in [(rho_A, rho_B), (rho_A, rho_C), (rho_B, rho_C)]:
        val = math.pi * (r**2 + r1**2 + r2**2)
        best = max(best, val)

    # Also try 3 gap circles (one at each vertex, no incircle)
    val3 = math.pi * (rho_A**2 + rho_B**2 + rho_C**2)
    # But wait, these gap circles are defined relative to the incircle...
    # Without the incircle, the gap circles would be different (larger)

    return best


def malfatti_radii(a, b, c):
    """Compute Malfatti circle radii for triangle with sides a, b, c."""
    s = (a + b + c) / 2.0
    area = math.sqrt(s * (s-a) * (s-b) * (s-c))
    r = area / s  # inradius

    A = math.acos(max(-1, min(1, (b**2 + c**2 - a**2) / (2*b*c))))
    B = math.acos(max(-1, min(1, (a**2 + c**2 - b**2) / (2*a*c))))
    C = math.pi - A - B

    # Malfatti circle radii (using the standard formulas)
    # References: various sources give these formulas
    # Using the formulation with IB, IC, IA (distances from incenter to vertices)
    d = 1.0 / (2.0 * (math.cos(A/2) + math.cos(B/2) + math.cos(C/2)) - 1)
    # Actually, let me use a known formula.
    # From Weisstein / Mathworld:
    # r_1 = (r / (2 * s)) * (s - a) * sec^2(A/2) / (1 + ...)
    # This is getting complicated. Let me use a different approach.

    # From the formulas given by Malfatti:
    # Let's use the Ajima-Malfatti points approach
    # Actually, let me use the explicit formulas from:
    # https://en.wikipedia.org/wiki/Malfatti_circles

    # The radii are:
    # r_i = r * (1 - tan(alpha_j/2))(1 - tan(alpha_k/2)) / (... )
    # Actually the Wikipedia formulas involve IB, IC (exradii or something)

    # Let me compute using the formula involving half-angle tangents
    ta = math.tan(A/2)
    tb = math.tan(B/2)
    tc = math.tan(C/2)

    # From Steiner's construction, the Malfatti radii can be computed as:
    # This is complex. Let me use a numerical approach instead.
    return None


def R_numerical(a, b, c):
    """
    Numerically optimize 3 circle packing in triangle using scipy.
    Each circle: (x, y, radius). Maximize pi*(r1^2 + r2^2 + r3^2).
    """
    from scipy.optimize import minimize

    # Place triangle with vertices
    # A = (0, 0), B = (c, 0), C = (x_C, y_C)
    x_C = (a**2 - b**2 + c**2) / (2*c)  # wait, let me redo
    # Side a is opposite vertex A, side b opposite B, side c opposite C
    # Vertices: A opposite side a (between B and C)
    # Let B = (0,0), C = (a, 0) (side a = BC)
    # Then AB = c, AC = b
    # B at origin, C at (a, 0)
    # A at (x, y) where |AB| = c, |AC| = b
    # x^2 + y^2 = c^2
    # (x-a)^2 + y^2 = b^2
    # x^2 - 2ax + a^2 + y^2 = b^2
    # c^2 - 2ax + a^2 = b^2
    # x = (a^2 + c^2 - b^2) / (2a)
    # y = sqrt(c^2 - x^2)

    Bx, By = 0.0, 0.0
    Cx, Cy = float(a), 0.0
    Ax = (a**2 + c**2 - b**2) / (2.0 * a)
    Ay = math.sqrt(max(c**2 - Ax**2, 0))

    # Triangle edges as (nx, ny, d) where nx*x + ny*y + d >= 0 means inside
    # Edge BC: from (0,0) to (a,0), normal pointing inward (upward): (0, 1, 0)
    # Edge CA: from (a,0) to (Ax, Ay)
    # Edge AB: from (Ax, Ay) to (0, 0)

    def edge_normal(p1, p2, p_inside):
        """Return (nx, ny, d) such that nx*x + ny*y + d >= 0 for inside points."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        # Normal: (-dy, dx) or (dy, -dx)
        nx, ny = -dy, dx
        # Check which direction
        if nx * p_inside[0] + ny * p_inside[1] + (-(nx*p1[0] + ny*p1[1])) >= 0:
            d = -(nx*p1[0] + ny*p1[1])
        else:
            nx, ny = dy, -dx
            d = -(nx*p1[0] + ny*p1[1])
        # Normalize
        norm = math.sqrt(nx**2 + ny**2)
        return nx/norm, ny/norm, d/norm

    # Incenter as reference inside point
    ss = (a + b + c) / 2.0
    incenter_x = (a * Ax + b * Bx + c * Cx) / (a + b + c)  # wait, weighted by opposite sides
    # Incenter = (a*Ax + b*Bx + c*Cx) / (a+b+c) where a,b,c are side lengths opposite vertices A,B,C
    incenter_x = (a * Ax + b * Bx + c * Cx) / (a + b + c)
    incenter_y = (a * Ay + b * By + c * Cy) / (a + b + c)
    inside_pt = (incenter_x, incenter_y)

    edges = [
        edge_normal((Bx, By), (Cx, Cy), inside_pt),  # BC
        edge_normal((Cx, Cy), (Ax, Ay), inside_pt),  # CA
        edge_normal((Ax, Ay), (Bx, By), inside_pt),  # AB
    ]

    def signed_dist_to_edge(x, y, edge):
        nx, ny, d = edge
        return nx*x + ny*y + d

    # Optimization: minimize negative total area
    # Variables: x1, y1, r1, x2, y2, r2, x3, y3, r3
    inr = math.sqrt(ss * (ss-a) * (ss-b) * (ss-c)) / ss

    def objective(params):
        r1, r2, r3 = params[2], params[5], params[8]
        return -(r1**2 + r2**2 + r3**2)

    def constraint_inside(params, i, j):
        """Circle i is inside edge j."""
        x, y, r = params[3*i], params[3*i+1], params[3*i+2]
        return signed_dist_to_edge(x, y, edges[j]) - r

    def constraint_no_overlap(params, i, k):
        """Circles i and k don't overlap."""
        x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
        x2, y2, r2 = params[3*k], params[3*k+1], params[3*k+2]
        return math.sqrt((x1-x2)**2 + (y1-y2)**2) - r1 - r2

    def constraint_positive_r(params, i):
        return params[3*i+2]

    constraints = []
    for i in range(3):
        for j in range(3):
            constraints.append({'type': 'ineq', 'fun': constraint_inside, 'args': (i, j)})
        constraints.append({'type': 'ineq', 'fun': constraint_positive_r, 'args': (i,)})
    for i in range(3):
        for k in range(i+1, 3):
            constraints.append({'type': 'ineq', 'fun': constraint_no_overlap, 'args': (i, k)})

    # Try multiple starting points
    best = 0
    # Start 1: incircle + two gap circles
    # Compute gap circles
    A_angle = math.acos(max(-1, min(1, (b**2 + c**2 - a**2) / (2*b*c))))
    B_angle = math.acos(max(-1, min(1, (a**2 + c**2 - b**2) / (2*a*c))))
    C_angle = math.pi - A_angle - B_angle

    # Incircle center
    ix, iy = incenter_x, incenter_y

    # Gap circle at vertex A (angle A_angle)
    sin_ha = math.sin(A_angle/2)
    rho_A = inr * (1 - sin_ha) / (1 + sin_ha)
    # Center on bisector from A toward incenter
    dist_A_to_I = math.sqrt((Ax - ix)**2 + (Ay - iy)**2)
    dir_x = (ix - Ax) / dist_A_to_I
    dir_y = (iy - Ay) / dist_A_to_I
    gap_A_x = Ax + dir_x * rho_A / math.sin(A_angle/2)
    gap_A_y = Ay + dir_y * rho_A / math.sin(A_angle/2)

    # Gap circle at vertex B
    sin_hb = math.sin(B_angle/2)
    rho_B = inr * (1 - sin_hb) / (1 + sin_hb)
    dist_B_to_I = math.sqrt((Bx - ix)**2 + (By - iy)**2)
    dir_x = (ix - Bx) / dist_B_to_I
    dir_y = (iy - By) / dist_B_to_I
    gap_B_x = Bx + dir_x * rho_B / math.sin(B_angle/2)
    gap_B_y = By + dir_y * rho_B / math.sin(B_angle/2)

    # Gap circle at vertex C
    sin_hc = math.sin(C_angle/2)
    rho_C = inr * (1 - sin_hc) / (1 + sin_hc)
    dist_C_to_I = math.sqrt((Cx - ix)**2 + (Cy - iy)**2)
    dir_x = (ix - Cx) / dist_C_to_I
    dir_y = (iy - Cy) / dist_C_to_I
    gap_C_x = Cx + dir_x * rho_C / math.sin(C_angle/2)
    gap_C_y = Cy + dir_y * rho_C / math.sin(C_angle/2)

    starts = []

    # Incircle + 2 largest gap circles
    gaps = [(rho_A, gap_A_x, gap_A_y), (rho_B, gap_B_x, gap_B_y), (rho_C, gap_C_x, gap_C_y)]
    gaps_sorted = sorted(gaps, key=lambda g: g[0], reverse=True)

    starts.append([
        ix, iy, inr,
        gaps_sorted[0][1], gaps_sorted[0][2], gaps_sorted[0][0],
        gaps_sorted[1][1], gaps_sorted[1][2], gaps_sorted[1][0],
    ])

    # All 3 gap circles (no incircle)
    starts.append([
        gap_A_x, gap_A_y, rho_A,
        gap_B_x, gap_B_y, rho_B,
        gap_C_x, gap_C_y, rho_C,
    ])

    # Incircle + largest gap + smallest gap
    starts.append([
        ix, iy, inr,
        gaps_sorted[0][1], gaps_sorted[0][2], gaps_sorted[0][0],
        gaps_sorted[2][1], gaps_sorted[2][2], gaps_sorted[2][0],
    ])

    for x0 in starts:
        try:
            res = minimize(objective, x0, method='SLSQP', constraints=constraints,
                         options={'maxiter': 500, 'ftol': 1e-14})
            val = -res.fun * math.pi
            if val > best:
                best = val
        except:
            pass

    return best


# Enumerate S(5) triangles
triangles_5 = []
for a in range(1, 5):
    for b in range(a, 5):
        if a + b > 5:
            continue
        for c in range(b, a + b):
            if a + b <= 5:
                triangles_5.append((a, b, c))

print("S(5) triangles:")
total = 0
for t in triangles_5:
    a, b, c = t
    r_greedy = R_greedy(a, b, c)
    r_num = R_numerical(a, b, c)
    print(f"  ({a},{b},{c}): greedy={r_greedy:.6f}, numerical={r_num:.6f}, diff={r_num-r_greedy:.6f}")
    total += r_num

print(f"\nS(5) numerical = {total/len(triangles_5):.5f} (expected ~1.25899)")
print(f"S(5) greedy = {sum(R_greedy(*t) for t in triangles_5)/len(triangles_5):.5f}")
