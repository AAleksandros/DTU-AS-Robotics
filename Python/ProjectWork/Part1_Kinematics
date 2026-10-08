import sympy as sp
import numpy as np

# Problem 1: Forward kinematics of the 4-joint arm.

q1, q2, q3, q4 = sp.symbols("q1 q2 q3 q4")


def A(theta, d, a, alpha):
    ct, st = sp.cos(theta), sp.sin(theta)
    ca, sa = sp.cos(alpha), sp.sin(alpha)
    return sp.Matrix([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1],
    ])


A1 = A(q1, 50, 0, sp.pi / 2)
A2 = A(q2 + sp.pi / 2, 0, 93, 0)
A3 = A(q3, 0, 93, 0)
A4 = A(q4, 0, 50, 0)

T04 = sp.trigsimp(A1 * A2 * A3 * A4)

# Camera: pure translation, not an additional DH row.
T45 = sp.Matrix([
    [1, 0, 0, -15],
    [0, 1, 0, 45],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])
T05 = sp.trigsimp(T04 * T45)

for index, matrix in enumerate((A1, A2, A3, A4), start=1):
    print(f"A{index} =\n{matrix}\n")

print(f"T04 =\n{T04}\n")
print(f"T45 =\n{T45}\n")
print(f"T05 =\n{T05}\n")

# Null-position check: o4=(0, 0, 286), o5=(-45, 0, 271).
zero = {q1: 0, q2: 0, q3: 0, q4: 0}
print("o4 at zero position:", T04.subs(zero)[:3, 3].T)
print("o5 at zero position:", T05.subs(zero)[:3, 3].T)


"""
Problem 2: Inverse kinematics of the 4-joint arm.

Given   x_z : z-component of x_4^0   (= cos(q2+q3+q4))
        p   : stylus tip position o_4^0 = [px, py, pz] in mm
Find    q = [q1, q2, q3, q4]  (rad)

All positions are in mm, all angles in rad.
"""
# Link dimensions from Figure 1 (mm)
L1, L2, L3, L4 = 50.0, 93.0, 93.0, 50.0


def wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- forward kinematics
def dh(theta, d, a, alpha):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca,  st * sa, a * ct],
                     [st,  ct * ca, -ct * sa, a * st],
                     [0,   sa,       ca,      d],
                     [0,   0,        0,       1]])


def fk(q):
    """T_4^0 from the DH table of Problem 1."""
    q1, q2, q3, q4 = q
    return (dh(q1,           L1, 0,  np.pi / 2) @
            dh(q2 + np.pi / 2, 0, L2, 0) @
            dh(q3,            0, L3, 0) @
            dh(q4,            0, L4, 0))


# ---------------------------------------------------------------- inverse kinematics
def ik(xz, p, tol=1e-9):
    """Return ALL analytic solutions (up to 8) as an (n,4) array."""
    px, py, pz = p
    if abs(xz) > 1 + tol:
        raise ValueError("|x_z| must be <= 1")
    xz = np.clip(xz, -1.0, 1.0)

    # q1: two branches (q1 and q1+pi); undefined if px = py = 0
    q1_base = np.arctan2(py, px) if np.hypot(px, py) > tol else 0.0

    sols = []
    for q1 in (q1_base, q1_base + np.pi):
        c1, s1 = np.cos(q1), np.sin(q1)
        u = -(c1 * px + s1 * py)          # horizontal coordinate in arm plane
        v = pz - L1                       # vertical coordinate in arm plane

        for phi in (np.arccos(xz), -np.arccos(xz)):      # phi = q2+q3+q4
            # wrist point: remove last link
            uw = u - L4 * np.sin(phi)
            vw = v - L4 * np.cos(phi)

            c3 = (uw**2 + vw**2 - L2**2 - L3**2) / (2 * L2 * L3)
            if abs(c3) > 1 + tol:
                continue                  # not reachable with this phi
            c3 = np.clip(c3, -1.0, 1.0)

            for sgn in (+1, -1):          # two elbow configurations
                q3 = np.arctan2(sgn * np.sqrt(1 - c3**2), c3)
                q2 = (np.arctan2(uw, vw)
                      - np.arctan2(L3 * np.sin(q3), L2 + L3 * np.cos(q3)))
                q4 = phi - q2 - q3
                sols.append(wrap([q1, q2, q3, q4]))
    return np.array(sols).reshape(-1, 4)


def pick(sols, q1_max=np.pi / 2, elbow="up"):
    """Select one solution:
       - |q1| <= q1_max            (arm works in front of the base)
       - stylus points away from the base (x_4 . radial direction >= 0)
       - elbow 'up' (higher elbow joint) or 'down'
    """
    cand = []
    for q in sols:
        if abs(q[0]) > q1_max + 1e-9:
            continue
        T = fk(q)
        radial = T[:2, 3]
        if T[:2, 0] @ radial < -1e-9:
            continue
        cand.append(q)
    if not cand:
        return None
    height = [np.cos(q[1]) for q in cand]          # elbow height ~ L1 + L2 cos(q2)
    return cand[int(np.argmax(height) if elbow == "up" else np.argmin(height))]


# ---------------------------------------------------------------- demo / self-test
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    # 1) Round trip: q -> FK -> IK must contain q
    q_test = np.deg2rad([20, -30, 50, -40])
    T = fk(q_test)
    sols = ik(T[2, 0], T[:3, 3])
    err = np.min(np.linalg.norm(wrap(sols - q_test), axis=1))
    print("Round-trip: nearest solution error [rad] =", err)

    # 2) Check that every returned solution reproduces o4 and x_z
    for q in sols:
        Tq = fk(q)
        assert np.allclose(Tq[:3, 3], T[:3, 3], atol=1e-6)
        assert np.isclose(Tq[2, 0], T[2, 0], atol=1e-9)
    print(f"All {len(sols)} solutions satisfy position and x_z.\n")

    # 3) Circle of Problem 3 (R = 32 mm, pc = [150, 0, 120], stylus horizontal => x_z = 0)
    R, pc = 32.0, np.array([150.0, 0.0, 120.0])
    omega = np.linspace(0, 2 * np.pi, 37)
    Q = []
    for w in omega:
        p = pc + R * np.array([0.0, np.cos(w), np.sin(w)])
        q = pick(ik(0.0, p), elbow="up")
        Q.append(q)
    Q = np.array(Q)
    print("j   omega[deg]   q1..q4 [deg]")
    for j in (0, 9, 18, 27, 36):
        print(f"{j:2d}  {np.rad2deg(omega[j]):7.1f}   {np.rad2deg(Q[j])}")
    P = np.array([fk(q)[:3, 3] for q in Q])
    P_des = pc + R * np.c_[np.zeros(37), np.cos(omega), np.sin(omega)]
    print("\nmax position error on circle [mm]:", np.abs(P - P_des).max())
    print("max |x_z| over circle:", max(abs(fk(q)[2, 0]) for q in Q))