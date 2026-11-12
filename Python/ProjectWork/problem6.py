"""Problem 6: quintic joint trajectories through the circle knots j = 0, 9, 18, 27, 36 (T = 2 s per segment).

Each segment and joint gets q(t) = c0 + c1 t + ... + c5 t^5 with the knot position, joint speed
and zero acceleration at both ends. Prints the coefficients and checks the end conditions.
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # MyRobot.py is one folder up
import math
import numpy as np
from MyRobot import jacobian
from problem3 import qs

T = 2.0  # segment duration [s]
knots = [qs[j] for j in (0, 9, 18, 27, 36)]  # joint angles [deg]
tip_v = [(0, 0, 0), (0, -0.027, 0), (0, 0, -0.027), (0, 0.027, 0), (0, 0, 0)]  # tool velocity at each knot [m/s]


def joint_speeds(q, v):
    """Tool velocity (x, y, z) [m/s] -> joint speeds [deg/s], with zero pitch rate."""
    s, c = math.sin(math.radians(q[0])), math.cos(math.radians(q[0]))
    plane = [-v[0] * s + v[1] * c, v[0] * c + v[1] * s, v[2], 0]  # tangential, radial, vertical, pitch rate
    return np.degrees(np.linalg.solve(jacobian(*q), plane))


speeds = [joint_speeds(q, v) for q, v in zip(knots, tip_v)]

# End conditions at t = T = 2 for the unknowns (c3, c4, c5)
M = np.array([[8, 16, 32],      # position:     T^3,   T^4,    T^5
              [12, 32, 80],     # velocity:   3T^2,  4T^3,   5T^4
              [12, 48, 160]])   # acceleration: 6T,  12T^2,  20T^3

coeffs = np.zeros((4, 4, 6))  # [segment A-D, joint 1-4, c0..c5]
for seg in range(4):
    for joint in range(4):
        q0, qT = knots[seg][joint], knots[seg + 1][joint]
        v0, vT = speeds[seg][joint], speeds[seg + 1][joint]
        a0, aT = 0, 0
        # Start conditions fix c0..c2 directly; the end conditions give M · (c3, c4, c5) = b
        c0 = q0
        c1 = v0
        c2 = a0 / 2
        b = [qT - c0 - 2*c1 - 4*c2, vT - c1 - 4*c2, aT - 2*c2]
        c3, c4, c5 = np.linalg.solve(M, b)
        coeffs[seg, joint] = [c0, c1, c2, c3, c4, c5]

for seg in range(4):
    print("ABCD"[seg], np.round(coeffs[seg], 4))

# Check the end conditions at t = T (the start conditions hold by construction)
for seg in range(4):
    for joint in range(4):
        c = coeffs[seg, joint]
        end = (np.polyval(c[::-1], T), np.polyval(np.polyder(c[::-1]), T), np.polyval(np.polyder(c[::-1], 2), T))
        assert np.allclose(end, (knots[seg + 1][joint], speeds[seg + 1][joint], 0)), ("ABCD"[seg], joint + 1)
print("all 16 polynomials hit their end position, speed and acceleration")
