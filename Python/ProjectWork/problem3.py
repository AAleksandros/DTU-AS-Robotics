"""Problem 3: joint configurations for 37 points on the circle p(φ) = (150, 32 cos φ, 120 + 32 sin φ) mm.

The stylus is kept horizontal (pitch = 0). Importing this module gives `qs` (used by Problem 6
and the lab scripts); running it prints the table.
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # MyRobot.py is one folder up
import math
from MyRobot import inverse_kinematics

R = 0.032                       # circle radius [m]
xc, yc, zc = 0.150, 0.0, 0.120  # circle centre [m]

qs = []                         # joint angles [deg] for φ_j = 2πj/36, j = 0..36
for j in range(37):
    phi = 2 * math.pi * j / 36
    p = (xc, yc + R * math.cos(phi), zc + R * math.sin(phi))
    qs.append(inverse_kinematics(*p, 0))

if __name__ == "__main__":
    for j, q in enumerate(qs):
        print(j, [round(v, 2) for v in q])
