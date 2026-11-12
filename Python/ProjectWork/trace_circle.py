"""Trace the Problem 3 circle point by point on the real arm.

Run: python trace_circle.py   (Ctrl+C stops early)
The motors are switched off at the end, so hold the arm.
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # MyRobot.py is one folder up
from MyRobot import MyRobot
from problem3 import qs

robot = MyRobot()  # moves slowly to (0, 0, 0, 0) = straight up
try:
    for j, q in enumerate(qs):
        print(f"point {j:2d}: q = {[round(v, 1) for v in q]}")
        robot.move_j(*q)
    robot.move_j(0, 0, 0, 0)
finally:
    robot.disable_motors()
