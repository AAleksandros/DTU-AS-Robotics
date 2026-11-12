"""Step-by-step check of the kinematics on the real arm. Each step waits for Enter.

Run: python lab_test.py   (Ctrl+C stops the arm where it is)
The motors stay on at the end so the arm doesn't fall. Hold the arm and switch the power off when done.
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # MyRobot.py is one folder up
from MyRobot import MyRobot, inverse_kinematics

robot = None


def step(title, expect, q):
    input(f"\n--- {title}\n    Expect: {expect}\n    Press Enter to move (Ctrl+C to stop) ")
    robot.move_j(*q)
    print("    commanded:", [round(v, 1) for v in q])
    print("    measured: ", [round(v, 1) for v in robot.read_joint_angles()])


try:
    input("Clear 30 cm around the arm, hand near the power switch. Press Enter to switch the motors on ")
    robot = MyRobot()  # moves slowly to (0, 0, 0, 0)
    if not robot.init_status:
        raise SystemExit("The robot didn't start or couldn't reach straight up: see the 'Initialization error' line above.")
    print("Should now be straight up.")

    # Joint directions
    step("1. Shoulder only", "the whole arm leans 30° FORWARD (the way the camera looks)", (0, -30, 0, 0))
    step("2. Base only", "the arm turns 30° to ITS LEFT (anticlockwise seen from above)", (30, -30, 0, 0))
    step("3. Back to straight up", "straight up", (0, 0, 0, 0))

    # IK pose inside the joint limits (link 4 straight down would need q4 = -60°, past the wrist stop at -29°)
    q = inverse_kinematics(0.130, 0.075, 0.100, -30)
    step("4. IK test pose (30, -9.5, -90.9, -19.6)",
         "base 30° left, link 4 points 30° below horizontal. Wrist axle about 125 mm above the base plate "
         "and about 107 mm out from the base axis (Figure 1 lengths; the real arm differs by a few mm)", q)

    from problem3 import qs
    input("\n--- 5. Problem 3 circle\n    Expect: link 4 level, its end traces a 64 mm upright circle "
          "150 mm in front of the base (left, top, right, bottom)\n    Press Enter to start ")
    for j, qj in enumerate(qs):
        robot.move_j(*qj)
        print(f"    point {j:2d} done", end="\r", flush=True)
    print()

    step("6. Finish", "straight up, and it stays there: the motors keep holding it", (0, 0, 0, 0))
except KeyboardInterrupt:
    print("\nStopped. The arm holds its current pose.")
finally:
    # Torque stays on so the arm doesn't collapse; only the port is closed
    if robot is not None and robot.init_status:
        robot.port_handler.closePort()
        print("Motors still on and holding. Hold the arm, then switch the robot's power off when you're done.")
