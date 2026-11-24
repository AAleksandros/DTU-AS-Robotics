"""Problem 11a, fast version 3: like type_hello_world_fast2.py (one photo, the slow version's press),
but quicker:

1. Between keys the motors go faster (TRAVEL_SPEED). Going down and pressing is done at the slow
   version's speed (PRESS_SPEED = 0.1) and with its own press() function and press time, unchanged.
2. Between letters it does NOT go back to the look pose. It lifts to HOVER_MM above the keys, moves
   across to above the next key at that height and goes down from there. With HOVER_MM = 50 the
   stylus stays at least ~18 mm above the keys while it moves across (60 mm: ~27 mm; 40 mm: only ~7 mm).
   These in-between moves don't need precision, so they skip move_exact's corrections.

The press itself (30 mm above the key with move_exact, press, back up) is the slow version's own
press() function, unchanged.

Usage (from the Python/vision folder):
    python type_hello_world_fast3.py
    python type_hello_world_fast3.py "hello"
"""
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))              # MyRobot.py is one folder up
from MyRobot import MyRobot, inverse_kinematics           # noqa: E402
from red_detector import open_camera                      # noqa: E402
from calibrate_homography import CALIB_FILE, grab_fresh_frame   # noqa: E402
import click_touch                                        # noqa: E402
from click_touch import pixel_to_base, plan_touch, within_limits   # noqa: E402
from arm import move_exact, send, wait_until_still        # noqa: E402
from detect_keys import load_templates, detect_keys, draw  # noqa: E402
import type_hello_world as slow                           # noqa: E402  (only read, never changed on disk)

# ------------------------------ Settings ---------------------------------------
TRAVEL_SPEED = 0.2            # motor speed 0..1 for the moves BETWEEN keys (MyRobot default 0.1)
PRESS_SPEED = 0.1             # motor speed for going down and pressing: same as the slow version
HOVER_MM = 50.0               # height above the keys while moving across (stylus stays ~18 mm above them)
SAVE_DIR = os.path.join(HERE, "..", "images", time.strftime("typing_fast3_%H%M%S"))


def hover_pose(x, y, z, pitch, limits):
    """Joint angles HOVER_MM above (x, y, z) with the same stylus angle as the press. None if out of limits."""
    try:
        q = inverse_kinematics(x / 1000, y / 1000, (z + HOVER_MM) / 1000, pitch)
    except ValueError:
        return None
    return np.asarray(q, dtype=float) if within_limits(q, limits) else None


def go(robot, q):
    """Plain move (no corrections) and wait until the arm stops."""
    send(robot, q)
    wait_until_still(robot)


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else slow.TEXT
    calib = np.load(CALIB_FILE)
    H, z_key = calib["H"], float(calib["z_keys"])
    look_pose, camera_index = tuple(calib["look_pose"]), int(calib["camera_index"])
    templates, space_rule = load_templates()
    needed = {slow.key_name(c) for c in text}
    os.makedirs(SAVE_DIR, exist_ok=True)

    robot = MyRobot()                                     # connects and moves slowly to straight up
    if not robot.init_status:
        sys.exit("Robot did not start (see 'Initialization error' above).")
    cam = open_camera(camera_index)
    click_touch.TOUCH_DEPTH_MM = slow.PRESS_DEPTH_MM      # same press depth as the slow version

    # 1. Preview, then ONE fresh photo
    move_exact(robot, look_pose)
    while True:
        if not slow.preview(cam, templates, space_rule, needed):
            cam.release()
            robot.port_handler.closePort()
            sys.exit("Cancelled.")
        frame = grab_fresh_frame(cam)
        found, report = detect_keys(frame, templates, space_rule)
        cv2.imwrite(os.path.join(SAVE_DIR, "photo.jpg"), draw(frame, found, templates))
        print("\n".join(report))
        if needed <= set(found):
            break
        print(f"  missing {sorted(needed - set(found))} in the photo, back to the preview")
    cam.release()

    # 2. Plan every key once: press (as the slow version) + hover point above it
    plans = {}
    for key in needed:
        x, y = pixel_to_base(*found[key], H)
        plan = plan_touch(x, y, z_key, robot.joint_limits)
        if plan is None:
            robot.port_handler.closePort()
            sys.exit(f"Stopping: '{key}' at ({x:.1f}, {y:.1f}) mm can't be reached within the joint limits.")
        q_above, q_press, pitch = plan
        plans[key] = (q_above, q_press, hover_pose(x, y, z_key, pitch, robot.joint_limits))
        print(f"  {key}: ({x:.1f}, {y:.1f}) mm, pitch {pitch} deg")

    for s in range(slow.COUNTDOWN_S, 0, -1):
        print(f"Click inside the text editor now... {s}")
        time.sleep(1)

    # 3. Type: hover above the key -> press (slow version's press) -> back up to hover -> next key
    try:
        for step, char in enumerate(text, start=1):
            key = slow.key_name(char)
            q_above, q_press, q_hover = plans[key]
            print(f"\n[{step}/{len(text)}] '{key}'")
            robot.set_speed([TRAVEL_SPEED] * 4)          # fast across
            if q_hover is None:                           # can't hover there: go via the look pose instead
                move_exact(robot, look_pose)
            else:
                go(robot, q_hover)                        # across, high above the keys
            robot.set_speed([PRESS_SPEED] * 4)           # slow, exactly like the slow version, for the press
            slow.press(robot, q_above, q_press)           # down to 30 mm (exact), press, back to 30 mm
            robot.set_speed([TRAVEL_SPEED] * 4)
            if q_hover is not None:
                go(robot, q_hover)                        # straight up before moving on
        print(f"\nDone: typed '{text}'.")
        robot.set_speed([PRESS_SPEED] * 4)
        move_exact(robot, look_pose)
    except KeyboardInterrupt:
        print("\nStopped by the user.")
    finally:
        robot.port_handler.closePort()
        print("The arm keeps its pose with the motors on: hold it and switch the power off when done.")


if __name__ == "__main__":
    main()
