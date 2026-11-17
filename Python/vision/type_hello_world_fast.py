"""Problem 11a, fast version: ONE photo, then all the keys are pressed one after another.

Faster than type_hello_world.py (which takes a new photo before every letter), but nothing may move
(keyboard, robot base) while it types, because it doesn't look again.

Usage (from the Python/vision folder):
    python type_hello_world_fast.py
    python type_hello_world_fast.py "hello"
"""
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))              # MyRobot.py is one folder up
from MyRobot import MyRobot                               # noqa: E402
from red_detector import open_camera                      # noqa: E402
from calibrate_homography import CALIB_FILE, grab_fresh_frame   # noqa: E402
import click_touch                                        # noqa: E402
from click_touch import pixel_to_base, plan_touch         # noqa: E402
from arm import move_exact                                # noqa: E402
from detect_keys import load_templates, detect_keys, draw  # noqa: E402
from type_hello_world import key_name, press, TEXT, PRESS_DEPTH_MM, PHOTO_TRIES, COUNTDOWN_S  # noqa: E402

EXTRA_DEPTH_MM = {"space": 2.0}   # the space bar is long and stiff: press it a bit deeper
SAVE_DIR = os.path.join(HERE, "..", "images", time.strftime("typing_fast_%H%M%S"))


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else TEXT
    calib = np.load(CALIB_FILE)
    H, z_key = calib["H"], float(calib["z_keys"])
    look_pose, camera_index = tuple(calib["look_pose"]), int(calib["camera_index"])
    templates, space_rule = load_templates()
    needed = {key_name(c) for c in text}
    os.makedirs(SAVE_DIR, exist_ok=True)

    robot = MyRobot()
    if not robot.init_status:
        sys.exit("Robot did not start (see 'Initialization error' above).")
    cam = open_camera(camera_index)

    # 1. One look: photo(s) until every key we need is found
    found = {}
    for attempt in range(1, PHOTO_TRIES + 1):
        move_exact(robot, look_pose)
        frame = grab_fresh_frame(cam)
        found, report = detect_keys(frame, templates, space_rule)
        cv2.imwrite(os.path.join(SAVE_DIR, f"photo_try{attempt}.jpg"), draw(frame, found, templates))
        print("\n".join(report))
        if needed <= set(found):
            break
        print(f"  missing {sorted(needed - set(found))} (photo {attempt}/{PHOTO_TRIES})")
    cam.release()
    if not needed <= set(found):
        robot.port_handler.closePort()
        sys.exit("Stopping: not all keys found. Check the photos in " + os.path.abspath(SAVE_DIR))

    # 2. Plan every key once (pixel -> mm -> joint angles)
    plans = {}
    for key in needed:
        x, y = pixel_to_base(*found[key], H)
        click_touch.TOUCH_DEPTH_MM = PRESS_DEPTH_MM + EXTRA_DEPTH_MM.get(key, 0.0)
        plan = plan_touch(x, y, z_key, robot.joint_limits)
        if plan is None:
            robot.port_handler.closePort()
            sys.exit(f"Stopping: '{key}' at ({x:.1f}, {y:.1f}) mm can't be reached.")
        plans[key] = plan
        print(f"  {key}: ({x:.1f}, {y:.1f}) mm, pitch {plan[2]} deg")

    for s in range(COUNTDOWN_S, 0, -1):
        print(f"Click inside the text editor now... {s}")
        time.sleep(1)

    # 3. Type: straight from key to key, no more photos
    try:
        for step, char in enumerate(text, start=1):
            key = key_name(char)
            print(f"[{step}/{len(text)}] '{key}'")
            q_above, q_press, _ = plans[key]
            press(robot, q_above, q_press)
        print(f"\nDone: typed '{text}'.")
        move_exact(robot, look_pose)
    except KeyboardInterrupt:
        print("\nStopped by the user.")
    finally:
        robot.port_handler.closePort()
        print("The arm keeps its pose with the motors on: hold it and switch the power off when done.")


if __name__ == "__main__":
    main()
