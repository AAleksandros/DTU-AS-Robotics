"""Problem 11a, fast version 2: the SAME moves as type_hello_world.py (which works), but only ONE photo.

The slow version, for every letter: look pose -> photo -> find the key -> press.
This one: preview + ONE photo at the start -> all keys found and planned once -> for every letter:
look pose -> press. The arm moves exactly like the slow version (same functions, same settings,
imported from type_hello_world.py); it only skips the photo and the search before each letter.

Nothing may move (keyboard, robot base) after the photo, because it doesn't look again.

Usage (from the Python/vision folder):
    python type_hello_world_fast2.py
    python type_hello_world_fast2.py "hello"
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
# Everything about HOW to press comes from the slow version, unchanged:
from type_hello_world import key_name, preview, press, TEXT, PRESS_DEPTH_MM, COUNTDOWN_S  # noqa: E402

SAVE_DIR = os.path.join(HERE, "..", "images", time.strftime("typing_fast2_%H%M%S"))


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else TEXT
    calib = np.load(CALIB_FILE)
    H, z_key = calib["H"], float(calib["z_keys"])
    look_pose, camera_index = tuple(calib["look_pose"]), int(calib["camera_index"])
    templates, space_rule = load_templates()
    needed = {key_name(c) for c in text}
    os.makedirs(SAVE_DIR, exist_ok=True)

    robot = MyRobot()                                     # connects and moves slowly to straight up
    if not robot.init_status:
        sys.exit("Robot did not start (see 'Initialization error' above).")
    cam = open_camera(camera_index)
    click_touch.TOUCH_DEPTH_MM = PRESS_DEPTH_MM           # same press depth as the slow version

    # 1. Preview (same as the slow version), then ONE fresh photo
    move_exact(robot, look_pose)
    while True:
        if not preview(cam, templates, space_rule, needed):
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

    # 2. Plan every key once (same planning as the slow version)
    plans = {}
    for key in needed:
        x, y = pixel_to_base(*found[key], H)
        plan = plan_touch(x, y, z_key, robot.joint_limits)
        if plan is None:
            robot.port_handler.closePort()
            sys.exit(f"Stopping: '{key}' at ({x:.1f}, {y:.1f}) mm can't be reached within the joint limits.")
        plans[key] = plan
        print(f"  {key}: pixel ({found[key][0]:.0f}, {found[key][1]:.0f}) -> ({x:.1f}, {y:.1f}) mm, pitch {plan[2]} deg")

    for s in range(COUNTDOWN_S, 0, -1):
        print(f"Click inside the text editor now... {s}")
        time.sleep(1)

    # 3. Type: like the slow version (back to the look pose between keys), without the photos
    try:
        for step, char in enumerate(text, start=1):
            key = key_name(char)
            print(f"\n[{step}/{len(text)}] '{key}'")
            move_exact(robot, look_pose)
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
