"""Problem 11a: the robot finds the keys with the camera and types "hello world".

Usage (from the Python/vision folder, after calibrate_homography.py and make_templates.py):
    python type_hello_world.py
    python type_hello_world.py "hello"        # another text, using only keys that have a template

Before the countdown ends, click inside a text editor (Notepad) on the computer the keyboard is plugged into.
Ctrl+C stops; the arm keeps its pose with the motors on.
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
from click_touch import pixel_to_base, plan_touch         # noqa: E402
from arm import move_exact, send, wait_until_still        # noqa: E402
import click_touch                                        # noqa: E402
from detect_keys import load_templates, detect_keys, draw  # noqa: E402

# ------------------------------ Settings ---------------------------------------
TEXT = "hello world"
PRESS_DEPTH_MM = 3.0          # how far below the key top the stylus goes, so the key actually types
PRESS_TIME_S = 0.5            # time spent pressing before going back up
PHOTO_TRIES = 3               # photos to try if a key is not found
COUNTDOWN_S = 5
SAVE_DIR = os.path.join(HERE, "..", "images", time.strftime("typing_%H%M%S"))


def key_name(char):
    return "space" if char == " " else char.upper()


def find_key(robot, cam, look_pose, templates, space_rule, key, step):
    """Go to the look pose, take photos until the key is found. Returns its (u, v) or None."""
    for attempt in range(1, PHOTO_TRIES + 1):
        q_cmd, error = move_exact(robot, look_pose)       # camera exactly where it was at calibration
        print(f"    look pose: correction {np.round(q_cmd - np.asarray(look_pose), 1)} deg")
        frame = grab_fresh_frame(cam)
        found, report = detect_keys(frame, templates, space_rule)
        cv2.imwrite(os.path.join(SAVE_DIR, f"{step:02d}_{key}_try{attempt}.jpg"), draw(frame, found, templates))
        if key in found:
            return found[key]
        print(f"    '{key}' not found (photo {attempt}/{PHOTO_TRIES}):\n      " + "\n      ".join(report))
    return None


def press(robot, q_above, q_press):
    q_above_cmd, error = move_exact(robot, q_above)       # really above the key, not "within 2 deg"
    correction = q_above_cmd - np.asarray(q_above)
    print(f"  above the key: correction {np.round(correction, 1)} deg, remaining error {np.round(error, 1)} deg")
    send(robot, np.asarray(q_press) + correction)         # same correction for the press (similar pose)
    time.sleep(PRESS_TIME_S)                              # no "arrived" check: the key bottom stops the arm
    send(robot, q_above_cmd)
    wait_until_still(robot)


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else TEXT
    calib = np.load(CALIB_FILE)
    H, z_key = calib["H"], float(calib["z_keys"])
    look_pose, camera_index = tuple(calib["look_pose"]), int(calib["camera_index"])
    templates, space_rule = load_templates()
    available = set(templates) | {"space"}
    missing = sorted({key_name(c) for c in text} - available)
    if missing:
        sys.exit(f"No template for: {missing}. Add them in make_templates.py (KEYS) first.")
    os.makedirs(SAVE_DIR, exist_ok=True)

    robot = MyRobot()                                     # connects and moves slowly to straight up
    if not robot.init_status:
        sys.exit("Robot did not start (see 'Initialization error' above).")
    cam = open_camera(camera_index)

    click_touch.TOUCH_DEPTH_MM = PRESS_DEPTH_MM           # plan_touch goes this far below the key top
    for s in range(COUNTDOWN_S, 0, -1):
        print(f"Click inside the text editor now... {s}")
        time.sleep(1)

    try:
        for step, char in enumerate(text, start=1):
            key = key_name(char)
            print(f"\n[{step}/{len(text)}] '{key}'")
            pixel = find_key(robot, cam, look_pose, templates, space_rule, key, step)
            if pixel is None:
                print("  Stopping: key not found. Check the photos in", os.path.abspath(SAVE_DIR))
                break
            x, y = pixel_to_base(*pixel, H)
            plan = plan_touch(x, y, z_key, robot.joint_limits)
            if plan is None:
                print(f"  Stopping: ({x:.1f}, {y:.1f}) mm can't be reached within the joint limits.")
                break
            q_above, q_press, pitch = plan
            print(f"  pixel ({pixel[0]:.0f}, {pixel[1]:.0f}) -> ({x:.1f}, {y:.1f}) mm, pitch {pitch} deg")
            press(robot, q_above, q_press)
        else:
            print(f"\nDone: typed '{text}'.")
        move_exact(robot, look_pose)
    except KeyboardInterrupt:
        print("\nStopped by the user.")
    finally:
        cam.release()
        robot.port_handler.closePort()
        print("Photos of every step in", os.path.abspath(SAVE_DIR))
        print("The arm keeps its pose with the motors on: hold it and switch the power off when done.")


if __name__ == "__main__":
    main()
