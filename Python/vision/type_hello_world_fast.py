"""Problem 11a, fast version: ONE photo, then all the keys are pressed one after another.

1. Preview: the arm goes to the look pose and shows the camera live with the keys it finds.
   Move the keyboard until every key is green, then press ENTER in the window (q = quit).
2. One photo -> every key's position in mm.
3. Typing, for each key: lift to LIFT_MM above the keys, go above the key, down to ABOVE_MM
   (reached exactly with move_exact), one short press (PRESS_TIME_S, so the keyboard doesn't repeat
   the letter), back up and lift again before the next key.

Nothing may move (keyboard, robot base) after the photo, because it doesn't look again.

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
from MyRobot import MyRobot, inverse_kinematics           # noqa: E402
from red_detector import open_camera                      # noqa: E402
from calibrate_homography import CALIB_FILE, grab_fresh_frame   # noqa: E402
from click_touch import pixel_to_base, within_limits, PITCHES   # noqa: E402
from arm import move_exact, send, wait_until_still        # noqa: E402
from detect_keys import load_templates, detect_keys, draw  # noqa: E402
from type_hello_world import key_name, TEXT, COUNTDOWN_S  # noqa: E402

LIFT_MM = 60.0                # height above the keys while moving from one key to the next
ABOVE_MM = 30.0               # stop here above the key (exact), then press straight down
PRESS_DEPTH_MM = 4          # how far below the key top the press goes
EXTRA_DEPTH_MM = {"space": 4}   # the space bar is long and stiff: press it deeper
PRESS_TIME_S = 0.23         
   # short press: the keyboard repeats a letter if it is held ~0.5 s
SAVE_DIR = os.path.join(HERE, "..", "images", time.strftime("typing_fast_%H%M%S"))


def solve(p, limits):
    """Joint angles for the stylus tip at p = (x, y, z) mm, steepest pitch that fits (same as click_touch)."""
    for pitch in PITCHES:
        try:
            q = inverse_kinematics(p[0] / 1000, p[1] / 1000, p[2] / 1000, pitch)
        except ValueError:
            continue
        if within_limits(q, limits):
            return np.asarray(q, dtype=float)
    return None


def type_key(robot, p_key, depth_mm):
    """Lift, go above the key at LIFT_MM, down to ABOVE_MM (exact), short press, back up to ABOVE_MM.
    Only whole moves (no small steps), so the arm moves smoothly."""
    limits = robot.joint_limits
    move_exact(robot, solve(p_key + [0, 0, LIFT_MM], limits))           # high above the key
    q_above = solve(p_key + [0, 0, ABOVE_MM], limits)
    q_above_cmd, _ = move_exact(robot, q_above)                          # exactly 30 mm above
    correction = q_above_cmd - q_above
    send(robot, solve(p_key - [0, 0, depth_mm], limits) + correction)   # short press
    time.sleep(PRESS_TIME_S)
    send(robot, q_above_cmd)
    wait_until_still(robot)
    move_exact(robot, solve(p_key + [0, 0, LIFT_MM], limits))           # lift before moving on


# ------------------------------ Preview ------------------------------------------
def preview(cam, templates, space_rule, needed):
    """Live view until every needed key is found and ENTER is pressed. Returns the photo used, or None (q)."""
    cv2.namedWindow("place the keyboard", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cam.read()
        if not ok:
            continue
        found, _ = detect_keys(frame, templates, space_rule)
        view = draw(frame, found, templates)
        missing = sorted(needed - set(found))
        text = "All keys found: ENTER = start   q = quit" if not missing else "Move the keyboard until all keys are green"
        cv2.putText(view, text, (20, view.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 200, 0) if not missing else (0, 0, 255), 2)
        cv2.imshow("place the keyboard", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            return None
        if key == 13 and not missing:
            cv2.destroyAllWindows()
            return frame


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

    # 1. Preview, then one fresh photo (taken again so nothing changed after ENTER)
    move_exact(robot, look_pose)
    while True:
        if preview(cam, templates, space_rule, needed) is None:
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

    # 2. Every key's position in mm (on the key plane)
    keys_mm = {}
    for key in needed:
        x, y = pixel_to_base(*found[key], H)
        keys_mm[key] = np.array([x, y, z_key])
        reachable = all(solve([x, y, z], robot.joint_limits) is not None
                        for z in (z_key + LIFT_MM, z_key + ABOVE_MM, z_key - PRESS_DEPTH_MM - EXTRA_DEPTH_MM.get(key, 0.0)))
        if not reachable:
            robot.port_handler.closePort()
            sys.exit(f"Stopping: '{key}' at ({x:.1f}, {y:.1f}) mm can't be reached within the joint limits.")
        print(f"  {key}: ({x:.1f}, {y:.1f}) mm")

    for s in range(COUNTDOWN_S, 0, -1):
        print(f"Click inside the text editor now... {s}")
        time.sleep(1)

    # 3. Type: straight lines at a safe height between keys, straight down and up at each key
    try:
        for step, char in enumerate(text, start=1):
            key = key_name(char)
            print(f"[{step}/{len(text)}] '{key}'")
            type_key(robot, keys_mm[key], PRESS_DEPTH_MM + EXTRA_DEPTH_MM.get(key, 0.0))
        print(f"\nDone: typed '{text}'.")
        move_exact(robot, look_pose)
    except KeyboardInterrupt:
        print("\nStopped by the user.")
    finally:
        robot.port_handler.closePort()
        print("The arm keeps its pose with the motors on: hold it and switch the power off when done.")


if __name__ == "__main__":
    main()
