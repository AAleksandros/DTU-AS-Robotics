"""Vision step 3 (Problem 11a): click a key in the camera image and the robot touches it.

Tests the chain  pixel -> H -> (x, y) mm -> inverse kinematics -> arm,  without automatic detection yet.

Usage (from the Python/vision folder, after calibrate_homography.py):
    python click_touch.py
In the window: click a key = touch it,  q = quit.
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
from arm import move_exact, send, wait_until_still   # noqa: E402

# ------------------------------ Settings ---------------------------------------
APPROACH_MM = 30.0            # height above the key before going down
TOUCH_DEPTH_MM = 0.0          # 0 = just touch the key top; e.g. 3 = press 3 mm down (later, for typing)
PITCHES = range(-80, -25, 5)  # stylus angles to try [deg], steepest first (-80, -75, ..., -30)


def pixel_to_base(u, v, H):
    """Pixel (u, v) -> (x, y) in mm in the robot base frame, on the key plane."""
    p = cv2.perspectiveTransform(np.array([[[u, v]]], dtype=np.float32), H)
    return float(p[0, 0, 0]), float(p[0, 0, 1])


def within_limits(q, limits):
    return all(lo <= a <= hi for a, (lo, hi) in zip(q, limits))


def plan_touch(x, y, z_key, limits):
    """Joint angles for 'above the key' and 'on the key' with the same pitch, steepest pitch that fits the limits."""
    for pitch in PITCHES:
        try:
            q_above = inverse_kinematics(x / 1000, y / 1000, (z_key + APPROACH_MM) / 1000, pitch)
            q_touch = inverse_kinematics(x / 1000, y / 1000, (z_key - TOUCH_DEPTH_MM) / 1000, pitch)
        except ValueError:                                # out of reach with this pitch
            continue
        if within_limits(q_above, limits) and within_limits(q_touch, limits):
            return q_above, q_touch, pitch
    return None


def wait_for_click(frame):
    """Show the photo and return the clicked (u, v), or None if q is pressed."""
    click = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click.append((x, y))

    cv2.namedWindow("click_touch", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("click_touch", on_click)
    view = frame.copy()
    cv2.putText(view, "Click a key to touch it   q = quit", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    while not click:
        cv2.imshow("click_touch", view)
        if (cv2.waitKey(20) & 0xFF) == ord("q"):
            return None
    u, v = click[0]
    cv2.drawMarker(view, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
    cv2.imshow("click_touch", view)
    cv2.waitKey(1)
    return u, v


def main():
    calib = np.load(CALIB_FILE)
    H, z_key = calib["H"], float(calib["z_keys"])
    look_pose, camera_index = tuple(calib["look_pose"]), int(calib["camera_index"])
    print(f"Calibration loaded: keys at z = {z_key:.1f} mm, look pose = {look_pose}")

    robot = MyRobot()                                     # connects and moves slowly to straight up
    if not robot.init_status:
        sys.exit("Robot did not start (see 'Initialization error' above).")
    cam = open_camera(camera_index)

    try:
        while True:
            move_exact(robot, look_pose)                  # camera exactly where it was at calibration
            frame = grab_fresh_frame(cam)
            pixel = wait_for_click(frame)
            if pixel is None:
                break

            x, y = pixel_to_base(*pixel, H)
            print(f"\npixel {pixel}  ->  ({x:.1f}, {y:.1f}) mm")
            plan = plan_touch(x, y, z_key, robot.joint_limits)
            if plan is None:
                print("  Can't reach that point within the joint limits. Click another key.")
                continue
            q_above, q_touch, pitch = plan
            print(f"  pitch {pitch} deg   above {np.round(q_above, 1)}   touch {np.round(q_touch, 1)}")

            q_above_cmd, error = move_exact(robot, q_above)
            correction = q_above_cmd - np.asarray(q_above)
            print(f"  correction {np.round(correction, 1)} deg, remaining error {np.round(error, 1)} deg")
            send(robot, np.asarray(q_touch) + correction)   # same correction just below: similar pose, similar shortfall
            wait_until_still(robot)
            input("  Touching. Measure the error with a ruler, then press Enter to go back up ")
            send(robot, q_above_cmd)
            wait_until_still(robot)
    finally:
        cam.release()
        cv2.destroyAllWindows()
        robot.port_handler.closePort()
        print("Done. The arm keeps its pose with the motors on: hold it and switch the power off.")


if __name__ == "__main__":
    main()
