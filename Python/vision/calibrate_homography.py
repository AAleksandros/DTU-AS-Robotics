"""Vision step 2 (Problem 11a): calibrate the homography H (image pixel -> robot base x, y in mm).

Usage (from the Python/vision folder, robot powered on, camera plugged in):
    python calibrate_homography.py

What it does:
  Part 1  motors off, you place the stylus tip on each reference key -> forward kinematics gives (x, y, z) mm
  Part 2  motors on, arm goes to the look pose, photo, you click the centre of each key in the same order
  Part 3  H from the pairs, error check per key, everything saved to calib_homography.npz
"""
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))          # MyRobot.py is one folder up
from MyRobot import MyRobot, forward_kinematics       # noqa: E402
from red_detector import open_camera                  # noqa: E402
from arm import set_torque, read_angles, hold_current_position, move_exact   # noqa: E402

# ------------------------------ Settings ---------------------------------------
CAMERA_INDEX = 1
LOOK_POSE = (0.0, 7.6, -70.0, -105.0)                 # [deg] the pose the photo is always taken from
KEYS = ["W", "T", "O", "X", "B", "M", "space"]                 # reference keys, spread around the "hello world" letters
MAX_Z_SPREAD = 5.0                                    # [mm] warn if the touched keys are not at the same height
CALIB_FILE = os.path.join(HERE, "calib_homography.npz")
PHOTO_FILE = os.path.join(HERE, "..", "images", "calib_photo.jpg")


# ------------------------------ Part 1: touch ----------------------------------
def measure_keys(robot):
    """You put the stylus tip on each key; returns an (n, 3) array of (x, y, z) in mm."""
    points = []
    for key in KEYS:
        input(f"  Put the stylus tip on the centre of '{key}' and press Enter ")
        q = read_angles(robot)
        x, y, z, _ = forward_kinematics(*q)
        p = np.array([x, y, z]) * 1000.0
        points.append(p)
        print(f"    {key}: q = {np.round(q, 1)} deg  ->  ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) mm")
    return np.array(points)


# ------------------------------ Part 2: click ----------------------------------
def grab_fresh_frame(cam, discard=15):
    """The camera keeps a few old frames in a buffer; throw them away and return a new one."""
    for _ in range(discard):
        cam.read()
    ok, frame = cam.read()
    if not ok:
        sys.exit("No image from the camera.")
    return frame


def click_keys(frame):
    """Show the photo; you click the centre of each key in KEYS order. Returns an (n, 2) array of (u, v)."""
    clicks = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(KEYS):
            clicks.append((x, y))

    cv2.namedWindow("calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibration", on_click)
    while True:
        view = frame.copy()
        for (u, v), key in zip(clicks, KEYS):
            cv2.drawMarker(view, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
            cv2.putText(view, key, (u + 12, v - 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if len(clicks) < len(KEYS):
            text = f"Click the centre of '{KEYS[len(clicks)]}' ({len(clicks) + 1}/{len(KEYS)})   z = undo"
        else:
            text = "Done: Enter = accept   z = undo"
        cv2.putText(view, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("calibration", view)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("z") and clicks:
            clicks.pop()
        elif key == 13 and len(clicks) == len(KEYS):     # 13 = Enter
            break
        elif key == ord("q"):
            sys.exit("Cancelled.")
    cv2.destroyWindow("calibration")
    return np.array(clicks, dtype=float)


# ------------------------------ Part 3: compute --------------------------------
def compute_homography(pixels, xy_mm):
    """H such that [x, y, 1] ~ H @ [u, v, 1]. Uses all points (least squares). Returns H and error per point [mm]."""
    H, _ = cv2.findHomography(pixels.astype(np.float32), xy_mm.astype(np.float32), 0)
    predicted = cv2.perspectiveTransform(pixels.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
    errors = np.linalg.norm(predicted - xy_mm, axis=1)
    return H, errors


def main():
    robot = MyRobot()                                  # connects and moves slowly to straight up
    if not robot.init_status:
        sys.exit("Robot did not start (see 'Initialization error' above).")

    print("\nPART 1: touch the keys")
    input("  HOLD THE ARM, then press Enter to switch the motors off ")
    set_torque(robot, False)
    touched = measure_keys(robot)
    z_spread = touched[:, 2].max() - touched[:, 2].min()
    if z_spread > MAX_Z_SPREAD:
        print(f"  WARNING: the keys differ {z_spread:.1f} mm in height. A touch may be wrong; consider repeating.")

    print("\nPART 2: look pose and photo")
    input("  Keep holding the arm and press Enter: the motors switch on and it goes to the look pose ")
    hold_current_position(robot)
    set_torque(robot, True)
    q_cmd, error = move_exact(robot, LOOK_POSE)      # measured angles = LOOK_POSE, not just "within 2 deg"
    print(f"  look pose reached; correction sent {np.round(q_cmd - np.array(LOOK_POSE), 1)} deg, "
          f"remaining error {np.round(error, 1)} deg")
    cam = open_camera(CAMERA_INDEX)
    time.sleep(1.0)                                    # let the arm settle
    frame = grab_fresh_frame(cam)
    cam.release()
    os.makedirs(os.path.dirname(PHOTO_FILE), exist_ok=True)
    cv2.imwrite(PHOTO_FILE, frame)
    pixels = click_keys(frame)

    print("\nPART 3: homography")
    H, errors = compute_homography(pixels, touched[:, :2])
    for key, (u, v), e in zip(KEYS, pixels, errors):
        print(f"  {key}: pixel ({u:.0f}, {v:.0f})  error {e:.1f} mm")
    print(f"  mean error {errors.mean():.1f} mm, max {errors.max():.1f} mm")

    np.savez(CALIB_FILE, H=H, z_keys=touched[:, 2].mean(), look_pose=np.array(LOOK_POSE),
             camera_index=CAMERA_INDEX, keys=np.array(KEYS), pixels=pixels, touched_mm=touched)
    print(f"\nSaved {CALIB_FILE}\nPhoto  {os.path.abspath(PHOTO_FILE)}")
    print("The arm stays at the look pose with the motors on. Hold it and switch the power off when done.")
    robot.port_handler.closePort()


if __name__ == "__main__":
    main()
