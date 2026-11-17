"""Vision step 1 (Problem 11a): show the camera and find a red sticker.

Usage (from the Python/vision folder):
    python red_detector.py        # camera 0 (usually the laptop webcam)
    python red_detector.py 1      # camera 1 (usually the robot's USB camera)

Keys in the window:  q = quit,  s = save the photo to ../images/
Left click on the image = print the HSV colour of that pixel (to tune the thresholds).
"""
import os
import sys
import time

import cv2
import numpy as np

# ------------------------------ Settings ---------------------------------------
WIDTH, HEIGHT = 1280, 720        # resolution we ask the camera for (same as the test photos)
MIN_AREA = 200                   # [pixels] smaller blobs are ignored (noise)

# Red sits at both ends of the hue range (H goes from 0 to 180 in OpenCV), hence two ranges.
# Each range is (H, S, V) minimum and maximum. High S = strong colour, high V = well lit.
RED_LOW_1, RED_HIGH_1 = (0, 120, 70), (10, 255, 255)
RED_LOW_2, RED_HIGH_2 = (170, 120, 70), (180, 255, 255)

PHOTO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")


def detect_red(frame):
    """Frame (BGR) -> (u, v) of the centre of the largest red blob, or None. Also returns the mask."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, RED_LOW_1, RED_HIGH_1) | cv2.inRange(hsv, RED_LOW_2, RED_HIGH_2)
    # Remove small specks (erode, then dilate)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_AREA:
        return None, mask

    m = cv2.moments(largest)                     # centroid = mean position of the blob's pixels
    u, v = m["m10"] / m["m00"], m["m01"] / m["m00"]
    return (u, v), mask


def open_camera(index):
    # On Windows CAP_DSHOW opens faster and lets us change the resolution
    cam = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    if not cam.isOpened():
        sys.exit(f"Could not open camera {index}. Try another number: python red_detector.py 1")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    print(f"Camera {index} opened at {int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return cam


def main():
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cam = open_camera(index)

    latest = {"frame": None}

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and latest["frame"] is not None:
            h, s, v = cv2.cvtColor(latest["frame"], cv2.COLOR_BGR2HSV)[y, x]
            print(f"pixel (u={x}, v={y})  HSV = ({h}, {s}, {v})")

    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("camera", on_click)

    while True:
        ok, frame = cam.read()
        if not ok:
            print("No image from the camera.")
            break
        latest["frame"] = frame.copy()

        point, mask = detect_red(frame)
        if point is not None:
            u, v = point
            cv2.circle(frame, (int(u), int(v)), 12, (0, 255, 0), 3)
            cv2.putText(frame, f"u={u:.0f}  v={v:.0f}", (int(u) + 15, int(v) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "no red", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("camera", frame)
        cv2.imshow("mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            os.makedirs(PHOTO_DIR, exist_ok=True)
            path = os.path.join(PHOTO_DIR, time.strftime("photo_%H%M%S.jpg"))
            cv2.imwrite(path, latest["frame"])
            print("Saved:", os.path.abspath(path), "| red at:", None if point is None else (round(point[0]), round(point[1])))

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
