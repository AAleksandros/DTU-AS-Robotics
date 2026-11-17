"""Vision step 4a (Problem 11a): cut one template per key from a photo taken at the look pose.

Usage (from the Python/vision folder, no robot needed):
    python make_templates.py ..\\images\\photo_153155.jpg

Click the centre of each key in the order shown (z = undo, Enter = save, q = cancel).
Saves templates/<letter>.png and templates/boxes.json (crop boxes + where the space bar is relative to D and L).
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(HERE, "templates")

KEYS = ["H", "E", "L", "O", "W", "R", "D", "space"]   # every key "hello world" needs (clicked in this order)
LETTERS = [k for k in KEYS if k != "space"]            # these get a template
SPACE_REF = ("D", "L")                                  # the space bar is located from these two letters

# Crop box per letter: (width, height, x of the key centre inside the crop, y of the key centre inside the crop) [px]
# A square about one key wide, centred on the key.
DEFAULT_BOX = (110, 110, 55, 55)


def box_for(key):
    return DEFAULT_BOX


def space_relative(p_space, p_a, p_b):
    """Space bar centre as (s, t): s along a->b, t along a->b turned 90 deg (downwards), both in units of |a->b|."""
    a = np.subtract(p_b, p_a, dtype=float)
    n = np.array([-a[1], a[0]])
    d = np.subtract(p_space, p_a, dtype=float)
    return float(d @ a / (a @ a)), float(d @ n / (a @ a))


def click_points(image, labels, window="make_templates"):
    """Show the image; you click one point per label, in order. Returns a list of (u, v)."""
    clicks = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(labels):
            clicks.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_click)
    while True:
        view = image.copy()
        for (u, v), label in zip(clicks, labels):
            cv2.drawMarker(view, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            if label in LETTERS:                         # show the crop box (space has none)
                w, h, ox, oy = box_for(label)
                cv2.rectangle(view, (u - ox, v - oy), (u - ox + w, v - oy + h), (0, 255, 0), 2)
            cv2.putText(view, label, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if len(clicks) < len(labels):
            text = f"Click the centre of '{labels[len(clicks)]}' ({len(clicks) + 1}/{len(labels)})   z = undo"
        else:
            text = "Done: Enter = save   z = undo"
        cv2.putText(view, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(window, view)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("z") and clicks:
            clicks.pop()
        elif key == 13 and len(clicks) == len(labels):     # 13 = Enter
            break
        elif key == ord("q"):
            sys.exit("Cancelled.")
    cv2.destroyWindow(window)
    return clicks


def cut(gray, u, v, key):
    """Crop the template for one key; None if the box sticks out of the image."""
    w, h, ox, oy = box_for(key)
    x0, y0 = u - ox, v - oy
    if x0 < 0 or y0 < 0 or x0 + w > gray.shape[1] or y0 + h > gray.shape[0]:
        return None
    return gray[y0:y0 + h, x0:x0 + w]


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python make_templates.py <photo taken at the look pose>")
    photo = cv2.imread(sys.argv[1])
    if photo is None:
        sys.exit(f"Could not read {sys.argv[1]}")
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

    clicks = click_points(photo, KEYS)

    centres = dict(zip(KEYS, clicks))

    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    boxes = {}
    for key in LETTERS:
        u, v = centres[key]
        template = cut(gray, u, v, key)
        if template is None:
            print(f"  {key}: too close to the edge of the photo, not saved. Use a photo where it is fully visible.")
            continue
        cv2.imwrite(os.path.join(TEMPLATE_DIR, f"{key}.png"), template)
        boxes[key] = box_for(key)
        print(f"  {key}: saved ({template.shape[1]}x{template.shape[0]} px) from centre ({u}, {v})")

    s, t = space_relative(centres["space"], centres[SPACE_REF[0]], centres[SPACE_REF[1]])
    space = {"from": list(SPACE_REF), "s": s, "t": t}
    print(f"  space: s = {s:.3f}, t = {t:.3f} relative to {SPACE_REF[0]} -> {SPACE_REF[1]}")

    with open(os.path.join(TEMPLATE_DIR, "boxes.json"), "w") as f:
        json.dump({"boxes": boxes, "space": space}, f, indent=2)
    print(f"Templates in {TEMPLATE_DIR}")


if __name__ == "__main__":
    main()
