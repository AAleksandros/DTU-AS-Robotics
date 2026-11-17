"""Vision step 4b (Problem 11a): find the "hello world" keys in a photo with template matching.

Usage (from the Python/vision folder, after make_templates.py):
    python detect_keys.py ..\\images\\calib_photo.jpg    # a saved photo (no robot needed)
    python detect_keys.py                               # live camera 1 (put the robot at the look pose first)
Live mode: q = quit, s = save the annotated frame.
"""
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(HERE, "templates")
PHOTO_DIR = os.path.join(HERE, "..", "images")

MIN_SCORE = 0.75      # best match must be at least this similar (1 = identical)
MIN_MARGIN = 0.05     # ...and beat the second-best place by at least this much
TOPHAT_SIZE = 25      # [px] bright details thinner than this are kept (the letters)
BLUR_SIZE = 5         # [px] small blur so 1-pixel differences don't lower the score


def letters_only(gray):
    """White top-hat: original minus its opening = only thin bright details (the printed letters)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_SIZE, TOPHAT_SIZE))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    return cv2.GaussianBlur(tophat, (BLUR_SIZE, BLUR_SIZE), 0)


def load_templates():
    """Returns {letter: (template after top-hat, box)} and the space bar rule from boxes.json."""
    with open(os.path.join(TEMPLATE_DIR, "boxes.json")) as f:
        info = json.load(f)
    templates = {}
    for letter, box in info["boxes"].items():
        gray = cv2.imread(os.path.join(TEMPLATE_DIR, f"{letter}.png"), cv2.IMREAD_GRAYSCALE)
        templates[letter] = (letters_only(gray), box)
    return templates, info["space"]


def match_one(image, template, box):
    """Best and second-best match of one template. Returns (u, v) of the key centre, best score, second score."""
    scores = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, best, _, (x, y) = cv2.minMaxLoc(scores)
    w, h, ox, oy = box
    # Hide the area around the best match and look again: that is the place it could be confused with
    hidden = scores.copy()
    hidden[max(0, y - h + 1):y + h, max(0, x - w + 1):x + w] = -1
    _, second, _, _ = cv2.minMaxLoc(hidden)
    return (x + ox, y + oy), best, second


def space_from(p_a, p_b, s, t):
    """Space bar centre from two letters: p_a + s * (a->b) + t * (a->b turned 90 deg downwards)."""
    a = np.subtract(p_b, p_a, dtype=float)
    n = np.array([-a[1], a[0]])
    p = np.asarray(p_a, dtype=float) + s * a + t * n
    return float(p[0]), float(p[1])


def detect_keys(photo, templates, space_rule):
    """Photo (BGR) -> (found, report). found = {key: (u, v)} for accepted keys; report = one line per key."""
    image = letters_only(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY))
    candidates = {}
    for letter, (template, box) in templates.items():
        candidates[letter] = match_one(image, template, box)

    # Two letters can't be on the same key: if a letter is off-screen, its best match can land on a
    # similar-looking key (D on O). Keep the better score there and reject the other.
    clash = {}
    letters = list(candidates)
    for i, a in enumerate(letters):
        for b in letters[i + 1:]:
            (pa, sa, _), (pb, sb, _) = candidates[a], candidates[b]
            if np.hypot(pa[0] - pb[0], pa[1] - pb[1]) < templates[a][1][0] / 2:
                loser, winner = (a, b) if sa < sb else (b, a)
                clash[loser] = winner

    found, report = {}, []
    for letter, ((u, v), best, second) in candidates.items():
        ok = best >= MIN_SCORE and best - second >= MIN_MARGIN and letter not in clash
        if ok:
            found[letter] = (u, v)
        reason = f"  (same key as {clash[letter]})" if letter in clash else ""
        report.append(f"{letter}: ({u}, {v})  score {best:.2f}  second {second:.2f}  {'OK' if ok else 'REJECTED'}{reason}")

    a, b = space_rule["from"]
    if a in found and b in found:
        found["space"] = space_from(found[a], found[b], space_rule["s"], space_rule["t"])
        u, v = found["space"]
        report.append(f"space: ({u:.0f}, {v:.0f})  from {a} and {b}")
    else:
        report.append(f"space: not computed ({a} or {b} missing)")
    return found, report


def draw(photo, found, templates):
    view = photo.copy()
    for key, (u, v) in found.items():
        u, v = int(round(u)), int(round(v))
        if key in templates:
            w, h, ox, oy = templates[key][1]
            cv2.rectangle(view, (u - ox, v - oy), (u - ox + w, v - oy + h), (0, 255, 0), 2)
        cv2.drawMarker(view, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(view, key, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    missing = [k for k in list(templates) + ["space"] if k not in found]
    if missing:
        cv2.putText(view, "missing: " + " ".join(missing), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return view


def main():
    templates, space_rule = load_templates()
    cv2.namedWindow("detect_keys", cv2.WINDOW_NORMAL)

    if len(sys.argv) > 1:                                # a saved photo
        photo = cv2.imread(sys.argv[1])
        if photo is None:
            sys.exit(f"Could not read {sys.argv[1]}")
        found, report = detect_keys(photo, templates, space_rule)
        print("\n".join(report))
        view = draw(photo, found, templates)
        out = os.path.splitext(sys.argv[1])[0] + "_keys.jpg"
        cv2.imwrite(out, view)
        print(f"Annotated photo saved to {os.path.abspath(out)}  (press any key in the window to close)")
        cv2.imshow("detect_keys", view)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    from red_detector import open_camera                # live camera
    with open(os.path.join(TEMPLATE_DIR, "..", "calib_homography.npz"), "rb") as f:
        camera_index = int(np.load(f)["camera_index"])
    cam = open_camera(camera_index)
    while True:
        ok, photo = cam.read()
        if not ok:
            print("No image from the camera.")
            break
        found, report = detect_keys(photo, templates, space_rule)
        view = draw(photo, found, templates)
        cv2.imshow("detect_keys", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            os.makedirs(PHOTO_DIR, exist_ok=True)
            out = os.path.join(PHOTO_DIR, time.strftime("keys_%H%M%S.jpg"))
            cv2.imwrite(out, view)
            print("\n".join(report))
            print("Saved:", os.path.abspath(out))
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
