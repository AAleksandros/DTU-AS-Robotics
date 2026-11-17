"""Small helpers for moving the real arm precisely (used by the vision scripts).

Why: move_j sends the target angles and accepts the move once every motor is within 2 deg. Under load
(the arm's weight, the cables pulling the base) a motor can stop a few degrees short, always on the same
side, and at ~15 cm from the base 1 deg is ~3 mm at the stylus. move_exact reads where the motors really
stopped and sends a corrected target until they are within ~0.6 deg (two motor steps) of where we want them.
"""
import time

import numpy as np

STEP_DEG = 0.29               # one AX-12A position step


def set_torque(robot, on):
    for dxl_id in robot.motor_ids:
        robot._write1(dxl_id, robot.ADDR_MX_TORQUE_ENABLE, 1 if on else 0)


def read_angles(robot, tries=5):
    """Joint angles [deg]; reads sometimes fail, so retry a few times."""
    last_error = None
    for _ in range(tries):
        try:
            return robot.read_joint_angles()
        except IOError as e:
            last_error = e
            time.sleep(0.05)
    raise IOError(f"Could not read the joint angles. Last error: {last_error}")


def hold_current_position(robot):
    """Make each motor's goal = where it is now, so switching torque on doesn't make it jump."""
    q = read_angles(robot)
    for i, dxl_id in enumerate(robot.motor_ids):
        robot._write2(dxl_id, robot.ADDR_MX_GOAL_POSITION, robot.deg_to_ticks(q[i] + robot.joint_offsets[i]))


def wait_until_still(robot, still_deg=0.3, timeout_s=8.0):
    """Wait until no joint moves more than still_deg between two readings 0.1 s apart. Returns the angles."""
    t_end = time.time() + timeout_s
    previous = read_angles(robot)
    while time.time() < t_end:
        time.sleep(0.1)
        current = read_angles(robot)
        if np.all(np.abs(current - previous) < still_deg):
            return current
        previous = current
    return previous


def send(robot, q):
    """Send target angles without move_j's 'arrived' check (we check ourselves)."""
    q = np.clip(np.asarray(q, dtype=float), robot.joint_limits[:, 0], robot.joint_limits[:, 1])
    robot.move_j(*q, wait=False)
    return q


def move_exact(robot, q_target, tol_deg=2 * STEP_DEG + 0.05, max_iter=4, max_correction_deg=10.0):
    """Move so the MEASURED angles end up at q_target.

    Returns (q_command, error): q_command = the angles we had to send to get there (target + correction),
    error = target - measured at the end [deg].
    """
    q_target = np.asarray(q_target, dtype=float)
    q_cmd = send(robot, q_target)
    error = q_target - wait_until_still(robot)
    for _ in range(max_iter):
        if np.all(np.abs(error) <= tol_deg):
            break
        q_cmd = q_cmd + error                       # aim past the target by what we fell short
        if np.any(np.abs(q_cmd - q_target) > max_correction_deg):
            print(f"    WARNING: correction above {max_correction_deg} deg ({np.round(q_cmd - q_target, 1)}); stopping corrections")
            break
        q_cmd = send(robot, q_cmd)
        error = q_target - wait_until_still(robot)
    return q_cmd, error
