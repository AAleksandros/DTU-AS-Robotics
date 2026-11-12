import os
import sys
import inspect
import math
import numpy as np
import time

# --- Dynamixel SDK path bootstrap -----------------------
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sdk_path = os.path.join(parentdir, "DynamixelSDK-3.7.31", "python", "src", "dynamixel_sdk")
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)

from dynamixel_sdk import PortHandler, PacketHandler


def cosd(x_deg):  # cos with degrees
    return math.cos(math.radians(x_deg))


def sind(x_deg):  # sin with degrees
    return math.sin(math.radians(x_deg))

# --------------------------- Kinematics (Problems 1-4) ---------------------------
# Link lengths [m] from Figure 1 of the assignment. The real arm differs by a few mm (see Part 5).
D1, L2, L3, L4 = 0.05, 0.093, 0.093, 0.05
# Joint zero is the arm pointing straight up; the formulas below measure the shoulder from horizontal.
THETA2_OFFSET = 90.0


def forward_kinematics(q1, q2, q3, q4):
    """Joint angles [deg] -> tool point (x, y, z) [m] at the end of link 4, and pitch [deg].

    Pitch is the angle of link 4 above horizontal (0 = horizontal stylus)."""
    t2 = q2 + THETA2_OFFSET
    r = L2 * cosd(t2) + L3 * cosd(t2 + q3) + L4 * cosd(t2 + q3 + q4)  # horizontal reach from the base axis
    z = D1 + L2 * sind(t2) + L3 * sind(t2 + q3) + L4 * sind(t2 + q3 + q4)
    x = r * cosd(q1)
    y = r * sind(q1)
    pitch = t2 + q3 + q4
    return x, y, z, pitch


def inverse_kinematics(x, y, z, pitch):
    """Tool point (x, y, z) [m] and pitch [deg] -> joint angles [deg], elbow-up solution.

    Assumes the target is in front of the base axis. Raises ValueError if it is out of reach."""
    q1 = math.degrees(math.atan2(y, x))
    r = math.sqrt(x**2 + y**2)
    # Wrist centre: step back from the tool point along link 4
    r_w = r - L4 * cosd(pitch)
    z_w = z - L4 * sind(pitch)
    # Elbow angle from the law of cosines on the shoulder-elbow-wrist triangle
    up = z_w - D1
    D2 = r_w**2 + up**2
    c3 = (D2 - L2**2 - L3**2) / (2 * L2 * L3)
    if abs(c3) > 1:
        raise ValueError("target out of reach")
    t3 = -math.degrees(math.acos(c3))  # negative = elbow-up
    # Shoulder angle: direction to the wrist minus the angle link 3 adds
    up2 = L3 * sind(t3)
    out2 = L2 + L3 * cosd(t3)
    t2 = math.degrees(math.atan2(up, r_w)) - math.degrees(math.atan2(up2, out2))
    # Wrist supplies the remaining pitch
    t4 = pitch - t2 - t3
    return q1, t2 - THETA2_OFFSET, t3, t4


def jacobian(q1, q2, q3, q4):
    """Joint angles [deg] -> 4x4 Jacobian expressed in the arm's vertical plane.

    Rows: tangential, radial, vertical velocity [m/rad] and pitch rate [rad/rad].
    Columns: base, shoulder, elbow, wrist."""
    t2 = q2 + THETA2_OFFSET
    # Joint positions in the arm plane as (radial, vertical)
    shoulder = (0, D1)
    elbow = (L2 * cosd(t2), D1 + L2 * sind(t2))
    wrist = (elbow[0] + L3 * cosd(t2 + q3), elbow[1] + L3 * sind(t2 + q3))
    tool = (wrist[0] + L4 * cosd(t2 + q3 + q4), wrist[1] + L4 * sind(t2 + q3 + q4))
    J = np.zeros((4, 4))
    J[0, 0] = tool[0]  # base rotation moves the tool tangentially by its reach r
    for col, joint in ((1, shoulder), (2, elbow), (3, wrist)):
        dr = tool[0] - joint[0]
        dz = tool[1] - joint[1]
        J[1, col], J[2, col] = -dz, dr  # z × (dr, dz): velocity is perpendicular to the joint-to-tool vector
        J[3, col] = 1  # pitch is the sum of the three pitch joints
    return J

class MyRobot:
    # ----------------------------- Constants (AX-12A) -------------------------
    ADDR_MX_TORQUE_ENABLE = 24       # 1 byte
    ADDR_MX_GOAL_POSITION = 30       # 2 bytes
    ADDR_MX_MOVING_SPEED = 32        # 2 bytes
    ADDR_MX_TORQUE_LIMIT = 34        # 2 bytes
    ADDR_MX_PRESENT_POSITION = 36    # 2 bytes

    PROTOCOL_VERSION = 1.0
    BAUDRATE = 1000000
    DEVICENAME = "COM5"              # e.g., Windows: 'COM3' | Linux: '/dev/ttyUSB0' | Mac: '/dev/tty.usbserial-*'

    TORQUE_ENABLE = 1
    TORQUE_DISABLE = 0
    COMM_SUCCESS = 0
    DXL_MOVING_STATUS_THRESHOLD = 10  # [ticks], not used directly here
    TICKS_PER_DEG = 1.0 / 0.29        # ≈ 3.448275862 ticks per degree
    DEG_PER_TICK = 0.29
    # Motor angle [deg] that counts as joint angle 0 (arm straight up); 150 is the middle of the motor's range.
    # Calibrated with a spirit level on 10 Nov (the shoulder leaned back ~2.5°). Recalibrate if the arm is rebuilt.
    JOINT_OFFSETS = (150.0, 147.5, 150.0, 150.0)

    # ------------------------------- Init -------------------------------------
    def __init__(self):
        # Robot + kinematics state
        self.motor_ids = [1, 2, 3, 4]
        self.gripper_motor_id = 5
        self.forward_transform = np.zeros((4, 4))
        self.joint_angles = np.array([0.0, 0.0, 0.0, 0.0])  # deg
        self.joint_pos = np.zeros((4, 4))
        self.draw_robot_flag = False
        self.use_smooth_speed_flag = False
        self.gripper_open_flag = True

        # Joint limits [deg]: measured hard stops (10 Nov) q2 [-104, 102], q3 [-141, 118], q4 [-29, 140],
        # minus a 5° margin. q1 is limited to ±90° by the cables. Re-measure if the arm is rebuilt.
        self.joint_limits = np.array([[-90, 90], [-99, 97], [-136, 113], [-24, 135]])

        # IK config (simple analytical below; no external toolbox)
        self.ik_weights = np.array([0.25, 0.25, 0.25, 1, 1, 1])

        # Motor offsets (deg)
        self.joint_offsets = np.array(self.JOINT_OFFSETS, dtype=float)

        self.joint_angle_error = np.array([0.0, 0.0, 0.0, 0.0])
        self.init_status = 0

        self.movement_history = []  # list of dicts
        self.motor_speed = np.array([0.1, 0.1, 0.1, 0.1])   # [0..1]
        self.motor_torque = np.array([1.0, 1.0, 1.0, 1.0])  # [0..1]
        self.pitch = 0.0

        # --- SDK setup ---
        try:
            self.port_handler = PortHandler(self.DEVICENAME)
            self.packet_handler = PacketHandler(self.PROTOCOL_VERSION)

            if self.port_handler.openPort():
                print("Succeeded to open the port!")
            else:
                print("Failed to open the port! Reconnect Robot.")
                raise RuntimeError("openPort failed")

            if self.port_handler.setBaudRate(self.BAUDRATE):
                print("Succeeded to change the baudrate!")
            else:
                print("Failed to change the baudrate! Reconnect Robot.")
                raise RuntimeError("setBaudRate failed")

            # Initialize defaults
            self.set_speed(self.motor_speed, overwrite_speeds=True)
            self.set_torque_limit(self.motor_torque)
            self.enable_motors()  # must come before move_j, which waits for the joints to arrive
            self.move_j(0, 0, 0, 0)
            self.init_status = 1

        except Exception as e:
            print(f"Initialization error: {e}")
            self.init_status = 0

    # ----------------------------- Utilities ----------------------------------
    @staticmethod
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    @classmethod
    def deg_to_ticks(cls, deg):
        return int(round(deg * cls.TICKS_PER_DEG))

    @classmethod
    def ticks_to_deg(cls, ticks):
        return ticks * cls.DEG_PER_TICK

    def _check_limits_one(self, deg, idx):
        lo, hi = self.joint_limits[idx]
        if not (lo <= deg <= hi):
            raise ValueError(f"Joint {idx + 1}: {deg:.1f}° is outside its limits [{lo}°, {hi}°]")
        return deg

    # --------------------------- Motor I/O wrappers ---------------------------
    DXL_ERRORS = {1: "input voltage", 2: "angle limit", 4: "overheating", 8: "range",
                  16: "checksum", 32: "overload (motor switched itself off; power-cycle the robot)", 64: "instruction"}

    @classmethod
    def _err(cls, code):
        return ", ".join(name for bit, name in cls.DXL_ERRORS.items() if code & bit) or str(code)

    def _write1(self, dxl_id, addr, value):
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, dxl_id, addr, int(value))
        if dxl_comm_result != self.COMM_SUCCESS or dxl_error != 0:
            raise IOError(f"DXL write1 error id={dxl_id}, addr={addr}, result={dxl_comm_result}, error={self._err(dxl_error)}")

    def _write2(self, dxl_id, addr, value):
        dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(self.port_handler, dxl_id, addr, int(value))
        if dxl_comm_result != self.COMM_SUCCESS or dxl_error != 0:
            raise IOError(f"DXL write2 error id={dxl_id}, addr={addr}, result={dxl_comm_result}, error={self._err(dxl_error)}")

    def _read2(self, dxl_id, addr):
        value, dxl_comm_result, dxl_error = self.packet_handler.read2ByteTxRx(self.port_handler, dxl_id, addr)
        if dxl_comm_result != self.COMM_SUCCESS or dxl_error != 0:
            raise IOError(f"DXL read2 error id={dxl_id}, addr={addr}, result={dxl_comm_result}, error={self._err(dxl_error)}")
        return value

    # ------------------------------- Speeds/Torque ----------------------------
    def smooth_speed(self, joint_angles_delta_deg):
        max_angle = float(np.max(np.abs(joint_angles_delta_deg)))
        if max_angle == 0:
            return
        speed_per_deg = max_angle / 100.0
        if speed_per_deg != 0:
            new_speeds = np.abs(joint_angles_delta_deg / speed_per_deg) * 0.01
            # preserve zeros by falling back to previous speeds
            for i in range(len(self.motor_speed)):
                if new_speeds[i] == 0:
                    new_speeds[i] = self.motor_speed[i]
                else:
                    new_speeds[i] = new_speeds[i] * self.motor_speed[i]
            self.set_speed(new_speeds, overwrite_speeds=False)

    def set_speed(self, speeds, overwrite_speeds=True):
        speeds = np.asarray(speeds, dtype=float)
        if overwrite_speeds:
            self.motor_speed = speeds.copy()

        for i, dxl_id in enumerate(self.motor_ids):
            s = float(speeds[i])
            if 0 < s <= 1:
                raw = int(round(s * 1023))
                self._write2(dxl_id, self.ADDR_MX_MOVING_SPEED, raw)
            else:
                print("Movement speed out of range, enter value between (0, 1].")

    def set_torque_limit(self, torques):
        torques = np.asarray(torques, dtype=float)
        self.motor_torque = torques.copy()
        for i, dxl_id in enumerate(self.motor_ids):
            t = float(torques[i])
            if 0 < t <= 1:
                self._write2(dxl_id, self.ADDR_MX_TORQUE_LIMIT, int(round(t * 1023)))
            else:
                print("Torque limit out of range, enter value between (0, 1].")

    # --------------------------------- Power ----------------------------------
    def enable_motors(self):
        for dxl_id in self.motor_ids:
            self._write1(dxl_id, self.ADDR_MX_TORQUE_ENABLE, self.TORQUE_ENABLE)
        print("Dynamixel torque enabled on all motors.")

    def disable_motors(self):
        for dxl_id in self.motor_ids:
            try:
                self._write1(dxl_id, self.ADDR_MX_TORQUE_ENABLE, self.TORQUE_DISABLE)
            except Exception as e:
                print(f"Disable motor {dxl_id} error: {e}")
        try:
            self.port_handler.closePort()
        finally:
            self.init_status = 0
        print("Dynamixels disabled and port closed.")

    # ------------------------------- Telemetry --------------------------------
    def get_position(self, motor_id):
        ticks = self._read2(motor_id, self.ADDR_MX_PRESENT_POSITION)
        return self.ticks_to_deg(ticks)

    def read_joint_angles(self):
        j_a = np.zeros(4, dtype=float)
        for i, dxl_id in enumerate(self.motor_ids):
            ticks = self._read2(dxl_id, self.ADDR_MX_PRESENT_POSITION)
            deg = self.ticks_to_deg(ticks) - self.joint_offsets[i]
            j_a[i] = deg
            self.joint_angle_error[i] = deg - self.joint_angles[i]
        return j_a

    # Kinematics (forward_kinematics, inverse_kinematics, jacobian) are module-level functions at the top of this file.

    # ------------------------------- Motion -----------------------------------
    def move_j(self, j1_deg, j2_deg, j3_deg, j4_deg, wait=True, tol_deg=2.0, poll_period=0.05, timeout_s=10.0):
        # Limits
        desired = np.array([j1_deg, j2_deg, j3_deg, j4_deg], dtype=float)
        for i in range(4):
            desired[i] = self._check_limits_one(desired[i], i)

        # Smooth speed option
        if self.use_smooth_speed_flag:
            self.smooth_speed(desired - self.joint_angles)

        # Update internal & FK
        #self.joint_angles = desired.copy()
        #self.forward(desired)

        # Apply motor offsets for command
        cmd_deg = desired + self.joint_offsets
        for i, dxl_id in enumerate(self.motor_ids):
            self._write2(dxl_id, self.ADDR_MX_GOAL_POSITION, self.deg_to_ticks(cmd_deg[i]))

        if not wait:
            return


        # Wait until all joints within tolerance, but not forever
        t_end = time.time() + timeout_s
        ja, last_err = None, None
        while True:
            try:
                ja = self.read_joint_angles()
                if np.all(np.abs(ja - desired) < tol_deg):
                    break
            except Exception as e:
                last_err = e  # reads occasionally fail; retry until the timeout
            if time.time() > t_end:
                raise TimeoutError(f"Joints did not reach {np.round(desired, 1)} within {timeout_s} s; "
                                   f"last measured {None if ja is None else np.round(ja, 1)}; last error: {last_err}")
            time.sleep(poll_period)

    #def move_c(self, x, y, z, pitch_deg, **kwargs):
     #   j_deg = self.inverse(x, y, z, math.radians(pitch_deg))
      #  self.move_j(j_deg[0], j_deg[1], j_deg[2], j_deg[3], **kwargs)

    #def read_ee_position(self):
     #   j_a = self.read_joint_angles()
      #  return self.forward(j_a)

    # --------------------------- History / Macros ------------------------------
    def record_configuration(self):
        entry = {
            "j1": float(self.joint_angles[0]),
            "j2": float(self.joint_angles[1]),
            "j3": float(self.joint_angles[2]),
            "j4": float(self.joint_angles[3]),
            "speed": float(self.motor_speed[0]),
            "torque": float(self.motor_torque[0]),
            "gripper_open": bool(self.gripper_open_flag),
        }
        self.movement_history.append(entry)
        print(
            f"Recorded Speed: {entry['speed']:.3f}, Torque: {entry['torque']:.3f}\n"
            f"Joint Positions: {entry['j1']:.2f}, {entry['j2']:.2f}, {entry['j3']:.2f}, {entry['j4']:.2f}\n"
            f"Gripper open: {int(entry['gripper_open'])}"
        )

    def delete_last_recorded_configuration(self):
        if not self.movement_history:
            print("No last history position")
            return
        self.movement_history.pop()

    def play_configuration_history(self, dwell_after_move_s=1.0, dwell_after_grip_s=3.0):
        if not self.movement_history:
            print("No recorded configurations.")
            return
        for rec in self.movement_history:
            s = rec["speed"]
            t = rec["torque"]
            self.set_speed([s, s, s, s], overwrite_speeds=True)
            self.set_torque_limit([t, t, t, t])
            self.move_j(rec["j1"], rec["j2"], rec["j3"], rec["j4"])
            time.sleep(dwell_after_move_s)
            if self.gripper_open_flag != rec["gripper_open"]:
                self.actuate_gripper()
                time.sleep(dwell_after_grip_s)


# ------------------------------- Example usage --------------------------------
# Example 1:
# Demonstrates how to move the robot joints into desired joint positions [degrees]
# You might need to adjust the offsets in the code above to make the zero point suit your needs.
# Just comment out the code snippet, input desired joint angles in robot.move_j() and run.

if __name__ == "__main__":
    robot = MyRobot()
    if robot.init_status:
        try:

            #Reading off position for calibration:
            joint_angles = robot.read_joint_angles()
            print("Joint angles (deg):", joint_angles)


            # Example motions:
            robot.move_j(-12.02, -7.99, -95.76, 13.75) #Degrees

            robot.record_configuration()

            # Play it back
            #robot.play_configuration_history()

        finally:
            pass
            #robot.disable_motors()

#Example 2:
#Demonstrates how to move the robot joints into desired a list of joint positions [degrees]
#You might need to adjust the offsets in the code above to make the zero point suit your needs.
#Just comment out the code snippet, input desired joint angle sequence in robot.move_j() and run.

'''
if __name__ == "__main__":
    import time
    import numpy as np

    # Create robot instance
    robot = MyRobot()

    if robot.init_status:
        try:
            # ----------------- Define the sequence of joint angles -----------------
            # Each row = [J1, J2, J3, J4] in DEGREES
            # Adjust these based on your mechanical limits / workspace
            joint_sequence = np.array([
                [-12.02, -7.99, -95.76, 13.75],
                [-11.75, -7.21, -93.14, 10.35],
                [-11.24, -6.60, -90.47, 7.07],
                [-10.49, -6.14, -87.85, 3.99],
                [-9.33, -5.47, -85.76, 1.24],
                [-7.69, -5.20, -83.63, -1.17],
                [-5.96, -5.05, -81.82, -3.13],
                [-4.25, -4.67, -80.78, -4.55],
                [-2.04, -4.61, -79.95, -5.44],
                [-0.00, -4.60, -79.66, -5.74],
                [2.04, -4.61, -79.95, -5.44],
                [4.25, -4.67, -80.78, -4.55],
                [5.96, -5.05, -81.82, -3.13],
                [7.69, -5.20, -83.63, -1.17],
                [9.33, -5.47, -85.76, 1.24],
                [10.49, -6.14, -87.85, 3.99],
                [11.24, -6.60, -90.47, 7.07],
                [11.75, -7.21, -93.14, 10.35],
                [12.02, -7.99, -95.76, 13.75],
                [11.75, -8.85, -98.32, 17.17],
                [11.24, -9.82, -100.68, 20.50],
                [10.49, -10.87, -102.80, 23.67],
                [9.33, -11.60, -105.02, 26.62],
                [7.69, -12.54, -106.61, 29.14],
                [5.96, -13.37, -107.84, 31.21],
                [4.25, -13.76, -109.06, 32.82],
                [2.04, -14.16, -109.60, 33.76],
                [-0.00, -14.30, -109.78, 34.08],
                [-2.04, -14.16, -109.60, 33.76],
                [-4.25, -13.76, -109.06, 32.82],
                [-5.96, -13.37, -107.84, 31.21],
                [-7.69, -12.54, -106.61, 29.14],
                [-9.33, -11.60, -105.02, 26.62],
                [-10.49, -10.87, -102.80, 23.67],
                [-11.24, -9.82, -100.68, 20.50],
                [-11.75, -8.85, -98.32, 17.17],
                [-12.02, -7.99, -95.76, 13.75]
                # return home
            ])

            # ----------------- Run the motion sequence -----------------
            robot.enable_motors()
            print("\nStarting joint sequence...")
            time.sleep(1.0)

            #Number of loops
            loops = 5

            for loop in range(loops):
                print(f"\n🔁 Loop {loop + 1} of {loops} ----------------------------")
                for idx, pose in enumerate(joint_sequence):
                    print(f"\n➡️  Moving to pose {idx+1}: {pose}°")
                    robot.move_j(*pose)
                    robot.record_configuration()  # save to history
                    # Read and print measured joint positions
                    measured = robot.read_joint_angles()
                    print("Measured (deg):", np.round(measured, 2))
                    time.sleep(0)  # pause between poses



        except Exception as e:
            print("Error during sequence:", e)

        finally:
            robot.disable_motors()
            print("\nMotors disabled. Program finished.")
'''