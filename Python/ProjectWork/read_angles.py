"""Live joint-angle readout with torque off, for calibration: move the arm by hand and watch the angles.

Run: python read_angles.py   (Ctrl+C to stop)
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # MyRobot.py is one folder up
import time
from dynamixel_sdk import PortHandler, PacketHandler
from MyRobot import MyRobot

port, pk = PortHandler(MyRobot.DEVICENAME), PacketHandler(MyRobot.PROTOCOL_VERSION)
port.openPort()
port.setBaudRate(MyRobot.BAUDRATE)
try:
    for i in (1, 2, 3, 4):
        pk.write1ByteTxRx(port, i, MyRobot.ADDR_MX_TORQUE_ENABLE, 0)  # torque off so the arm moves freely
    while True:
        q = [pk.read2ByteTxRx(port, i, MyRobot.ADDR_MX_PRESENT_POSITION)[0] * MyRobot.DEG_PER_TICK - MyRobot.JOINT_OFFSETS[i - 1] for i in (1, 2, 3, 4)]
        print("  ".join(f"q{i+1} = {v:+7.1f}°" for i, v in enumerate(q)), end="\r", flush=True)
        time.sleep(0.2)
except KeyboardInterrupt:
    print()
finally:
    port.closePort()
