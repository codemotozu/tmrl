"""
This file defines the A1Robot Python object that enables controlling the real A1 robot.
A1Robot wraps the C++ Unitree Legged SDK.
The SDK was compiled as an .so file and translated to Python via pybind under the name robot_interface.
TMRL ships with this .so file.
"""

import numpy as np

from robot_interface import RobotInterface


class A1Robot:
    def __init__(self):
        self._interface = RobotInterface()
        self._initialize_interface()

        self._state = None
        self.orientation = None
        self.acceleration = None
        self.motor_angles = None
        self.motor_velocities = None
        self.motor_torques = None
        self.motor_temperatures = None

        self.update()

    def _initialize_interface(self):
        self._interface.send_command(np.zeros(60, dtype=np.float32))

    def update(self):
        self._state = self._interface.receive_observation()
        quat = self._state.imu.quaternion
        self.orientation = np.array([quat[1], quat[2], quat[3], quat[0]])
        self.acceleration = np.array(self._state.imu.accelerometer)
        self.motor_angles = np.array([motor.q for motor in self._state.motorState[:12]])
        self.motor_velocities = np.array([motor.dq for motor in self._state.motorState[:12]])
        self.motor_torques = np.array([motor.tauEst for motor in self._state.motorState[:12]])
        self.motor_temperatures = np.array([motor.temperature for motor in self._state.motorState[:12]])




