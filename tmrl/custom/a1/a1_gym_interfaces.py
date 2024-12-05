# rtgym interfaces for the Unitree A1 robot

# standard library imports

import logging

# third-party imports
import gymnasium.spaces as spaces
import numpy as np
import time

import gymnasium as gym
import gymnasium.spaces as spaces

from dm_control.utils import rewards
from tmrl.custom.a1.utils import resetters
from tmrl.custom.a1.utils import env_builder
from tmrl.custom.a1.utils import a1, a1_robot, robot_config
from tmrl.custom.a1.utils import pose3d

# third-party imports
from rtgym import RealTimeGymInterface

# local imports
import tmrl.config.config_constants as cfg


# Interface for Unitree A1 ========================================================================================


def get_run_reward(x_velocity: float, move_speed: float,
                   cos_pitch_cos_roll: float, terminate_pitch_roll_deg: float):
    termination = np.cos(np.deg2rad(terminate_pitch_roll_deg))
    upright = rewards.tolerance(cos_pitch_cos_roll,
                                bounds=(termination, float('inf')),
                                sigmoid='linear',
                                margin=termination + 1,
                                value_at_margin=0)

    forward = rewards.tolerance(x_velocity,
                                bounds=(move_speed, 2 * move_speed),
                                margin=move_speed,
                                value_at_margin=0,
                                sigmoid='linear')

    return upright * forward  # [0, 1] => [0, 10]


class A1WalkInterface(RealTimeGymInterface):
    def __init__(self,
                 zero_action: np.ndarray = np.asarray([0.05, 0.9, -1.8] * 4),
                 action_offset: np.ndarray = np.asarray([0.2, 0.4, 0.4] * 4),):

        logging.info("WARNING: this code executes low-level control on the robot.")
        input("Press enter to continue...")

        self.env = env_builder.build_imitation_env()
        self.resetter = resetters.GetupResetter(self.env, True, standing_pose=zero_action)
        self.original_kps = self.env.robot._motor_kps.copy()
        self.original_kds = self.env.robot._motor_kds.copy()

        min_actions = zero_action - action_offset
        max_actions = zero_action + action_offset

        self.action_space = gym.spaces.Box(min_actions, max_actions)
        self._estimated_velocity = np.zeros(3)
        self._reset_var()

        obs = self.observation()

        self.observation_space = gym.spaces.Box(float("-inf"),
                                                float("inf"),
                                                shape=obs.shape,
                                                dtype=np.float32)

    def get_observation_space(self):
        pass

    def get_action_space(self):
        pass

    def get_default_action(self):
        pass

    def send_control(self, control):
        pass

    def reset(self, seed=None, options=None):
        pass

    def get_obs_rew_terminated_info(self):
        pass

    def wait(self):
        pass
