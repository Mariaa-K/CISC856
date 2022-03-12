# DeepMind Control Suite
# The basic mujoco wrapper.
import copy

from dm_control import mujoco

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

# PyMJCF
from dm_control import mjcf

# Composer high level imports
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer import variation

# Imports for Composer tutorial example
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises
from dm_control.locomotion.arenas import floors

# Control Suite
from dm_control import suite

# Run through corridor example
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.arenas import corridors as corridor_arenas
from dm_control.locomotion.tasks import corridors as corridor_tasks

# Soccer
# from dm_control.locomotion import soccer

# Manipulation
from dm_control import manipulation

import numpy as np


class Simulation:
    def __init__(self, domain='cartpole', task='swingup', screen_width=64, screen_height=64, num_cameras=1,
                 observation_type='image'):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.num_cameras = num_cameras
        self.observation_type = observation_type

        self.env = suite.load(domain, task)

        self.frames = []
        self.ticks = []
        self.rewards = []
        self.observations = []

        self.spec = self.env.action_spec()
        self.time_step = self.env.reset()
        self.frames.append([self.env.physics.render(
                                        camera_id=x,
                                        height=self.screen_height,
                                        width=self.screen_width
                                        ) for x in range(self.num_cameras)])

    def step(self, action, record=False):
        time_step = self.env.step(action)

        cameras = [self.env.physics.render(
                                        camera_id=x,
                                        height=self.screen_height,
                                        width=self.screen_width
                                        ) for x in range(self.num_cameras)]
        if record:
            self.frames.append(np.hstack(cameras))
        self.rewards.append(time_step.reward)
        self.observations.append(copy.deepcopy(time_step.observation))
        self.ticks.append(self.env.physics.data.time)
        terminal = False
        if self.observation_type == 'image':
            return cameras, time_step.reward, terminal
        elif self.observation_type == 'position':
            return np.concatenate(list(time_step.observation.values())), time_step.reward, terminal

