import os
import copy
import glob
import numpy as np
from environments import natural_imgsource
# Control Suite
from dm_control import suite


class DMCSimulation:
    """
For our DMC experiments, we ran a grid search over the learning rate in [1e − 4, 3e − 4, 7e − 4, 1e − 3], the number of
minibatches in [32, 8, 16, 64], the entropy coefficient in [0.0, 0.01, 0.001, 0.0001], and the number of PPO epochs per
update in [3, 5, 10, 20]. For Walker Walk and Finger Spin we use 2 action repeats and for the others we use 4. We also
use 3 stacked frames as observations. For Finger Spin, we found 10 ppo epochs, 0.0 entropy coefficient, 16 minibatches,
and 0.0001 learning rate to be the best. For Cartpole Balance, we used the same except for 0.001 learning rate. For
Walker Walk, we used 5 ppo epochs, 0.001 entropy coefficient, 32 minibatches, and 0.001 learning rate. For Cheetah Run,
we used 3 ppo epochs, 0.0001 entropy coefficient, 64 minibatches, and 0.001 learning rate. We use γ = 0.99, γ = 0.95 for
the generalized advantage estimates, 2048 steps, 1 process, value loss coefficient 0.5, and linear rate decay over
1 million environment steps. We also found crop to be the best augmentation from our set of eight transformations.
Any other hyperaparameters not mentioned here were set to the same values as the ones used for Procgen as described
above
    """
    def __init__(self, domain='cartpole', task='swingup', screen_width=64, screen_height=64, num_cameras=1,
                 observation_type='image', resource_files=None, img_source=None, total_frames=0):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.num_cameras = num_cameras
        self.observation_type = observation_type
        self._img_source = img_source
        if img_source is not None:
            shape2d = (screen_height, screen_width)
            if img_source == "color":
                self._bg_source = natural_imgsource.RandomColorSource(shape2d)
            elif img_source == "noise":
                self._bg_source = natural_imgsource.NoiseSource(shape2d)
            else:
                files = glob.glob(os.path.expanduser(resource_files))
                assert len(files), "Pattern {} does not match any files".format(
                    resource_files
                )
                if img_source == "images":
                    self._bg_source = natural_imgsource.RandomImageSource(shape2d,
                                                                          files,
                                                                          grayscale=True,
                                                                          total_frames=total_frames)
                elif img_source == "video":
                    self._bg_source = natural_imgsource.RandomVideoSource(shape2d,
                                                                          files,
                                                                          grayscale=True,
                                                                          total_frames=total_frames)
                else:
                    raise Exception("img_source %s not defined." % img_source)

        self.env = suite.load(domain, task)

        self.frames, self.ticks, self.rewards, self.observations = [], [], [], []
        self.spec, self.time_step = None, None
        self.reward_range = [0]
        self.reset()
        self.actions_count = self.env.action_spec().shape[0]
        self.observation_shape = self._get_obs().shape

    def reset(self):
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
        return self._get_obs()

    def _get_obs(self):
        if self.observation_type == 'position':
            return np.concatenate(list(self.time_step.observation.values()))
        else:
            # Check the following link for distractors.
            # https://github.com/facebookresearch/deep_bisim4control/blob/main/dmc2gym/natural_imgsource.py
            # Stolen from deep_bism4control
            obs = self.env.physics.render(
                                        camera_id=0,
                                        height=self.screen_height,
                                        width=self.screen_width)
            if self._img_source is not None:
                mask = np.logical_and((obs[:, :, 2] > obs[:, :, 1]), (obs[:, :, 2] > obs[:, :, 0]))  # hardcoded for dmc
                bg = self._bg_source.get_image()
                obs[mask] = bg[mask]
            obs = obs.transpose(2, 0, 1).copy()  # I'm not sure why this needs to happen.
            return obs

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
        return self._get_obs(), time_step.reward, time_step.last()
