import os
import gym
import cv2
import numpy as np
import glob
from environments.dmc2gym import natural_imgsource


class gymWrapper:
    def __init__(self, env_name, resource_files=None, img_source=None, total_frames=0, num_envs=1):
        self.observation_type = 'image'
        self.screen_height = 64
        self.screen_width = 64
        self.env = gym.make(env_name, num_envs=num_envs)
        self.reset()
        self._img_source = img_source
        if img_source is not None:
            shape2d = (self.screen_height, self.screen_width)
            if img_source == 'color':
                self._bg_source = natural_imgsource.RandomColorSource(shape2d)
            elif img_source == 'noise':
                self._bg_source = natural_imgsource.NoiseSource(shape2d)
            else:
                files = glob.glob(os.path.expanduser(resource_files))
                assert len(files), "Pattern {} does not match any files".format(resource_files)
                if img_source == 'images':
                    self._bg_source = natural_imgsource.RandomImageSource(shape2d,
                                                                          files,
                                                                          grayscale=True,
                                                                          total_frames=total_frames)
                elif img_source == 'video':
                    self._bg_source = natural_imgsource.RandomVideoSource(shape2d,
                                                                          files,
                                                                          grayscale=True,
                                                                          total_frames=total_frames)
                else:
                    raise Exception('img_source %s not defined' % img_source)

    @property
    def observation_space(self):
        return self._observe()

    @property
    def actions_count(self):
        return self.env.action_space.n

    def reset(self):
        self.env.reset()
        return self._observe()

    def _observe(self):
        return cv2.resize(self.env.render(mode='rgb_array'), dsize=(64, 64), interpolation=cv2.INTER_CUBIC)

    def _get_obs(self):
        if self.observation_type == 'position':
            return np.concatenate(list(self.time_step.observation.values()))
        else:
            obs = cv2.resize(self.env.render(mode='rgb_array'),
                             dsize=(self.screen_height, self.screen_width),
                             interpolation=cv2.INTER_CUBIC)
            if self._img_source is not None:
                mask = np.logical_and(obs[:, :, 0] == np.asarray([255]),
                                      obs[:, :, 1] == np.asarray([255]),
                                      obs[:, :, 2] == np.asarray([255]))
                bg = self._bg_source.get_image()
                obs[mask] = bg[mask]
            return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._get_obs()
        return obs, reward, done, info

    @property
    def reward_range(self):
        return [1]
