import gym
import cv2


class gymWrapper:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.reset()

    @property
    def observation_space(self):
        return self._observe()

    def reset(self):
        self.env.reset()
        return self._observe()

    def _observe(self):
        return cv2.resize(self.env.render(mode='rgb_array'), dsize=(64, 64), interpolation=cv2.INTER_CUBIC)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._observe()
        return obs, reward, done, info

    @property
    def reward_range(self):
        return [1]
