# need to: pip install procgen

import gym  # must be version 0.19.0
import procgen  # This seems to be required for Jeremy's computer.  I hope it works for you guys.


class ProcgenEnv:

    def __init__(self, env_name='coinrun', num_levels=200, start_level=0, distribution_mode='easy'):

        self.env_name = env_name
        self.full_env_name = 'procgen-'+self.env_name+'-v0'
        self.num_levels = num_levels
        self.start_level = start_level
        self.distribution_mode = distribution_mode
        self.reward_range = [0]
        
        self.env = gym.make(self.full_env_name, render_mode="human")
        self.obs = self.env.reset() / 255.
    
    def step(self, action):
        # self.env.action_space.sample()
        # if isinstance(action, tuple):
        #     action = (action[0],)
        obs, rew, done, info = self.env.step(action)  # removed fourth output info
        self.env.render()
        self.obs = obs / 255.

        return obs, rew, done, info

    def reset(self):
        self.obs = self.env.reset()
        return self.obs / 255.

    @property
    def actions_count(self):
        return self.env.action_space.n

    @property
    def observation_space(self):
        return self.env.observation_space
