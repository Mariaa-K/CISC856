# need to: pip install procgen

import gym  # must be version 0.19.0


class ProcgenEnv:

    def __init__(self, env_name='coinrun', num_levels=200, start_level=0, distribution_mode='hard'):

        self.env_name = env_name
        self.full_env_name = 'procgen:procgen-'+self.env_name+'-v0'
        self.num_levels = num_levels
        self.start_level = start_level
        self.distribution_mode = distribution_mode
        self.reward_range = [0]
        
        self.env = gym.make(self.full_env_name, render_mode="human")
        self.obs = self.env.reset()
    
    def step(self, action):
        # self.env.action_space.sample()
        obs, rew, done, info = self.env.step(action)  # removed fourth output info
        self.env.render()
        self.obs = obs

        return obs, rew, done, info

    def reset(self):
        self.obs = self.env.reset()
        return self.obs

    @property
    def actions_count(self):
        return self.env.action_space.n

    @property
    def observation_space(self):
        return self.observation_space
