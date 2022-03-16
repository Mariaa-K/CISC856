# need to: pip install procgen

import gym  # must be version 0.19.0


class ProcgenEnv:

    def __init__(self, env_name='coinrun', num_levels=200, start_level=0, distribution_mode='hard'):

        self.env_name = env_name
        self.full_env_name = 'procgen:procgen-'+self.env_name+'-v0'
        self.num_levels = num_levels
        self.start_level = start_level
        self.distribution_mode = distribution_mode
        
        self.env = gym.make(self.full_env_name, render_mode="human")
        self.obs = self.env.reset()
    
    def step(self):

        obs, rew, done, _ = self.env.step(self.env.action_space.sample())  # removed fourth output info
        self.env.render()
        
        # pass obs, rew, done to Maria's code
