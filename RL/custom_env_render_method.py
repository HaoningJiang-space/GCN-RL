from re import I
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from PIL import Image, ImageDraw

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.rllib.examples.envs.env_rendering_and_recording import EnvRenderCallback
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    add_rllib_example_script_experiment,
)
from ray import tune

""" 
default_iters is the number of training iterations to run.
default_reward is the reward to achieve in the test.
default_timesteps is the number of timesteps to run the test for.

"""


parser = add_rllib_example_script_args(
    default_iters=10,
    default_reward=9.0,
    default_timesteps = 10000,
)                                                

class CunstomRenderedCorridorEnv(gym.Env):
    def __init__(self,config):
        self.end_pos = config.get("corridor_length", 10)
        self.max_steps = config.get("max_steps", 100)
        self.cur_pos = 0
        self.steps = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(0.0,999.0, shape(1,), dtype=np.float32)
        
    def reset(self,*,seed=None,options = None):
        self.cur_pos = 0.0
        self.steps = 0
        return np.array([self.cur_pos],np.float32),{}
    def step(self,action):
        self.steps +=1
        assert action in [0,1], action
        if action ==0 and self.cur_pos >0 :
            self.cur_pos -= 1.0
        elif action ==1 :
            self.cur_pos +=1.0
        truncated = self.steps >= self.max_steps
        terminated = self.cur_pos >= self.end_pos
        return (
            np.array([self.cur_pos],np.float32)
            10.0 if terminated else -0.1,
            terminated,
            truncated,
            {},
        )
            
    
    