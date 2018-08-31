import gym
import gym.spaces
import numpy as np
from .bot0 import Bot0Strategy
from .bot1 import Bot1Strategy
from .bot2 import Bot2Strategy
from .bot3 import Bot3Strategy


class MadCarsAIEnv(gym.Env):
    observation_space = gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
    action_space = gym.spaces.Discrete(3)
    strategies = [Bot0Strategy, Bot1Strategy, Bot2Strategy, Bot3Strategy]

    def __init__(self):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def step(self, action):
        pass

