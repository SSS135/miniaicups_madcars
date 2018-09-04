import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, curdir)
sys.path.insert(1, os.path.join(curdir, os.pardir))


try:
    from gym.envs.registration import register
    register(
        id='MadCarsAI-v0',
        entry_point='miniaicups_mad_cars.common.gym_env:MadCarsAIEnv',
        reward_threshold=0.95,
    )
except:
    pass