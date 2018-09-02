import json
import random
import math
from miniaicups_mad_cars.common.strategy import Strategy
from miniaicups_mad_cars.common.types import TickStep, Car
from miniaicups_mad_cars.common.state_processor import StateProcessor
from miniaicups_mad_cars.common.gym_env import MadCarsAIEnv
from ppo_pytorch.ppo import PPO_RNN


class RNNBotStrategy(Strategy):
    def __init__(self, model_path):
        self.rl = self._load_model(model_path)
        self.proc = StateProcessor()
        self.cur_cmd = None
        self.memory = None

    def tick(self, step: TickStep):
        state = self.proc.update_state(step)
        if state is not None:
            action = self.rl.eval(state)
            self.cur_cmd = MadCarsAIEnv.commands[action]

        assert self.cur_cmd is not None
        return {"command": self.cur_cmd, 'debug': self.cur_cmd}

    def _load_model(self, path):
        return PPO_RNN(
            MadCarsAIEnv.observation_space,
            MadCarsAIEnv.action_space,
            num_actors=1,
            disable_training=True,
            cuda_eval=True,
            model_init_path=path,
        )