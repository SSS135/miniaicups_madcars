import json
import random
import math
from functools import partial

from ..common.strategy import Strategy
from ..common.types import TickStep, NewMatchStep, Car
from ..common.state_processor import StateProcessor
from ..common.gym_env import MadCarsAIEnv
from ppo_pytorch.ppo import PPO_RNN, PPO
from ppo_pytorch.models import RNNActor, FCActor


class NNBotStrategy(Strategy):
    def __init__(self, model_path):
        self.rl = self._load_model(model_path)
        self.proc = None
        self.cur_cmd = None
        self.memory = None

    def new_match(self, data: NewMatchStep):
        self.proc = StateProcessor(data)

    def tick(self, step: TickStep):
        state = self.proc.update_state(step)
        if state is not None:
            action = self.rl.eval([state])
            self.cur_cmd = MadCarsAIEnv.commands[action[0]]

        assert self.cur_cmd is not None
        return {"command": self.cur_cmd, 'debug': self.cur_cmd}

    def _load_model(self, path):
        return PPO(
            MadCarsAIEnv.observation_space,
            MadCarsAIEnv.action_space,
            model_factory=partial(FCActor),
            num_actors=1,
            disable_training=True,
            cuda_eval=True,
            model_init_path=path,
        )


class RNNBotStrategy(NNBotStrategy):
    def _load_model(self, path):
        return PPO_RNN(
            MadCarsAIEnv.observation_space,
            MadCarsAIEnv.action_space,
            model_factory=partial(RNNActor, rnn_kind='qrnn', num_layers=4),
            num_actors=1,
            disable_training=True,
            cuda_eval=True,
            model_init_path=path,
        )