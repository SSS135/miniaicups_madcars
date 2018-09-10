from typing import Tuple

import torch
from ppo_pytorch.models import FCActor
from ppo_pytorch.ppo import PPO_RNN, PPO
from ppo_pytorch.common import RLBase

from ..common import STATE_SIZE_V1, STATE_SIZE_V2
from ..common.bot_env import get_spaces
from ..common.state_processor import StateProcessor
from ..common.strategy import Strategy
from ..common.types import TickStep, NewMatchStep


class TorchBotStrategy(Strategy):
    def __init__(self, model_path: str):
        self.rl, self.version = self._load_model(model_path)
        self.proc = None
        self.cur_cmd = None

    def new_match(self, data: NewMatchStep):
        self.proc = StateProcessor(data, self.version)
        self.rl.drop_collected_steps()

    def tick(self, step: TickStep):
        state = self.proc.update_state(step)
        if state is not None:
            action = self.rl.eval([state])
            self.cur_cmd = self.proc.get_command(action[0])

        assert self.cur_cmd is not None
        return {"command": self.cur_cmd, 'debug': self.cur_cmd}

    @staticmethod
    def _load_model(path: str) -> Tuple[RLBase, int]:
        model = torch.load(path)
        version = {STATE_SIZE_V2: 2, STATE_SIZE_V1: 1}[model.observation_space.shape[0]]
        alg = PPO if isinstance(model, FCActor) else PPO_RNN
        return alg(
            *get_spaces(version),
            num_actors=1,
            disable_training=True,
            cuda_eval=False,
            cuda_train=False,
            model_init_path=path,), version
