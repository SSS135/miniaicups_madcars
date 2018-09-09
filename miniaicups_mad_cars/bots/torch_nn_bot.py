import torch
from ppo_pytorch.models import FCActor
from ppo_pytorch.ppo import PPO_RNN, PPO

from ..common.bot_env import MadCarsAIEnv
from ..common.state_processor import StateProcessor
from ..common.strategy import Strategy
from ..common.types import TickStep, NewMatchStep


class TorchBotStrategy(Strategy):
    def __init__(self, model_path):
        self.rl = self._load_model(model_path)
        self.proc = None
        self.cur_cmd = None

    def new_match(self, data: NewMatchStep):
        self.proc = StateProcessor(data)
        self.rl.drop_collected_steps()

    def tick(self, step: TickStep):
        state = self.proc.update_state(step)
        if state is not None:
            action = self.rl.eval([state])
            self.cur_cmd = self.proc.get_command(action[0])

        assert self.cur_cmd is not None
        return {"command": self.cur_cmd, 'debug': self.cur_cmd}

    def _load_model(self, path):
        model = torch.load(path)
        alg = PPO if isinstance(model, FCActor) else PPO_RNN
        return alg(
            MadCarsAIEnv.observation_space,
            MadCarsAIEnv.action_space,
            num_actors=1,
            disable_training=True,
            cuda_eval=False,
            cuda_train=False,
            model_init_path=path,
        )