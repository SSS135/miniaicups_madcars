from ..common.state_processor import StateProcessor
from ..common.strategy import Strategy
from ..common.types import TickStep, NewMatchStep
from ..common.numpy_nn_utils import FFNet


commands = ['left', 'right', 'stop']


class NumpyFFBotStrategy(Strategy):
    def __init__(self, model_path):
        self.net = FFNet(model_path)
        self.proc = None
        self.cur_cmd = None

    def new_match(self, data: NewMatchStep):
        self.proc = StateProcessor(data)

    def tick(self, step: TickStep):
        state = self.proc.update_state(step)
        if state is not None:
            action = self.net(state)
            self.cur_cmd = commands[action]

        assert self.cur_cmd is not None
        return {"command": self.cur_cmd, 'debug': self.cur_cmd}