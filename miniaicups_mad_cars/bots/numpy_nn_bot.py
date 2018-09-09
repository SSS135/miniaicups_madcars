from ..common.numpy_ff_net import FFNet
from ..common.state_processor import StateProcessor
from ..common.strategy import Strategy
from ..common.types import TickStep, NewMatchStep
from ..common import STATE_SIZE_V1, STATE_SIZE_V2


class NumpyFFBotStrategy(Strategy):
    def __init__(self, model_path):
        self.net = FFNet(model_path)
        self.version = {STATE_SIZE_V2: 2, STATE_SIZE_V1: 1}[self.net.input_size]
        self.proc = None
        self.cur_cmd = None

    def new_match(self, data: NewMatchStep):
        self.proc = StateProcessor(data, self.version)

    def tick(self, step: TickStep):
        state = self.proc.update_state(step)
        if state is not None:
            action = self.net(state)
            self.cur_cmd = self.proc.get_command(action)

        assert self.cur_cmd is not None
        return {"command": self.cur_cmd, 'debug': self.cur_cmd}