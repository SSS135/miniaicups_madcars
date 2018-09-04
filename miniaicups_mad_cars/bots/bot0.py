import random

from miniaicups_mad_cars.common.strategy import Strategy
from miniaicups_mad_cars.common.types import TickStep


class Bot0Strategy(Strategy):
    def tick(self, step: TickStep):
        commands = ['left', 'right', 'stop']
        cmd = random.choice(commands)
        return {"command": cmd, 'debug': cmd}