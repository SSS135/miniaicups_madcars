import json
import random
import math
from .strategy import Strategy
from .types import TickStep, Car


class Bot0Strategy(Strategy):
    def tick(self, step: TickStep):
        commands = ['left', 'right', 'stop']
        cmd = random.choice(commands)
        return {"command": cmd, 'debug': cmd}