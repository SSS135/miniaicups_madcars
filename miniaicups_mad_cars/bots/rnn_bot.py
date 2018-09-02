import json
import random
import math
from miniaicups_mad_cars.common.strategy import Strategy
from miniaicups_mad_cars.common.types import TickStep, Car


class RNNBotStrategy(Strategy):
    def __init__(self, model):
        self.model = model

    def tick(self, step: TickStep):
        commands = ['left', 'right', 'stop']
        cmd = random.choice(commands)
        return {"command": cmd, 'debug': cmd}