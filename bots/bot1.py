import json
import random
import numpy as np
import math
from ..common.strategy import Strategy
from ..common.types import TickStep, Car


class Bot1Strategy(Strategy):
    move_ticks = 40
    angle_limit = 40

    def __init__(self):
        self.cur_tick = 0
        self.cur_dir = 'left'
        self.dir_change_time = 0

    def tick(self, step: TickStep):
        commands = ['left', 'right']

        if self.dir_change_time < self.cur_tick:
            self.dir_change_time = self.cur_tick + self.move_ticks
            self.cur_dir = random.choice(commands)

        cmd = self.cur_dir

        angle = step.my_car.angle
        angle = angle / math.pi * 180
        if angle > self.angle_limit:
            cmd = 'left'
        if angle < -self.angle_limit:
            cmd = 'right'

        self.cur_tick += 1

        return {"command": cmd, 'debug': cmd}