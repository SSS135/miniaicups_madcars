import math
import random

from miniaicups_mad_cars.common.strategy import Strategy
from miniaicups_mad_cars.common.types import TickStep


class Bot2Strategy(Strategy):
    move_ticks = 40
    angle_limit = 35
    max_closing_dist = 200

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

        diff_x = step.enemy_car.pos.x - step.my_car.pos.x
        if abs(diff_x) > self.max_closing_dist:
            cmd = 'right' if diff_x > 0 else 'left'

        angle = step.my_car.angle
        angle = angle / math.pi * 180
        if angle > self.angle_limit:
            cmd = 'left'
        if angle < -self.angle_limit:
            cmd = 'right'

        self.cur_tick += 1

        return {"command": cmd, 'debug': cmd}