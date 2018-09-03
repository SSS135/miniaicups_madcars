import json
import random
import math
from ..common.strategy import Strategy
from ..common.types import TickStep, Car


class Bot3Strategy(Strategy):
    move_ticks = 40
    angle_limit = 40
    min_closing_dist = 300
    max_flee_dist = 150

    def __init__(self):
        self.cur_tick = 0
        self.cur_dir = 'left'
        self.dir_change_time = 0

    def tick(self, data: TickStep):
        commands = ['left', 'right']

        if self.dir_change_time < self.cur_tick:
            self.dir_change_time = self.cur_tick + self.move_ticks
            self.cur_dir = random.choice(commands)

        cmd = self.cur_dir

        # close
        diff_x = data.enemy_car.pos.x - data.my_car.pos.x
        if abs(diff_x) > self.min_closing_dist:
            cmd = 'right' if diff_x > 0 else 'left'

        # flee
        my_nearest_wheel = self.get_nearest_wheel(data.my_car, data.enemy_car)
        enemy_nearest_wheel = self.get_nearest_wheel(data.enemy_car, data.my_car)
        enemy_wheel_higher = my_nearest_wheel.y < enemy_nearest_wheel.y
        in_flee_range = (my_nearest_wheel - enemy_nearest_wheel).magnitude < self.max_flee_dist
        if enemy_wheel_higher and in_flee_range:
            cmd = 'right' if diff_x < 0 else 'left'

        # balance
        angle = data.my_car.angle
        angle = angle / math.pi * 180
        if angle > self.angle_limit:
            cmd = 'left'
        if angle < -self.angle_limit:
            cmd = 'right'

        self.cur_tick += 1

        return {"command": cmd, 'debug': cmd}

    def get_nearest_wheel(self, me: Car, target: Car):
        fw_dist = abs(me.fw_pos.x - target.pos.x)
        bw_dist = abs(me.bw_pos.x - target.pos.x)
        return me.fw_pos if fw_dist < bw_dist else me.bw_pos