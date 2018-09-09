import math

from miniaicups_mad_cars.common.strategy import Strategy
from miniaicups_mad_cars.common.types import TickStep


class Bot4Strategy(Strategy):
    def __init__(self):
        self.cur_tick = 0

    def tick(self, data: TickStep):
        if self.cur_tick < 20:
            cmd = 'stop'
        else:
            my_car = data.my_car
            enemy_car = data.enemy_car
            my_pos = my_car.pos
            enemy_pos = enemy_car.pos
            my_angle = my_car.angle

            cmd = 'left' if my_pos.x > enemy_pos.x else 'right'

            while my_angle > math.pi:
                my_angle -= 2 * math.pi
            while my_angle < -math.pi:
                my_angle += 2 * math.pi

            if my_angle > math.pi / 4:
                cmd = 'left'
            if my_angle < -math.pi / 4:
                cmd = 'right'

        self.cur_tick += 1

        return {"command": cmd, 'debug': cmd}
