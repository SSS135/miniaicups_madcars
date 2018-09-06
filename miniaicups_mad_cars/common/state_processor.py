import math

import numpy as np

from .types import TickStep, Car, NewMatchStep
from .vec2 import Vec2


class StateProcessor:
    state_size = 10 * 2
    static_state_size = 3 + 3 + 6
    stacked_state_idx = [0, 1, 3, 15]
    frameskip = 4
    pos_mean = Vec2(600, 150)
    pos_std = Vec2(500, 400)
    deadline_std = 200
    commands = ['left', 'stop', 'right']

    def __init__(self, game_info):
        self._game_info = game_info
        self._states = [np.zeros(self.state_size, dtype=np.float32) for _ in range(max(self.stacked_state_idx) + 1)]
        self._frame_index = 0
        self._side = 1

    def get_action_name(self, index):
        if self._side == -1:
            index = 2 - index
        return self.commands[index]

    def update_state(self, tick: TickStep) -> np.ndarray or None:
        self._side = tick.my_car.side
        if self._frame_index % self.frameskip == 0:
            state = [
                *self._get_car_state(tick.my_car),
                *self._get_car_state(tick.enemy_car),
            ]
            static_state = [
                tick.deadline_pos / self.deadline_std,
                (tick.my_car.pos.y - tick.deadline_pos) / 100,
                (tick.enemy_car.pos.y - tick.deadline_pos) / 100,
                *self._one_hot(self._game_info.proto_car.external_id - 1, 3),
                *self._one_hot(self._game_info.proto_map.external_id - 1, 6),
            ]
            state = np.array(state, dtype=np.float32)
            self._states.pop(-1)
            self._states.insert(0, state)
            cur_state = np.concatenate([self._states[i] for i in self.stacked_state_idx]).flatten()
            cur_state = np.concatenate((cur_state, static_state))
        else:
            cur_state = None
        self._frame_index += 1
        return cur_state

    def _one_hot(self, n, count):
        return [int(n == i) for i in range(count)]

    def _get_car_state(self, c: Car):
        return (*self._norm_pos(c.pos),
                *self._norm_pos(c.fw_pos),
                *self._norm_pos(c.bw_pos),
                *self.polar_angle(c.angle),
                math.sin(c.fw_angle / 3 * self._side),
                math.sin(c.bw_angle / 10 * self._side))

    def _norm_pos(self, p):
        p = (p - self.pos_mean) / self.pos_std
        p.x *= self._side
        return p

    def polar_angle(self, rads):
        return math.sin(rads * self._side), math.cos(rads * self._side)
