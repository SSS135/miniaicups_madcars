import math

import numpy as np
from ..common.types import TickStep, Car
from ..common.vec2 import Vec2


class StateProcessor:
    state_size = 20 + 3
    stacked_states = 4
    frameskip = 4
    pos_mean = Vec2(600, 150)
    pos_std = Vec2(1000, 400)
    deadline_std = 300

    def __init__(self, game_info):
        self.game_info = game_info
        self.states = [np.zeros(self.state_size, dtype=np.float32) for _ in range(self.stacked_states)]
        self.frame_index = 0

    def update_state(self, tick: TickStep) -> np.ndarray or None:
        if self.frame_index % self.frameskip == 0:
            state = [*self._get_car_state(tick.my_car),
                     *self._get_car_state(tick.enemy_car),
                     tick.deadline_pos / self.deadline_std,
                     self.game_info.proto_car.external_id - 2,
                     (self.game_info.proto_map.external_id - 3.5) / 2.5]
            state = np.array(state, dtype=np.float32)
            self.states.pop(-1)
            self.states.append(state)
            cur_state = np.array(self.states).flatten()
        else:
            cur_state = None
        self.frame_index += 1
        return cur_state

    def _get_car_state(self, c: Car):
        pos = self._norm_pos(c.pos)
        return (*pos,
                math.sin(pos.x * 30),
                math.cos(pos.y * 30),
                *self._norm_pos(c.fw_pos),
                *self._norm_pos(c.bw_pos),
                math.sin(c.angle),
                math.cos(c.angle))

    def _norm_pos(self, p):
        return (p - self.pos_mean) / self.pos_std