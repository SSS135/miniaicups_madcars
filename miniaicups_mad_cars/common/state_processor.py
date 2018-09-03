import math
import numpy as np

from .types import TickStep, Car, NewMatchStep
from .vec2 import Vec2


class StateProcessor:
    state_size = 8 * 2
    static_state_size = 3 + 3 + 6
    stacked_state_idx = [0, 1, 3, 15]
    frameskip = 4
    pos_mean = Vec2(600, 150)
    pos_std = Vec2(1000, 400)
    deadline_std = 200

    def __init__(self, game_info):
        self.game_info: NewMatchStep = game_info
        self.states = [np.zeros(self.state_size, dtype=np.float32) for _ in range(max(self.stacked_state_idx) + 1)]
        self.frame_index = 0

    def update_state(self, tick: TickStep) -> np.ndarray or None:
        if self.frame_index % self.frameskip == 0:
            mc = np.array(self._get_car_state(tick.my_car))
            ec = self._get_car_state(tick.enemy_car)
            state = [
                *mc,
                *ec,
            ]
            static_state = [
                tick.deadline_pos / self.deadline_std,
                (tick.my_car.pos.y - tick.deadline_pos) / 100,
                (tick.enemy_car.pos.y - tick.deadline_pos) / 100,
                *self._one_hot(self.game_info.proto_car.external_id - 1, 3),
                *self._one_hot(self.game_info.proto_map.external_id - 1, 6),
            ]
            state = np.array(state, dtype=np.float32)
            self.states.pop(-1)
            self.states.insert(0, state)
            cur_state = np.concatenate([self.states[i] for i in self.stacked_state_idx]).flatten()
            cur_state = np.concatenate((cur_state, static_state))
        else:
            cur_state = None
        self.frame_index += 1
        return cur_state

    def _one_hot(self, n, nmax):
        return [int(n == i) for i in range(nmax)]

    def _get_car_state(self, c: Car):
        pos = self._norm_pos(c.pos)
        return (*pos,
                *self._norm_pos(c.fw_pos),
                *self._norm_pos(c.bw_pos),
                math.sin(c.angle),
                math.cos(c.angle))

    def _norm_pos(self, p):
        return (p - self.pos_mean) / self.pos_std