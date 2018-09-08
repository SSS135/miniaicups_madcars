import math
from collections import deque

import numpy as np
import numpy.random
import random

from .types import TickStep, Car, NewMatchStep
from .vec2 import Vec2


class StateProcessor:
    state_size = 16 * 2
    static_state_size = 2 * (3 + 3 + 6 + 1)
    stacked_state_idx = [0, 1, 3, 15]
    frameskip = 4
    extra_frameskip = 3
    extra_frameskip_chance = 0.25
    observation_noise_scale = 0.01
    pos_mean = Vec2(600, 150)
    pos_std = Vec2(500, 400)
    deadline_std = 200
    velocity_interval = 2
    commands = ['left', 'stop', 'right']
    num_actions = 3

    def __init__(self, game_info):
        self._game_info: NewMatchStep = game_info
        states = [np.zeros(self.state_size, dtype=np.float32) for _ in range(max(self.stacked_state_idx) + 1)]
        self._states: deque[np.ndarray] = deque(states, maxlen=len(states))
        self._ticks: deque[TickStep] = deque(maxlen=len(states))
        self._frame_index: int = 0
        self._next_frame: int = 0
        self.side: int = 1

    def get_command(self, index) -> str:
        """ (left, left-stop, stop, right-stop, right) """
        if self.side == -1:
            index = self.num_actions - 1 - index
        # if index == 1 or index == 3:
        #     stop = (self._next_frame - self._frame_index) % 2 != 0
        #     index = 1 if stop else (0 if index == 1 else 2)
        # elif index == 2:
        #     index = 1
        # elif index == 4:
        #     index = 2
        return self.commands[index]

    def update_state(self, tick: TickStep) -> np.ndarray or None:
        self.side = tick.my_car.side
        if self._frame_index == self._next_frame:
            next_frameskip = self.frameskip if random.random() > self.extra_frameskip_chance else self.extra_frameskip
            self._next_frame = self._frame_index + next_frameskip
            prev_tick = self._ticks[self.velocity_interval - 1] if len(self._ticks) >= self.velocity_interval else tick
            state = [
                *self._get_car_state(tick.my_car, prev_tick.my_car),
                *self._get_car_state(tick.enemy_car, prev_tick.enemy_car),
            ]
            static_state = [
                tick.deadline_pos / self.deadline_std,
                (tick.my_car.pos.y - tick.deadline_pos) / 100,
                (tick.enemy_car.pos.y - tick.deadline_pos) / 100,
                *self._one_hot(self._game_info.proto_car.external_id - 1, 3),
                *self._one_hot(self._game_info.proto_map.external_id - 1, 6),
                self.side,
            ]
            static_state = np.array(static_state)
            state = np.array(state, dtype=np.float32)
            self._states.appendleft(state)
            self._ticks.appendleft(tick)
            cur_state = np.concatenate([self._states[i] for i in self.stacked_state_idx]).flatten()
            cur_state = np.concatenate((cur_state, static_state, -static_state))
            cur_state += np.random.normal(0, self.observation_noise_scale, cur_state.shape)
        else:
            cur_state = None
        self._frame_index += 1
        return cur_state

    def _one_hot(self, n, count):
        return [int(n == i) for i in range(count)]

    def _get_car_state(self, cur: Car, prev: Car):
        pos = self._norm_pos(cur.pos)
        fw_pos = self._norm_pos(cur.fw_pos)
        bw_pos = self._norm_pos(cur.bw_pos)
        cfw_pos = fw_pos * math.pi * 20
        cbw_pos = bw_pos * math.pi * 10

        wheel_pos = np.array([*fw_pos, *bw_pos])
        prev_wheel_pos = np.array([*self._norm_pos(prev.fw_pos), *self._norm_pos(prev.bw_pos)])

        avg_frameskip = self.extra_frameskip_chance * self.extra_frameskip + \
                        (1 - self.extra_frameskip_chance) * self.frameskip
        vel_mult = 60 / (self.velocity_interval * avg_frameskip)

        vel = vel_mult * (wheel_pos - prev_wheel_pos)
        vel = np.tanh(vel)

        return [*pos,
                *fw_pos,
                *bw_pos,
                *vel,
                math.sin(cfw_pos.x),
                math.cos(cfw_pos.y),
                math.sin(cbw_pos.x),
                math.cos(cbw_pos.y),
                *self.polar_angle(cur.angle)]

    def _norm_pos(self, p):
        p = (p - self.pos_mean) / self.pos_std
        p.x *= self.side
        return p

    def polar_angle(self, rads):
        return math.sin(rads * self.side), math.cos(rads * self.side)
