from common.types import TickStep
import numpy as np


class RewardShaper:
    def __init__(self):
        self.initial_aux_reward: float = None
        self.prev_tick: TickStep = None
        self.cur_vel: None = 0
        self.vel_blend = 0.9
        self.ended = False
        self.prev_max_y = 0

    def get_reward(self, tick: TickStep, have_won: bool, done: bool) -> (float, dict):
        assert not self.ended
        if done:
            self.ended = True
            reward = 1 if have_won else -1
            reward_info = dict(reward_info=dict(raw=reward))
            return reward, reward_info
        else:
            if self.prev_tick is None:
                self.prev_tick = tick

            new_vel = (tick.my_car.fw_pos - self.prev_tick.my_car.fw_pos).magnitude + \
                      (tick.my_car.bw_pos - self.prev_tick.my_car.bw_pos).magnitude
            self.cur_vel += (1 - self.vel_blend) * (30 * new_vel - self.cur_vel)

            r_dist_x = -0.0000001 * abs(tick.my_car.pos.x - tick.enemy_car.pos.x) ** 1.5
            r_higher = 0.0003 * (tick.my_car.pos.y - tick.enemy_car.pos.y)
            r_vel = 0.000001 * self.cur_vel ** 1.5
            r_max_y = 0.01 * max(0, tick.my_car.pos.y - self.prev_max_y)

            new_aux_reward = r_dist_x + r_higher + r_vel + r_max_y
            reward = new_aux_reward - self.initial_aux_reward if self.initial_aux_reward is not None else 0
            if self.prev_tick != tick and self.initial_aux_reward is None:
                self.initial_aux_reward = new_aux_reward

            reward_info = dict(reward_info=dict(
                dist_x=r_dist_x, higher=r_higher, vel=r_vel, max_y=r_max_y, reward=reward, raw=0))

            self.prev_tick = tick
            self.prev_max_y = max(self.prev_max_y, tick.my_car.pos.y)

            return reward, reward_info
