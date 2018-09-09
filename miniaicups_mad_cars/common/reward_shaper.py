from common.types import TickStep
import numpy as np


class RewardShaper:
    def __init__(self):
        self.prev_tick: TickStep = None
        self.cur_vel: None = 0
        self.vel_blend = 0.9
        self.ended = False
        self.prev_max_y = None
        self.prev_min_dist_x = None

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
                return 0, {}

            new_vel = (tick.my_car.fw_pos - self.prev_tick.my_car.fw_pos).magnitude + \
                      (tick.my_car.bw_pos - self.prev_tick.my_car.bw_pos).magnitude
            self.cur_vel += (1 - self.vel_blend) * (30 * new_vel - self.cur_vel)
            cur_dist_x = abs(tick.my_car.pos.x - tick.enemy_car.pos.x) ** 1.5
            if self.prev_max_y is None:
                self.prev_max_y = max(0, tick.my_car.pos.y)
                self.prev_min_dist_x = cur_dist_x

            r_dist_x = 0.0001 * max(0, self.prev_min_dist_x - cur_dist_x)
            r_higher = 0.00003 * (tick.my_car.pos.y - tick.enemy_car.pos.y)
            r_vel = 0.0000005 * self.cur_vel ** 1.5
            r_max_y = 0.0002 * max(0, max(0, tick.my_car.pos.y) ** 1.5 - self.prev_max_y ** 1.5)

            reward = r_dist_x + r_higher + r_vel + r_max_y

            reward_info = dict(reward_info=dict(
                dist_x=r_dist_x, higher=r_higher, vel=r_vel, max_y=r_max_y, reward=reward, raw=0))

            self.prev_tick = tick
            self.prev_max_y = max(self.prev_max_y, tick.my_car.pos.y)
            self.prev_min_dist_x = min(self.prev_min_dist_x, cur_dist_x)

            return reward, reward_info
