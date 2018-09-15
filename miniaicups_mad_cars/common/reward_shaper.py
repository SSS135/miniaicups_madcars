from common.types import TickStep, NewMatchStep
from enum import Enum


class Winner(Enum):
    No = 0
    Self = 1
    Enemy = -1


class RewardShaper:
    reward_mult_by_map = {
        # PillMap
        1: {
            'aux_dist_x': 1,
            'aux_higher': 0,
            'aux_vel': 0,
            'aux_max_y': 0,
        },
        # PillHubbleMap
        2: {
            'aux_dist_x': 1,
            'aux_higher': 0,
            'aux_vel': 0,
            'aux_max_y': 0,
        },
        # PillHillMap
        3: {
            'aux_dist_x': 1,
            'aux_higher': 0,
            'aux_vel': 0,
            'aux_max_y': 0,
        },
        # PillCarcassMap
        4: {
            'aux_dist_x': 0,
            'aux_higher': 2,
            'aux_vel': 1,
            'aux_max_y': 2,
        },
        # IslandMap
        5: {
            'aux_dist_x': 0,
            'aux_higher': 0,
            'aux_vel': 1,
            'aux_max_y': 0,
        },
        # IslandHoleMap
        6: {
            'aux_dist_x': -0.5,
            'aux_higher': 0,
            'aux_vel': 0,
            'aux_max_y': 0,
        },
    }

    def __init__(self, game_info: NewMatchStep):
        self.game_info = game_info
        self.prev_tick: TickStep = None
        self.cur_vel: None = 0
        self.vel_blend = 0.9
        self.ended = False
        self.prev_max_y = None
        self.prev_min_dist_x = None

    def get_reward(self, tick: TickStep, winner: Winner, done: bool) -> (float, dict):
        assert not self.ended
        if done:
            self.ended = True
            reward = winner.value
            reward_info = dict(reward_info=dict(true_reward=reward))
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

            aux_rewards = dict(
                aux_dist_x=0.0001 * max(0, self.prev_min_dist_x - cur_dist_x),
                aux_higher=0.00004 * (tick.my_car.pos.y - tick.enemy_car.pos.y),
                aux_vel=0.0000004 * self.cur_vel ** 1.5,
                aux_max_y=0.00015 * max(0, max(0, tick.my_car.pos.y) ** 1.5 - self.prev_max_y ** 1.5),
            )
            reward_scale = self.reward_mult_by_map[self.game_info.proto_map.external_id]
            aux_rewards = {k: v * reward_scale[k] for k, v in aux_rewards.items()}
            aux_sum = sum(aux_rewards.values())

            reward_info = dict(reward_info=dict(aux_total=aux_sum, true_reward=0, **aux_rewards))

            self.prev_tick = tick
            self.prev_max_y = max(self.prev_max_y, tick.my_car.pos.y)
            self.prev_min_dist_x = min(self.prev_min_dist_x, cur_dist_x)

            return aux_sum, reward_info
