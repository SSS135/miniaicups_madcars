from common.types import TickStep


class RewardShaper:
    def __init__(self):
        self.prev_aux_reward = None

    def get_reward(self, tick: TickStep, have_won: bool, done: bool) -> float:
        if done:
            self.prev_aux_reward = 0
            return 1 if have_won else -1
        else:
            new_aux_reward = 0.003 * (tick.my_car.pos.y - tick.enemy_car.pos.y) + \
                             -0.002 * abs(tick.my_car.pos.x - tick.enemy_car.pos.x)
            reward = new_aux_reward - self.prev_aux_reward if self.prev_aux_reward is not None else 0
            self.prev_aux_reward = new_aux_reward
            # reward = -0.001
            return reward