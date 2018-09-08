import random
import time
from itertools import product, count
from typing import List

import gym.spaces
import numpy as np

from .inverse_client import DetachedClient, DetachedGame, BotClient, NoGraphicsGame
from .strategy import parse_step
from .types import NewMatchStep, TickStep
from ..bots.bot0 import Bot0Strategy
from ..bots.bot1 import Bot1Strategy
from ..bots.bot2 import Bot2Strategy
from ..bots.bot3 import Bot3Strategy
from .strategy import Strategy
from ..common.state_processor import StateProcessor
from .detached_mad_cars import DetachedMadCars


class MadCarsAIEnv(gym.Env):
    observation_len = StateProcessor.state_size * len(StateProcessor.stacked_state_idx) + \
                      StateProcessor.static_state_size
    observation_space = gym.spaces.Box(low=-1, high=1, shape=(observation_len,), dtype=np.float32)
    action_space = gym.spaces.Discrete(StateProcessor.num_actions)
    strategies = [Bot0Strategy, Bot1Strategy, Bot2Strategy, Bot3Strategy]

    def __init__(self):
        self.game = DetachedMadCars()
        self.proc: StateProcessor = None
        self.prev_aux_reward: float = 0
        self.internal_bot_index: int = None
        self.player_index: int = None
        self.bot: Strategy = None
        self.ticks: List[TickStep] = None
        self.state: np.ndarray = None

    def reset(self) -> np.ndarray:
        self.ticks = self.game.reset()
        self.internal_bot_index = random.randrange(2)
        self.player_index = (self.internal_bot_index + 1) % 2
        self.proc = StateProcessor(self.game.game_infos[self.player_index])
        self.bot = self._get_bot()
        self.bot.process_data(self.game.game_infos[self.internal_bot_index])
        self.state = None
        return self.proc.update_state(self.ticks[self.player_index])

    def step(self, action: int) -> (np.ndarray, float, bool, dict):
        while True:
            player_cmd = self.proc.get_command(action)
            bot_cmd = self.bot.process_data(self.ticks[self.internal_bot_index])['command']
            commands = [player_cmd, bot_cmd]
            if self.player_index == 1:
                commands.reverse()
            self.ticks, winner, done = self.game.step(commands)
            reward = self._get_reward(winner, done)
            if done:
                return self.state, reward, done, {}
            new_state = self.proc.update_state(self.ticks[self.player_index])
            if new_state is not None:
                self.state = new_state
                return self.state, reward, done, {}

    def render(self, mode='human'):
        pass

    def _get_bot(self) -> Strategy:
        strategy = random.choice(self.strategies)
        return strategy()

    def _get_reward(self, winner: int or None, done: bool) -> float:
        if done:
            self.prev_aux_reward = 0
            if winner is None:
                return 0
            else:
                return 1 if winner == self.player_index else -1
        else:
            data = self.ticks[self.player_index]
            new_aux_reward = 0.003 * (data.my_car.pos.y - data.enemy_car.pos.y) + \
                             -0.002 * abs(data.my_car.pos.x - data.enemy_car.pos.x)
            reward = new_aux_reward - self.prev_aux_reward
            self.prev_aux_reward = new_aux_reward
            # reward = -0.001
            return reward
