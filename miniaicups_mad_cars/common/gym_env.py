import math
from itertools import product

import gym
import gym.spaces
import numpy as np
from ..bots.bot0 import Bot0Strategy
from ..bots.bot1 import Bot1Strategy
from ..bots.bot2 import Bot2Strategy
from ..bots.bot3 import Bot3Strategy
from .inverse_client import InverseClient, InverseGame, BotClient
from ..mechanic.game import Game
import random
from .strategy import parse_step
from .types import Car
from .vec2 import Vec2


class MadCarsAIEnv(gym.Env):
    observation_space = gym.spaces.Box(low=-1, high=1, shape=(16,), dtype=np.float32)
    action_space = gym.spaces.Discrete(3)
    strategies = [Bot0Strategy, Bot1Strategy, Bot2Strategy, Bot3Strategy]
    maps = ['PillMap', 'PillHubbleMap', 'PillHillMap', 'PillCarcassMap', 'IslandMap', 'IslandHoleMap']
    cars = ['Buggy', 'Bus', 'SquareWheelsBuggy']
    games = [','.join(t) for t in product(maps, cars)]
    commands = ['left', 'right', 'stop']
    pos_mean = Vec2(600, 200)
    pos_std = Vec2(1000, 400)

    def __init__(self):
        self.inv_game: InverseGame = None
        self.inv_client: InverseClient = None
        self.game_info = None

    def reset(self):
        if self.inv_game is not None:
            while not self.inv_game.done:
                self._send_action(0)

        strategy = random.choice(self.strategies)
        self.inv_client = InverseClient()
        bots = [BotClient(strategy()), self.inv_client]
        random.shuffle(bots)
        game = Game(bots, self.games, extended_save=False)
        for p in game.all_players:
            p.lives = 1
        self.inv_game = InverseGame(game)

        self.game_info = self._receive_message()
        tick = self._receive_message()
        return self._get_state(tick)

    def step(self, action):
        self._send_action(action)
        data = self._receive_message()
        return self._get_state(data)

    def render(self, mode='human'):
        pass

    def _get_state(self, data):
        state = [*self._get_car_state(data.my_car), *self._get_car_state(data.enemy_car)]
        state = np.array(state, dtype=np.float32)
        return state

    def _get_car_state(self, c: Car):
        return *self._norm_pos(c.pos), \
               *self._norm_pos(c.fw_pos),\
               *self._norm_pos(c.bw_pos), \
               math.sin(c.angle), \
               math.cos(c.angle)

    def _norm_pos(self, p):
        return (p - self.pos_mean) / self.pos_std

    def _send_action(self, action):
        cmd = self.commands[action]
        out = {"command": cmd, 'debug': cmd}
        self.inv_client.command_queue.put(out)

    def _receive_message(self):
        type, params = self.inv_client.message_queue.get()
        return parse_step(dict(type=type, params=params))

