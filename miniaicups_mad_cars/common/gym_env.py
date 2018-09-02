import math
from itertools import product
import time

import gym
import gym.spaces
import numpy as np
from ..bots.bot0 import Bot0Strategy
from ..bots.bot1 import Bot1Strategy
from ..bots.bot2 import Bot2Strategy
from ..bots.bot3 import Bot3Strategy
from .inverse_client import DetachedClient, DetachedGame, BotClient, NoGraphicsGame
from .types import NewMatchStep
import random
from .strategy import parse_step
from .types import Car
from .vec2 import Vec2


class MadCarsAIEnv(gym.Env):
    state_size = 20 + 3
    stacked_states = 4
    frameskip = 4
    observation_space = gym.spaces.Box(low=-1, high=1, shape=(state_size * stacked_states,), dtype=np.float32)
    action_space = gym.spaces.Discrete(3)
    strategies = [Bot0Strategy, Bot1Strategy, Bot2Strategy, Bot3Strategy]
    maps = ['PillMap', 'PillHubbleMap', 'PillHillMap', 'PillCarcassMap', 'IslandMap', 'IslandHoleMap']
    cars = ['Buggy', 'Bus', 'SquareWheelsBuggy']
    games = [','.join(t) for t in product(maps, cars)]
    commands = ['left', 'right', 'stop']
    pos_mean = Vec2(600, 150)
    pos_std = Vec2(1000, 400)
    deadline_std = 300

    def __init__(self):
        self.inv_game: DetachedGame = None
        self.inv_client: DetachedClient = None
        self.game_info: NewMatchStep = None
        self.bots = None
        self.states = [np.zeros(self.state_size, dtype=np.float32) for _ in range(self.stacked_states)]
        self.cur_state = None

    def reset(self):
        if self.inv_game is not None:
            while not self.inv_game.done:
                self._send_action(0)

        strategy = random.choice(self.strategies)
        self.inv_client = DetachedClient()
        self.bots = [BotClient(strategy()), self.inv_client]
        random.shuffle(self.bots)
        game = NoGraphicsGame(self.bots, self.games, extended_save=False)
        for p in game.all_players:
            p.lives = 1
        self.inv_game = DetachedGame(game)

        self.game_info = self._receive_message(False)
        data = self._receive_message(False)
        self._update_state(data)
        return self.cur_state

    def step(self, action):
        assert self.inv_game is not None
        assert not self.inv_game.done

        self._send_action(action)

        if self.inv_game.done:
            reward = 1 if self.inv_game.winner == self.inv_client else -1
        else:
            data = self._receive_message(True)
            self._update_state(data)
            reward = 0

        return self.cur_state, reward, self.inv_game.done, dict(game_info=self.game_info)

    def render(self, mode='human'):
        pass

    def _update_state(self, data):
        state = [*self._get_car_state(data.my_car),
                 *self._get_car_state(data.enemy_car),
                 data.deadline_pos / self.deadline_std,
                 self.game_info.proto_car.external_id - 2,
                 (self.game_info.proto_map.external_id - 3.5) / 2.5]
        state = np.array(state, dtype=np.float32)
        self.states.pop(-1)
        self.states.append(state)
        self.cur_state = np.array(self.states).flatten()

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

    def _send_action(self, action):
        cmd = self.commands[action]
        out = {"command": cmd, 'debug': cmd}
        for _ in range(self.frameskip):
            self.inv_client.command_queue.put(out)
        while len(self.inv_client.message_queue.queue) < self.frameskip and not self.inv_game.done:
            time.sleep(0.0001)

    def _receive_message(self, use_frameskip):
        for _ in range(self.frameskip if use_frameskip else 1):
            type, params = self.inv_client.message_queue.get()
        return parse_step(dict(type=type, params=params))

