import random
import time
from itertools import product

import gym.spaces
import numpy as np

from .inverse_client import DetachedClient, DetachedGame, BotClient, NoGraphicsGame
from .strategy import parse_step
from .types import NewMatchStep
from ..bots.bot0 import Bot0Strategy
from ..bots.bot1 import Bot1Strategy
from ..bots.bot2 import Bot2Strategy
from ..bots.bot3 import Bot3Strategy
from ..common.state_processor import StateProcessor


class MadCarsAIEnv(gym.Env):
    observation_space = gym.spaces.Box(low=-1, high=1, shape=(
        StateProcessor.state_size * len(StateProcessor.stacked_state_idx) + StateProcessor.static_state_size,), dtype=np.float32)
    action_space = gym.spaces.Discrete(3)
    strategies = [Bot0Strategy, Bot1Strategy, Bot2Strategy, Bot3Strategy]
    maps = ['PillMap', 'PillHubbleMap', 'PillHillMap', 'PillCarcassMap', 'IslandMap', 'IslandHoleMap']
    cars = ['Buggy', 'Bus', 'SquareWheelsBuggy']
    games = [','.join(t) for t in product(maps, cars)]
    commands = ['left', 'right', 'stop']

    def __init__(self):
        self.inv_game: DetachedGame = None
        self.inv_client: DetachedClient = None
        self.game_info: NewMatchStep = None
        self.bots = None
        self.proc = None

    def reset(self):
        if self.inv_game is not None:
            while not self.inv_game.done:
                self._send_action(0)

        strategy = random.choice(self.strategies)
        self.inv_client = DetachedClient()
        self.bots = [BotClient(strategy()), self.inv_client]
        random.shuffle(self.bots)
        games = self.games.copy()
        random.shuffle(games)
        game = NoGraphicsGame(self.bots, games, extended_save=False)
        for p in game.all_players:
            p.lives = 1
        self.inv_game = DetachedGame(game)

        self.game_info = self._receive_message()
        self.proc = StateProcessor(self.game_info)
        data = self._receive_message()
        return self.proc.update_state(data)

    def step(self, action):
        assert self.inv_game is not None
        assert not self.inv_game.done

        while True:
            self._send_action(action)
            if self.inv_game.done:
                reward = 1 if self.inv_game.winner == self.inv_client else -1
                state = self.observation_space.sample()
                state.fill(0)
                break
            data = self._receive_message()
            state = self.proc.update_state(data)
            if state is not None:
                reward = 0
                break

        return state, reward, self.inv_game.done, dict(game_info=self.game_info)

    def render(self, mode='human'):
        pass

    def _send_action(self, action):
        assert not self.inv_game.done
        cmd = self.commands[action]
        out = {"command": cmd, 'debug': cmd}
        self.inv_client.command_queue.put(out)
        while self.inv_client.message_queue.empty() and not self.inv_game.done:
            time.sleep(0.0001)

    def _receive_message(self):
        assert not self.inv_game.done
        type, params = self.inv_client.message_queue.get()
        return parse_step(dict(type=type, params=params))