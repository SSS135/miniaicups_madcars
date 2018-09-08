import random
import time
from itertools import product, count
from typing import List

from .inverse_client import DetachedClient, DetachedGame, NoGraphicsGame
from .strategy import parse_step
from .types import NewMatchStep, TickStep


class DetachedMadCars:
    maps = ['PillMap', 'PillHubbleMap', 'PillHillMap', 'PillCarcassMap', 'IslandMap', 'IslandHoleMap']
    cars = ['Buggy', 'Bus', 'SquareWheelsBuggy']
    games = [','.join(t) for t in product(maps, cars)]

    def __init__(self):
        self.inv_game: DetachedGame = None
        self.clients: List[DetachedClient] = None
        self.game_infos: List[NewMatchStep] = None
        self.states: List[TickStep] = None

    def reset(self) -> List[TickStep]:
        if self.inv_game is not None:
            while not self.inv_game.done:
                self._send_commands(['stop', 'stop'])

        self.clients = [DetachedClient() for _ in range(2)]
        games = self.games[:]
        random.shuffle(games)
        game = NoGraphicsGame(self.clients, games, extended_save=False)
        for p in game.all_players:
            p.lives = 1
        self.inv_game = DetachedGame(game)

        self.game_infos = self._receive_states()
        self.states = self._receive_states()
        return self.states

    def step(self, commands: List[str]) -> (List[TickStep], int or None, bool):
        assert self.inv_game is not None
        assert not self.inv_game.done

        winner = None
        self._send_commands(commands)
        if self.inv_game.done:
            try:
                winner = self.clients.index(self.inv_game.winner)
            except ValueError:
                pass
        else:
            self.states = self._receive_states()

        return self.states, winner, self.inv_game.done

    def render(self, mode='human'):
        pass

    def _send_commands(self, commands: List[str]):
        assert not self.inv_game.done
        assert len(commands) == 2
        for client, cmd in zip(self.clients, commands):
            assert cmd in ('left', 'stop', 'right')
            out = {"command": cmd, 'debug': cmd}
            client.command_queue.put(out)
        wait_start_time = time.time()
        while any(c.message_queue.empty() for c in self.clients) and not self.inv_game.done:
            time.sleep(0.0001)
            if wait_start_time + 60 < time.time():
                raise RuntimeError('wait_start_time + 60 < time.time()')

    def _receive_states(self) -> List[TickStep or NewMatchStep]:
        assert not self.inv_game.done
        out = [c.message_queue.get(timeout=60) for c in self.clients]
        steps = [parse_step(dict(type=type, params=params)) for (type, params) in out]
        return steps