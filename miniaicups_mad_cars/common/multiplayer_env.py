import random
from itertools import count
from typing import List

import numpy as np
from common.reward_shaper import RewardShaper
from ppo_pytorch.common.multiplayer_env import MultiplayerEnv

from .types import NewMatchStep, TickStep
from ..common.state_processor import StateProcessor
from .detached_mad_cars import DetachedMadCars
from .bot_env import MadCarsAIEnv


class PlayerProcessor:
    def __init__(self, game_info: NewMatchStep):
        self.proc: StateProcessor = None
        self.prev_aux_reward: float = None
        self.ticks: List[TickStep] = None
        self.proc = StateProcessor(game_info)
        self.reward_shaper = RewardShaper()

    def get_command(self, index, rand_state) -> str:
        random.setstate(rand_state)
        return self.proc.get_command(index)

    def step(self, tick: TickStep, have_won: bool, done: bool, rand_state) -> (np.ndarray, float):
        random.setstate(rand_state)
        reward = self.reward_shaper.get_reward(tick, have_won, done)
        if done:
            return None, reward
        state = self.proc.update_state(tick)
        return state, reward


class MadCarsMultiplayerEnv(MultiplayerEnv):
    observation_space = MadCarsAIEnv.observation_space
    action_space = MadCarsAIEnv.action_space

    def __init__(self):
        super().__init__(num_players=2)
        self.game = DetachedMadCars()
        self.processors: List[PlayerProcessor] = None
        self.states = None

    def reset(self) -> np.ndarray:
        ticks = self.game.reset()
        rand_state = random.getstate()
        self.processors = [PlayerProcessor(inf) for inf in self.game.game_infos]
        self.states, _ = zip(*[p.step(t, False, False, rand_state) for (p, t) in zip(self.processors, ticks)])
        self.states = np.array(self.states)
        return self.states

    def step(self, actions: List[int]) -> (np.ndarray, np.ndarray, bool, List[dict]):
        while True:
            rand_state = random.getstate()
            commands = [p.get_command(a, rand_state) for (p, a) in zip(self.processors, actions)]
            ticks, winner_id, done = self.game.step(commands)
            rand_state = random.getstate()
            new_states, rewards = zip(*[p.step(t, i == winner_id, done, rand_state)
                                        for (i, p, t) in zip(count(), self.processors, ticks)])
            assert sum(s is None for s in new_states) in (0, len(new_states))
            if new_states[0] is not None:
                self.states = np.array(new_states)
            if done or new_states[0] is not None:
                return self.states, np.array(rewards), done, [{}, {}]

    def render(self, mode='human'):
        pass
