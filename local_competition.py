import os
from typing import List, Tuple
import numpy as np
import random
import trueskill as ts
import argparse
from miniaicups_mad_cars.mechanic.strategy import Client
from miniaicups_mad_cars.common.inverse_client import BotClient, NoGraphicsGame
from miniaicups_mad_cars.bots.bot0 import Bot0Strategy
from miniaicups_mad_cars.bots.bot1 import Bot1Strategy
from miniaicups_mad_cars.bots.bot2 import Bot2Strategy
from miniaicups_mad_cars.bots.bot3 import Bot3Strategy
from miniaicups_mad_cars.bots.bot4 import Bot4Strategy
from miniaicups_mad_cars.bots.torch_nn_bot import TorchBotStrategy
from miniaicups_mad_cars.mechanic.constants import MATCHES_COUNT
import asyncio
from asyncio import events
from queue import Queue
from threading import Thread
import json


def main():
    parser = argparse.ArgumentParser(description='MadCars Bot Competition')
    parser.add_argument('-n', '--num-games', type=int,
                        help='Total number of games', required=True)
    parser.add_argument('-d', '--model-dir', type=str,
                        help='NN model root directory', required=True)
    parser.add_argument('-l', '--log-interval', type=int,
                        help='Rating print interval', default=10)
    parser.add_argument('-c', '--cache-path', type=str, default='competition_cache.json',
                        help='Ragings JSON cache')
    args = parser.parse_args()

    model_dir: str = args.model_dir
    games_remaining: int = args.num_games
    cache_path: str = args.cache_path
    log_interval: int = args.log_interval

    loop = events.new_event_loop()
    events.set_event_loop(loop)

    clients = get_simple_bots() + get_nn_bots(model_dir)
    clients = load_ratings(cache_path, clients)

    while games_remaining > 0:
        games_remaining -= log_interval
        ratings = run_competition(clients, log_interval)
        ratings = sorted(ratings, key=lambda t: -ts.expose(t[1]))
        save_ratings(cache_path, clients)
        print('-- RATINGS --')
        for name, rating in ratings:
            print(f'{ts.expose(rating):6.1f}: {name:<32}')


def load_ratings(path: str, clients: List[Tuple[str, Client]]) -> List[Tuple[str, Client, ts.Rating]]:
    ratings = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            for name, (mu, sigma) in json.load(f).items():
                ratings[name] = ts.Rating(mu, sigma)
    for name, _ in clients:
        if name not in ratings:
            ratings[name] = ts.Rating()
    return [(n, c, ratings[n]) for (n, c) in clients]


def save_ratings(path: str, clients: List[Tuple[str, Client, ts.Rating]]):
    clients = {n: (r.mu, r.sigma) for (n, c, r) in clients}
    with open(path, 'w') as f:
        json.dump(clients, f)


def get_simple_bots() -> List[Tuple[str, Client]]:
    bots = Bot0Strategy, Bot1Strategy, Bot2Strategy, Bot3Strategy, Bot4Strategy
    return [(b.__name__, BotClient(b())) for b in bots]


def get_nn_bots(dir: str) -> List[Tuple[str, Client]]:
    files = get_model_files(dir)
    return [(os.path.relpath(f, dir), BotClient(TorchBotStrategy(f))) for f in files]


def get_model_files(dir: str) -> List[str]:
    return [os.path.join(root, name)
            for root, dirs, files in os.walk(dir)
            for name in files
            if name.endswith('.pth')]


def run_competition(clients: List[Tuple[str, Client, ts.Rating]], num_games: int) -> List[Tuple[str, ts.Rating]]:
    for game_id in range(num_games):
        idx = np.random.choice(range(len(clients)), 2, replace=False)
        (n1, c1, r1), (n2, c2, r2) = np.take(clients, idx, axis=0)
        first_won = run_game(c1, c2)
        c_win, c_los, n_win, n_los, r_win, r_los, i_win, i_los = \
            (c1, c2, n1, n2, r1, r2, idx[0], idx[1]) if first_won \
                else (c2, c1, n2, n1, r2, r1, idx[1], idx[0])
        # print(f'{n_win:<32} {ts.expose(r_win):6.1f}   -- WON --   {n_los:<32} {ts.expose(r_los):6.1f}')
        r_win, r_los = ts.rate_1vs1(r_win, r_los)
        clients[i_win] = (n_win, c_win, r_win)
        clients[i_los] = (n_los, c_los, r_los)
    return [(n, r) for (n, c, r) in clients]


def run_game(ca: Client, cb: Client) -> bool:
    game = NoGraphicsGame([ca, cb], NoGraphicsGame.generate_matches(MATCHES_COUNT))
    # for p in game.all_players:
    #     p.lives = 1

    loop = events.get_event_loop()
    while not game.game_complete:
        loop.run_until_complete(game.tick())

    return game.get_winner().client == ca


if __name__ == '__main__':
    main()