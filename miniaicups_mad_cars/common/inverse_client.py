from ..mechanic.strategy import Client
from ..mechanic.game import Game
from queue import Queue
from threading import Thread
from asyncio import events


class InverseClient(Client):
    def __init__(self):
        self.message_queue = Queue()
        self.command_queue = Queue()

    def get_command(self):
        return self.command_queue.get()

    def send_message(self, t, d):
        self.message_queue.put((t, d))


class BotClient(Client):
    def __init__(self, strategy):
        self.strategy = strategy
        self.command = None

    def get_command(self):
        return self.command

    def send_message(self, t, d):
        self.command = self.strategy.receive_message(t, d)


class InverseGame:
    def __init__(self, game: Game):
        self.game = game
        self.thread = Thread(target=self.run_thread)
        self.thread.start()

    @property
    def done(self):
        return self.game.game_complete

    def run_thread(self):
        loop = events.new_event_loop()
        events.set_event_loop(loop)

        while not self.done:
            loop.run_until_complete(self.game.tick())