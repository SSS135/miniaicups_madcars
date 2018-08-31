import json

from .dict_ex import DictEx
from .types import TickStep, NewMatchStep


class Strategy:
    def loop(self):
        while True:
            data = self.parse_step(input())
            if isinstance(data, TickStep):
                out = self.tick(data)
                print(json.dumps(out))
            if isinstance(data, NewMatchStep):
                self.new_match(data)

    def tick(self, data: TickStep):
        pass

    def new_match(self, data: NewMatchStep):
        pass

    @staticmethod
    def parse_step(text):
        data = DictEx(json.loads(text))
        params = DictEx(data.params)
        type = data.type
        if type == 'tick':
            return TickStep(params)
        if type == 'new_match':
            return NewMatchStep(params)