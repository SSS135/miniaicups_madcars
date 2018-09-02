import json

from .dict_ex import DictEx
from .types import TickStep, NewMatchStep, Step


def parse_step(data) -> Step:
    data = DictEx(data)
    params = DictEx(data.params)
    type = data.type
    if type == 'tick':
        return TickStep(params)
    if type == 'new_match':
        return NewMatchStep(params)


class Strategy:
    def tick(self, data: TickStep) -> dict:
        return dict(command='stop', debug='')

    def new_match(self, data: NewMatchStep):
        pass

    def loop(self):
        while True:
            data = parse_step(json.loads(input()))
            out = self.process_data(data)
            if out is not None:
                print(json.dumps(out))

    def receive_message(self, type, params):
        data = parse_step(dict(type=type, params=params))
        return self.process_data(data)

    def process_data(self, data: Step) -> dict or None:
        if isinstance(data, TickStep):
            return self.tick(data)
        if isinstance(data, NewMatchStep):
            self.new_match(data)
        return None
