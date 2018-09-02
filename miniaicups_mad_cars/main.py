import sys
import argparse
from bots.bot0 import Bot0Strategy
from bots.bot1 import Bot1Strategy
from bots.bot2 import Bot2Strategy
from bots.bot3 import Bot3Strategy
from bots.rnn_bot import RNNBotStrategy

parser = argparse.ArgumentParser(description='Bot for MadCars')

parser.add_argument('-b', '--bot-index', type=int,
                    help='Simple bot index', default=None)
parser.add_argument('-p', '--model-path', type=str,
                    help='RNN model path', default=None)

args = parser.parse_args()
bot_index = args.bot_index
model_path = args.model_path

assert bot_index is not None or model_path is not None

if bot_index is not None:
    bot = [Bot0Strategy, Bot1Strategy, Bot2Strategy, Bot3Strategy][bot_index]()
else:
    bot = RNNBotStrategy(model_path)

bot.loop()