import sys
from bots.bot0 import Bot0Strategy
from bots.bot1 import Bot1Strategy
from bots.bot2 import Bot2Strategy
from bots.bot3 import Bot3Strategy


bot_num = int(sys.argv[1]) if len(sys.argv) > 1 else 3
s = [Bot0Strategy, Bot1Strategy, Bot2Strategy, Bot3Strategy][bot_num]()
s.loop()