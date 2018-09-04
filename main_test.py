import json
import random
import numpy as np
import pickle

with open('./model.pkl', 'rb') as file:
    data = pickle.load(file)
    mul = np.zeros(128) @ data['linear.1.0.weight']

while True:
    z = input()
    commands = ['left', 'right', 'stop']
    cmd = random.choice(commands)
    print(json.dumps({"command": cmd, 'debug': cmd}))