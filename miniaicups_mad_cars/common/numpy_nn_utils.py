import pickle
from itertools import count

import numpy as np
import numpy.random as rng


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1)


class FFNet:
    def __init__(self, model_path):
        with open(model_path, 'rb') as file:
            self.data = pickle.load(file)

    def __call__(self, x):
        for i in count():
            weight_name = str.format('linear.{}.0.weight', i)
            bias_name = str.format('linear.{}.0.bias', i)
            if weight_name not in self.data:
                break
            x = self.data[weight_name] @ x + self.data[bias_name]
            x = np.tanh(x)

        probs_weight = self.data['head_probs.linear.weight']
        probs_bias = self.data['head_probs.linear.bias']

        probs = probs_weight @ x + probs_bias
        probs = softmax(probs)
        action = rng.choice(len(probs), p=probs)

        return action