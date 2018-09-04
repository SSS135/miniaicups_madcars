import pickle
import torch


def pytorch_to_numpy(pt_path, np_path):
    data = torch.load(pt_path)
    data = {k: v.cpu().numpy() for (k, v) in data.items()}
    with open(np_path, 'w+b') as file:
        pickle.dump(data, file)