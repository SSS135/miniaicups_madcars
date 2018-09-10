import pickle
import torch
import sys
from pytorch_rl_kit.models import FCActor


def pytorch_to_numpy(pt_path, np_path):
    model = torch.load(pt_path)
    if isinstance(model, FCActor):
        model = model.state_dict()
    model = {k: v.cpu().numpy() for (k, v) in model.items()}
    with open(np_path, 'w+b') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    inp = sys.argv[1]
    outp = sys.argv[2]
    print(f'converting "{inp}" -> "{outp}"')
    pytorch_to_numpy(inp, outp)