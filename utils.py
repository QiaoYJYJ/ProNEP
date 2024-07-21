import os
import random
import numpy as np
import torch


CHARPROTLEN = 25


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def graph_collate_func(x):
    d, p, y = zip(*x)
    p = np.asarray(p,dtype=object)
    d = np.asarray(d,dtype=object)
    p=np.stack(p).astype(float)
    d=np.stack(d).astype(float)
    return torch.tensor(d), torch.tensor(p), torch.tensor(y)


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)

