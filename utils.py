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

    # 如果输入张量已经在GPU上，则不需要转换为NumPy数组
    d = torch.stack(d).float()
    p = torch.stack(p).float()
    y = torch.stack(y).float()

    return d, p, y

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


