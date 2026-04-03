"""seeding.py — 全局随机种子固定"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    固定所有随机源，保证实验可复现。

    Parameters
    ----------
    seed : int
        随机种子（建议 42）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
