import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int] = None) -> None:
    """Set random seeds for reproducibility.

    Parameters
    ----------
    seed: Optional[int]
        Seed value to use. If None, the function does nothing.
    """
    if seed is None:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
