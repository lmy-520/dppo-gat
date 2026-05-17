"""Seed utilities."""

def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
