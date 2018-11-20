#!/usr/bin/env python
import numpy as np

from flowder import choice


def random_choice(source, num: int, seed: int = None):
    assert len(source) >= num
    if seed is not None:
        st = np.random.get_state()
        np.random.seed(seed)
        ind = np.random.choice(len(source), num)
        np.random.set_state(st)
    else:
        ind = np.random.choice(len(source), num)

    return choice(source, list(ind))


def permutation(source, seed=None):
    if seed is not None:
        st = np.random.get_state()
        np.random.seed(seed)
        ind = np.random.permutation(np.arange(len(source)))
        np.random.set_state(st)
    else:
        ind = np.random.permutation(np.arange(len(source)))

    return choice(source, list(ind))
