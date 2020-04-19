import numpy as np

"""
Misc. utility functions
"""

def rand_seed_wrapper(random_seed = None):
    """Sets the random seed using numpy

    Parameters
    ----------
    random_seed: int

    Returns
    ----------
    None
    """
    if random_seed is None:
        return None
    np.random_seed(random_seed)
