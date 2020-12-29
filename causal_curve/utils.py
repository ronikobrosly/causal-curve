"""
Misc. utility functions
"""

import numpy as np
from scipy.stats import norm


def rand_seed_wrapper(random_seed=None):
    """Sets the random seed using numpy

    Parameters
    ----------
    random_seed: int, random seed number

    Returns
    ----------
    None
    """
    if random_seed is None:
        pass
    else:
        np.random_seed(random_seed)


def calculate_z_score(ci):
    """Calculates the critical z-score for a desired two-sided,
    confidence interval width.

    Parameters
    ----------
    ci: float, the confidence interval width (e.g. 0.95)

    Returns
    -------
    Float, critical z-score value
    """
    return norm.ppf((1 + ci) / 2)


def clip_negatives(number):
    """Helper function to clip negative numbers to zero

    Parameters
    ----------
    number: int or float, any number that needs a floor at zero

    Returns
    -------
    Int or float of modified value

    """
    if number < 0:
        return 0
    return number
