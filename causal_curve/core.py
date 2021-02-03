"""
Core classes (with basic methods) that will be invoked when other, model classes are defined
"""

import numpy as np
from scipy.stats import norm


class Core:
    """Base class for causal_curve module"""

    def __init__(self):
        pass

    __version__ = "1.0.4"

    def get_params(self):
        """Returns a dict of all of the object's user-facing parameters

        Parameters
        ----------
        None

        Returns
        -------
        self: object
        """
        attrs = self.__dict__
        return dict(
            [(k, v) for k, v in list(attrs.items()) if (k[0] != "_") and (k[-1] != "_")]
        )

    def if_verbose_print(self, string):
        """Prints the input statement if verbose is set to True

        Parameters
        ----------
        string: str, some string to be printed

        Returns
        ----------
        None

        """
        if self.verbose:
            print(string)

    @staticmethod
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
            np.random.seed(random_seed)

    @staticmethod
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

    @staticmethod
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
