"""
Core classes (with basic methods) that will be invoked when other, model classes are defined
"""


class Core:
    """Base class for causal_curve module"""

    def __init__(self):
        pass

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
