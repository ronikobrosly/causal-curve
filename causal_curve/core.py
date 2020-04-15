"""
Core classes (with basic methods) that will be invoked when other, model classes are defined
"""


class Core():

    def __init__(self):
        pass


    def get_params(self):
        """
        returns a dict of all of the object's user-facing parameters

        Parameters
        ----------
        None

        Returns
        -------
        Dict of these parameters
        """
        attrs = self.__dict__
        return dict([(k,v) for k,v in list(attrs.items()) if (k[0] != '_') and (k[-1] != '_')])
