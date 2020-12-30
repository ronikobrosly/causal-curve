"""
Defines the Generalized Prospensity Score (GPS) classifier model class
"""

import contextlib
import io
from pprint import pprint

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype, is_numeric_dtype
from pygam import LinearGAM, LogisticGAM, s
from scipy.special import logit
from scipy.stats import gamma, norm
import statsmodels.api as sm
from statsmodels.genmod.families.links import inverse_power as Inverse_Power
from statsmodels.tools.tools import add_constant

from causal_curve import GPS_core
from causal_curve.utils import calculate_z_score, rand_seed_wrapper


class GPS_classifier(GPS_core):
    """
    The GPS tool that handles binary outcomes. Inherits the GPS_core
    base class. See that base class code and docstring for more details.
    """

    def __init__():
        pass

    def _cdrc_predictions_binary(self, ci):
        """Returns the predictions of CDRC for each value of the treatment grid. Essentially,
        we're making predictions using the original treatment and gps_at_grid.
        To be used when the outcome of interest is binary.
        """
        # To keep track of cdrc predictions, we create an empty 2d array of shape
        # (n_samples, treatment_grid_num, 2). The last dimension is of length 2 because
        # we are going to keep track of the point estimate (log-odds) of the prediction, as well as
        # the standard error of the prediction interval (again, this is for the log odds)
        cdrc_preds = np.zeros((len(self.T), self.treatment_grid_num, 2), dtype=float)

        # Loop through each of the grid values, predict point estimate and get prediction interval
        for i in range(0, self.treatment_grid_num):

            temp_T = np.repeat(self.grid_values[i], repeats=len(self.T))
            temp_gps = self.gps_at_grid[:, i]

            temp_cdrc_preds = logit(
                self.gam_results.predict_proba(np.column_stack((temp_T, temp_gps)))
            )

            temp_cdrc_interval = logit(
                self.gam_results.confidence_intervals(
                    np.column_stack((temp_T, temp_gps)), width=ci
                )
            )

            standard_error = (
                temp_cdrc_interval[:, 1] - temp_cdrc_preds
            ) / calculate_z_score(ci)

            cdrc_preds[:, i, 0] = temp_cdrc_preds
            cdrc_preds[:, i, 1] = standard_error

        return np.round(cdrc_preds, 3)

    def predict_log_odds(self, T):
        """Calculates the predicted log odds of the highest integer class. Can
        only be used when the outcome is binary. Can be estimate for a single
        data point or can be run in batch for many observations. Extrapolation will produce
        untrustworthy results; the provided treatment should be within
        the range of the training data.

        Parameters
        ----------
        T: Numpy array, shape (n_samples,)
            A continuous treatment variable.

        Returns
        ----------
        array: Numpy array
            Contains a set of log odds
        """
        if self.outcome_type != "binary":
            raise TypeError("Your outcome must be binary to use this function!")

        return np.apply_along_axis(self._create_log_odds, 0, T.reshape(1, -1))

    def _create_log_odds(self, T):
        """Take a single treatment value and produces the log odds of the higher
        integer class, in the case of a binary outcome.
        """
        return logit(
            self.gam_results.predict_proba(
                np.array([T, self.gps_function(T).mean()]).reshape(1, -1)
            )
        )
