"""
Defines the Generalized Prospensity Score (GPS) regressor model class
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


class GPS_regressor(GPS_core):
    """
    A GPS tool that handles continuous outcomes. Inherits the GPS_core
    base class. See that base class code its docstring for more details.
    """

    def __init__():
        pass

    def _cdrc_predictions_continuous(self, ci):
        """Returns the predictions of CDRC for each value of the treatment grid. Essentially,
        we're making predictions using the original treatment and gps_at_grid.
        To be used when the outcome of interest is continuous.
        """

        # To keep track of cdrc predictions, we create an empty 3d array of shape
        # (n_samples, treatment_grid_num, 3). The last dimension is of length 3 because
        # we are going to keep track of the point estimate of the prediction, as well as
        # the lower and upper bounds of the prediction interval
        cdrc_preds = np.zeros((len(self.T), self.treatment_grid_num, 3), dtype=float)

        # Loop through each of the grid values, predict point estimate and get prediction interval
        for i in range(0, self.treatment_grid_num):
            temp_T = np.repeat(self.grid_values[i], repeats=len(self.T))
            temp_gps = self.gps_at_grid[:, i]
            temp_cdrc_preds = self.gam_results.predict(
                np.column_stack((temp_T, temp_gps))
            )
            temp_cdrc_interval = self.gam_results.confidence_intervals(
                np.column_stack((temp_T, temp_gps)), width=ci
            )
            temp_cdrc_lower_bound = temp_cdrc_interval[:, 0]
            temp_cdrc_upper_bound = temp_cdrc_interval[:, 1]
            cdrc_preds[:, i, 0] = temp_cdrc_preds
            cdrc_preds[:, i, 1] = temp_cdrc_lower_bound
            cdrc_preds[:, i, 2] = temp_cdrc_upper_bound

        return np.round(cdrc_preds, 3)

    def predict(self, T):
        """Calculates point estimate within the CDRC given treatment values.
        Can only be used when outcome is continuous. Can be estimate for a single
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
            Contains a set of CDRC point estimates

        """
        if self.outcome_type != "continuous":
            raise TypeError("Your outcome must be continuous to use this function!")

        return np.apply_along_axis(self._create_predict, 0, T.reshape(1, -1))

    def _create_predict(self, T):
        """Takes a single treatment value and produces a single point estimate
        in the case of a continuous outcome.
        """
        return self.gam_results.predict(
            np.array([T, self.gps_function(T).mean()]).reshape(1, -1)
        )

    def predict_interval(self, T, ci=0.95):
        """Calculates the prediction confidence interval associated with a point estimate
        within the CDRC given treatment values. Can only be used
        when outcome is continuous. Can be estimate for a single data point
        or can be run in batch for many observations. Extrapolation will produce
        untrustworthy results; the provided treatment should be within
        the range of the training data.

        Parameters
        ----------
        T: Numpy array, shape (n_samples,)
            A continuous treatment variable.
        ci: float (default = 0.95)
            The desired confidence interval to produce. Default value is 0.95, corresponding
            to 95% confidence intervals. bounded (0, 1.0).

        Returns
        ----------
        array: Numpy array
            Contains a set of CDRC prediction intervals ([lower bound, higher bound])

        """
        if self.outcome_type != "continuous":
            raise TypeError("Your outcome must be continuous to use this function!")

        return np.apply_along_axis(
            self._create_predict_interval, 0, T.reshape(1, -1), width=ci
        ).T.reshape(-1, 2)

    def _create_predict_interval(self, T, width):
        """Takes a single treatment value and produces confidence interval
        associated with a point estimate in the case of a continuous outcome.
        """
        return self.gam_results.prediction_intervals(
            np.array([T, self.gps_function(T).mean()]).reshape(1, -1), width=width
        )
