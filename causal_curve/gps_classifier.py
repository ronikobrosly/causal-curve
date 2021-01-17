"""
Defines the Generalized Prospensity Score (GPS) classifier model class
"""
from pprint import pprint

import numpy as np
from scipy.special import logit

from causal_curve.gps_core import GPS_Core


class GPS_Classifier(GPS_Core):
    """
    A GPS tool that handles binary outcomes. Inherits the GPS_core
    base class. See that base class code its docstring for more details.


    Methods
    ----------

    estimate_log_odds: (self, T)
        Calculates the predicted log odds of the highest integer class. Can
        only be used when the outcome is binary.

    """

    def __init__(
        self,
        gps_family=None,
        treatment_grid_num=100,
        lower_grid_constraint=0.01,
        upper_grid_constraint=0.99,
        spline_order=3,
        n_splines=30,
        lambda_=0.5,
        max_iter=100,
        random_seed=None,
        verbose=False,
    ):

        self.gps_family = gps_family
        self.treatment_grid_num = treatment_grid_num
        self.lower_grid_constraint = lower_grid_constraint
        self.upper_grid_constraint = upper_grid_constraint
        self.spline_order = spline_order
        self.n_splines = n_splines
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.verbose = verbose

        # Validate the params
        self._validate_init_params()
        self.rand_seed_wrapper()

        self.if_verbose_print("Using the following params for GPS model:")
        if self.verbose:
            pprint(self.get_params(), indent=4)

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
            ) / self.calculate_z_score(ci)

            cdrc_preds[:, i, 0] = temp_cdrc_preds
            cdrc_preds[:, i, 1] = standard_error

        return np.round(cdrc_preds, 3)

    def estimate_log_odds(self, T):
        """Calculates the estimated log odds of the highest integer class. Can
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
