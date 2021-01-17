"""
Defines the Targetted Maximum likelihood Estimation (TMLE) regressor model class
"""
from pprint import pprint

import numpy as np

from causal_curve.tmle_core import TMLE_Core


class TMLE_Regressor(TMLE_Core):
    """
    A TMLE tool that handles continuous outcomes. Inherits the TMLE_core
    base class. See that base class code its docstring for more details.

    Methods
    ----------

    point_estimate: (self, T)
        Calculates point estimate within the CDRC given treatment values.
        Can only be used when outcome is continuous.
    """

    def __init__(
        self,
        treatment_grid_num=100,
        lower_grid_constraint=0.01,
        upper_grid_constraint=0.99,
        n_estimators=200,
        learning_rate=0.01,
        max_depth=3,
        bandwidth=0.5,
        random_seed=None,
        verbose=False,
    ):

        self.treatment_grid_num = treatment_grid_num
        self.lower_grid_constraint = lower_grid_constraint
        self.upper_grid_constraint = upper_grid_constraint
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.bandwidth = bandwidth
        self.random_seed = random_seed
        self.verbose = verbose

        # Validate the params
        self._validate_init_params()
        self.rand_seed_wrapper()

        self.if_verbose_print("Using the following params for TMLE model:")
        if self.verbose:
            pprint(self.get_params(), indent=4)

    def _cdrc_predictions_continuous(self, ci):
        """Returns the predictions of CDRC for each value of the treatment grid. Essentially,
        we're making predictions using the original treatment against the pseudo-outcome.
        To be used when the outcome of interest is continuous.
        """

        # To keep track of cdrc predictions, we create an empty 2d array of shape
        # (treatment_grid_num, 4). The last dimension is of length 4 because
        # we are going to keep track of the treatment, point estimate of the prediction, as well as
        # the lower and upper bounds of the prediction interval
        cdrc_preds = np.zeros((self.treatment_grid_num, 4), dtype=float)

        # Loop through each of the grid values, get point estimate and get estimate interval
        for i in range(0, self.treatment_grid_num):
            temp_T = self.grid_values[i]
            temp_cdrc_preds = self.final_gam.predict(temp_T)
            temp_cdrc_interval = self.final_gam.confidence_intervals(temp_T, width=ci)
            temp_cdrc_lower_bound = temp_cdrc_interval[:, 0]
            temp_cdrc_upper_bound = temp_cdrc_interval[:, 1]
            cdrc_preds[i, 0] = temp_T
            cdrc_preds[i, 1] = temp_cdrc_preds
            cdrc_preds[i, 2] = temp_cdrc_lower_bound
            cdrc_preds[i, 3] = temp_cdrc_upper_bound

        return cdrc_preds

    def point_estimate(self, T):
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
        return np.apply_along_axis(self._create_point_estimate, 0, T.reshape(1, -1))

    def _create_point_estimate(self, T):
        """Takes a single treatment value and produces a single point estimate
        in the case of a continuous outcome.
        """
        return self.final_gam.predict(np.array([T]).reshape(1, -1))

    def point_estimate_interval(self, T, ci=0.95):
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
        return np.apply_along_axis(
            self._create_point_estimate_interval, 0, T.reshape(1, -1), width=ci
        ).T.reshape(-1, 2)

    def _create_point_estimate_interval(self, T, width):
        """Takes a single treatment value and produces confidence interval
        associated with a point estimate in the case of a continuous outcome.
        """
        return self.final_gam.prediction_intervals(
            np.array([T]).reshape(1, -1), width=width
        )
