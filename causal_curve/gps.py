# pylint: disable=bad-continuation
"""
Defines the Generalized Prospensity Score (GPS) model class
"""

import contextlib
import io
from pprint import pprint

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_numeric_dtype
from pygam import LinearGAM, s
from scipy.stats import gamma, norm
import statsmodels.api as sm
from statsmodels.genmod.families.links import inverse_power as Inverse_Power
from statsmodels.tools.tools import add_constant

from causal_curve.core import Core
from causal_curve.utils import rand_seed_wrapper


class GPS(Core):
    """
    In a multi-stage approach, this computes the generalized propensity score (GPS) function,
    and uses this in a generalized additive model (GAM) to correct treatment prediction of
    the outcome variable. Assumes continuous treatment and outcome variable.

    WARNING:

    -This algorithm assumes you've already performed the necessary transformations to
    categorical covariates (i.e. these variables are already one-hot encoded and
    one of the categories is excluded for each set of dummy variables).

    -Please take care to ensure that the "ignorability" assumption is met (i.e.
    all strong confounders are captured in your covariates and there is no
    informative censoring), otherwise your results will be biased, sometimes strongly so.

    Parameters
    ----------

    gps_family: str, optional (default = None)
        Is used to determine the family of the glm used to model the GPS function.
        Look at the distribution of your treatment variable to determine which
        family is more appropriate.
        Possible values:

        - 'normal'
        - 'lognormal'
        - 'gamma'
        - None : (best-fitting family automatically chosen)

    treatment_grid_num: int, optional (default = 100)
        Takes the treatment, and creates a quantile-based grid across its values.
        For instance, if the number 6 is selected, this means the algorithm will only take
        the 6 treatment variable values at approximately the 0, 20, 40, 60, 80, and 100th
        percentiles to estimate the causal dose response curve.
        Higher value here means the final curve will be more finely estimated,
        but also increases computation time. Default is usually a reasonable number.

    lower_grid_constraint:  float, optional(default = 0.01)
        This adds an optional constraint of the lower side of the treatment grid.
        Sometimes data near the minimum values of the treatment are few in number
        and thus generate unstable estimates. By default, this clips the bottom 1 percentile
        or lower of treatment values. This can be as low as 0, indicating there is no
        lower limit to how much treatment data is considered.

    upper_grid_constraint: float, optional (default = 0.99)
        See above parameter. Just like above, but this is an upper constraint.
        By default, this clips the top 99th percentile or higher of treatment values.
        This can be as high as 1.0, indicating there is no upper limit to how much
        treatment data is considered.

    spline_order: int, optional (default = 3)
        Order of the splines to use fitting the final GAM.
        Must be integer >= 1. Default value creates cubic splines.

    n_splines: int, optional (default = 30)
        Number of splines to use for the treatment and GPS in the final GAM.
        Must be integer >= 2. Must be non-negative.

    lambda_: int or float, optional (default = 0.5)
        Strength of smoothing penalty. Must be a positive float.
        Larger values enforce stronger smoothing.

    max_iter: int, optional (default = 100)
        Maximum number of iterations allowed for the maximum likelihood algo to converge.

    random_seed: int, optional (default = None)
        Sets the random seed.

    verbose: bool, optional (default = False)
        Determines whether the user will get verbose status updates.


    Attributes
    ----------

    grid_values: array of shape (treatment_grid_num, )
        The gridded values of the treatment variable. Equally spaced.

    best_gps_family: str
        If no gps_family is specified and the algorithm chooses the best glm family, this is
        the name of the family that was chosen.

    gps_deviance: float
        The GPS model deviance

    gps: array of shape (number of observations, )
        The calculated GPS for each observation

    gam_results: `pygam.LinearGAM` class
        trained model of `LinearGAM` class, from pyGAM library


    Methods
    ----------
    fit: (self, T, X, y)
        Fits the causal dose-response model

    calculate_CDRC: (self, ci)
        Calculates the CDRC (and confidence interval) from trained model

    print_gam_summary: (self)
        Prints pyGAM text summary of GAM predicting outcome from the treatment and the GPS.


    Examples
    --------
    >>> from causal_curve import GPS
    >>> gps = GPS(treatment_grid_num = 200, random_seed = 512)
    >>> gps.fit(T = df['Treatment'], X = df[['X_1', 'X_2']], y = df['Outcome'])
    >>> gps_results = gps.calculate_CDRC(0.95)


    References
    ----------

    Galagate, D. Causal Inference with a Continuous Treatment and Outcome: Alternative
    Estimators for Parametric Dose-Response function with Applications. PhD thesis, 2016.

    Moodie E and Stephens DA. Estimation of dose–response functions for
    longitudinal data using the generalised propensity score. In: Statistical Methods in
    Medical Research 21(2), 2010, pp.149–166.

    Hirano K and Imbens GW. The propensity score with continuous treatments.
    In: Gelman A and Meng XL (eds) Applied bayesian modeling and causal inference
    from incomplete-data perspectives. Oxford, UK: Wiley, 2004, pp.73–84.

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
        rand_seed_wrapper()

        if self.verbose:
            print("Using the following params for GPS model:")
            pprint(self.get_params(), indent=4)

    def _validate_init_params(self):
        """
        Checks that the params used when instantiating GPS model are formatted correctly
        """
        # Checks for gps_family param
        if not isinstance(self.gps_family, (str, type(None))):
            raise TypeError(
                f"gps_family parameter must be a string or None "
                f"but found type {type(self.gps_family)}"
            )

        if (isinstance(self.gps_family, str)) and (
            self.gps_family not in ["normal", "lognormal", "gamma"]
        ):
            raise ValueError(
                f"gps_family parameter must take on values of "
                f"'normal', 'lognormal', or 'gamma', but found {self.gps_family}"
            )

        # Checks for treatment_grid_num
        if not isinstance(self.treatment_grid_num, int):
            raise TypeError(
                f"treatment_grid_num parameter must be an integer, "
                f"but found type {type(self.treatment_grid_num)}"
            )

        if (isinstance(self.treatment_grid_num, int)) and self.treatment_grid_num < 10:
            raise ValueError(
                f"treatment_grid_num parameter should be >= 10 so your final curve "
                f"has enough resolution, but found value {self.treatment_grid_num}"
            )

        if (
            isinstance(self.treatment_grid_num, int)
        ) and self.treatment_grid_num >= 1000:
            raise ValueError(f"treatment_grid_num parameter is too high!")

        # Checks for lower_grid_constraint
        if not isinstance(self.lower_grid_constraint, float):
            raise TypeError(
                f"lower_grid_constraint parameter must be a float, "
                f"but found type {type(self.lower_grid_constraint)}"
            )

        if (
            isinstance(self.lower_grid_constraint, float)
        ) and self.lower_grid_constraint < 0:
            raise ValueError(
                f"lower_grid_constraint parameter cannot be < 0, "
                f"but found value {self.lower_grid_constraint}"
            )

        if (
            isinstance(self.lower_grid_constraint, float)
        ) and self.lower_grid_constraint >= 1.0:
            raise ValueError(
                f"lower_grid_constraint parameter cannot >= 1.0, "
                f"but found value {self.lower_grid_constraint}"
            )

        # Checks for upper_grid_constraint
        if not isinstance(self.upper_grid_constraint, float):
            raise TypeError(
                f"upper_grid_constraint parameter must be a float, "
                f"but found type {type(self.upper_grid_constraint)}"
            )

        if (
            isinstance(self.upper_grid_constraint, float)
        ) and self.upper_grid_constraint <= 0:
            raise ValueError(
                f"upper_grid_constraint parameter cannot be <= 0, "
                f"but found value {self.upper_grid_constraint}"
            )

        if (
            isinstance(self.upper_grid_constraint, float)
        ) and self.upper_grid_constraint > 1.0:
            raise ValueError(
                f"upper_grid_constraint parameter cannot > 1.0, "
                f"but found value {self.upper_grid_constraint}"
            )

        # Checks for lower_grid_constraint isn't higher than upper_grid_constraint
        if self.lower_grid_constraint >= self.upper_grid_constraint:
            raise ValueError(
                "lower_grid_constraint should be lower than upper_grid_constraint!"
            )

        # Checks for spline_order
        if not isinstance(self.spline_order, int):
            raise TypeError(
                f"spline_order parameter must be an integer, "
                f"but found type {type(self.spline_order)}"
            )

        if (isinstance(self.spline_order, int)) and self.spline_order < 1:
            raise ValueError(
                f"spline_order parameter should be >= 1, but found {self.spline_order}"
            )

        if (isinstance(self.spline_order, int)) and self.spline_order >= 30:
            raise ValueError(f"spline_order parameter is too high!")

        # Checks for n_splines
        if not isinstance(self.n_splines, int):
            raise TypeError(
                f"n_splines parameter must be an integer, but found type {type(self.n_splines)}"
            )

        if (isinstance(self.n_splines, int)) and self.n_splines < 2:
            raise ValueError(
                f"n_splines parameter should be >= 2, but found {self.n_splines}"
            )

        if (isinstance(self.n_splines, int)) and self.n_splines >= 100:
            raise ValueError(f"n_splines parameter is too high!")

        # Checks for lambda_
        if not isinstance(self.lambda_, (int, float)):
            raise TypeError(
                f"lambda_ parameter must be an int or float, but found type {type(self.lambda_)}"
            )

        if (isinstance(self.lambda_, (int, float))) and self.lambda_ <= 0:
            raise ValueError(
                f"lambda_ parameter should be >= 2, but found {self.lambda_}"
            )

        if (isinstance(self.lambda_, (int, float))) and self.lambda_ >= 1000:
            raise ValueError(f"lambda_ parameter is too high!")

        # Checks for max_iter
        if not isinstance(self.max_iter, int):
            raise TypeError(
                f"max_iter parameter must be an int, but found type {type(self.max_iter)}"
            )

        if (isinstance(self.max_iter, int)) and self.max_iter <= 10:
            raise ValueError(
                f"max_iter parameter is too low! Results won't be reliable!"
            )

        if (isinstance(self.max_iter, int)) and self.max_iter >= 1e6:
            raise ValueError(f"max_iter parameter is unnecessarily high!")

        # Checks for random_seed
        if not isinstance(self.random_seed, (int, type(None))):
            raise TypeError(
                f"random_seed parameter must be an int, but found type {type(self.random_seed)}"
            )

        if (isinstance(self.random_seed, int)) and self.random_seed < 0:
            raise ValueError(f"random_seed parameter must be > 0")

        # Checks for verbose
        if not isinstance(self.verbose, bool):
            raise TypeError(
                f"verbose parameter must be a boolean type, but found type {type(self.verbose)}"
            )

    def _validate_fit_data(self):
        """Verifies that T, X, and y are formatted the right way
        """
        # Checks for T column
        if not is_float_dtype(self.T):
            raise TypeError(f"Treatment data must be of type float")

        # Make sure all X columns are float or int
        if isinstance(self.X, pd.Series):
            if not is_numeric_dtype(self.X):
                raise TypeError(
                    f"All covariate (X) columns must be int or float type (i.e. must be numeric)"
                )

        elif isinstance(self.X, pd.DataFrame):
            for column in self.X:
                if not is_numeric_dtype(self.X[column]):
                    raise TypeError(
                        f"All covariate (X) columns must be int or float type "
                        f"(i.e. must be numeric)"
                    )

        # Checks for Y column
        if not is_float_dtype(self.y):
            raise TypeError(f"Outcome data must be of type float")

    def _grid_values(self):
        """Produces initial grid values for the treatment variable
        """
        return np.quantile(
            self.T,
            q=np.linspace(
                start=self.lower_grid_constraint,
                stop=self.upper_grid_constraint,
                num=self.treatment_grid_num,
            ),
        )

    def fit(self, T, X, y):
        """Fits the GPS causal dose-response model. For now, this only accepts pandas columns.

        Parameters
        ----------
        T: array-like, shape (n_samples,)
            A continuous treatment variable
        X: array-like, shape (n_samples, m_features)
            Covariates, where n_samples is the number of samples
            and m_features is the number of features
        y: array-like, shape (n_samples,)
            Outcome variable

        Returns
        ----------
        self : object

        """
        self.T = T.reset_index(drop=True, inplace=False)
        self.X = X.reset_index(drop=True, inplace=False)
        self.y = y.reset_index(drop=True, inplace=False)

        # Validate this input data
        self._validate_fit_data()

        # Create grid_values
        self.grid_values = self._grid_values()

        # Estimating the GPS
        self.best_gps_family = self.gps_family

        # If no family specified, pick the best family
        if self.gps_family == None:
            if self.verbose:
                print(f"Fitting several GPS models and picking the best fitting one...")

            (
                self.best_gps_family,
                self.gps_function,
                self.gps_deviance,
            ) = self._find_best_gps_model()

            if self.verbose:
                print(
                    f"Best fitting model was {self.best_gps_family}, which "
                    f"produced a deviance of {self.gps_deviance}"
                )

        # Otherwise, go with the what the user provided...
        else:
            if self.verbose:
                print(f"Fitting GPS model of family '{self.best_gps_family}'...")

            if self.best_gps_family == "normal":
                (
                    self.gps_function,
                    self.gps_deviance,
                ) = self._create_normal_gps_function()
            elif self.best_gps_family == "lognormal":
                (
                    self.gps_function,
                    self.gps_deviance,
                ) = self._create_lognormal_gps_function()
            elif self.best_gps_family == "gamma":
                self.gps_function, self.gps_deviance = self._create_gamma_gps_function()

        # Estimate the GPS
        if self.verbose:
            print(f"Saving GPS values...")

        self.gps = self.gps_function(self.T)

        # Create GAM that predicts outcome from the treatment and GPS
        if self.verbose:
            print(f"Fitting GAM using treatment and GPS...")

        # Save model results
        self.gam_results = self._fit_gam()

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.gam_results.summary()

        self._gam_summary_str = f.getvalue()

        if self.verbose:
            print(f"Calculating many CDRC estimates for each treatment grid value...")

        # Loop over all grid values (`treatment_grid_num` in total)
        # and give GPS loading for each observation in the dataset
        self.gps_at_grid = self._gps_values_at_grid()

    def calculate_CDRC(self, ci=0.95):
        """Using the results of the fitted model, this generates point estimates for the CDRC
        at each of the values of the treatment grid. Connecting these estimates will produce
        the overall estimated CDRC. Confidence interval is returned as well.

        Parameters
        ----------
        ci: float (default = 0.95)
            The desired confidence interval to produce. Default value is 0.95, corresponding
            to 95% confidence intervals. bounded (0, 1.0).

        Returns
        ----------
        dataframe: Pandas dataframe
            Contains treatment grid values, the CDRC point estimate at that value,
            and the associated lower and upper confidence interval bounds at that point.

        self: object

        """
        self._validate_calculate_CDRC_params(ci)

        # Create CDRC predictions from trained GAM
        self._cdrc_preds = self._cdrc_predictions(ci)

        if self.verbose:
            print(
                """Generating predictions for each value of treatment grid,
                and averaging to get the CDRC..."""
            )

        # For each column of _cdrc_preds, calculate the mean and confidence interval bounds
        results = []

        for i in range(0, self.treatment_grid_num):
            temp_grid_value = self.grid_values[i]
            temp_point_estimate = self._cdrc_preds[:, i, 0].mean()
            mean_ci_width = (
                self._cdrc_preds[:, i, 2].mean() - self._cdrc_preds[:, i, 1].mean()
            ) / 2
            temp_lower_bound = temp_point_estimate - mean_ci_width
            temp_upper_bound = temp_point_estimate + mean_ci_width
            results.append(
                [
                    temp_grid_value,
                    temp_point_estimate,
                    temp_lower_bound,
                    temp_upper_bound,
                ]
            )

        return pd.DataFrame(
            results, columns=["Treatment", "CDRC", "Lower_CI", "Upper_CI"]
        ).round(3)

    def _validate_calculate_CDRC_params(self, ci):
        """Validates the parameters given to `calculate_CDRC`
        """

        if not isinstance(ci, float):
            raise TypeError(
                f"`ci` parameter must be an float, but found type {type(ci)}"
            )

        if isinstance(ci, float) and ((ci <= 0) or (ci >= 1.0)):
            raise ValueError("`ci` parameter should be between (0, 1)")

    def _cdrc_predictions(self, ci):
        """Returns the predictions of CDRC for each value of the treatment grid. Essentially,
        we're making predictions using the original treatment and gps_at_grid
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

    def _gps_values_at_grid(self):
        """Returns an array where we get the GPS-derived values for each element
        of the treatment grid. Resulting array will be of shape (n_samples, treatment_grid_num)
        """
        # Creates an empty 2d array of shape (n_samples, treatment_grid_num)
        gps_at_grid = np.zeros((len(self.T), self.treatment_grid_num), dtype=float)

        # Loop over all grid values
        for i in range(0, self.treatment_grid_num):
            gps_at_grid[:, i] = self.gps_function(self.grid_values[i])

        return gps_at_grid

    def print_gam_summary(self):
        """Prints the GAM model summary (uses pyGAM's output)

        Parameters
        ----------
        None

        Returns
        ----------
        self: object


        """
        print(self._gam_summary_str)

    def _fit_gam(self):
        """Fits a GAM that predicts the outcome from the treatment and GPS
        """

        X = np.column_stack((self.T.values, self.gps))
        y = np.asarray(self.y)

        return LinearGAM(
            s(0, n_splines=self.n_splines, spline_order=self.spline_order)
            + s(1, n_splines=self.n_splines, spline_order=self.spline_order),
            max_iter=self.max_iter,
            lam=self.lambda_,
        ).fit(X, y)

    def _create_normal_gps_function(self):
        """Models the GPS using a GLM of the Gaussian family
        """
        normal_gps_model = sm.GLM(
            self.T, add_constant(self.X), family=sm.families.Gaussian()
        ).fit()

        pred_treat = normal_gps_model.fittedvalues
        sigma = np.std(normal_gps_model.resid_response)

        def gps_function(treatment_val, pred_treat=pred_treat, sigma=sigma):
            return norm.pdf(treatment_val, pred_treat, sigma)

        return gps_function, normal_gps_model.deviance

    def _create_lognormal_gps_function(self):
        """Models the GPS using a GLM of the Gaussian family (assumes treatment is lognormal)
        """
        lognormal_gps_model = sm.GLM(
            np.log(self.T), add_constant(self.X), family=sm.families.Gaussian()
        ).fit()

        pred_log_treat = lognormal_gps_model.fittedvalues
        sigma = np.std(lognormal_gps_model.resid_response)

        def gps_function(treatment_val, pred_log_treat=pred_log_treat, sigma=sigma):
            return norm.pdf(np.log(treatment_val), pred_log_treat, sigma)

        return gps_function, lognormal_gps_model.deviance

    def _create_gamma_gps_function(self):
        """Models the GPS using a GLM of the Gamma family
        """
        gamma_gps_model = sm.GLM(
            self.T, add_constant(self.X), family=sm.families.Gamma(Inverse_Power())
        ).fit()

        mu = gamma_gps_model.mu
        scale = gamma_gps_model.scale
        shape = mu / gamma_gps_model.scale

        def gps_function(treatment_val):
            return gamma.pdf(treatment_val, a=shape, loc=0, scale=scale)

        return gps_function, gamma_gps_model.deviance

    def _find_best_gps_model(self):
        """If user doesn't provide a GLM family for modeling the GPS, this function compares
        a few different gps models and picks the one with the lowest deviance
        """
        models_to_try_dict = {
            "normal_gps_model": self._create_normal_gps_function(),
            "lognormal_gps_model": self._create_lognormal_gps_function(),
            "gamma_gps_model": self._create_gamma_gps_function(),
        }

        model_comparison_dict = {}

        for key, value in models_to_try_dict.items():
            model_comparison_dict[key] = value[1]

        # Return model with lowest deviance
        best_model = min(model_comparison_dict, key=model_comparison_dict.get)

        return (
            best_model,
            models_to_try_dict[best_model][0],
            models_to_try_dict[best_model][1],
        )
