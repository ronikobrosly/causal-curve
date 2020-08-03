# pylint: disable=bad-continuation
"""
Defines the Targetted Maximum likelihood Estimation (TMLE) model class
"""
from pprint import pprint

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_numeric_dtype
from scipy.interpolate import interp1d
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from statsmodels.genmod.generalized_linear_model import GLM

from causal_curve.core import Core
from causal_curve.utils import rand_seed_wrapper


class TMLE(Core):
    """
    Constructs a causal dose response curve through a series of TMLE comparisons across a grid
    of the treatment values. Gradient boosting is used for prediction in Q model and G model.
    Assumes continuous treatment and outcome variable.

    WARNING:

    -In choosing `treatment_grid_bins` be very careful to respect the "positivity" assumption.
    There must be sufficient data and variability of treatment within each bin the treatment
    is split into.

    -This algorithm assumes you've already performed the necessary transformations to
    categorical covariates (i.e. these variables are already one-hot encoded and
    one of the categories is excluded for each set of dummy variables).

    -Please take care to ensure that the "ignorability" assumption is met (i.e.
    all strong confounders are captured in your covariates and there is no
    informative censoring), otherwise your results will be biased, sometimes strongly so.

    Parameters
    ----------

    treatment_grid_bins: list of floats or ints
        Represents the edges of bins of treatment values that are used to construct a smooth curve
        Look at the distribution of your treatment variable to determine which
        family is more appropriate. Be mindful of the "positivity" assumption when determining
        bins. In other words, make sure there aren't too few data points in each bin. Mean
        treatment values between the bin edges will be used to generate the CDRC.

    n_estimators: int, optional (default = 100)
        Optional argument to set the number of learners to use when sklearn
        creates TMLE's Q and G models.

    learning_rate: float, optional (default = 0.1)
        Optional argument to set the sklearn's learning rate for TMLE's Q and G models.

    max_depth: int, optional (default = 5)
        Optional argument to set sklearn's maximum depth when creating TMLE's Q and G models.

    random_seed: int, optional (default = None)
        Sets the random seed.

    verbose: bool, optional (default = False)
        Determines whether the user will get verbose status updates.


    Attributes
    ----------
    psi_list: array of shape (len(treatment_grid_bins) - 2, )
        The estimated causal difference between treatment bins

    std_error_ic_list: array of shape (len(treatment_grid_bins) - 2, )
        The standard errors for the psi estimates found in `psi_list`


    Methods
    ----------
    fit: (self, T, X, y)
        Fits the causal dose-response model

    calculate_CDRC: (self, ci, CDRC_grid_num)
        Calculates the CDRC (and confidence interval) from TMLE estimation


    Examples
    --------
    >>> from causal_curve import TMLE
    >>> tmle = TMLE(treatment_grid_bins = [0, 0.03, 0.05, 0.25, 0.5, 1.0, 2.0],
        n_estimators=500,
        learning_rate = 0.1,
        max_depth = 5,
        random_seed=111
    )
    >>> tmle.fit(T = df['Treatment'], X = df[['X_1', 'X_2']], y = df['Outcome'])
    >>> tmle_results = tmle.calculate_CDRC(0.95)


    References
    ----------

    van der Laan MJ and Rubin D. Targeted maximum likelihood learning. In: The International
    Journal of Biostatistics, 2(1), 2006.

    van der Laan MJ and Gruber S. Collaborative double robust penalized targeted
    maximum likelihood estimation. In: The International Journal of Biostatistics 6(1), 2010.

    """

    def __init__(
        self,
        treatment_grid_bins,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_seed=None,
        verbose=False,
    ):

        self.treatment_grid_bins = treatment_grid_bins
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_seed = random_seed
        self.verbose = verbose

        # Validate the params
        self._validate_init_params()
        rand_seed_wrapper()

        if self.verbose:
            print("Using the following params for TMLE model:")
            pprint(self.get_params(), indent=4)

    def _validate_init_params(self):
        """
        Checks that the params used when instantiating TMLE model are formatted correctly
        """

        # Checks for treatment_grid_bins
        if not isinstance(self.treatment_grid_bins, list):
            raise TypeError(
                f"treatment_grid_bins parameter must be a list, "
                f"but found type {type(self.treatment_grid_bins)}"
            )

        for element in self.treatment_grid_bins:
            if not isinstance(element, (int, float)):
                raise TypeError(
                    f"'{element}' in `treatment_grid_bins` list is not of type float or int, "
                    f"it is {type(element)}"
                )

        if len(self.treatment_grid_bins) < 2:
            raise TypeError("treatment_grid_bins list must, at minimum, of length >= 2")

        # Checks for n_estimators
        if not isinstance(self.n_estimators, int):
            raise TypeError(
                f"n_estimators parameter must be an integer, "
                f"but found type {type(self.n_estimators)}"
            )

        if (self.n_estimators < 10) or (self.n_estimators > 100000):
            raise TypeError("n_estimators parameter must be between 10 and 100000")

        # Checks for learning_rate
        if not isinstance(self.learning_rate, (int, float)):
            raise TypeError(
                f"learning_rate parameter must be an integer or float, "
                f"but found type {type(self.learning_rate)}"
            )

        if (self.learning_rate <= 0) or (self.learning_rate >= 1000):
            raise TypeError(
                "learning_rate parameter must be greater than 0 and less than 1000"
            )

        # Checks for max_depth
        if not isinstance(self.max_depth, int):
            raise TypeError(
                f"max_depth parameter must be an integer, "
                f"but found type {type(self.max_depth)}"
            )

        if self.max_depth <= 0:
            raise TypeError("max_depth parameter must be greater than 0")

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
        if not is_float_dtype(self.t_data):
            raise TypeError(f"Treatment data must be of type float")

        # Make sure all X columns are float or int
        if isinstance(self.x_data, pd.Series):
            if not is_numeric_dtype(self.x_data):
                raise TypeError(
                    f"All covariate (X) columns must be int or float type (i.e. must be numeric)"
                )

        elif isinstance(self.x_data, pd.DataFrame):
            for column in self.x_data:
                if not is_numeric_dtype(self.x_data[column]):
                    raise TypeError(
                        """All covariate (X) columns must be int or float type
                        (i.e. must be numeric)"""
                    )

        # Checks for Y column
        if not is_float_dtype(self.y_data):
            raise TypeError(f"Outcome data must be of type float")

    def _validate_calculate_CDRC_params(self, ci):
        """Validates the parameters given to `calculate_CDRC`
        """

        if not isinstance(ci, float):
            raise TypeError(
                f"`ci` parameter must be an float, but found type {type(ci)}"
            )

        if isinstance(ci, float) and ((ci <= 0) or (ci >= 1.0)):
            raise ValueError("`ci` parameter should be between (0, 1)")

    def _initial_bucket_mean_prediction(self):
        """Creates a model to predict the outcome variable given the provided inputs within
        the first bucket of treatment_grid_bins. This returns the mean predicted outcome.
        """

        y = self.y_data[self.t_data < self.treatment_grid_bins[1]]
        X = pd.concat([self.t_data, self.x_data], axis=1)[
            self.t_data < self.treatment_grid_bins[1]
        ]

        init_model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_seed,
        ).fit(X, y)

        return init_model.predict(X).mean()

    def _create_treatment_comparison_df(
        self, low_boundary, med_boundary, high_boundary
    ):
        """Given the current boundaries chosen from treatment_grid_bins, this filters
        the input data appropriately.
        """
        temp_y = self.y_data[
            ((self.t_data >= low_boundary) & (self.t_data <= high_boundary))
        ]
        temp_x = self.x_data[
            ((self.t_data >= low_boundary) & (self.t_data <= high_boundary))
        ]
        temp_t = self.t_data.copy()
        temp_t = temp_t[((temp_t >= low_boundary) & (temp_t <= high_boundary))]
        temp_t[((temp_t >= low_boundary) & (temp_t < med_boundary))] = 0
        temp_t[((temp_t >= med_boundary) & (temp_t <= high_boundary))] = 1

        return temp_y, temp_x, temp_t

    def _collect_mean_t_levels(self):
        """Collects the mean treatment value within each treatment bucket in treatment_grid_bins
        """

        t_bin_means = []

        for index, _ in enumerate(self.treatment_grid_bins):
            if index == (len(self.treatment_grid_bins) - 1):
                continue

            t_bin_means.append(
                self.t_data[
                    (
                        (self.t_data >= self.treatment_grid_bins[index])
                        & (self.t_data <= self.treatment_grid_bins[index + 1])
                    )
                ].mean()
            )

        return t_bin_means

    def fit(self, T, X, y):
        """Fits the TMLE causal dose-response model. For now, this only accepts pandas columns
        with the same index.

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
        self.t_data = T.reset_index(drop=True, inplace=False)
        self.x_data = X.reset_index(drop=True, inplace=False)
        self.y_data = y.reset_index(drop=True, inplace=False)

        # Validate this input data
        self._validate_fit_data()

        # Get the mean, predicted outcome value within the first bucket
        if self.verbose:
            print(
                "Calculating the mean, predicted outcome value within the first bucket..."
            )
        self.outcome_start_val = self._initial_bucket_mean_prediction()

        # Loop through the comparisons in the treatment_grid_bins
        if self.verbose:
            print("Beginning main loop through treatment bins...")

        # Collect loop results in these lists
        self.t_bin_means = self._collect_mean_t_levels()
        self.psi_list = []
        self.std_error_ic_list = []

        for index, _ in enumerate(self.treatment_grid_bins):
            if (index == 0) or (index == len(self.treatment_grid_bins) - 1):
                continue
            if self.verbose:
                print(
                    f"***** Starting iteration {index} of {len(self.treatment_grid_bins) - 2} *****"
                )
            low_boundary = self.treatment_grid_bins[index - 1]
            med_boundary = self.treatment_grid_bins[index]
            high_boundary = self.treatment_grid_bins[index + 1]

            # Create comparison dataset
            temp_y, temp_x, temp_t = self._create_treatment_comparison_df(
                low_boundary, med_boundary, high_boundary
            )

            self.n_obs = len(temp_y)

            # Fit Q-model and get relevent predictions
            if self.verbose:
                print("Fitting Q-model and making predictions...")
            self.y_hat_a, self.y_hat_1, self.y_hat_0 = self._q_model(
                temp_y, temp_x, temp_t
            )

            # Fit G-model and get relevent predictions
            if self.verbose:
                print("Fitting G-model and making predictions...")
            self.pi_hat1, self.pi_hat0 = self._g_model(temp_x, temp_t)

            # Estimate delta_hat
            if self.verbose:
                print("Estimating delta hat...")
            self.delta_hat = self._delta_hat_estimation(temp_y, temp_x, temp_t)

            # Estimate targeted and corrected Psi
            if self.verbose:
                print("Estimating Psi...")
            psi, std_error_IC = self._psi_estimation(temp_y, temp_t)

            self.psi_list.append(psi)
            self.std_error_ic_list.append(std_error_IC)

            if self.verbose:
                print(f"Finished this loop!")

    def calculate_CDRC(self, ci=0.95, CDRC_grid_num=100):
        """Using the results of the fitted model, this generates the CDRC by interpolation
        of the binned treatment comparisons. This returns a confidence interval as well, which
        is also generated by interpolation.

        Parameters
        ----------
        ci: float (default = 0.95)
            The desired confidence interval to produce. Default value is 0.95, corresponding
            to 95% confidence intervals. bounded (0, 1.0).

        CDRC_grid_num: int, optional (default = 100)
            Linear interpolation over a quantile-based grid of treatment values is used
            to produce the final CDRC. This parameter determines how many points
            to include on that grid. Higher values will produce a finer estimate of the CDRC,
            but this increases computation time. Default is usually a reasonable number.

        Returns
        ----------
        dataframe: Pandas dataframe
            Contains treatment grid values, the CDRC point estimate at that value,
            and the associated lower and upper confidence interval bounds at that point.

        self: object
        """

        self._validate_calculate_CDRC_params(ci)
        z_star = norm.ppf((1 + ci) / 2)

        if self.verbose:
            print(
                f"Estimating the CDRC and confidence intervals via cubic interpolation..."
            )

        # Collect discrete t_values
        t_values = self.t_bin_means

        # Collect discrete y_values
        y_values = [self.outcome_start_val]

        for index, item in enumerate(self.psi_list):
            y_values.append(self.outcome_start_val + sum(self.psi_list[: (index + 1)]))

        # Collect discrete lower confidence bounds
        lower_values = [self.outcome_start_val - (z_star * self.std_error_ic_list[0])]
        for index, y_value in enumerate(y_values[1:]):
            lower_values.append(y_value - (z_star * self.std_error_ic_list[index]))

        # Collect discrete upper confidence bounds
        upper_values = [self.outcome_start_val + (z_star * self.std_error_ic_list[0])]
        for index, y_value in enumerate(y_values[1:]):
            upper_values.append(y_value + (z_star * self.std_error_ic_list[index]))

        # Perform cubic interpolation
        CDRC_interp = interp1d(t_values, y_values, kind="cubic")
        lower_interp = interp1d(t_values, lower_values, kind="cubic")
        upper_interp = interp1d(t_values, upper_values, kind="cubic")

        CDRC_t_grid = self._grid_values(CDRC_grid_num, t_values)

        Treatment = CDRC_t_grid
        CDRC = CDRC_interp(CDRC_t_grid)
        Lower_CI = lower_interp(CDRC_t_grid)
        Upper_CI = upper_interp(CDRC_t_grid)

        if self.verbose:
            print(f"Done!")

        return pd.DataFrame(
            {
                "Treatment": Treatment,
                "CDRC": CDRC,
                "Lower_CI": Lower_CI,
                "Upper_CI": Upper_CI,
            }
        ).round(3)

    def _grid_values(self, CDRC_grid_num, t_values):
        """Produces grid values for use in estimating the final CDRC and confidence intervals.
        """

        return np.quantile(
            self.t_data[((self.t_data > t_values[0]) & (self.t_data < t_values[-1]))],
            q=np.linspace(start=0, stop=1, num=CDRC_grid_num,),
        )

    def _q_model(self, temp_y, temp_x, temp_t):
        """Produces the Q-model and gets outcome predictions using the provided treatment
        values, when the treatment is completely present and not present.
        """

        X = pd.concat([temp_t, temp_x], axis=1).to_numpy()
        y = temp_y.to_numpy()

        Q_model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_seed,
        ).fit(X, y)

        # Make predictions with provided treatment values
        y_hat_a = Q_model.predict(X)

        temp = np.column_stack((np.repeat(1, self.n_obs), np.asarray(temp_x)))

        # Make predictions when treatment is completely present
        y_hat_1 = Q_model.predict(temp)

        # Make predictions when treatment is completely not present
        temp = np.column_stack((np.repeat(0, self.n_obs), np.asarray(temp_x)))

        y_hat_0 = Q_model.predict(temp)

        return y_hat_a, y_hat_1, y_hat_0

    def _g_model(self, temp_x, temp_t):
        """Produces the G-model and gets treatment assignment predictions
        """

        X = temp_x.to_numpy()
        t = temp_t.to_numpy()

        G_model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_seed,
        ).fit(X, t)

        # Make predictions of receiving treatment
        pi_hat1 = G_model.predict_proba(X)[:, 1]

        # Predictions of not receiving treatment
        pi_hat0 = 1 - pi_hat1

        return pi_hat1, pi_hat0

    def _delta_hat_estimation(self, temp_y, temp_x, temp_t):
        """Estimates delta to correct treatment estimation
        """
        H_a = []

        for idx, treatment in enumerate(np.asarray(temp_t)):
            if treatment == 1:
                H_a.append(1 / self.pi_hat1[idx])
            elif treatment == 0:
                H_a.append(-1 / self.pi_hat0[idx])

        H_a = np.array(H_a)

        # Create GLM using H_a as a forced offset
        targetting_model = GLM(
            endog=np.asarray(temp_y), exog=H_a, offset=np.asarray(self.y_hat_a)
        ).fit()

        return targetting_model.params[0]

    def _psi_estimation(self, temp_y, temp_t):
        """Estimates final Psi for the treatment comparison, also estimates the
        standard error via the influence curve
        """

        # Estimate Psi
        H_1 = 1 / self.pi_hat1
        H_0 = -1 / self.pi_hat0

        Y_star_1 = self.y_hat_1 + (self.delta_hat * H_1)
        Y_star_0 = self.y_hat_0 + (self.delta_hat * H_0)

        Psi = round((Y_star_1 - Y_star_0).mean(), 3)

        # Use Psi and other various variables to estimate the standard error
        D1 = (
            ((temp_t / self.pi_hat1) * (temp_y - self.y_hat_1))
            + self.y_hat_1
            - Y_star_1
        )
        D0 = (
            (((1 - temp_t) / (1 - self.pi_hat1)) * (temp_y - self.y_hat_0))
            + self.y_hat_0
            - Y_star_0
        )
        EIC = D1 - D0

        std_error_IC = np.sqrt(EIC.var() / self.n_obs)

        return Psi, std_error_IC
