"""
Defines the Targetted Maximum likelihood Estimation (TMLE) model class
"""
import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_numeric_dtype
from pygam import LinearGAM, s
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.nonparametric.kernel_regression import KernelReg

from causal_curve.core import Core


class TMLE_Core(Core):
    """
    Constructs a causal dose response curve via a modified version of Targetted
    Maximum Likelihood Estimation (TMLE) across a grid of the treatment values.
    Gradient boosting is used for prediction of the Q model and G models, simple
    kernel regression is used processing those model results, and a generalized
    additive model is used in the final step to contruct the final curve.
    Assumes continuous treatment and outcome variable.

    WARNING:

    -The treatment values should be roughly normally-distributed for this tool
    to work. Otherwise you may encounter internal math errors.

    -This algorithm assumes you've already performed the necessary transformations to
    categorical covariates (i.e. these variables are already one-hot encoded and
    one of the categories is excluded for each set of dummy variables).

    -Please take care to ensure that the "ignorability" assumption is met (i.e.
    all strong confounders are captured in your covariates and there is no
    informative censoring), otherwise your results will be biased, sometimes strongly so.

    Parameters
    ----------

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

    n_estimators: int, optional (default = 200)
        Optional argument to set the number of learners to use when sklearn
        creates TMLE's Q and G models.

    learning_rate: float, optional (default = 0.01)
        Optional argument to set the sklearn's learning rate for TMLE's Q and G models.

    max_depth: int, optional (default = 3)
        Optional argument to set sklearn's maximum depth when creating TMLE's Q and G models.

    bandwidth: float, optional (default = 0.5)
        Optional argument to set the bandwidth parameter of the internal
        kernel density estimation and kernel regression methods.

    random_seed: int, optional (default = None)
        Sets the random seed.

    verbose: bool, optional (default = False)
        Determines whether the user will get verbose status updates.


    Attributes
    ----------

    grid_values: array of shape (treatment_grid_num, )
        The gridded values of the treatment variable. Equally spaced.

    final_gam: `pygam.LinearGAM` class
        trained final model of `LinearGAM` class, from pyGAM library

    pseudo_out: array of shape (observations, )
        Adjusted, pseudo-outcome observations


    Methods
    ----------
    fit: (self, T, X, y)
        Fits the causal dose-response model

    calculate_CDRC: (self, ci, CDRC_grid_num)
        Calculates the CDRC (and confidence interval) from TMLE estimation


    Examples
    --------

    >>> # With continuous outcome
    >>> from causal_curve import TMLE_Regressor
    >>> tmle = TMLE_Regressor()
    >>> tmle.fit(T = df['Treatment'], X = df[['X_1', 'X_2']], y = df['Outcome'])
    >>> tmle_results = tmle.calculate_CDRC(0.95)
    >>> point_estimate = tmle.point_estimate(np.array([5.0]))
    >>> point_estimate_interval = tmle.point_estimate_interval(np.array([5.0]), 0.95)


    References
    ----------

    Kennedy EH, Ma Z, McHugh MD, Small DS. Nonparametric methods for doubly robust estimation
    of continuous treatment effects. Journal of the Royal Statistical Society, Series B. 79(4), 2017, pp.1229-1245.

    van der Laan MJ and Rubin D. Targeted maximum likelihood learning. In: The International
    Journal of Biostatistics, 2(1), 2006.

    van der Laan MJ and Gruber S. Collaborative double robust penalized targeted
    maximum likelihood estimation. In: The International Journal of Biostatistics 6(1), 2010.

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

    def _validate_init_params(self):
        """
        Checks that the params used when instantiating TMLE model are formatted correctly
        """

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
            raise ValueError("treatment_grid_num parameter is too high!")

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

        # Checks for bandwidth
        if not isinstance(self.bandwidth, (int, float)):
            raise TypeError(
                f"bandwidth parameter must be an integer or float, "
                f"but found type {type(self.bandwidth)}"
            )

        if (self.bandwidth <= 0) or (self.bandwidth >= 1000):
            raise TypeError(
                "bandwidth parameter must be greater than 0 and less than 1000"
            )

        # Checks for random_seed
        if not isinstance(self.random_seed, (int, type(None))):
            raise TypeError(
                f"random_seed parameter must be an int, but found type {type(self.random_seed)}"
            )

        if (isinstance(self.random_seed, int)) and self.random_seed < 0:
            raise ValueError("random_seed parameter must be > 0")

        # Checks for verbose
        if not isinstance(self.verbose, bool):
            raise TypeError(
                f"verbose parameter must be a boolean type, but found type {type(self.verbose)}"
            )

    def _validate_fit_data(self):
        """Verifies that T, X, and y are formatted the right way"""
        # Checks for T column
        if not is_float_dtype(self.t_data):
            raise TypeError("Treatment data must be of type float")

        # Make sure all X columns are float or int
        if isinstance(self.x_data, pd.Series):
            if not is_numeric_dtype(self.x_data):
                raise TypeError(
                    "All covariate (X) columns must be int or float type (i.e. must be numeric)"
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
            raise TypeError("Outcome data must be of type float")

    def _validate_calculate_CDRC_params(self, ci):
        """Validates the parameters given to `calculate_CDRC`"""

        if not isinstance(ci, float):
            raise TypeError(
                f"`ci` parameter must be an float, but found type {type(ci)}"
            )

        if isinstance(ci, float) and ((ci <= 0) or (ci >= 1.0)):
            raise ValueError("`ci` parameter should be between (0, 1)")

    def fit(self, T, X, y):
        """Fits the TMLE causal dose-response model. For now, this only
        accepts pandas columns. You *must* provide at least one covariate column.

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
        self.rand_seed_wrapper(self.random_seed)

        self.t_data = T.reset_index(drop=True, inplace=False)
        self.x_data = X.reset_index(drop=True, inplace=False)
        self.y_data = y.reset_index(drop=True, inplace=False)

        # Validate this input data
        self._validate_fit_data()

        # Capture covariate and treatment column names
        self.treatment_col_name = self.t_data.name

        if len(self.x_data.shape) == 1:
            self.covariate_col_names = [self.x_data.name]
        else:
            self.covariate_col_names = self.x_data.columns.values.tolist()

        # Note the size of the data
        self.num_rows = len(self.t_data)

        # Produce expanded versions of the inputs
        self.if_verbose_print("Transforming data for the Q-model and G-model")
        self.grid_values, self.fully_expanded_x , self.fully_expanded_t_and_x = self._transform_inputs()

        # Fit G-model and get relevent predictions
        self.if_verbose_print("Fitting G-model and making treatment assignment predictions...")
        self.g_model_preds, self.g_model_2_preds = self._g_model()

        # Fit Q-model and get relevent predictions
        self.if_verbose_print("Fitting Q-model and making outcome predictions...")
        self.q_model_preds = self._q_model()

        # Calculating treatment assignment adjustment using G-model's predictions
        self.if_verbose_print("Calculating treatment assignment adjustment using G-model's predictions...")
        self.n_interpd_values, self.var_n_interpd_values = self._treatment_assignment_correction()

        # Adjusting outcome using Q-model's predictions
        self.if_verbose_print("Adjusting outcome using Q-model's predictions...")
        self.outcome_adjust, self.expand_outcome_adjust = self._outcome_adjustment()

        # Calculating corrected pseudo-outcome values
        self.if_verbose_print("Calculating corrected pseudo-outcome values...")
        self.pseudo_out = (self.y_data - self.outcome_adjust) / (self.n_interpd_values / self.var_n_interpd_values) + self.expand_outcome_adjust

        # Training final GAM model using pseudo-outcome values
        self.if_verbose_print("Training final GAM model using pseudo-outcome values...")
        self.final_gam = self._fit_final_gam()


    def calculate_CDRC(self, ci=0.95):
        """Using the results of the fitted model, this generates a dataframe of CDRC point estimates
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
        self.rand_seed_wrapper(self.random_seed)

        self._validate_calculate_CDRC_params(ci)

        self.if_verbose_print("""
            Generating predictions for each value of treatment grid,
            and averaging to get the CDRC..."""
        )

        # Create CDRC predictions from the trained, final GAM

        self._cdrc_preds = self._cdrc_predictions_continuous(ci)

        return pd.DataFrame(
            self._cdrc_preds, columns=["Treatment", "Causal_Dose_Response", "Lower_CI", "Upper_CI"]
        ).round(3)


    def _transform_inputs(self):
        """Takes the treatment and covariates and transforms so they can
        be used by the algo"""

        # Create treatment grid
        grid_values = np.linspace(
            start=self.t_data.min(),
            stop=self.t_data.max(),
            num=self.treatment_grid_num
        )

        # Create expanded treatment array
        expanded_t = np.array([])
        for treat_value in grid_values:
        	expanded_t = np.append(expanded_t, ([treat_value] * self.num_rows))

        # Create expanded treatment array with covariates
        expanded_t_and_x = pd.concat(
            [
                pd.DataFrame(expanded_t),
                pd.concat(
                    [self.x_data] * self.treatment_grid_num
                ).reset_index(drop = True, inplace = False),
            ],
	        axis = 1,
            ignore_index = True
        )

        expanded_t_and_x.columns = [self.treatment_col_name] + self.covariate_col_names

        fully_expanded_t_and_x = pd.concat(
        	[
        		pd.concat([self.x_data, self.t_data], axis=1),
        		expanded_t_and_x
        	],
        	axis = 0,
        	ignore_index = True
        )

        fully_expanded_x = fully_expanded_t_and_x[self.covariate_col_names]

        return grid_values, fully_expanded_x, fully_expanded_t_and_x

    def _g_model(self):
        """Produces the G-model and gets treatment assignment predictions"""

        t = self.t_data.to_numpy()
        X = self.x_data.to_numpy()

        g_model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_seed
        ).fit(X = X, y = t)
        g_model_preds = g_model.predict(self.fully_expanded_x)

        g_model2 = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_seed
        ).fit(X = X, y = ((t - g_model_preds[0:self.num_rows])**2))
        g_model_2_preds = g_model2.predict(self.fully_expanded_x)

        return g_model_preds, g_model_2_preds

    def _q_model(self):
        """Produces the Q-model and gets outcome predictions using the provided treatment
        values, when the treatment is completely present and not present.
        """

        X = pd.concat([self.t_data, self.x_data], axis=1).to_numpy()
        y = self.y_data.to_numpy()

        q_model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_seed
        ).fit(X = X, y = y)
        q_model_preds = q_model.predict(self.fully_expanded_t_and_x)

        return q_model_preds


    def _treatment_assignment_correction(self):
        """Uses the G-model and its predictions to adjust treatment assignment
        """

        t_standard = (
            (self.fully_expanded_t_and_x[self.treatment_col_name] - self.g_model_preds) / np.sqrt(self.g_model_2_preds)
        )

        interpd_values = interp1d(
            self.one_dim_estimate_density(t_standard.values)[0],
            self.one_dim_estimate_density(t_standard.values[0:self.num_rows])[1],
            kind='linear'
        )(t_standard) / np.sqrt(self.g_model_2_preds)

        n_interpd_values = interpd_values[0:self.num_rows]

        temp_interpd = interpd_values[self.num_rows:]

        zeros_mat = np.zeros((self.num_rows, self.treatment_grid_num))

        for i in range(0, self.treatment_grid_num):
        	lower = i * self.num_rows
        	upper = i * self.num_rows + self.num_rows
        	zeros_mat[:,i] = temp_interpd[lower:upper]

        var_n_interpd_values = self.pred_from_loess(
            train_x = self.grid_values,
            train_y = zeros_mat.mean(axis = 0),
            x_to_pred = self.t_data
        )

        return n_interpd_values, var_n_interpd_values


    def _outcome_adjustment(self):
        """Uses the Q-model and its predictions to adjust the outcome
        """

        outcome_adjust = self.q_model_preds[0:self.num_rows]

        temp_outcome_adjust = self.q_model_preds[self.num_rows:]

        zero_mat = np.zeros((self.num_rows, self.treatment_grid_num))
        for i in range(0, self.treatment_grid_num):
        	lower = i * self.num_rows
        	upper = i * self.num_rows + self.num_rows
        	zero_mat[:,i] = temp_outcome_adjust[lower:upper]

        expand_outcome_adjust = self.pred_from_loess(
            train_x = self.grid_values,
            train_y = zero_mat.mean(axis = 0),
            x_to_pred = self.t_data
        )

        return outcome_adjust, expand_outcome_adjust

    def _fit_final_gam(self):
        """We now regress the original treatment values against the pseudo-outcome values
        """

        return LinearGAM(
        	s(0, n_splines=30, spline_order=3),
            max_iter=500,
            lam=self.bandwidth
        ).fit(self.t_data, y = self.pseudo_out)

    def one_dim_estimate_density(self, series):
    	"""
    	Takes in a numpy array, returns grid values for KDE and predicted probabilities
    	"""
    	series_grid = np.linspace(
            start=series.min(),
            stop=series.max(),
            num=self.num_rows
        )

    	kde = KernelDensity(
            kernel='gaussian',
            bandwidth=self.bandwidth
        ).fit(series.reshape(-1, 1))

    	return series_grid, np.exp(kde.score_samples(series_grid.reshape(-1, 1)))

    def pred_from_loess(self, train_x, train_y, x_to_pred):
    	"""
    	Trains simple loess regression and returns predictions
    	"""
    	kr_model = KernelReg(
            endog = train_y,
            exog = train_x,
            var_type = 'c',
            bw = [self.bandwidth]
        )

    	return kr_model.fit(x_to_pred)[0]
