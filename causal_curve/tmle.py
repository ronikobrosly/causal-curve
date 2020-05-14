"""
Defines the Targetted Maximum likelihood Estimation (TMLE) model class
"""
import pdb
from pprint import pprint

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_numeric_dtype
from statsmodels.genmod.generalized_linear_model import GLM
from xgboost import XGBClassifier, XGBRegressor

from causal_curve.core import Core
from causal_curve.utils import rand_seed_wrapper


class TMLE(Core):
    """
    Constructs a causal dose response curve through a series of TMLE comparisons across a grid
    of the treatment values. XGBoost is used for prediction in Q model and G model.
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
        bins. In other words, make sure there aren't too few data points in each bin. The lowest
        element in the list should be minimum treatment value, the highest element should be the
        maximum treatment value.

    n_estimators: int, optional (default = 100)
        Optional argument to set the number of learners to use when XGBoost creates TMLE's Q and G models.

    learning_rate: float, optional (default = 0.1)
        Optional argument to set the XGBoost's learning rate for TMLE's Q and G models.

    max_depth: int, optional (default = 5)
        Optional argument to set XGBoost's maximum depth when creating TMLE's Q and G models.

    gamma: float, optional (default = 1.0)
        Optional argument to set XGBoost's gamma parameter (regularization) when creating TMLE's Q and G models.

    random_seed: int, optional (default = None)
        Sets the random seed.

    verbose: bool, optional (default = False)
        Determines whether the user will get verbose status updates.


    Attributes
    ----------


    Methods
    ----------
    fit: (self, T, X, y)
        Fits the causal dose-response model

    Examples
    --------


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
        learning_rate = 0.1,
        max_depth = 5,
        gamma = 1.0,
        random_seed=None,
        verbose=False,
    ):

        self.treatment_grid_bins = treatment_grid_bins
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.gamma = gamma
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
                f"treatment_grid_bins parameter must be a list, \
                 but found type {type(self.treatment_grid_bins)}"
            )

        for element in self.treatment_grid_bins:
            if not isinstance(element, (int, float)):
                raise TypeError(
                    f"'{element}' in `treatment_grid_bins` list is not of type float or int, \
                     it is {type(element)}"
                )

        if len(self.treatment_grid_bins) < 2:
            raise TypeError(
                "treatment_grid_bins list must, at minimum, of length >= 2"
            )

        # Checks for n_estimators
        if not isinstance(self.n_estimators, int):
            raise TypeError(
                f"n_estimators parameter must be an integer, \
                but found type {type(self.n_estimators)}"
            )

        if (self.n_estimators < 10) or (self.n_estimators > 100000) :
            raise TypeError(
                "n_estimators parameter must be between 10 and 100000"
            )

        # Checks for learning_rate
        if not isinstance(self.learning_rate, (int, float)):
            raise TypeError(
                f"learning_rate parameter must be an integer or float, \
                but found type {type(self.learning_rate)}"
            )

        if (self.learning_rate <= 0 ) or (self.learning_rate >= 1000) :
            raise TypeError(
                "learning_rate parameter must be greater than 0 and less than 1000"
            )

        # Checks for max_depth
        if not isinstance(self.max_depth, int):
            raise TypeError(
                f"max_depth parameter must be an integer, \
                but found type {type(self.max_depth)}"
            )

        if self.max_depth <= 0:
            raise TypeError(
                "max_depth parameter must be greater than 0"
            )

        # Checks for gamma
        if not isinstance(self.gamma, float):
            raise TypeError(
                f"gamma parameter must be a float, \
                but found type {type(self.gamma)}"
            )

        if self.gamma <= 0:
            raise TypeError(
                "gamma parameter must be greater than 0"
            )

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
                        f"All covariate (X) columns must be int or float type (i.e. must be numeric)"
                    )

        # Checks for Y column
        if not is_float_dtype(self.y):
            raise TypeError(f"Outcome data must be of type float")




    def _initial_bucket_mean_prediction(self):
        """Creates a model to predict the outcome variable given the provided inputs within
        the first bucket of treatment_grid_bins. This returns the mean predicted outcome.
        """

        y = self.y[self.T < self.treatment_grid_bins[1]]
        X = pd.concat([self.T, self.X], axis = 1)[self.T < self.treatment_grid_bins[1]]

        init_model = XGBRegressor(
            n_estimators = self.n_estimators,
            max_depth = self.max_depth,
            learning_rate = self.learning_rate,
            gamma = self.gamma,
            random_state = self.random_seed
        ).fit(X, y)

        return init_model.predict(X).mean()


    def _create_treatment_comparison_df(self, low_boundary, med_boundary, high_boundary):
        """Given the current boundaries chosen from treatment_grid_bins, this filters
        the input data appropriately.
        """
        temp_y = self.y[((self.T >= low_boundary) & (self.T <= high_boundary))]
        temp_x = self.X[((self.T >= low_boundary) & (self.T <= high_boundary))]
        temp_t = self.T.copy()
        temp_t = temp_t[((temp_t >= low_boundary) & (temp_t <= high_boundary))]
        temp_t[((temp_t >= low_boundary) & (temp_t < med_boundary))] = 0
        temp_t[((temp_t >= med_boundary) & (temp_t <= high_boundary))] = 1

        return temp_y, temp_x, temp_t


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
        self.T = T
        self.X = X
        self.y = y
        self.n = len(y)

        # Validate this input data
        self._validate_fit_data()

        # Get the mean, predicted outcome value within the first bucket
        if self.verbose:
            print("Calculating the mean, predicted outcome value within the first bucket...")
        self.outcome_start_val = self._initial_bucket_mean_prediction()

        # Loop through the comparisons in the treatment_grid_bins
        if self.verbose:
            print("Beginning main loop through treatment bins...")

        for index, _ in enumerate(self.treatment_grid_bins):
            if (index == 0) or (index == len(self.treatment_grid_bins) - 1):
                continue
            if self.verbose:
                print(f"***** Starting loop {index} of {len(self.treatment_grid_bins) - 2} *****")
            low_boundary = self.treatment_grid_bins[index - 1]
            med_boundary = self.treatment_grid_bins[index]
            high_boundary = self.treatment_grid_bins[index + 1]

            # Create comparison dataset
            temp_y, temp_x, temp_t = self._create_treatment_comparison_df(low_boundary, med_boundary, high_boundary)

            # Fit Q-model and get relevent predictions
            if self.verbose:
                print("Fitting Q-model and making predictions...")
            self.Y_hat_a, self.Y_hat_1, self.Y_hat_0 = self._q_model(temp_y, temp_x, temp_t)

            # Fit G-model and get relevent predictions
            if self.verbose:
                print("Fitting G-model and making predictions...")
            self.pi_hat1, self.pi_hat0 = self._g_model(temp_x, temp_t)

            if self.verbose:
                print(f"Finished this loop!")


    def _q_model(self, temp_y, temp_x, temp_t):
        """Produces the Q-model and gets outcome predictions using the provided treatment
        values, when the treatment is completely present and not present.
        """

        X = pd.concat([temp_t, temp_x], axis = 1).to_numpy()
        y = temp_y.to_numpy()

        Q_model = XGBRegressor(
            n_estimators = self.n_estimators,
            max_depth = self.max_depth,
            learning_rate = self.learning_rate,
            gamma = self.gamma,
            random_state = self.random_seed
        ).fit(X, y)

        # Make predictions with provided treatment values
        Y_hat_a = Q_model.predict(X)

        temp =  np.column_stack(
        (
            np.repeat(1, self.n),
            np.asarray(self.X)
    	))

        # Make predictions when treatment is completely present
        Y_hat_1 = Q_model.predict(temp)

        # Make predictions when treatment is completely not present
        temp = np.column_stack(
        (
            np.repeat(0, self.n),
            np.asarray(self.X)
        ))

        Y_hat_0 = Q_model.predict(temp)

        return Y_hat_a, Y_hat_1, Y_hat_0



    def _g_model(self, temp_x, temp_t):
        """Produces the G-model and gets treatment assignment predictions
        """

        X = temp_x.to_numpy()
        t = temp_t.to_numpy()

        G_model = XGBClassifier(
            n_estimators = self.n_estimators,
            max_depth = self.max_depth,
            learning_rate = self.learning_rate,
            gamma = self.gamma,
            random_state = self.random_seed
        ).fit(X, t)

        # Make predictions of receiving treatment
        pi_hat1 = G_model.predict_proba(X)

        # Predictions of not receiving treatment
        pi_hat0 = 1 - pi_hat1

        return pi_hat1, pi_hat0
