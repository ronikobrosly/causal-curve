"""
Defines the Mediation test class
"""

from pprint import pprint

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype
from pygam import LinearGAM, s

from causal_curve.core import Core


class Mediation(Core):
    """
    Given three continuous variables (a treatment or independent variable of interest,
    a potential mediator, and an outcome variable of interest), Mediation provides a method
    to determine the average direct and indirect effect.

    Parameters
    ----------

    treatment_grid_num: int, optional (default = 10)
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

    bootstrap_draws: int, optional (default = 500)
        Bootstrapping is used as part of the mediation test. The parameter determines
        the number of draws from the original data to create a single bootstrap replicate.

    bootstrap_replicates: int, optional (default = 100)
        Bootstrapping is used as part of the mediation test. The parameter determines
        the number of bootstrapping runs to perform / number of new datasets to create.

    spline_order: int, optional (default = 3)
        Order of the splines to use fitting the final GAM.
        Must be integer >= 1. Default value creates cubic splines.

    n_splines: int, optional (default = 5)
        Number of splines to use for the mediation and outcome GAMs.
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


    Methods
    ----------
    fit: (self, T, M, y)
        Fits the trio of relevant variables using generalized additive models.

    calculate_effects: (self, ci)
        Calculates the average direct and indirect effects.


    Examples
    --------
    >>> from causal_curve import Mediation
    >>> med = Mediation(treatment_grid_num = 200, random_seed = 512)
    >>> med.fit(T = df['Treatment'], M = df['Mediator'], y = df['Outcome'])
    >>> med_results = med.calculate_effects(0.95)


    References
    ----------

    Imai K., Keele L., Tingley D. A General Approach to Causal Mediation Analysis. Psychological
    Methods. 15(4), 2010, pp.309–334.

    """

    def __init__(
        self,
        treatment_grid_num=10,
        lower_grid_constraint=0.01,
        upper_grid_constraint=0.99,
        bootstrap_draws=500,
        bootstrap_replicates=100,
        spline_order=3,
        n_splines=5,
        lambda_=0.5,
        max_iter=100,
        random_seed=None,
        verbose=False,
    ):

        self.treatment_grid_num = treatment_grid_num
        self.lower_grid_constraint = lower_grid_constraint
        self.upper_grid_constraint = upper_grid_constraint
        self.bootstrap_draws = bootstrap_draws
        self.bootstrap_replicates = bootstrap_replicates
        self.spline_order = spline_order
        self.n_splines = n_splines
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.verbose = verbose

        # Validate the params
        self._validate_init_params()
        self.rand_seed_wrapper()

        if self.verbose:
            print("Using the following params for the mediation analysis:")
            pprint(self.get_params(), indent=4)

    def _validate_init_params(self):
        """
        Checks that the params used when instantiating mediation tool are formatted correctly
        """

        # Checks for treatment_grid_num
        if not isinstance(self.treatment_grid_num, int):
            raise TypeError(
                f"treatment_grid_num parameter must be an integer, "
                f"but found type {type(self.treatment_grid_num)}"
            )

        if (isinstance(self.treatment_grid_num, int)) and self.treatment_grid_num < 4:
            raise ValueError(
                f"treatment_grid_num parameter should be >= 4 so the internal models "
                f"have enough resolution, but found value {self.treatment_grid_num}"
            )

        if (isinstance(self.treatment_grid_num, int)) and self.treatment_grid_num > 100:
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

        # Checks for bootstrap_draws
        if not isinstance(self.bootstrap_draws, int):
            raise TypeError(
                f"bootstrap_draws parameter must be a int, "
                f"but found type {type(self.bootstrap_draws)}"
            )

        if (isinstance(self.bootstrap_draws, int)) and self.bootstrap_draws < 100:
            raise ValueError(
                f"bootstrap_draws parameter cannot be < 100, "
                f"but found value {self.bootstrap_draws}"
            )

        if (isinstance(self.bootstrap_draws, int)) and self.bootstrap_draws > 500000:
            raise ValueError(
                f"bootstrap_draws parameter cannot > 500000, "
                f"but found value {self.bootstrap_draws}"
            )

        # Checks for bootstrap_replicates
        if not isinstance(self.bootstrap_replicates, int):
            raise TypeError(
                f"bootstrap_replicates parameter must be a int, "
                f"but found type {type(self.bootstrap_replicates)}"
            )

        if (
            isinstance(self.bootstrap_replicates, int)
        ) and self.bootstrap_replicates < 50:
            raise ValueError(
                f"bootstrap_replicates parameter cannot be < 50, "
                f"but found value {self.bootstrap_replicates}"
            )

        if (
            isinstance(self.bootstrap_replicates, int)
        ) and self.bootstrap_replicates > 100000:
            raise ValueError(
                f"bootstrap_replicates parameter cannot > 100000, "
                f"but found value {self.bootstrap_replicates}"
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

        if (isinstance(self.spline_order, int)) and self.spline_order < 3:
            raise ValueError(
                f"spline_order parameter should be >= 1, but found {self.spline_order}"
            )

        if (isinstance(self.spline_order, int)) and self.spline_order >= 30:
            raise ValueError("spline_order parameter is too high!")

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
            raise ValueError("n_splines parameter is too high!")

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
            raise ValueError("lambda_ parameter is too high!")

        # Checks for max_iter
        if not isinstance(self.max_iter, int):
            raise TypeError(
                f"max_iter parameter must be an int, but found type {type(self.max_iter)}"
            )

        if (isinstance(self.max_iter, int)) and self.max_iter <= 10:
            raise ValueError(
                "max_iter parameter is too low! Results won't be reliable!"
            )

        if (isinstance(self.max_iter, int)) and self.max_iter >= 1e6:
            raise ValueError("max_iter parameter is unnecessarily high!")

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
        """Verifies that T, M, and y are formatted the right way"""
        # Checks for T column
        if not is_float_dtype(self.T):
            raise TypeError("Treatment data must be of type float")

        # Checks for M column
        if not is_float_dtype(self.M):
            raise TypeError("Mediation data must be of type float")

        # Checks for Y column
        if not is_float_dtype(self.y):
            raise TypeError("Outcome data must be of type float")

    def _grid_values(self):
        """Produces initial grid values for the treatment variable"""
        return np.quantile(
            self.T,
            q=np.linspace(
                start=self.lower_grid_constraint,
                stop=self.upper_grid_constraint,
                num=self.treatment_grid_num,
            ),
        )

    def _collect_mean_t_levels(self):
        """Collects the mean treatment value within each treatment bucket in the grid_values"""

        t_bin_means = []

        for index, _ in enumerate(self.grid_values):
            if index == (len(self.grid_values) - 1):
                continue

            t_bin_means.append(
                self.T[
                    (
                        (self.T >= self.grid_values[index])
                        & (self.T <= self.grid_values[index + 1])
                    )
                ].mean()
            )

        return t_bin_means

    def fit(self, T, M, y):
        """Fits models so that mediation analysis can be run.
        For now, this only accepts pandas columns.

        Parameters
        ----------
        T: array-like, shape (n_samples,)
            A continuous treatment variable
        M: array-like, shape (n_samples,)
            A continuous mediation variable
        y: array-like, shape (n_samples,)
            A continuous outcome variable

        Returns
        ----------
        self : object

        """
        self.rand_seed_wrapper(self.random_seed)

        self.T = T.reset_index(drop=True, inplace=False)
        self.M = M.reset_index(drop=True, inplace=False)
        self.y = y.reset_index(drop=True, inplace=False)

        # Validate this input data
        self._validate_fit_data()

        self.n = len(y)

        # Create grid_values
        self.grid_values = self._grid_values()

        # Loop through the comparisons in the grid_values
        if self.verbose:
            print("Beginning main loop through treatment bins...")

        # Collect loop results in this list
        self.final_bootstrap_results = []

        # Begin main loop
        for index, _ in enumerate(self.grid_values):
            if index == 0:
                continue
            if self.verbose:
                print(
                    f"***** Starting iteration {index} of {len(self.grid_values) - 1} *****"
                )

            temp_low_treatment = self.grid_values[index - 1]
            temp_high_treatment = self.grid_values[index]

            bootstrap_results = self._bootstrap_analysis(
                temp_low_treatment, temp_high_treatment
            )

            self.final_bootstrap_results.append(bootstrap_results)

    def calculate_mediation(self, ci=0.95):
        """Conducts mediation analysis on the fit data

        Parameters
        ----------
        ci: float (default = 0.95)
            The desired bootstrap confidence interval to produce. Default value is 0.95,
            corresponding to 95% confidence intervals. bounded (0, 1.0).

        Returns
        ----------
        dataframe: Pandas dataframe
            Contains the estimate of the direct and indirect effects
            and the proportion of indirect effects across the treatment grid values.
            The bootstrap confidence interval that is returned might not be symmetric.

        self : object

        """
        self.rand_seed_wrapper(self.random_seed)

        # Collect effect results in these lists
        self.t_bin_means = self._collect_mean_t_levels()
        self.prop_direct_list = []
        self.prop_indirect_list = []
        general_indirect = []

        lower = (1 - ci) / 2
        upper = ci + lower

        # Calculate results for each treatment bin
        for index, _ in enumerate(self.grid_values):

            if index == (len(self.grid_values) - 1):
                continue

            temp_bootstrap_results = self.final_bootstrap_results[index]

            mean_results = {
                key: temp_bootstrap_results[key].mean()
                for key in temp_bootstrap_results
            }

            tau_coef = (
                mean_results["d1"]
                + mean_results["d0"]
                + mean_results["z1"]
                + mean_results["z0"]
            ) / 2
            n0 = mean_results["d0"] / tau_coef
            n1 = mean_results["d1"] / tau_coef
            n_avg = (n0 + n1) / 2

            tau_general = (
                temp_bootstrap_results["d1"]
                + temp_bootstrap_results["d0"]
                + temp_bootstrap_results["z1"]
                + temp_bootstrap_results["z0"]
            ) / 2
            nu_0_general = temp_bootstrap_results["d0"] / tau_general
            nu_1_general = temp_bootstrap_results["d1"] / tau_general
            nu_avg_general = (nu_0_general + nu_1_general) / 2

            self.prop_direct_list.append(1 - n_avg)
            self.prop_indirect_list.append(n_avg)
            general_indirect.append(nu_avg_general)

        general_indirect = pd.concat(general_indirect)

        # Bootstrap these general_indirect values
        bootstrap_overall_means = []
        for _ in range(0, 1000):
            bootstrap_overall_means.append(
                general_indirect.sample(frac=0.25, replace=True).mean()
            )

        bootstrap_overall_means = np.array(bootstrap_overall_means)

        final_results = pd.DataFrame(
            {
                "Treatment_Value": self.t_bin_means,
                "Proportion_Direct_Effect": self.prop_direct_list,
                "Proportion_Indirect_Effect": self.prop_indirect_list,
            }
        ).round(4)

        # Clip Proportion_Direct_Effect and Proportion_Indirect_Effect
        final_results["Proportion_Direct_Effect"].clip(lower=0, upper=1.0, inplace=True)
        final_results["Proportion_Indirect_Effect"].clip(
            lower=0, upper=1.0, inplace=True
        )

        # Calculate overall, mean, indirect effect
        total_prop_mean = round(np.array(self.prop_indirect_list).mean(), 4)
        total_prop_lower = self.clip_negatives(
            round(np.percentile(bootstrap_overall_means, q=lower * 100), 4)
        )
        total_prop_upper = self.clip_negatives(
            round(np.percentile(bootstrap_overall_means, q=upper * 100), 4)
        )

        print(
            f"""\n\nMean indirect effect proportion:
            {total_prop_mean} ({total_prop_lower} - {total_prop_upper})
            """
        )
        return final_results

    def _bootstrap_analysis(self, temp_low_treatment, temp_high_treatment):
        """The top-level function used in the fitting method"""

        bootstrap_collection = []

        for _ in range(0, self.bootstrap_replicates):
            # Create single bootstrap replicate
            temp_t, temp_m, temp_y = self._create_bootstrap_replicate()
            # Create the models from this
            temp_mediator_model, temp_outcome_model = self._fit_gams(
                temp_t, temp_m, temp_y
            )
            # Make mediator predictions
            predict_m1, predict_m0 = self._mediator_prediction(
                temp_mediator_model,
                temp_t,
                temp_m,
                temp_low_treatment,
                temp_high_treatment,
            )
            # Make outcome predictions
            outcome_preds = self._outcome_prediction(
                temp_low_treatment,
                temp_high_treatment,
                predict_m1,
                predict_m0,
                temp_outcome_model,
            )
            # Collect the replicate results here
            bootstrap_collection.append(outcome_preds)

        # Convert this into a dataframe
        bootstrap_results = pd.DataFrame(bootstrap_collection)

        return bootstrap_results

    def _create_bootstrap_replicate(self):
        """Creates a single bootstrap replicate from the data"""
        temp_t = self.T.sample(n=self.bootstrap_draws, replace=True)
        temp_m = self.M.iloc[temp_t.index]
        temp_y = self.y.iloc[temp_t.index]

        return temp_t, temp_m, temp_y

    def _fit_gams(self, temp_t, temp_m, temp_y):
        """Fits the mediator and outcome GAMs"""
        temp_mediator_model = LinearGAM(
            s(0, n_splines=self.n_splines, spline_order=self.spline_order),
            fit_intercept=True,
            max_iter=self.max_iter,
            lam=self.lambda_,
        )
        temp_mediator_model.fit(temp_t, temp_m)

        temp_outcome_model = LinearGAM(
            s(0, n_splines=self.n_splines, spline_order=self.spline_order)
            + s(1, n_splines=self.n_splines, spline_order=self.spline_order),
            fit_intercept=True,
            max_iter=self.max_iter,
            lam=self.lambda_,
        )
        temp_outcome_model.fit(pd.concat([temp_t, temp_m], axis=1), temp_y)

        return temp_mediator_model, temp_outcome_model

    def _mediator_prediction(
        self,
        temp_mediator_model,
        temp_t,
        temp_m,
        temp_low_treatment,
        temp_high_treatment,
    ):
        """Makes predictions based on the mediator models"""

        m1_mean = temp_mediator_model.predict(temp_high_treatment)[0]
        m0_mean = temp_mediator_model.predict(temp_low_treatment)[0]

        std_dev = (
            (temp_mediator_model.deviance_residuals(temp_t, temp_m) ** 2).sum()
        ) / (self.n - (len(temp_mediator_model.get_params()["terms"]._terms) + 1))

        est_error = np.random.normal(loc=0, scale=std_dev, size=self.n)

        predict_m1 = m1_mean + est_error
        predict_m0 = m0_mean + est_error

        return predict_m1, predict_m0

    def _outcome_prediction(
        self,
        temp_low_treatment,
        temp_high_treatment,
        predict_m1,
        predict_m0,
        temp_outcome_model,
    ):
        """Makes predictions based on the outcome models"""

        outcome_preds = {}

        inputs = [
            ["d1", temp_high_treatment, temp_high_treatment, predict_m1, predict_m0],
            ["d0", temp_low_treatment, temp_low_treatment, predict_m1, predict_m0],
            ["z1", temp_high_treatment, temp_low_treatment, predict_m1, predict_m1],
            ["z0", temp_high_treatment, temp_low_treatment, predict_m0, predict_m0],
        ]

        for element in inputs:

            # Set treatment values
            t_1 = element[1]
            t_0 = element[2]

            # Set mediator values
            m_1 = element[3]
            m_0 = element[4]

            pr_1 = temp_outcome_model.predict(
                np.column_stack((np.repeat(t_1, self.n), m_1))
            )

            pr_0 = temp_outcome_model.predict(
                np.column_stack((np.repeat(t_0, self.n), m_0))
            )

            outcome_preds[element[0]] = (pr_1 - pr_0).mean()

        return outcome_preds
