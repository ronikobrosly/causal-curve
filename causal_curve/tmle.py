"""
Defines the Targetted Maximum likelihood Estimation (TMLE) model class
"""

from pprint import pprint


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

    treatment_grid_bins: list of floats
        Represents the edges of bins of treatment values that are used to construct a smooth curve
        Look at the distribution of your treatment variable to determine which
        family is more appropriate. Be mindful of the "positivity" assumption when determining
        bins.

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
        if not isinstance(self.treatment_grid_bins, type(list)):
            raise TypeError(
                f"treatment_grid_bins parameter must be a list, \
                 but found type {type(self.treatment_grid_bins)}"
            )

        for element in self.treatment_grid_bins:
            if not isinstance(element, float):
                raise TypeError(
                    f"'{element}' in `treatment_grid_bins` list is not of type float, \
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
                "n_estimators parameter must be between 10 and 100000}"
            )
