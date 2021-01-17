""" Unit tests of the tmle.py module """

import numpy as np
import pytest

from causal_curve import TMLE_Regressor


def test_point_estimate_method_good(TMLE_fitted_model_continuous_fixture):
    """
    Tests the GPS `point_estimate` method using appropriate data (with a continuous outcome)
    """

    observed_result = TMLE_fitted_model_continuous_fixture.point_estimate(
        np.array([50])
    )
    assert isinstance(observed_result[0][0], float)
    assert len(observed_result[0]) == 1

    observed_result = TMLE_fitted_model_continuous_fixture.point_estimate(
        np.array([40, 50, 60])
    )
    assert isinstance(observed_result[0][0], float)
    assert len(observed_result[0]) == 3


def test_point_estimate_interval_method_good(TMLE_fitted_model_continuous_fixture):
    """
    Tests the GPS `point_estimate_interval` method using appropriate data (with a continuous outcome)
    """
    observed_result = TMLE_fitted_model_continuous_fixture.point_estimate_interval(
        np.array([50])
    )
    assert isinstance(observed_result[0][0], float)
    assert observed_result.shape == (1, 2)

    observed_result = TMLE_fitted_model_continuous_fixture.point_estimate_interval(
        np.array([40, 50, 60])
    )
    assert isinstance(observed_result[0][0], float)
    assert observed_result.shape == (3, 2)


@pytest.mark.parametrize(
    (
        "treatment_grid_num",
        "lower_grid_constraint",
        "upper_grid_constraint",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "bandwidth",
        "random_seed",
        "verbose",
    ),
    [
        # treatment_grid_num
        (100.0, 0.01, 0.99, 200, 0.01, 3, 0.5, None, False),
        ("100.0", 0.01, 0.99, 200, 0.01, 3, 0.5, None, False),
        (2, 0.01, 0.99, 200, 0.01, 3, 0.5, None, False),
        (500000, 0.01, 0.99, 200, 0.01, 3, 0.5, None, False),
        # lower_grid_constraint
        (100, {0.01: "a"}, 0.99, 200, 0.01, 3, 0.5, None, False),
        (100, -0.01, 0.99, 200, 0.01, 3, 0.5, None, False),
        (100, 6.05, 0.99, 200, 0.01, 3, 0.5, None, False),
        # upper_grid_constraint
        (100, 0.01, [1, 2, 3], 200, 0.01, 3, 0.5, None, False),
        (100, 0.01, -0.05, 200, 0.01, 3, 0.5, None, False),
        (100, 0.01, 5.99, 200, 0.01, 3, 0.5, None, False),
        (100, 0.9, 0.2, 200, 0.01, 3, 0.5, None, False),
        # n_estimators
        (100, 0.01, 0.99, "3.0", 0.01, 3, 0.5, None, False),
        (100, 0.01, 0.99, -5, 0.01, 3, 0.5, None, False),
        (100, 0.01, 0.99, 10000000, 0.01, 3, 0.5, None, False),
        # learning_rate
        (100, 0.01, 0.99, 200, ["a", "b"], 3, 0.5, None, False),
        (100, 0.01, 0.99, 200, 5000000, 3, 0.5, None, False),
        # max_depth
        (100, 0.01, 0.99, 200, 0.01, "a", 0.5, None, False),
        (100, 0.01, 0.99, 200, 0.01, -6, 0.5, None, False),
        # bandwidth
        (100, 0.01, 0.99, 200, 0.01, 3, "b", None, False),
        (100, 0.01, 0.99, 200, 0.01, 3, -10, None, False),
        # random seed
        (100, 0.01, 0.99, 200, 0.01, 3, 0.5, "b", False),
        (100, 0.01, 0.99, 200, 0.01, 3, 0.5, -10, False),
        # verbose
        (100, 0.01, 0.99, 200, 0.01, 3, 0.5, None, "Verbose"),
    ],
)
def test_bad_tmle_instantiation(
    treatment_grid_num,
    lower_grid_constraint,
    upper_grid_constraint,
    n_estimators,
    learning_rate,
    max_depth,
    bandwidth,
    random_seed,
    verbose,
):
    with pytest.raises(Exception) as bad:
        TMLE_Regressor(
            treatment_grid_num=treatment_grid_num,
            lower_grid_constraint=lower_grid_constraint,
            upper_grid_constraint=upper_grid_constraint,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            bandwidth=bandwidth,
            random_seed=random_seed,
            verbose=verbose,
        )
