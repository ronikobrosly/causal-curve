""" Unit tests for the GPS_Core class """

import numpy as np
from pygam import LinearGAM
import pytest

from causal_curve import GPS_Regressor


def test_point_estimate_method_good(GPS_fitted_model_continuous_fixture):
    """
    Tests the GPS `point_estimate` method using appropriate data (with a continuous outcome)
    """

    observed_result = GPS_fitted_model_continuous_fixture.point_estimate(np.array([50]))
    assert isinstance(observed_result[0][0], float)
    assert len(observed_result[0]) == 1

    observed_result = GPS_fitted_model_continuous_fixture.point_estimate(
        np.array([40, 50, 60])
    )
    assert isinstance(observed_result[0][0], float)
    assert len(observed_result[0]) == 3


def test_point_estimate_interval_method_good(GPS_fitted_model_continuous_fixture):
    """
    Tests the GPS `point_estimate_interval` method using appropriate data (with a continuous outcome)
    """
    observed_result = GPS_fitted_model_continuous_fixture.point_estimate_interval(
        np.array([50])
    )
    assert isinstance(observed_result[0][0], float)
    assert observed_result.shape == (1, 2)

    observed_result = GPS_fitted_model_continuous_fixture.point_estimate_interval(
        np.array([40, 50, 60])
    )
    assert isinstance(observed_result[0][0], float)
    assert observed_result.shape == (3, 2)


def test_point_estimate_method_bad(GPS_fitted_model_continuous_fixture):
    """
    Tests the GPS `point_estimate` method using appropriate data (with a continuous outcome)
    """

    GPS_fitted_model_continuous_fixture.outcome_type = "binary"

    with pytest.raises(Exception) as bad:
        observed_result = GPS_fitted_model_continuous_fixture.point_estimate(
            np.array([50])
        )


def test_point_estimate_interval_method_bad(GPS_fitted_model_continuous_fixture):
    """
    Tests the GPS `point_estimate_interval` method using appropriate data (with a continuous outcome)
    """

    GPS_fitted_model_continuous_fixture.outcome_type = "binary"

    with pytest.raises(Exception) as bad:
        observed_result = GPS_fitted_model_continuous_fixture.point_estimate_interval(
            np.array([50])
        )


@pytest.mark.parametrize(
    (
        "gps_family",
        "treatment_grid_num",
        "lower_grid_constraint",
        "upper_grid_constraint",
        "spline_order",
        "n_splines",
        "lambda_",
        "max_iter",
        "random_seed",
        "verbose",
    ),
    [
        (546, 10, 0, 1.0, 3, 10, 0.5, 100, 100, True),
        ("linear", 10, 0, 1.0, 3, 10, 0.5, 100, 100, True),
        (None, "hehe", 0, 1.0, 3, 10, 0.5, 100, 100, True),
        (None, 2, 0, 1.0, 3, 10, 0.5, 100, 100, True),
        (None, 100000, 0, 1.0, 3, 10, 0.5, 100, 100, True),
        (None, 10, "hehe", 1.0, 3, 10, 0.5, 100, 100, True),
        (None, 10, -1.0, 1.0, 3, 10, 0.5, 100, 100, True),
        (None, 10, 1.5, 1.0, 3, 10, 0.5, 100, 100, True),
        (None, 10, 0, "hehe", 3, 10, 0.5, 100, 100, True),
        (None, 10, 0, 1.5, 3, 10, 0.5, 100, 100, True),
        (None, 100, -3.0, 0.99, 3, 30, 0.5, 100, None, True),
        (None, 100, 0.01, 1, 3, 30, 0.5, 100, None, True),
        (None, 100, 0.01, -4.5, 3, 30, 0.5, 100, None, True),
        (None, 100, 0.01, 5.5, 3, 30, 0.5, 100, None, True),
        (None, 100, 0.99, 0.01, 3, 30, 0.5, 100, None, True),
        (None, 100, 0.01, 0.99, 3.0, 30, 0.5, 100, None, True),
        (None, 100, 0.01, 0.99, -2, 30, 0.5, 100, None, True),
        (None, 100, 0.01, 0.99, 30, 30, 0.5, 100, None, True),
        (None, 100, 0.01, 0.99, 3, 30.0, 0.5, 100, None, True),
        (None, 100, 0.01, 0.99, 3, -2, 0.5, 100, None, True),
        (None, 100, 0.01, 0.99, 3, 500, 0.5, 100, None, True),
        (None, 100, 0.01, 0.99, 3, 30, 0.5, 100.0, None, True),
        (None, 100, 0.01, 0.99, 3, 30, 0.5, -100, None, True),
        (None, 100, 0.01, 0.99, 3, 30, 0.5, 10000000000, None, True),
        (None, 100, 0.01, 0.99, 3, 30, 0.5, 100, 234.5, True),
        (None, 100, 0.01, 0.99, 3, 30, 0.5, 100, -5, True),
        (None, 100, 0.01, 0.99, 3, 30, 0.5, 100, None, 4.0),
        (None, 10, 0, -1, 3, 10, 0.5, 100, 100, True),
        (None, 10, 0, 1, 3, 10, 0.5, 100, 100, True),
        (None, 10, 0, 1, "splines", 10, 0.5, 100, 100, True),
        (None, 10, 0, 1, 0, 10, 0.5, 100, 100, True),
        (None, 10, 0, 1, 200, 10, 0.5, 100, 100, True),
        (None, 10, 0, 1, 3, 0, 0.5, 100, 100, True),
        (None, 10, 0, 1, 3, 1e6, 0.5, 100, 100, True),
        (None, 10, 0, 1, 3, 10, 0.5, 100, 100, True),
        (None, 10, 0, 1, 3, 10, 0.5, "many", 100, True),
        (None, 10, 0, 1, 3, 10, 0.5, 5, 100, True),
        (None, 10, 0, 1, 3, 10, 0.5, 1e7, 100, True),
        (None, 10, 0, 1, 3, 10, 0.5, 100, "random", True),
        (None, 10, 0, 1, 3, 10, 0.5, 100, -1.5, True),
        (None, 10, 0, 1, 3, 10, 0.5, 100, 111, "True"),
        (None, 100, 0.01, 0.99, 3, 30, "lambda", 100, None, True),
        (None, 100, 0.01, 0.99, 3, 30, -1.0, 100, None, True),
        (None, 100, 0.01, 0.99, 3, 30, 2000.0, 100, None, True),
    ],
)
def test_bad_gps_instantiation(
    gps_family,
    treatment_grid_num,
    lower_grid_constraint,
    upper_grid_constraint,
    spline_order,
    n_splines,
    lambda_,
    max_iter,
    random_seed,
    verbose,
):
    """
    Tests for exceptions when the GPS class if call with bad inputs.
    """
    with pytest.raises(Exception) as bad:
        GPS_Regressor(
            gps_family=gps_family,
            treatment_grid_num=treatment_grid_num,
            lower_grid_constraint=lower_grid_constraint,
            upper_grid_constraint=upper_grid_constraint,
            spline_order=spline_order,
            n_splines=n_splines,
            lambda_=lambda_,
            max_iter=max_iter,
            random_seed=random_seed,
            verbose=verbose,
        )
