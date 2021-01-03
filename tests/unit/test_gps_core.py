""" Unit tests for the GPS_Core class """

import numpy as np
from pygam import LinearGAM
import pytest

from causal_curve.gps_core import GPS_Core
from tests.conftest import full_continuous_example_dataset


@pytest.mark.parametrize(
    ("df_fixture", "family"),
    [
        (full_continuous_example_dataset, "normal"),
        (full_continuous_example_dataset, "lognormal"),
        (full_continuous_example_dataset, "gamma"),
        (full_continuous_example_dataset, None),
    ],
)
def test_gps_fit(df_fixture, family):
    """
    Tests the fit method of the GPS_Core tool
    """

    gps = GPS_Core(
        gps_family=family,
        treatment_grid_num=10,
        lower_grid_constraint=0.0,
        upper_grid_constraint=1.0,
        spline_order=3,
        n_splines=10,
        max_iter=100,
        random_seed=100,
        verbose=True,
    )
    gps.fit(
        T=df_fixture()["treatment"],
        X=df_fixture()["x1"],
        y=df_fixture()["outcome"],
    )

    assert isinstance(gps.gam_results, LinearGAM)
    assert gps.gps.shape == (500,)


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
        GPS_Core(
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


def test_bad_param_calculate_CDRC(GPS_fitted_model_continuous_fixture):
    """
    Tests the GPS `calculate_CDRC` when the `ci` param is bad
    """

    with pytest.raises(Exception) as bad:
        observed_result = GPS_fitted_model_continuous_fixture.calculate_CDRC(
            np.array([50]), ci={"param": 0.95}
        )

    with pytest.raises(Exception) as bad:
        observed_result = GPS_fitted_model_continuous_fixture.calculate_CDRC(
            np.array([50]), ci=1.05
        )
