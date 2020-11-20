""" Unit tests of the gps.py module """

import numpy as np
from pygam import LinearGAM
import pytest

from causal_curve import GPS
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
    Tests the fit method of the GPS tool
    """

    gps = GPS(
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
        "max_iter",
        "random_seed",
        "verbose",
    ),
    [
        (546, 10, 0, 1.0, 3, 10, 100, 100, True),
        ("linear", 10, 0, 1.0, 3, 10, 100, 100, True),
        (None, "hehe", 0, 1.0, 3, 10, 100, 100, True),
        (None, 2, 0, 1.0, 3, 10, 100, 100, True),
        (None, 1e6, 0, 1.0, 3, 10, 100, 100, True),
        (None, 10, "hehe", 1.0, 3, 10, 100, 100, True),
        (None, 10, -1, 1.0, 3, 10, 100, 100, True),
        (None, 10, 1.5, 1.0, 3, 10, 100, 100, True),
        (None, 10, 0, "hehe", 3, 10, 100, 100, True),
        (None, 10, 0, 1.5, 3, 10, 100, 100, True),
        (None, 10, 0, -1, 3, 10, 100, 100, True),
        (None, 10, 0, 1, 3, 10, 100, 100, True),
        (None, 10, 0, 1, "splines", 10, 100, 100, True),
        (None, 10, 0, 1, 0, 10, 100, 100, True),
        (None, 10, 0, 1, 200, 10, 100, 100, True),
        (None, 10, 0, 1, 3, 0, 100, 100, True),
        (None, 10, 0, 1, 3, 1e6, 100, 100, True),
        (None, 10, 0, 1, 3, 10, 100, 100, True),
        (None, 10, 0, 1, 3, 10, "many", 100, True),
        (None, 10, 0, 1, 3, 10, 5, 100, True),
        (None, 10, 0, 1, 3, 10, 1e7, 100, True),
        (None, 10, 0, 1, 3, 10, 100, "random", True),
        (None, 10, 0, 1, 3, 10, 100, -1.5, True),
        (None, 10, 0, 1, 3, 10, 100, 111, "True"),
    ],
)
def test_bad_gps_instantiation(
    gps_family,
    treatment_grid_num,
    lower_grid_constraint,
    upper_grid_constraint,
    spline_order,
    n_splines,
    max_iter,
    random_seed,
    verbose,
):
    """
    Tests for exceptions when the GPS class if call with bad inputs.
    """
    with pytest.raises(Exception) as bad:
        GPS(
            gps_family=gps_family,
            treatment_grid_num=treatment_grid_num,
            lower_grid_constraint=lower_grid_constraint,
            upper_grid_constraint=upper_grid_constraint,
            spline_order=spline_order,
            n_splines=n_splines,
            max_iter=max_iter,
            random_seed=random_seed,
            verbose=verbose,
        )


def test_calculate_z_score():
    """
    Tests that that `_calculate_z_score` methods returns expected z-scores
    """
    gps = GPS()
    assert round(gps._calculate_z_score(0.99), 2) == 2.58
    assert round(gps._calculate_z_score(0.95), 2) == 1.96
    assert round(gps._calculate_z_score(0.90), 2) == 1.64
    assert round(gps._calculate_z_score(0.80), 2) == 1.28


def test_predict_method_good(GPS_fitted_model_continuous_fixture):
    """
    Tests the GPS `predict` method using appropriate data (with a continuous outcome)
    """
    observed_result = GPS_fitted_model_continuous_fixture.predict(np.array([50]))
    assert isinstance(observed_result[0][0], float)
    assert len(observed_result[0]) == 1

    observed_result = GPS_fitted_model_continuous_fixture.predict(
        np.array([40, 50, 60])
    )
    assert isinstance(observed_result[0][0], float)
    assert len(observed_result[0]) == 3


def test_predict_method_bad(GPS_fitted_model_binary_fixture):
    """
    Tests the GPS `predict` method using inappropriate data (with a binary outcome)
    """
    with pytest.raises(Exception) as bad:
        observed_result = GPS_fitted_model_binary_fixture.predict(np.array([50]))


def test_predict_interval_method_good(GPS_fitted_model_continuous_fixture):
    """
    Tests the GPS `predict_interval` method using appropriate data (with a continuous outcome)
    """
    observed_result = GPS_fitted_model_continuous_fixture.predict_interval(
        np.array([50])
    )
    assert isinstance(observed_result[0][0], float)
    assert observed_result.shape == (1, 2)

    observed_result = GPS_fitted_model_continuous_fixture.predict_interval(
        np.array([40, 50, 60])
    )
    assert isinstance(observed_result[0][0], float)
    assert observed_result.shape == (3, 2)


def test_predict_interval_method_bad(GPS_fitted_model_binary_fixture):
    """
    Tests the GPS `predict_interval` method using appropriate data (with a continuous outcome)
    """
    with pytest.raises(Exception) as bad:
        observed_result = GPS_fitted_model_binary_fixture.predict_interval(
            np.array([50])
        )


def test_predict_log_odds_method_good(GPS_fitted_model_binary_fixture):
    """
    Tests the GPS `predict_log_odds` method using appropriate data (with a binary outcome)
    """
    observed_result = GPS_fitted_model_binary_fixture.predict_log_odds(np.array([0.5]))
    assert isinstance(observed_result[0][0], float)
    assert len(observed_result[0]) == 1

    observed_result = GPS_fitted_model_binary_fixture.predict_log_odds(
        np.array([0.5, 0.6, 0.7])
    )
    assert isinstance(observed_result[0][0], float)
    assert len(observed_result[0]) == 3


def test_predict_log_odds_method_bad(GPS_fitted_model_continuous_fixture):
    """
    Tests the GPS `predict_log_odds` method using appropriate data (with a continuous outcome)
    """
    with pytest.raises(Exception) as bad:
        observed_result = GPS_fitted_model_continuous_fixture.predict_log_odds(
            np.array([50])
        )
