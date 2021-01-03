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
