""" Unit tests for the GPS_Core class """

import numpy as np
from pygam import LinearGAM
import pytest

from causal_curve.gps_core import GPS_Core
from tests.conftest import full_continuous_example_dataset


def test_predict_log_odds_method_good(GPS_fitted_model_binary_fixture):
    """
    Tests the GPS `estimate_log_odds` method using appropriate data (with a binary outcome)
    """
    observed_result = GPS_fitted_model_binary_fixture.estimate_log_odds(np.array([0.5]))
    assert isinstance(observed_result[0][0], float)
    assert len(observed_result[0]) == 1

    observed_result = GPS_fitted_model_binary_fixture.estimate_log_odds(
        np.array([0.5, 0.6, 0.7])
    )
    assert isinstance(observed_result[0][0], float)
    assert len(observed_result[0]) == 3


def test_predict_log_odds_method_bad(GPS_fitted_model_continuous_fixture):
    """
    Tests the GPS `estimate_log_odds` method using appropriate data (with a continuous outcome)
    """
    with pytest.raises(Exception) as bad:
        observed_result = GPS_fitted_model_continuous_fixture.estimate_log_odds(
            np.array([50])
        )
