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
