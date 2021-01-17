""" Unit tests of the tmle.py module """

import pytest

from causal_curve.tmle_core import TMLE_Core


def test_tmle_fit(continuous_dataset_fixture):
    """
    Tests the fit method GPS tool
    """

    tmle = TMLE_Core(
        random_seed=100,
        verbose=True,
    )
    tmle.fit(
        T=continuous_dataset_fixture["treatment"],
        X=continuous_dataset_fixture[["x1", "x2"]],
        y=continuous_dataset_fixture["outcome"],
    )

    assert tmle.num_rows == 500
    assert tmle.fully_expanded_t_and_x.shape == (50500, 3)


def test_bad_param_calculate_CDRC_TMLE(TMLE_fitted_model_continuous_fixture):
    """
    Tests the TMLE `calculate_CDRC` when the `ci` param is bad
    """

    with pytest.raises(Exception) as bad:
        observed_result = TMLE_fitted_model_continuous_fixture.calculate_CDRC(
            np.array([50]), ci={"param": 0.95}
        )

    with pytest.raises(Exception) as bad:
        observed_result = TMLE_fitted_model_continuous_fixture.calculate_CDRC(
            np.array([50]), ci=1.05
        )
