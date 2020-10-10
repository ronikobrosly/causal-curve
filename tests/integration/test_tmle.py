""" Integration tests of the tmle.py module """

import pandas as pd

from causal_curve import TMLE


def test_full_tmle_flow(continuous_dataset_fixture):
    """
    Tests the full flow of the TMLE tool
    """

    tmle = TMLE(
        treatment_grid_bins=[22.1, 30, 40, 50, 60, 70, 80.1],
        random_seed=100,
        verbose=True,
    )
    tmle.fit(
        T=continuous_dataset_fixture["treatment"],
        X=continuous_dataset_fixture[["x1", "x2"]],
        y=continuous_dataset_fixture["outcome"],
    )
    tmle_results = tmle.calculate_CDRC(0.95)

    assert isinstance(tmle_results, pd.DataFrame)
    check = tmle_results.columns == ["Treatment", "Causal_Dose_Response", "Lower_CI", "Upper_CI"]
    assert check.all()
