""" Integration tests of the tmle.py module """

import pandas as pd

from causal_curve import TMLE
from tests.test_helpers import assert_df_equal


def test_full_tmle_flow(dataset_fixture):
    """
    Tests the full flow of the TMLE tool
    """

    tmle = TMLE(
        treatment_grid_bins=[22.1, 30, 40, 50, 60, 70, 80.1],
        random_seed=100,
        verbose=False,
    )
    tmle.fit(
        T=dataset_fixture["treatment"],
        X=dataset_fixture[["x1", "x2"]],
        y=dataset_fixture["outcome"],
    )
    tmle_results = tmle.calculate_CDRC(0.95)

    assert isinstance(tmle_results, pd.DataFrame)
    check = tmle_results.columns == ["Treatment", "CDRC", "Lower_CI", "Upper_CI"]
    assert check.all()
