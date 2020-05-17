""" Integration tests of the gps.py module """

import pandas as pd

from causal_curve import GPS
from tests.test_helpers import assert_df_equal


def test_full_gps_flow(dataset_fixture):
    """
    Tests the full flow of the GPS tool
    """

    gps = GPS(
        treatment_grid_num=10,
        lower_grid_constraint=0.0,
        upper_grid_constraint=1.0,
        spline_order=3,
        n_splines=10,
        max_iter=100,
        random_seed=100,
        verbose=False,
    )
    gps.fit(
        T=dataset_fixture["treatment"],
        X=dataset_fixture[["x1", "x2"]],
        y=dataset_fixture["outcome"],
    )
    gps_results = gps.calculate_CDRC(0.95)

    assert isinstance(gps_results, pd.DataFrame)
    check = gps_results.columns == ["Treatment", "CDRC", "Lower_CI", "Upper_CI"]
    assert check.all()
