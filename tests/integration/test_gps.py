""" Integration tests of the gps.py module """

import pandas as pd

from causal_curve import GPS_Regressor, GPS_Classifier


def test_full_continuous_gps_flow(continuous_dataset_fixture):
    """
    Tests the full flow of the GPS tool when used with a continuous outcome
    """

    gps = GPS_Regressor(
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
        T=continuous_dataset_fixture["treatment"],
        X=continuous_dataset_fixture[["x1", "x2"]],
        y=continuous_dataset_fixture["outcome"],
    )
    gps_results = gps.calculate_CDRC(0.95)

    assert isinstance(gps_results, pd.DataFrame)
    check = gps_results.columns == [
        "Treatment",
        "Causal_Dose_Response",
        "Lower_CI",
        "Upper_CI",
    ]
    assert check.all()


def test_binary_gps_flow(binary_dataset_fixture):
    """
    Tests the full flow of the GPS tool when used with a binary outcome
    """

    gps = GPS_Classifier(
        gps_family="normal",
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
        T=binary_dataset_fixture["treatment"],
        X=binary_dataset_fixture["x1"],
        y=binary_dataset_fixture["outcome"],
    )
    gps_results = gps.calculate_CDRC(0.95)

    assert isinstance(gps_results, pd.DataFrame)
    check = gps_results.columns == [
        "Treatment",
        "Causal_Odds_Ratio",
        "Lower_CI",
        "Upper_CI",
    ]
    assert check.all()
