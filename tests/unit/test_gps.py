""" Unit tests of the gps.py module """

from pygam import LinearGAM

from causal_curve import GPS


def test_gps_fit(dataset_fixture):
    """
    Tests the fit method of the GPS tool
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
        X=dataset_fixture["x1"],
        y=dataset_fixture["outcome"],
    )

    assert isinstance(gps.gam_results, LinearGAM)
    assert gps.gps.shape == (500,)
