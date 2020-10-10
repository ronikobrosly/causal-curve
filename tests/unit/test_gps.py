""" Unit tests of the gps.py module """

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
        T=df_fixture()["treatment"], X=df_fixture()["x1"], y=df_fixture()["outcome"],
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
