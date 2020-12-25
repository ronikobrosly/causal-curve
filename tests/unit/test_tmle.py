""" Unit tests of the tmle.py module """

import pytest

from causal_curve import TMLE


def test_tmle_fit(continuous_dataset_fixture):
    """
    Tests the fit method GPS tool
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

    assert tmle.n_obs == 72
    assert len(tmle.psi_list) == 5
    assert len(tmle.std_error_ic_list) == 5


@pytest.mark.parametrize(
    (
        "treatment_grid_bins",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "random_seed",
        "verbose",
    ),
    [
        ([0, 1, "2", 3, 4], 100, 0.1, 5, None, False),
        (5, 100, 0.1, 5, None, False),
        ("5", 100, 0.1, 5, None, False),
        ([22.1, 30, 40, 50, 60, 70, 80.1], "100", 0.1, 5, None, False),
        ([22.1, 30, 40, 50, 60, 70, 80.1], 1, 0.1, 5, None, False),
        ([22.1, 30, 40, 50, 60, 70, 80.1], 100, "0.1", 5, None, False),
        ([22.1, 30, 40, 50, 60, 70, 80.1], 100, 1000000, 5, None, False),
        ([22.1, 30, 40, 50, 60, 70, 80.1], 100, 0.1, "5", None, False),
        ([22.1, 30, 40, 50, 60, 70, 80.1], 100, 0.1, -5, None, False),
        ([22.1, 30, 40, 50, 60, 70, 80.1], 100, 0.1, 5, "None", False),
        ([22.1, 30, 40, 50, 60, 70, 80.1], 100, 0.1, 5, -10, False),
        ([22.1, 30, 40, 50, 60, 70, 80.1], 100, 0.1, 5, None, "False"),
        ({"a": 5, "b": 6}, 100, 0.1, 5, None, False),
        (["a", "b", "c"], 100, 0.1, 5, None, False),
        ([1.0], 100, 0.1, 5, None, False),
        ([22.1, 30, 40, 50, 60, 70, 80.1], 100.0, 0.1, 5, None, False),
        ([22.1, 30, 40, 50, 60, 70, 80.1], 1.0, 0.1, 5, None, False),
        ([22.1, 30, 40, 50, 60, 70, 80.1], 10000000, 0.1, 5, None, False),
        ([22.1, 30, 40, 50, 60, 70, 80.1], 100, "hehe", 5, None, False),
        ([22.1, 30, 40, 50, 60, 70, 80.1], 100, -0.1, 5, None, False),
        ([22.1, 30, 40, 50, 60, 70, 80.1], 100, 10000000, 5, None, False),
        ([22.1, 30, 40, 50, 60, 70, 80.1], 100, 0.1, "hehe", None, False),
        ([22.1, 30, 40, 50, 60, 70, 80.1], 100, 0.1, 5.0, None, False),
        ([22.1, 30, 40, 50, 60, 70, 80.1], 100, 0.1, -5, None, False),
        ([22.1, 30, 40, 50, 60, 70, 80.1], 100, 0.1, 5, "hehe", False),
        ([22.1, 30, 40, 50, 60, 70, 80.1], 100, 0.1, 5, -10, False),
        ([22.1, 30, 40, 50, 60, 70, 80.1], 100, 0.1, 5, None, "thirty two"),
    ],
)
def test_bad_tmle_instantiation(
    treatment_grid_bins,
    n_estimators,
    learning_rate,
    max_depth,
    random_seed,
    verbose,
):
    with pytest.raises(Exception) as bad:
        TMLE(
            treatment_grid_bins=treatment_grid_bins,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_seed=random_seed,
            verbose=verbose,
        )
