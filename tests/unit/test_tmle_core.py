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


@pytest.mark.parametrize(
    (
        "treatment_grid_num",
        "lower_grid_constraint",
        "upper_grid_constraint",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "bandwidth",
        "random_seed",
        "verbose",
    ),
    [
        # treatment_grid_num
        (100.0, 0.01, 0.99, 200, 0.01, 3, 0.5, None, False),
        ("100.0", 0.01, 0.99, 200, 0.01, 3, 0.5, None, False),
        (2, 0.01, 0.99, 200, 0.01, 3, 0.5, None, False),
        (500000, 0.01, 0.99, 200, 0.01, 3, 0.5, None, False),

        # lower_grid_constraint
        (100, {0.01: 'a'}, 0.99, 200, 0.01, 3, 0.5, None, False),
        (100, -0.01, 0.99, 200, 0.01, 3, 0.5, None, False),
        (100, 6.05, 0.99, 200, 0.01, 3, 0.5, None, False),

        # upper_grid_constraint
        (100, 0.01, [1,2,3], 200, 0.01, 3, 0.5, None, False),
        (100, 0.01, -0.05, 200, 0.01, 3, 0.5, None, False),
        (100, 0.01, 5.99, 200, 0.01, 3, 0.5, None, False),
        (100, 0.9, 0.2, 200, 0.01, 3, 0.5, None, False),

        # n_estimators
        (100, 0.01, 0.99, "3.0", 0.01, 3, 0.5, None, False),
        (100, 0.01, 0.99, -5, 0.01, 3, 0.5, None, False),
        (100, 0.01, 0.99, 10000000, 0.01, 3, 0.5, None, False),

        # learning_rate
        (100, 0.01, 0.99, 200, ['a', 'b'], 3, 0.5, None, False),
        (100, 0.01, 0.99, 200, 5000000, 3, 0.5, None, False),

        # max_depth
        (100, 0.01, 0.99, 200, 0.01, "a", 0.5, None, False),
        (100, 0.01, 0.99, 200, 0.01, -6, 0.5, None, False),

        # bandwidth
        (100, 0.01, 0.99, 200, 0.01, 3, "b", None, False),
        (100, 0.01, 0.99, 200, 0.01, 3, -10, None, False),

        # random seed
        (100, 0.01, 0.99, 200, 0.01, 3, 0.5, "b", False),
        (100, 0.01, 0.99, 200, 0.01, 3, 0.5, -10, False),

        # verbose
        (100, 0.01, 0.99, 200, 0.01, 3, 0.5, None, "Verbose")
    ],
)
def test_bad_tmle_instantiation(
    treatment_grid_num,
    lower_grid_constraint,
    upper_grid_constraint,
    n_estimators,
    learning_rate,
    max_depth,
    bandwidth,
    random_seed,
    verbose,
):
    with pytest.raises(Exception) as bad:
        TMLE_Core(
            treatment_grid_num=treatment_grid_num,
            lower_grid_constraint=lower_grid_constraint,
            upper_grid_constraint=upper_grid_constraint,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            bandwidth=bandwidth,
            random_seed=random_seed,
            verbose=verbose,
        )
