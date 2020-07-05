""" Unit tests of the Mediation.py module """

import numpy as np
import pytest

from causal_curve import Mediation


def test_mediation_fit(mediation_fixture):
    """
    Tests the fit method of Mediation tool
    """

    med = Mediation(
        treatment_grid_num=10,
        lower_grid_constraint=0.01,
        upper_grid_constraint=0.99,
        bootstrap_draws=100,
        bootstrap_replicates=50,
        spline_order=3,
        n_splines=5,
        lambda_=0.5,
        max_iter=20,
        random_seed=None,
        verbose=True,
    )
    med.fit(
        T=mediation_fixture["treatment"],
        M=mediation_fixture["mediator"],
        y=mediation_fixture["outcome"],
    )

    assert len(med.final_bootstrap_results) == 9


@pytest.mark.parametrize(
    (
        "treatment_grid_num",
        "lower_grid_constraint",
        "upper_grid_constraint",
        "bootstrap_draws",
        "bootstrap_replicates",
        "spline_order",
        "n_splines",
        "lambda_",
        "max_iter",
        "random_seed",
        "verbose",
    ),
    [
        (10.5, 0.01, 0.99, 10, 10, 3, 5, 0.5, 100, None, True),
        (0, 0.01, 0.99, 10, 10, 3, 5, 0.5, 100, None, True),
        (1e6, 0.01, 0.99, 10, 10, 3, 5, 0.5, 100, None, True),
        (10, "hehe", 0.99, 10, 10, 3, 5, 0.5, 100, None, True),
        (10, -1, 0.99, 10, 10, 3, 5, 0.5, 100, None, True),
        (10, 1.5, 0.99, 10, 10, 3, 5, 0.5, 100, None, True),
        (10, 0.1, "hehe", 10, 10, 3, 5, 0.5, 100, None, True),
        (10, 0.1, -1, 10, 10, 3, 5, 0.5, 100, None, True),
        (10, 0.1, 1.5, 10, 10, 3, 5, 0.5, 100, None, True),
        (10, 0.1, 0.9, 10.5, 10, 3, 5, 0.5, 100, None, True),
        (10, 0.1, 0.9, -2, 10, 3, 5, 0.5, 100, None, True),
        (10, 0.1, 0.9, 1e6, 10, 3, 5, 0.5, 100, None, True),
        (10, 0.1, 0.9, 100, "10", 3, 5, 0.5, 100, None, True),
        (10, 0.1, 0.9, 100, -1, 3, 5, 0.5, 100, None, True),
        (10, 0.1, 0.9, 100, 1e6, 3, 5, 0.5, 100, None, True),
        (10, 0.1, 0.9, 100, 200, "3", 5, 0.5, 100, None, True),
        (10, 0.1, 0.9, 100, 200, 1, 5, 0.5, 100, None, True),
        (10, 0.1, 0.9, 100, 200, 1e6, 5, 0.5, 100, None, True),
        (10, 0.1, 0.9, 100, 200, 5, "10", 0.5, 100, None, True),
        (10, 0.1, 0.9, 100, 200, 5, 1, 0.5, 100, None, True),
        (10, 0.1, 0.9, 100, 200, 5, 10, "0.5", 100, None, True),
        (10, 0.1, 0.9, 100, 200, 5, 10, -0.5, 100, None, True),
        (10, 0.1, 0.9, 100, 200, 5, 10, 1e7, 100, None, True),
        (10, 0.1, 0.9, 100, 200, 5, 10, 1, "100", None, True),
        (10, 0.1, 0.9, 100, 200, 5, 10, 1, 1, None, True),
        (10, 0.1, 0.9, 100, 200, 5, 10, 1, 1e8, None, True),
        (10, 0.1, 0.9, 100, 200, 5, 10, 1, 100, "None", True),
        (10, 0.1, 0.9, 100, 200, 5, 10, 1, 100, -5, True),
        (10, 0.1, 0.9, 100, 200, 5, 10, 1, 100, 123, "True"),
    ],
)
def test_bad_mediation_instantiation(
    treatment_grid_num,
    lower_grid_constraint,
    upper_grid_constraint,
    bootstrap_draws,
    bootstrap_replicates,
    spline_order,
    n_splines,
    lambda_,
    max_iter,
    random_seed,
    verbose,
):
    with pytest.raises(Exception) as bad:
        Mediation(
            treatment_grid_num=treatment_grid_num,
            lower_grid_constraint=lower_grid_constraint,
            upper_grid_constraint=upper_grid_constraint,
            bootstrap_draws=bootstrap_draws,
            bootstrap_replicates=bootstrap_replicates,
            spline_order=spline_order,
            n_splines=n_splines,
            lambda_=lambda_,
            max_iter=max_iter,
            random_seed=random_seed,
            verbose=verbose,
        )
