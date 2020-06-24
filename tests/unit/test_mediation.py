""" Unit tests of the Mediation.py module """

import numpy as np

from causal_curve import Mediation


def test_mediation_fit(mediation_fixture):
    """
    Tests the fit method of Mediation tool
    """

    med = Mediation(
        treatment_grid_num=10,
        lower_grid_constraint=0.01,
        upper_grid_constraint=0.99,
        bootstrap_draws=10,
        bootstrap_replicates=10,
        spline_order=3,
        n_splines=5,
        lambda_=0.5,
        max_iter=100,
        random_seed=None,
        verbose=False,
    )
    med.fit(
        T=mediation_fixture["treatment"],
        M=mediation_fixture["mediator"],
        y=mediation_fixture["outcome"],
    )

    assert len(med.final_bootstrap_results) == 9
