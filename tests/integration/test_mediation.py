""" Integration tests of the mediation.py module """

import pandas as pd

from causal_curve import Mediation


def test_full_mediation_flow(mediation_fixture):
    """
    Tests the full flow of the Mediation tool
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

    med_results = med.calculate_mediation(0.95)

    assert isinstance(med_results, pd.DataFrame)
    check = med_results.columns == [
        "Treatment_Value",
        "Proportion_Direct_Effect",
        "Proportion_Indirect_Effect",
    ]
    assert check.all()
