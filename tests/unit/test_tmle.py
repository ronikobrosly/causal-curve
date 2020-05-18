""" Unit tests of the tmle.py module """

from causal_curve import TMLE


def test_tmle_fit(dataset_fixture):
    """
    Tests the fit method GPS tool
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

    assert tmle.n_obs == 72
    assert len(tmle.psi_list) == 5
    assert len(tmle.std_error_ic_list) == 5
