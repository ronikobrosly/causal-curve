"""Common fixtures for tests using pytest framework"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def dataset_fixture():
    """Returns full_example_dataset"""
    return full_example_dataset()


def full_example_dataset():
    """Example dataset with a treatment, two covariates, and outcome variable"""

    np.random.seed(500)

    n_obs = 500

    treatment = np.random.normal(loc=50.0, scale=10.0, size=n_obs)
    x_1 = np.random.normal(loc=50.0, scale=10.0, size=n_obs)
    x_2 = np.random.normal(loc=0, scale=10.0, size=n_obs)
    outcome = treatment + x_1 + x_2 + np.random.normal(loc=50.0, scale=3.0, size=n_obs)

    fixture = pd.DataFrame(
        {"treatment": treatment, "x1": x_1, "x2": x_2, "outcome": outcome}
    )

    fixture.reset_index(drop=True, inplace=True)

    return fixture
