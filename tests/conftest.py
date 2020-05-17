"""Common fixtures for tests using pytest framework"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def dataset_fixture():
    return full_example_dataset()


def full_example_dataset():

    np.random.seed(500)

    n = 500

    treatment = np.random.normal(loc=50.0, scale=10.0, size=n)
    x1 = np.random.normal(loc=50.0, scale=10.0, size=n)
    outcome = treatment + x1 + np.random.normal(loc=50.0, scale=3.0, size=n)

    fixture = pd.DataFrame({"treatment": treatment, "x1": x1, "outcome": outcome})

    fixture.reset_index(drop=True, inplace=True)

    return fixture
