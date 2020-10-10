"""Common fixtures for tests using pytest framework"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm


@pytest.fixture(scope="module")
def continuous_dataset_fixture():
    """Returns full_continuous_example_dataset (with a continuous outcome)"""
    return full_continuous_example_dataset()


def full_continuous_example_dataset():
    """Example dataset with a treatment, two covariates, and continuous outcome variable"""

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


@pytest.fixture(scope="module")
def binary_dataset_fixture():
    """Returns full_binary_example_dataset (with a binary outcome)"""
    return full_binary_example_dataset()


def full_binary_example_dataset():
    """Example dataset with a treatment, two covariates, and binary outcome variable"""

    np.random.seed(500)
    treatment = np.linspace(
        start=0,
        stop=100,
        num=100,
    )
    x_1 = norm.rvs(size=100, loc = 50, scale = 5)
    outcome = [
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,1,0,0,
        0,0,0,0,0,0,0,0,1,1,
        0,0,0,0,0,0,0,0,0,1,
        1,1,1,1,1,0,1,1,1,1,
        1,0,1,1,1,1,1,0,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
    ]

    fixture = pd.DataFrame({'treatment':treatment, 'x1': x_1, 'outcome': outcome})
    fixture.reset_index(drop=True, inplace=True)

    return fixture


@pytest.fixture(scope="module")
def mediation_fixture():
    """Returns mediation_dataset"""
    return mediation_dataset()


def mediation_dataset():
    """Example dataset to test / demonstrate mediation with a treatment,
    a mediator, and an outcome variable"""

    np.random.seed(500)

    n_obs = 500

    treatment = np.random.normal(loc=50.0, scale=10.0, size=n_obs)
    mediator = np.random.normal(loc=70.0 + treatment, scale=8.0, size=n_obs)
    outcome = np.random.normal(loc=(treatment + mediator - 50), scale=10.0, size=n_obs)

    fixture = pd.DataFrame(
        {"treatment": treatment, "mediator": mediator, "outcome": outcome}
    )

    fixture.reset_index(drop=True, inplace=True)

    return fixture
