""" Integration tests of the cdrc.py module """

import pandas as pd

from causal_curve import CDRC
from tests.test_helpers import assert_df_equal


def test_full_cdrc_flow(dataset_fixture):
    """
    Tests the full flow of the CDRC tool
    """

    cdrc = CDRC(
        treatment_grid_num = 10,
        lower_grid_constraint=0.0,
        upper_grid_constraint=1.0,
        spline_order=3,
        n_splines=10,
        max_iter=100,
        random_seed=100,
        verbose=False
    )
    cdrc.fit(
        T = dataset_fixture['treatment'],
        X = dataset_fixture['x1'],
        y = dataset_fixture['outcome']
    )
    cdrc_results = cdrc.calculate_CDRC(0.95)

    expected_df = pd.DataFrame(
        {
            'Treatment': [22.104,37.955,42.553,45.957,48.494,51.139,53.994,57.607,61.19,80.168],
            'CDRC': [119.231,138.978,143.009,145.79,148.023,150.729,153.833,157.768,161.83,187.305],
            'Lower_CI': [108.507,136.278,140.41,143.233,145.545,148.375,151.452,155.517,159.391,174.926],
            'Upper_CI': [129.955,141.678,145.609,148.348,150.5,153.083,156.213,160.018,164.27,199.683],
        }
    )

    assert_df_equal(cdrc_results, expected_df)
