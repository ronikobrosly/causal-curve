"""Misc helper functions for tests"""

from pandas.testing import assert_frame_equal


def assert_df_equal(observed_frame, expected_frame, check_less_precise):
    """Assert that two pandas dataframes are equal, ignoring ordering of columns."""
    assert_frame_equal(
        observed_frame.sort_index(axis=1),
        expected_frame.sort_index(axis=1),
        check_names=True,
        check_less_precise=check_less_precise,
    )
