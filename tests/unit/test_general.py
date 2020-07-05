""" General unit tests of the causal-curve package """

from causal_curve.core import Core


def test_core():
    """
    Tests the `Core` base class
    """

    core = Core()
    core.a = 5
    core.b = 10

    observed_results = core.get_params()

    assert observed_results == {"a": 5, "b": 10}
