""" General unit tests of the causal-curve package """

import numpy as np

from causal_curve.core import Core


def test_get_params():
    """
    Tests the `get_params` method of the Core base class
    """

    core = Core()
    core.a = 5
    core.b = 10

    observed_results = core.get_params()

    assert observed_results == {"a": 5, "b": 10}


def test_if_verbose_print(capfd):
    """
    Tests the `if_verbose_print` method of the Core base class
    """

    core = Core()
    core.verbose = True

    core.if_verbose_print("This is a test")
    out, err = capfd.readouterr()

    assert out == "This is a test\n"

    core.verbose = False

    core.if_verbose_print("This is a test")
    out, err = capfd.readouterr()

    assert out == ""


def test_rand_seed_wrapper():
    """
    Tests the `rand_seed_wrapper` method of the Core base class
    """

    core = Core()
    core.rand_seed_wrapper(123)

    assert np.random.get_state()[1][0] == 123


def test_calculate_z_score():
    """
    Tests the `calculate_z_score` method of the Core base class
    """

    core = Core()
    assert round(core.calculate_z_score(0.95), 3) == 1.960
    assert round(core.calculate_z_score(0.90), 3) == 1.645


def test_clip_negatives():
    """
    Tests the `clip_negatives` method of the Core base class
    """

    core = Core()
    assert core.clip_negatives(0.5) == 0.5
    assert core.clip_negatives(-1.5) == 0
