"""
Defines the Targetted Maximum likelihood Estimation (TMLE) regressor model class
"""
from pprint import pprint

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_numeric_dtype
from scipy.interpolate import interp1d
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from statsmodels.genmod.generalized_linear_model import GLM

from causal_curve import TMLE_core
from causal_curve.utils import rand_seed_wrapper


class TMLE_regressor(Core):
    """
    A TMLE tool that handles continuous outcomes. Inherits the TMLE_core
    base class. See that base class code its docstring for more details.
    """
