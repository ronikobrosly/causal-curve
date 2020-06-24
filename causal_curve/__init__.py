"""causal_curve module"""

import warnings

from statsmodels.genmod.generalized_linear_model import DomainWarning

from causal_curve.gps import GPS
from causal_curve.tmle import TMLE
from causal_curve.mediation import Mediation


# Suppress statsmodel warning for gamma family GLM
warnings.filterwarnings("ignore", category=DomainWarning)
warnings.filterwarnings("ignore", category=UserWarning)
