"""causal_curve module"""

import warnings

from statsmodels.genmod.generalized_linear_model import DomainWarning

from causal_curve.cdrc import CDRC as CDRC


# Suppress statsmodel warning for gamma family GLM
warnings.filterwarnings("ignore", category=DomainWarning)
warnings.filterwarnings("ignore", category=UserWarning)
