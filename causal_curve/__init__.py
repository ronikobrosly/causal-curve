"""causal_curve module"""

import warnings

from statsmodels.genmod.generalized_linear_model import DomainWarning

from causal_curve.gps_classifier import GPS_Classifier
from causal_curve.gps_regressor import GPS_Regressor

from causal_curve.tmle_regressor import TMLE_Regressor
from causal_curve.mediation import Mediation


# Suppress statsmodel warning for gamma family GLM
warnings.filterwarnings("ignore", category=DomainWarning)
warnings.filterwarnings("ignore", category=UserWarning)
