import warnings

from statsmodels.genmod.generalized_linear_model import DomainWarning


# Suppress statsmodel warning for gamma family GLM
# See: https://github.com/statsmodels/statsmodels/issues/3316
warnings.filterwarnings("ignore", category = DomainWarning)
warnings.filterwarnings("ignore", category = UserWarning)
