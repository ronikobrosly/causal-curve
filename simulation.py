from os.path import expanduser

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
import numpy as np
import pandas as pd
from scipy.stats import expon

from causal_curve.cdrc import CDRC



np.random.seed(111)
n = 1000
x_1 = expon.rvs(size=n, scale = 1) # covariate 1
x_2 = expon.rvs(size=n, scale = 1) # covariate 2
treatment = expon.rvs(size=n, scale = (1/(x_1 + x_2))) # treatment variable

gps = ((x_1 + x_2) * np.exp(-(x_1 + x_2) * treatment))
outcome = treatment + gps + np.random.normal(size = n) # outcome is treatment + gps + some noise

truth_func = lambda treatment: (treatment + (2/np.power((1 + treatment), 3)))
vfunc = np.vectorize(truth_func)
true_outcome = vfunc(treatment)




df = pd.DataFrame(
    {
        'X_1': x_1,
        'X_2': x_2,
        'Treatment': treatment,
        'GPS': gps,
        'Outcome': outcome,
        'True_outcome': true_outcome
    }
)

df.sort_values('Treatment', ascending = True, inplace = True)

df.plot('Treatment', 'True_outcome')
