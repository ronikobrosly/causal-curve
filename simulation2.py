from os.path import expanduser

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
import numpy as np
import pandas as pd
from scipy.stats import expon
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess

from causal_curve.cdrc import CDRC



# SIMULATE THE DATA

np.random.seed(333)
n = 5000
x_1 = expon.rvs(size=n, scale = 1) # covariate 1
x_2 = expon.rvs(size=n, scale = 1) # covariate 2
treatment = expon.rvs(size=n, scale = (1/(x_1 + x_2))) # treatment variable

gps = ((x_1 + x_2) * np.exp(-(x_1 + x_2) * treatment))
outcome = treatment + gps + np.random.normal(size = n, scale = 1) # outcome is treatment + gps + some noise

truth_func = lambda treatment: (treatment + (2/(1 + treatment)**3))
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






# Try the CDRC
cdrc = CDRC()
cdrc.fit(T = df['Treatment'], X = df[['X_1', 'X_2']], y = df['Outcome'])
cdrc_results = cdrc.calculate_CDRC(0.95)
treatment_grid_values = cdrc.grid_values



center = cdrc.gam_results.predict(df[['Treatment', 'GPS']])
lower = cdrc.gam_results.confidence_intervals(df[['Treatment', 'GPS']])[:,0]
upper = cdrc.gam_results.confidence_intervals(df[['Treatment', 'GPS']])[:,1]


plt.clf()
plt.plot(df['Treatment'], center)
plt.plot(df['Treatment'], lower, color='b', ls='--')
plt.plot(df['Treatment'], upper, color='b', ls='--')
plt.show()
