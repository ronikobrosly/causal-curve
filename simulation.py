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





# CREATE LOESS CURVE

loess_data = lowess(endog = np.asarray(df['Outcome']), exog = np.asarray(df['Treatment']), is_sorted = False, return_sorted = True)

# unpack the lowess smoothed points to their values
lowess_x = loess_data[:,0]
lowess_y = loess_data[:,1]

f = interp1d(lowess_x, lowess_y, bounds_error=False)

lowess_x = treatment_grid_values
lowess_y = f(lowess_x)






# Compare the loess and CDRC
comparison = pd.DataFrame(
    {
        'treatment': treatment_grid_values,
        'loess': lowess_y,
        'CDRC': cdrc_results['CDRC'],
        'truth': vfunc(treatment_grid_values)
    }
)



plt.clf()
plt.scatter(treatment, outcome, s=5, color = "silver")
plt.plot('treatment', 'loess', data=comparison, lw=1, color='darkorange')
plt.plot('treatment', 'CDRC', data=comparison, lw=1, color='blue')
plt.plot('treatment', 'truth', data=comparison, lw=1, linestyle='dashed', color='black')
plt.show()





# Calculate ratio in bias
CDRC_abs_bias = (comparison['CDRC'] - comparison['truth']).abs().sum()
loess_abs_bias = (comparison['loess'] - comparison['truth']).abs().sum()

print(f"CDRC absolute bias is {round(CDRC_abs_bias, 3)}")
print(f"loess absolute bias is {round(loess_abs_bias, 3)}")
print(f"CDRC has {round(CDRC_abs_bias/loess_abs_bias, 3)} times the bias of loess")
