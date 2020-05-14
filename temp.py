from causal_curve import TMLE


##################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import expon


np.random.seed(123)
n = 50000
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
        'x1': x_1,
        'x2': x_2,
        't': treatment,
        'gps': gps,
        'y': outcome,
        'true_y': true_outcome
    }
)

df.sort_values('t', ascending = True, inplace = True)


# Restrict this to less than 3
df = df[df['t'] < 3]
df.reset_index(drop = True, inplace = True)


##################################################


# a = [0, 0.03, 0.05, 0.25, 0.5, 1.0, 2.0]
#
# for index, _ in enumerate(a):
#
#     if (index == 0) or (index == len(a) - 1):
#         continue
#     else:
#         print(f"starting loop {index} of {len(a) - 2}")
#         print(f"{a[index - 1]} - {a[index]}, {a[index]} - {a[index + 1]}")
#
#





tmle = TMLE(
    treatment_grid_bins = [0, 0.03, 0.05, 0.25, 0.5, 1.0, 2.0],
    n_estimators=100,
    learning_rate = 0.1,
    max_depth = 5,
    gamma = 1.0,
    random_seed=111,
    verbose=True,
)


tmle.fit(df['t'], df[['x1','x2']], df['y'])
