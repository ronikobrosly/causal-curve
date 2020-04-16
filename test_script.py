from os.path import expanduser

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
import numpy as np
import pandas as pd

from causal_curve.cdrc import CDRC


plt.figure(figsize=(11, 7))




data = pd.read_csv("~/Desktop/sim_data.csv")[['X1', 'X2', 'T', 'Y']]
T = data['T']
X = data[['X1', 'X2']]
y = data['Y']


#test = CDRC(treatment_grid_num = 100, spline_order = 3, n_splines = 20, lambda_ = 0.5, max_iter = 100, verbose = False)
test = CDRC(gps_family = 'normal', treatment_grid_num = 100, spline_order = 3, n_splines = 20, lambda_ = 0.5, max_iter = 100, verbose = False)


test.fit(T = T, X = X, y = y)

results = test.calculate_CDRC(95)




# Plot

plt.clf()
def plot_mean_and_CI(treatment, mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(treatment, lb, ub,
                     color=color_shading, alpha=0.3)
    # plot the mean on top
    plt.plot(treatment, mean, color_mean)

# generate 3 sets of random means and confidence intervals to plot
treat0 = results['Treatment']
mean0 = results['CDRC']
lb0 = results['Lower_CI']
ub0 = results['Upper_CI']


ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Ensure that the axis ticks only show up on the bottom and left of the plot.
# Ticks on the right and top of the plot are generally unnecessary chartjunk.
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()



# plot the data
fig = plt.figure(1, figsize=(7, 2.5))
plot_mean_and_CI(treat0, mean0, lb0, ub0, color_mean='b', color_shading='b')


plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xlabel('Treatment')
plt.ylabel('Outcome')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Causal Dose-Response Curve (with 95% confidence interval)')
plt.tight_layout()

plt.savefig(expanduser('~/Desktop/CDRC.png'), bbox_inches='tight', dpi=300)
plt.clf()
