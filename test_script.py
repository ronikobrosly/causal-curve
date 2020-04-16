import pandas as pd

from causal_curve.cdrc import CDRC



data = pd.read_csv("~/Desktop/sim_data.csv")[['X1', 'X2', 'T', 'Y']]
T = data['T']
X = data[['X1', 'X2']]
y = data['Y']


# test = CDRC(gps_family = 'gamma', treatment_grid_num = 100, spline_order = 3, n_splines = 20, lambda_ = 0.5, max_iter = 100, verbose = True)
test = CDRC(treatment_grid_num = 100, spline_order = 3, n_splines = 20, lambda_ = 0.5, max_iter = 100, verbose = True)



test.fit(T = T, X = X, y = y)


test.calculate_CDRC(95)
