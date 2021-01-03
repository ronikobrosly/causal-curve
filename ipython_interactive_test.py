import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 1000)



########## READ IN PRE-MADE SIM DATA ###########

np.random.seed(222)

treatment_grid_num = 100
lower_grid_constraint = 0
upper_grid_constraint = 1

n = 500

df = pd.read_csv("~/Desktop/causal-curve/my_data2.csv")[['x1', 'x2', 'a', 'y']]
df.columns = ['x1', 'x2', 'treatment', 'outcome']




from causal_curve import GPS_Regressor

gps_model = GPS_Regressor()

gps_model.fit(T = df['treatment'], X = df[['x1', 'x2']], y = df['outcome'])
gps_model.calculate_CDRC(0.90)
gps_model.point_estimate(np.array([0.5]))
gps_model.point_estimate_interval(np.array([0.5]))


########### READ IN PRE-MADE CLASSIFICATION DATA ###########

import pandas as pd
import numpy as np
np.random.seed(222)

treatment_grid_num = 100
lower_grid_constraint = 0
upper_grid_constraint = 1

n = 717

df = pd.read_csv("~/Desktop/causal-curve/binary_classification.csv")
df.columns = ['outcome', 'x1', 'x2', 'treatment']


import matplotlib.pyplot as plt
from causal_curve import GPS_Classifier

gps_model = GPS_Classifier()

gps_model.fit(T = df['treatment'], X = df[['x1', 'x2']], y = df['outcome'])
cdrc = gps_model.calculate_CDRC(0.95)


plt.clf()
plt.scatter(cdrc['Treatment'], cdrc['Causal_Odds_Ratio'], color='black', marker = 'o')
#plt.scatter(cdrc['Treatment'], cdrc['Lower_CI'], color='red', marker = '.')
#plt.scatter(cdrc['Treatment'], cdrc['Upper_CI'], color='red', marker = '.')
plt.show()





########### CREATE NON-LINEAR CUSTOM SIM DATA ###########

np.random.seed(200)

def generate_data(t, A, sigma, omega, noise=0, n_outliers=0, random_state=0):
	y = A * np.exp(-sigma * t) * np.sin(omega * t)
	rnd = np.random.RandomState(random_state)
	error = noise * rnd.randn(t.size)
	outliers = rnd.randint(0, t.size, n_outliers)
	error[outliers] *= 35
	return y + error


A = 2
sigma = 0.1
omega = 0.1 * 2 * np.pi
x_true = np.array([A, sigma, omega])

noise = 0.1

t_min = 0
t_max = 10

treatment = np.linspace(t_min, t_max, 1000)
outcome = generate_data(treatment, A, sigma, omega, noise=noise, n_outliers=5)
x1 = np.random.uniform(0,10,1000)
x2 = np.random.uniform(0,10,1000) * 3

df = pd.DataFrame(
	{
		'x1': x1,
		'x2': x2,
		'treatment': treatment,
		'outcome': outcome
	}
)

treatment_grid_num = 100
lower_grid_constraint = 0
upper_grid_constraint = 1

n = 1000




import matplotlib.pyplot as plt
from causal_curve import TMLE_Regressor


tmle = TMLE_Regressor()

tmle.fit(T = df['treatment'], X = df[['x1', 'x2']], y = df['outcome'])
cdrc = tmle.calculate_CDRC()

tmle.point_estimate(np.array([5.5]))
tmle.point_estimate_interval(np.array([5.5]))

#
# plt.clf()
# plt.scatter(cdrc['Treatment'], cdrc['Causal_Dose_Response'], color='black', marker = 'o')
# plt.scatter(cdrc['Treatment'], cdrc['Lower_CI'], color='red', marker = '.')
# plt.scatter(cdrc['Treatment'], cdrc['Upper_CI'], color='red', marker = '.')
# plt.show()
