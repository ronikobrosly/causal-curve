import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 1000)



########### READ IN PRE-MADE SIM DATA ###########

np.random.seed(222)

treatment_grid_num = 100
lower_grid_constraint = 0
upper_grid_constraint = 1

n = 500

df = pd.read_csv("my_data2.csv")[['x1', 'x2', 'a', 'y']]
df.columns = ['x1', 'x2', 'treatment', 'outcome']




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



########### TRY NEW causal-curve FUNCTIONS ###########
