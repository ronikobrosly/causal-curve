from causal_curve import GPS
import numpy as np
import pandas as pd
from scipy.stats import expon


np.random.seed(333)
n = 5000
x_1 = expon.rvs(size=n, scale = 1)
x_2 = expon.rvs(size=n, scale = 1)
treatment = expon.rvs(size=n, scale = (1/(x_1 + x_2)))

gps = ((x_1 + x_2) * np.exp(-(x_1 + x_2) * treatment))
outcome = treatment + gps + np.random.normal(size = n, scale = 1)

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
).sort_values('Treatment', ascending = True)


gps = GPS()
gps.fit(T = df['Treatment'], X = df[['X_1', 'X_2']], y = df['Outcome'])
gps_results = gps.calculate_CDRC(0.95)



####### EXPERIMENTING WITH PREDICTING GIVEN NEW COVARIATE AND TREATMENT DATA


df['Treatment'].describe()

# Using the `predict` method will work like this

temp_treatment_value = 4
temp_gps_value = gps.gps_function(temp_treatment_value).mean()

gps.gam_results.predict(np.array([temp_treatment_value, temp_gps_value]).reshape(1,-1))


# Using the `predict_interval` method will work like this


gps.gam_results.prediction_intervals(np.array([temp_treatment_value, temp_gps_value]).reshape(1,-1), width = 0.95)
