from causal_curve.gps import GPS
import numpy as np
import pandas as pd
from scipy.stats import expon
from scipy.special import logit


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



gps.predict(np.array([4.1, 4.3, 4.7]))
gps.predict_interval(np.array([4.1, 4.3, 4.7]))

gps.predict(np.array([5]))
gps.predict_interval(np.array([5]))








####### EXPERIMENTING WITH PREDICTING GIVEN NEW COVARIATE AND TREATMENT DATA

### Using the `predict` method will work like this
temp_treatment_value = 4
temp_gps_value = gps.gps_function(temp_treatment_value).mean()
gps.gam_results.predict(np.array([temp_treatment_value, temp_gps_value]).reshape(1,-1))





# Using the `predict_interval` method will work like this

gps.gam_results.prediction_intervals(np.array([temp_treatment_value, temp_gps_value]).reshape(1,-1), width = 0.95)





##########################################################
##########################################################


from causal_curve.gps import GPS
import numpy as np
import pandas as pd
from scipy.stats import expon


df = pd.read_csv("~/Desktop/advertising.csv")


gps = GPS()
gps.fit(T = df['treatment'], X = df[['age', 'income']], y = df['outcome'])
gps_results = gps.calculate_CDRC(0.95)



gps.predict_log_odds(np.array([65]))



# Using the `predict_log_odds` method

temp_treatment_value = 65
temp_gps_value = gps.gps_function(temp_treatment_value).mean()

logit(gps.gam_results.predict_proba(np.array([temp_treatment_value, temp_gps_value]).reshape(1,-1)))
