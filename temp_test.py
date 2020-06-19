import numpy as np
import pandas as pd

from causal_curve import Mediation




#######################
# Read in data (high mediation data)
#######################

df = pd.read_csv("~/Desktop/R_data.csv")



#######################
# Instantiate mediation class
#######################

med = Mediation(verbose = True, bootstrap_draws=100, bootstrap_replicates=100, n_splines = 5)

med.fit(df['x'], df['w'], df['y'])

med.calculate_mediation()




#######################
# Simulate low mediation data
#######################

df2 = pd.DataFrame(
    {
        'x': np.random.normal(100, 10, 100),
        'w': np.random.normal(200, 10, 100),
        'y': np.random.normal(300, 10, 100)
    }
)


#######################
# Instantiate mediation class
#######################

med = Mediation(verbose = True, bootstrap_draws=500, bootstrap_replicates=50)

med.fit(df2['x'], df2['w'], df2['y'])

med.calculate_mediation()
