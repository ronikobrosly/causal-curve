.. _TMLE_example:

=====================================================
TMLE Method for Causal Dose Response Curve Estimation
=====================================================

Targeted Maximum Likelihood Estimation method
---------------------------------------------


In this example, we use this package's Targeted Maximum Likelihood Estimation (TMLE)
tool to estimate the marginal causal curve of some continuous treatment on a continuous outcome,
accounting for some mild confounding effects.

Compared with the package's GPS method, this TMLE method is double robust against model
misspecification, incorporates more powerful machine learning techniques internally (gradient boosting),
produces significantly smaller confidence intervals, however it is not computationally efficient
and will take longer to run.

The quality of the estimate it produces is highly dependent on the user's choice
of the `treatment_grid_bins` parameter. If the bins are too small, you might violate the
'positivity' assumption, but if the buckets are too large, your final estimate of the CDRC will
not be smooth. We recommend ensure there are at least 100 treatment observations within
each of your buckets. Exploring the treatment distribution and quantiles is recommended.


>>> from causal_curve import TMLE
>>> tmle = TMLE(
    treatment_grid_bins = [1.0, 1.5, 2.0, ...],
    random_seed=111,
    verbose=True,
)

>>> tmle.fit(T = df['Treatment'], X = df[['X_1', 'X_2']], y = df['Outcome'])
>>> gps_results = tmle.calculate_CDRC(0.99)


References
----------

Kennedy EH, Ma Z, McHugh MD, Small DS. Nonparametric methods for doubly robust estimation
of continuous treatment effects. Journal of the Royal Statistical Society, Series B. 79(4), 2017, pp.1229-1245.

van der Laan MJ and Rubin D. Targeted maximum likelihood learning. In: ​U.C. Berkeley Division of
Biostatistics Working Paper Series, 2006.

van der Laan MJ and Gruber S. Collaborative double robust penalized targeted
maximum likelihood estimation. In: The International Journal of Biostatistics 6(1), 2010.
