.. _TMLE_Regressor:

================================================================
TMLE_Regressor Tool (continuous treatments, continuous outcomes)
================================================================


In this example, we use this package's Targeted Maximum Likelihood Estimation (TMLE)
tool to estimate the marginal causal curve of some continuous treatment on a continuous outcome,
accounting for some mild confounding effects.

The TMLE algorithm is doubly robust, meaning that as long as one of the two models contained
with the tool (the ``g`` or ``q`` models) performs well, then the overall tool will correctly
estimate the causal curve.

Compared with the package's GPS methods incorporates more powerful machine learning techniques internally (gradient boosting)
and produces significantly smaller confidence intervals. However it is less computationally efficient
and will take longer to run. In addition, **the treatment values provided should
be roughly normally-distributed**, otherwise you may encounter internal math errors.


>>> df.head(5) # a pandas dataframe with your data
           X_1       X_2  Treatment    Outcome
0     0.596685  0.162688   0.000039     12.3
1     1.014187  0.916101   0.000197     14.9
2     0.932859  1.328576   0.000223     19.01
3     1.140052  0.555203   0.000339     22.3
4     1.613471  0.340886   0.000438     24.98


>>> from causal_curve import TMLE_Regressor
>>> tmle = TMLE_Regressor(
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
