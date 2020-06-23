.. _Mediation_example:

==========================================================
Test for Mediation for continuous treatments and mediators
==========================================================

Mediation test
--------------


It trying to explore the causal relationships between various elements, it's common to use
your domain knowledge to sketch out your initial hypothesis of the connections. See the following
causal DAG:

.. image:: ../imgs/cdrc/CDRC.png

At some point though, it's helpful to validate these causal connections with empirical tests.
This tool provides a test that can estimate the amount of mediation that occurs between
a treatment, a purported mediator, and an outcome. In keeping with the causal curve theme,
this tool uses a test developed my Imai et al. when handling a continuous treatment and
mediator.







In this example we use simulated data originally developed by Hirano and Imbens but adapted by others
(see references). The advantage of this simulated data is it allows us
to compare the estimate we produce against the true, analytically-derived causal curve.

Let :math:`t_i` be the treatment for the i-th unit, let :math:`x_1` and :math:`x_2` be the
confounding covariates, and let :math:`y_i` be the outcome measure. We assume that the covariates
and treatment are exponentially-distributed, and the treatment variable is associated with the
covariates in the following way:

>>> import numpy as np
>>> import pandas as pd
>>> from scipy.stats import expon

>>> np.random.seed(333)
>>> n = 5000
>>> x_1 = expon.rvs(size=n, scale = 1)
>>> x_2 = expon.rvs(size=n, scale = 1)
>>> treatment = expon.rvs(size=n, scale = (1/(x_1 + x_2)))

The GPS is given by

.. math::

   f(t, x_1, x_2) = (x_1 + x_2) * e^{-(x_1 + x_2) * t}

If we generate the outcome variable by summing the treatment and GPS, the true causal
curve is derived analytically to be:

.. math::

   f(t) = t + \frac{2}{(1 + t)^3}


The following code completes the data generation:

>>> gps = ((x_1 + x_2) * np.exp(-(x_1 + x_2) * treatment))
>>> outcome = treatment + gps + np.random.normal(size = n, scale = 1)

>>> truth_func = lambda treatment: (treatment + (2/(1 + treatment)**3))
>>> vfunc = np.vectorize(truth_func)
>>> true_outcome = vfunc(treatment)

>>> df = pd.DataFrame(
>>>     {
>>>         'X_1': x_1,
>>>         'X_2': x_2,
>>>         'Treatment': treatment,
>>>         'GPS': gps,
>>>         'Outcome': outcome,
>>>         'True_outcome': true_outcome
>>>     }
>>> ).sort_values('Treatment', ascending = True)

With this dataframe, we can now calculate the GPS to estimate the causal relationship between
treatment and outcome. Let's use the default settings of the GPS tool:

>>> from causal_curve import GPS
>>> gps = GPS()
>>> gps.fit(T = df['Treatment'], X = df[['X_1', 'X_2']], y = df['Outcome'])
>>> gps_results = gps.calculate_CDRC(0.95)

You now have everything to produce the following plot with matplotlib. In this example with only mild confounding,
the GPS-calculated estimate of the true causal curve produces has approximately
half the error of a simple LOESS estimate using only the treatment and the outcome.

.. image:: ../imgs/cdrc/CDRC.png





References
----------

Imai K., Keele L., Tingley D. A General Approach to Causal Mediation Analysis. Psychological
Methods. 15(4), 2010, pp.309–334.
