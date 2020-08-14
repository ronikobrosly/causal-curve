Welcome to causal-curve's documentation!
========================================


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   intro
   install
   contribute


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: End-to-end demonstration

   full_example


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Tutorials of Individual Tools

   GPS_example
   TMLE_example
   Mediation_example

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Module details

   modules


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Additional Information

   changelog
   citation


.. toctree::
   :maxdepth: 2
   :caption: Contents:


**causal-curve** is a Python package with tools to perform causal inference
using observational data when the treatment of interest is continuous.

.. image:: ../imgs/welcome_plot.png


Summary
-------

There are many available methods to perform causal inference when your intervention of interest is binary,
but few methods exist to handle continuous treatments. This is unfortunate because there are many
scenarios (in industry and research) where these methods would be useful. This library attempts to
address this gap, providing tools to estimate causal curves (AKA causal dose-response curves).


Quick example (of the ``GPS`` tool)
-----------------------------------

**causal-curve** uses a sklearn-like API that should feel familiar to python machine learning users.
The following example estimates the causal dose-response curve (CDRC) by calculating
generalized propensity scores.

>>> from causal_curve import GPS

>>> gps = GPS(treatment_grid_num = 200, random_seed = 512)

>>> df # a pandas dataframe with your data
           X_1       X_2  Treatment    Outcome
0     0.596685  0.162688   0.000039  -0.270533
1     1.014187  0.916101   0.000197  -0.266979
2     0.932859  1.328576   0.000223   1.921979
3     1.140052  0.555203   0.000339   1.461526
4     1.613471  0.340886   0.000438   2.064511

>>> gps.fit(T = df['Treatment'], X = df[['X_1', 'X_2']], y = df['Outcome'])
>>> gps_results = gps.calculate_CDRC(ci = 0.95)

1. First we import our GPS class.

2. Then we instantiate the class, providing any of the optional parameters.

3. Prepare and organized your treatment, covariate, and outcome data into a pandas dataframe.

4. Fit the load the training and test sets by calling the ``.fit()`` method.

5. Estimate the points of the causal curve (along with 95% confidence interval bounds) with the ``.calculate_CDRC()`` method.

6. Explore or plot your results!
