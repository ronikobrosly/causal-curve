Welcome to causal-curve's documentation!
========================================

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   install
   contribute

.. toctree::
  :maxdepth: 1
  :hidden:
  :caption: Module details

  modules


.. toctree::
  :maxdepth: 1
  :hidden:
  :caption: Tutorial - Examples

  CDRC_example

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Additional Information

   changelog
   citation


.. toctree::
   :maxdepth: 2
   :caption: Contents:


**causal-curve** is a Python package dedicated with tools to perform causal inference
using observational data when the treatment of interest is continuous.


.. image:: ../imgs/welcome_plot.png



Summary
-------

Sometimes it would be nice to run a randomized, controlled experiment to determine whether drug `A`
is superior to drug `B`, whether the blue button gets more clicks than the orange button on your
e-commerce site, etc. Unfortunately, it isn't always possible (resources are finite, the
test might not be ethical, you lack a proper A/B testing infrastructure, etc).
In these situations, there are methods can be employed to help you infer causality from observational data.

There are many methods to perform causal inference when your intervention of interest is binary
(see the drug and button examples above), but few methods exist to handle continuous treatments.

This is unfortunate because there are many scenarios (in industry and research) where these methods would be useful.
For example, when you would like to:

* Estimate the causal response to increasing or decreasing the price of a product across a wide range.
* Understand how the number of hours per week of aerobic exercise causes positive health outcomes.
* Estimate how decreasing order wait time will impact customer satisfaction, after controlling for confounding effects.
* Estimate how changing neighborhood income inequality (Gini index) could be causally related to neighborhood crime rate.

This library attempts to address this gap, providing tools to estimate causal curves (AKA causal dose-response curves).


Quick example (of the ``CDRC`` tool)
--------------------------------------

**causal-curve** uses a sklearn-like API that should feel familiar to python machine learning users.

>>> from causal_curve.cdrc import CDRC

>>> cdrc = CDRC(treatment_grid_num = 200, random_seed = 512)

>>> df # a pandas dataframe with your data
           X_1       X_2  Treatment    Outcome
0     0.596685  0.162688   0.000039  -0.270533
1     1.014187  0.916101   0.000197  -0.266979
2     0.932859  1.328576   0.000223   1.921979
3     1.140052  0.555203   0.000339   1.461526
4     1.613471  0.340886   0.000438   2.064511

>>> cdrc.fit(T = df['Treatment'], X = df[['X_1', 'X_2']], y = df['Outcome'])
>>> cdrc_results = cdrc.calculate_CDRC(ci = 0.95)

1. First we import our CDRC (causal dose-response curve) class.

2. Then we instantiate the class, providing any of the optional parameters.

3. Prepare and organized your treatment, covariate, and outcome data into a pandas dataframe.

4. Fit the load the training and test sets by calling the ``.fit()`` method.

5. Estimate the points of the causal curve (along with 95% confidence interval bounds) with the ``.calculate_CDRC()`` method.

6. Explore or plot your results!


`Getting started <install.html>`_
---------------------------------

Information to install, test, and contribute to the package.




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
