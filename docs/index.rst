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

None of the methods provided in causal-curve rely on inference via instrumental variables, they only
rely on the data from the observed treatment, confounders, and the outcome of interest (like the above GPS example).



A caution about assumptions
---------------------------

There is a well-documented set of assumptions one must make to infer causal effects from
observational data. These are covered elsewhere in more detail, but briefly:

- Causes always occur before effects: The treatment variable needs to have occurred before the outcome.
- SUTVA: The treatment status of a given individual does not affect the potential outcomes of any other individuals.
- Positivity: Any individual has a positive probability of receiving all values of the treatment variable.
- Ignorability: All major confounding variables are included in the data you provide.

Violations of these assumptions will lead to biased results and incorrect conclusions!

In addition, any covariates that are included in `causal-curve` models are assumed to only
be **confounding** variables.



`Getting started <install.html>`_
---------------------------------

Information to install, test, and contribute to the package.
