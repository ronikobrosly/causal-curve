---
title: 'causal-curve: A Python Causal Inference Package to Estimate Causal Dose-Response Curves'
tags:
  - Python
  - causal inference
  - causality
  - machine learning

authors:
  - name: Roni W. Kobrosly^[Custom footnotes for e.g. denoting who the corresspoinding author is can be included like this.]
    orcid: 0000-0003-0363-9662
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Department of Environmental Medicine and Public Health, Icahn School of Medicine at Mount Sinai, New York, NY, USA
   index: 1
 - name: Flowcast, 44 Tehama St, San Francisco, CA, USA
   index: 2
date: 1 July 2020
bibliography: paper.bib

---

# Summary

In academia and industry, randomized controlled experiments (colloquially "A/B tests")
are considered the gold standard approach for assessing the impact of a treatment or intervention.
However, for ethical or financial reasons, these experiments may not always be feasible to carry out.
"Causal inference" methods are a set of approaches that attempt to estimate causal effects
from observational rather than experimental data, correcting for the biases that are inherent
to analyzing observational data (e.g. confounding and selection bias).

Although much significant research and implementation effort has gone towards methods in
causal inference to estimate the effects of binary treatments (e.g. did the population receive
treatment "A" or "B"), much less has gone towards estimating the effects of continuous treatments.
This is unfortunate because there are there are a large number of use cases in research
and industry that could benefit from tools to estimate the effect of
continuous treatments, such as estimating how:

- the number of minutes per week of aerobic exercise causes positive health outcomes,
after controlling for confounding effects.
- increasing or decreasing the price of a product would impact demand (price elasticity).
- changing neighborhood income inequality (via the continuous Gini index)
might or might not be causally related to the neighborhood crime rate.
- blood lead levels are causally related to neurodevelopment delays in children.

`causal-curve` is a Python package created to address this gap; it is designed to perform
causal inference when the treatment of interest is continuous in nature.
From the observational data that is provided by the user, it estimates the
"causal dose-response curve" (as known as the average or marginal dose-response function).

`causal-curve` attempts to make the user-experience as painless as possible:

- This API for this package was designed to resemble that of `scikit-learn`,
as this is a very commonly used predictive modeling framework in Python that most machine learning
practioners are familiar with.
- All of the major classes contained in `causal-curve` readily use Pandas DataFrames and Series as
inputs, to make this package more easily integrate with the standard Python data analysis tools.
- Full tutorials of the three major classes are available online in the documentation,
along with full documentation of all of their parameters, methods, and attributes.


# Methods

In the current release of `causal-curve`, there are two unique model classes for
constructing the causal dose-response curve: the Generalized Propensity Score (GPS) and the
Targetted Maximum Likelihood Estimation (TMLE) tools. In addition to this, there is also tool
to assess causal mediation effects in the presence of a continuous mediator and treatment.

The `GPS` method ...

The `TMLE` method ...

`causal-curve` allows for continuous mediation assessment with the `Mediation` tool

Pytest, Travis CI, code coverage > 90%


# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](welcome_plot.png)
and referenced from text using \autoref{fig:example}.


# Statement of Need

While there are a few established Python packages related to causal inference, to the best of
the author's knowledge, there is no Python package available that can provide support for
continuous treatments as `causal-curve` does. Similarly, the author isn't aware of any python
implementation of a causal mediation analysis for continuous treatments and mediators. Finally,
the tutorials available in the documentation introduce the concept of continuous treatments
and how their analysis might be interpretted and carried out.  


# Acknowledgements

We acknowledge contributions from Miguel-Angel Luque, Erica Moodie, and Mark van der Laan
during the creation of this project.

# References
