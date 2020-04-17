# causal-curve
Python tools to perform causal inference using observational data when the treatment of interest is continuous.

[![Antikythera mechanism](https://upload.wikimedia.org/wikipedia/commons/e/e8/Antikythera_mechanism.svg =460x630)](https://en.wikipedia.org/wiki/Antikythera_mechanism)

The Antikythera mechanism, an ancient analog computer, with lots of beautiful curves.


## Table of Contents

- [Overview](#overview)
- [Documentation](#documentation)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [References](#references)

## Overview

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
* Estimate how changing neighborhood income inequality (e.g. the Gini index) could be causally related to neighborhood crime rate.

This library attempts to address this gap, providing tools to estimate causal curves.

## Documentation

TBC

## Installation

TBC

## Usage

TBC

## Contributing

Your help is absolutely welcome! Please do reach out or create a feature branch!

## References

Galagate, D. Causal Inference with a Continuous Treatment and Outcome: Alternative
Estimators for Parametric Dose-Response function with Applications. PhD thesis, 2016.

Moodie E and Stephens DA. Estimation of dose–response functions for
longitudinal data using the generalised propensity score. In: Statistical Methods in
Medical Research 21(2), 2010, pp.149–166.

Hirano K and Imbens GW. The propensity score with continuous treatments.
In: Gelman A and Meng XL (eds) Applied bayesian modeling and causal inference
from incomplete-data perspectives. Oxford, UK: Wiley, 2004, pp.73–84.
