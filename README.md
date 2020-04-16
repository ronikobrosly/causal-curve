# causal-curve
Python tools to perform causal inference using observational data when the treatment of interest is continuous.

<img src=imgs/curves.png>

## Table of Contents

- [Overview](#overview)
- [Documentation](#documentation)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

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

This library attempts to address this gap, providing tools to estimate `causal curves`.

## Documentation

TBC

## Installation

causal-curves

TBC

## Usage

TBC

## Contributing

Your help is absolutely welcome! Please do reach out or create a feature branch!
