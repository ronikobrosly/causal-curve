# causal-curve

[![build status](http://img.shields.io/travis/ronikobrosly/causal-curve/master.svg?style=flat)](https://travis-ci.org/ronikobrosly/causal-curve)
[![codecov](https://codecov.io/gh/ronikobrosly/causal-curve/branch/master/graph/badge.svg)](https://codecov.io/gh/ronikobrosly/causal-curve)
[![DOI](https://zenodo.org/badge/256017107.svg)](https://zenodo.org/badge/latestdoi/256017107)

Python tools to perform causal inference when the treatment of interest is continuous.


<p align="center">
<img src="/imgs/curves.png" align="middle"/>
</p>



## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [References](#references)

## Overview

(**Version 1.0.0 released in January 2021!**)

There are many implemented methods to perform causal inference when your intervention of interest is binary,
but few methods exist to handle continuous treatments.

This is unfortunate because there are many scenarios (in industry and research) where these methods would be useful.
For example, when you would like to:

* Estimate the causal response to increasing or decreasing the price of a product across a wide range.
* Understand how the number of minutes per week of aerobic exercise causes positive health outcomes.
* Estimate how decreasing order wait time will impact customer satisfaction, after controlling for confounding effects.
* Estimate how changing neighborhood income inequality (Gini index) could be causally related to neighborhood crime rate.

This library attempts to address this gap, providing tools to estimate causal curves (AKA causal dose-response curves).
Both continuous and binary outcomes can be modeled against a continuous treatment.

## Installation

Available via PyPI:

`pip install causal-curve`

You can also get the latest version of causal-curve by cloning the repository::

```
git clone -b master https://github.com/ronikobrosly/causal-curve.git
cd causal-curve
pip install .
```

## Documentation

[Documentation is available at readthedocs.org](https://causal-curve.readthedocs.io/en/latest/)


## Contributing

Your help is absolutely welcome! Please do reach out or create a feature branch!

## Citation

Kobrosly, R. W., (2020). causal-curve: A Python Causal Inference Package to Estimate Causal Dose-Response Curves. Journal of Open Source Software, 5(52), 2523, [https://doi.org/10.21105/joss.02523](https://doi.org/10.21105/joss.02523)

## References

Galagate, D. Causal Inference with a Continuous Treatment and Outcome: Alternative
Estimators for Parametric Dose-Response function with Applications. PhD thesis, 2016.

Hirano K and Imbens GW. The propensity score with continuous treatments.
In: Gelman A and Meng XL (eds) Applied bayesian modeling and causal inference
from incomplete-data perspectives. Oxford, UK: Wiley, 2004, pp.73–84.

Imai K, Keele L, Tingley D. A General Approach to Causal Mediation Analysis. Psychological
Methods. 15(4), 2010, pp.309–334.

Kennedy EH, Ma Z, McHugh MD, Small DS. Nonparametric methods for doubly robust estimation
of continuous treatment effects. Journal of the Royal Statistical Society, Series B. 79(4), 2017, pp.1229-1245.

Moodie E and Stephens DA. Estimation of dose–response functions for
longitudinal data using the generalised propensity score. In: Statistical Methods in
Medical Research 21(2), 2010, pp.149–166.

van der Laan MJ and Gruber S. Collaborative double robust penalized targeted
maximum likelihood estimation. In: The International Journal of Biostatistics 6(1), 2010.

van der Laan MJ and Rubin D. Targeted maximum likelihood learning. In: ​U.C. Berkeley Division of
Biostatistics Working Paper Series, 2006.
