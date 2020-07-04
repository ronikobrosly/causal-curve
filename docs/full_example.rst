.. _full_example:

===================================================================
Health data: generating causal curves and examining mediation
===================================================================

Children, blood lead levels, and cognitive performance
------------------------------------------------------

To provide an end-to-end example of the sorts of analyses `cause-curve` can be used for, we'll
begin with an epidemiology topic. The full  Specific examples of the individual
`causal-curve` tools are available elsewhere in this documentation.

Despite the banning of the use of lead-based paint and the use of lead in gasoline, lead exposure
remains


In

>>> import numpy as np
>>> import pandas as pd
>>> from scipy.stats import expon

>>> np.random.seed(333)
>>> n = 5000
>>> x_1 = expon.rvs(size=n, scale = 1)
>>> x_2 = expon.rvs(size=n, scale = 1)
>>> treatment = expon.rvs(size=n, scale = (1/(x_1 + x_2)))



.. image:: ../imgs/full_example/CDRC.png





References
----------

Environmental Protection Agency. Learn about Lead. https://www.epa.gov/lead/learn-about-lead.
Accessed on July 2, 2020.

Pirkle JL, Kaufmann RB, Brody DJ, Hickman T, Gunter EW, Paschal DC. Exposure of the
U.S. population to lead, 1991-1994. Environmental Health Perspectives, 106(11), 1998, pp. 745–750.

Lanphear BP, Dietrich K, Auinger P, Cox C. Cognitive Deficits Associated with
Blood Lead Concentrations <10 pg/dL in US Children and Adolescents.
In: Public Health Reports, 115, 2000, pp.521-529.
