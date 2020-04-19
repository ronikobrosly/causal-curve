.. _install:

=====================================
Installation, testing and development
=====================================

Dependencies
------------

causal-curve requires:

- Python (>= 3.7.6)
- NumPy (>= 1.18.2)
- Pandas (>= 1.0.3)
- pyGAM (>= 0.8.0)
- SciPy (>= 1.4.1)
- Statsmodels (>= 0.11.1)


User installation
-----------------

If you already have a working installation of numpy, pandas, pygam, scipy, and statsmodels,
you can easily install causal-curve using ``pip``::

    pip install causal-curve


You can also get the latest version of pyts by cloning the repository::

    git clone https://github.com/ronikobrosly/causal-curve.git
    cd causal-curve
    pip install .


Testing
-------

After installation, you can launch the test suite from outside the source
directory using ``pytest``::

    pytest


Development
-----------

Please reach out if you are interested in adding additional tools,
or have ideas on how to improve the package!
