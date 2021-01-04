.. _changelog:

==========
Change Log
==========


Version 1.0.1
-------------
- Added to TMLE overview in the docs (including plot)


Version 1.0.0: **Major Update**
-------------------------------
- Overhaul of the TMLE tool to make it dramatically more accurate and user-friendly.
- Improved TMLE example documentation
- Much like with `scikit-learn`, there are now separate model classes used for predicting binary or continuous outcomes
- Updating documentation to reflect API changes
- Added more tests
- Linted with `pylint` (added `.pylintrc` file)


Version 0.5.2
-------------
- Fixed bug that prevented `causal-curve` modules from being shown in Sphinx documentation
- Augmented tests to capture more error states and improve code coverage


Version 0.5.1
-------------
- Removed working test file


Version 0.5.0
-------------
- Added new `predict`, `predict_interval`, and `predict_log_odds` methods to GPS tool
- Slight updates to doc to reflect new features


Version 0.4.1
-------------
- When using GPS tool with a treatment with negative values, only the normal GLM family can be picked
- Added 'sphinx_rtd_theme' to dependency list in `.travis.yml` and `install.rst`
- core.py base class now has __version__ attribute


Version 0.4.0
-------------
- Added support for binary outcomes in GPS tool
- Small changes to repo README


Version 0.3.8
-------------
- Added citation (yay!)


Version 0.3.7
-------------
- Bumped version for PyPi


Version 0.3.6
-------------
- Fixed bug in Mediation.calculate_mediation that would clip treatments < 0 or > 1
- Fixed incorrect horizontal axis labels in lead example
- Fixed typos in documentation
- Added links to resources so users could learn more about causal inference theory


Version 0.3.5
-------------
- Re-organized documentation
- Added `Introduction` section to explain purpose and need for the package


Version 0.3.4
-------------
- Removed XGBoost as dependency.
- Now using sklearn's gradient boosting implementation.


Version 0.3.3
-------------
- Misc edits to paper and bibliography


Version 0.3.2
-------------
- Fixed random seed issue with Mediation tool
- Fixed Mediation bootstrap issue. Confidence interval bounded [0,1]
- Fixed issue with all classes not accepting non-sequential indicies in pandas Dataframes/Series
- Class init checks for all classes now print cleaner errors if bad input


Version 0.3.1
-------------
- Small fixes to end-to-end example documentation
- Enlarged image in paper


Version 0.3.0
-------------
- Added full, end-to-end example of package usage to documentation
- Cleaned up documentation
- Added example folder with end-to-end notebook
- Added manuscript to paper folder


Version 0.2.4
-------------
- Strengthened unit tests


Version 0.2.3
-------------
- codecov integration


Version 0.2.2
-------------
- Travis CI integration


Version 0.2.1
-------------
- Fixed Mediation tool error / removed `tqdm` from requirements
- Misc documentation cleanup / revisions


Version 0.2.0
-------------
- Added new Mediation class
- Updated documentation to reflect this
- Added unit and integration tests for Mediation methods


Version 0.1.3
-------------
- Simplifying unit and integration tests.


Version 0.1.2
-------------

- Added unit and integration tests


Version 0.1.1
-------------

- setup.py fix


Version 0.1.0
-------------

- Added new TMLE class
- Updated documentation to reflect new TMLE method
- Renamed CDRC method to more appropriate `GPS` method
- Small docstring corrections to GPS method


Version 0.0.10
--------------

- Bug fix in GPS estimation method


Version 0.0.9
-------------

- Project created
