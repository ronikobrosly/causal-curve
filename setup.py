import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="causal-curve",
    version="0.1.1",
    author="Roni Kobrosly",
    author_email="roni.kobrosly@gmail.com",
    description="A python library with tools to perform causal inference using \
        observational data when the treatment of interest is continuous.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ronikobrosly/causal-curve",
    packages=setuptools.find_packages(include=['causal_curve']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'future',
        'joblib',
        'numpy',
        'pandas',
        'patsy',
        'progressbar2',
        'pygam',
        'python-dateutil',
        'python-utils',
        'pytz',
        'scikit-learn',
        'scipy',
        'six',
        'statsmodels',
        'xgboost'
    ]
)
