import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="causal-curve",
    version="0.4.0",
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
        'black',
        'coverage',
        'future',
        'joblib',
        'numpy',
        'numpydoc',
        'pandas',
        'patsy',
        'progressbar2',
        'pygam',
        'pytest',
        'python-dateutil',
        'python-utils',
        'pytz',
        'scikit-learn',
        'scipy',
        'six',
        'sphinx_rtd_theme',
        'statsmodels'
    ]
)
