#!/bin/bash -ef

python -m pip install --upgrade "pip!=20.3.0" build
# TODO: Use PST nightly once it exists and revert SG change
python -m pip install --upgrade --progress-bar off \
    --extra-index-url="https://pypi.anaconda.org/scientific-python-nightly-wheels/simple" \
    --only-binary "numpy,scipy,matplotlib,pandas,statsmodels,pydata-sphinx-theme" \
    "pydata-sphinx-theme>=0.15.4.dev0" \
    "git+https://github.com/jschueller/sphinx-gallery.git@paral" \
    -ve .[full,test,doc]
