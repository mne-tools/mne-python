#!/bin/bash -ef

python -m pip install --upgrade "pip!=20.3.0" build
# TODO: Use PST nightly once it exists and revert SG change
python -m pip install --upgrade --progress-bar off \
    --only-binary "numpy,scipy,matplotlib,pandas,statsmodels" \
    "git+https://github.com/jschueller/sphinx-gallery.git@paral" \
    "git+https://github.com/pydata/pydata-sphinx-theme.git" \
    -ve .[full,test,doc]
