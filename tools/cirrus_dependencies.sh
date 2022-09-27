#!/bin/bash -e

STD_ARGS="--progress-bar off --upgrade"
EXTRA_ARGS=""
python -m pip install $STD_ARGS pip setuptools wheel
# dipy does not have PyPi wheels yet so we have to use the --pre wheels until 1.6.0 comes out
pip install $STD_ARGS --only-binary ":all:" --no-deps numpy scipy
pip install $STD_ARGS --pre --only-binary ":all:" --no-deps --default-timeout=60 -i "https://pypi.anaconda.org/scipy-wheels-nightly/simple" dipy
pip install $STD_ARGS -r requirements.txt -r requirements_testing.txt -r requirements_testing_extra.txt
