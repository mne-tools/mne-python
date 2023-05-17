#!/bin/bash -e

STD_ARGS="--progress-bar off --upgrade"
pip install $STD_ARGS pip setuptools wheel
pip install $STD_ARGS --only-binary "numpy,scipy,dipy,statsmodels" -r requirements.txt -r requirements_testing.txt -r requirements_testing_extra.txt
