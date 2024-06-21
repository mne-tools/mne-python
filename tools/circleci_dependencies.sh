#!/bin/bash -ef

python -m pip install --upgrade "pip!=20.3.0" build
# https://github.com/dipy/dipy/issues/3265 for numpy, dipy
python -m pip install --upgrade --progress-bar off --only-binary "numpy<2,dipy!=1.9.0,scipy,matplotlib,pandas,statsmodels" git+https://github.com/sphinx-gallery/sphinx-gallery.git -ve .[full,test,doc]
