#!/bin/bash -ef

python -m pip install --upgrade "pip!=20.3.0" build
python -m pip install --upgrade --progress-bar off --only-binary "numpy,scipy,matplotlib,pandas,statsmodels" git+https://github.com/sphinx-gallery/sphinx-gallery.git git+https://github.com/Carreau/intersphinx_registry.git -ve .[full,test,doc]
