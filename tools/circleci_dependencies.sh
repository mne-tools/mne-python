#!/bin/bash -ef

python -m pip install --upgrade "pip!=20.3.0" build
# https://github.com/dipy/dipy/issues/3265 for numpy, dipy
python -m pip install --upgrade --progress-bar off \
    --only-binary "numpy,dipy,scipy,matplotlib,pandas,statsmodels" \
    -ve .[full,test,doc,doc_related] "numpy==1.26.4" "dipy!=1.9.0" \
    "git+https://github.com/larsoner/pyvista.git@refcycle" \
    git+https://github.com/sphinx-gallery/sphinx-gallery.git
