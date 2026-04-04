#!/bin/bash -ef

ONLY_BINARY="--only-binary \"numpy,dipy,scipy,matplotlib,pandas,statsmodels,netCDF4,h5py\""

set -x
python -m pip install --upgrade "pip>=25.1" build
# rpy2 3.6.7 (or its deps) cause problems with our installed R version, so pin them
python -m pip install --upgrade --progress-bar off $ONLY_BINARY \
    -ve .[full] \
    --group=test \
    --group=doc-full \
    "rpy2==3.6.6" "rpy2-rinterface==3.6.5" "rpy2-robjects==3.6.4" \
    -r doc/sphinxext/related_software.txt \
    "git+https://github.com/mne-tools/mne-bids.git" \
    "git+https://github.com/mne-tools/mne-qt-browser.git" \
    "git+https://github.com/pyvista/pyvista.git" \
    "git+https://github.com/sphinx-gallery/sphinx-gallery.git"
python -m pip install --upgrade --progress-bar off --no-deps $ONLY_BINARY \
    -r doc/sphinxext/related_software_nodeps.txt
