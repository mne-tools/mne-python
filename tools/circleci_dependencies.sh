#!/bin/bash -ef

set -x
python -m pip install --upgrade "pip>=25.1" build
ONLY_BINARY="--only-binary \"numpy,dipy,scipy,matplotlib,pandas,statsmodels,netCDF5,h5py\""
# netCDF5 is here to work around an annoying bug
python -m pip install --upgrade --progress-bar off $ONLY_BINARY \
    -ve .[full] \
    --group=test \
    --group=doc \
    netCDF5 \
    -r doc/sphinxext/related_software.txt \
    "git+https://github.com/mne-tools/mne-bids.git" \
    "git+https://github.com/mne-tools/mne-qt-browser.git" \
    "git+https://github.com/pyvista/pyvista.git" \
    "git+https://github.com/sphinx-gallery/sphinx-gallery.git"
python -m pip install --upgrade --progress-bar off --no-deps $ONLY_BINARY \
    -r doc/sphinxext/related_software_nodeps.txt
