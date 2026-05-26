#!/bin/bash -ef

set -x
python -m pip install --upgrade "pip>=25.1" build
# rpy2 3.6.7 (or its deps) cause problems with our installed R version, so pin them
python -m pip install --upgrade --only-binary=numpy,scipy \
    "rpy2==3.6.6" "rpy2-rinterface==3.6.5" "rpy2-robjects==3.6.4" mne-ari
python -m pip install --upgrade --only-binary=:all: \
    -ve .[full-pyside6] \
    --group=test \
    --group=doc-full \
    "mne-bids @ https://github.com/mne-tools/mne-bids/archive/refs/heads/main.zip" \
    "mne-qt-browser @ https://github.com/mne-tools/mne-qt-browser/archive/refs/heads/main.zip" \
    "pyvista @ https://github.com/pyvista/pyvista/archive/refs/heads/main.zip" \
    "pyvistaqt @ https://github.com/pyvista/pyvistaqt/archive/refs/heads/main.zip" \
    "sphinx-gallery @ https://github.com/sphinx-gallery/sphinx-gallery/archive/refs/heads/master.zip" \
    -r doc/sphinxext/related_software.txt
python -m pip install --upgrade --no-deps --only-binary=:all: \
    -r doc/sphinxext/related_software_nodeps.txt
