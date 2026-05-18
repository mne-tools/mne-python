#!/bin/bash -ef

ONLY_BINARY="--only-binary=:all:"

set -x
python -m pip install --upgrade $ONLY_BINARY "pip>=25.1" build
# rpy2 3.6.7 (or its deps) cause problems with our installed R version, so pin them
# and the interface doesn't have wheels so allow it to build
python -m pip install --upgrade \
    "rpy2==3.6.6" "rpy2-rinterface==3.6.5" "rpy2-robjects==3.6.4" \
python -m pip install --upgrade $ONLY_BINARY \
    -ve .[full-pyside6] \
    --group=test \
    --group=doc-full \
    -r doc/sphinxext/related_software.txt \
    "git+https://github.com/mne-tools/mne-bids.git" \
    "git+https://github.com/mne-tools/mne-qt-browser.git" \
    "git+https://github.com/pyvista/pyvista.git" \
    "git+https://github.com/sphinx-gallery/sphinx-gallery.git"
python -m pip install --upgrade --no-deps $ONLY_BINARY \
    -r doc/sphinxext/related_software_nodeps.txt
