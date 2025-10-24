#!/bin/bash -ef

python -m pip install --upgrade "pip>=25.1" build
python -m pip install --upgrade --progress-bar off \
    -ve .[full] \
    --group=test \
    --group=doc \
    -r doc/sphinxext/related_software.txt \
    --only-binary "numpy,dipy,scipy,matplotlib,pandas,statsmodels" \
    "git+https://github.com/mne-tools/mne-bids.git" \
    "git+https://github.com/mne-tools/mne-qt-browser.git" \
    "git+https://github.com/pyvista/pyvista.git" \
    "git+https://github.com/sphinx-gallery/sphinx-gallery.git"
python -m pip install --upgrade --progress-bar off --no-deps cross-domain-saliency-maps
