#!/bin/bash -ef

python -m pip install --upgrade "pip!=20.3.0" build
# https://github.com/dipy/dipy/issues/3265 for numpy, dipy
python -m pip install --upgrade --progress-bar off \
    --only-binary "numpy,dipy,scipy,matplotlib,pandas,statsmodels" \
    -ve .[full,test,doc] "numpy==1.26.4" "dipy!=1.9.0" \
    "git+https://github.com/larsoner/pyvista.git@refcycle" \
    git+https://github.com/sphinx-gallery/sphinx-gallery.git \
    \
    alphaCSC autoreject bycycle conpy emd fooof meggie \
    mne-ari mne-bids-pipeline mne-faster mne-features \
    mne-icalabel mne-lsl mne-microstates mne-nirs mne-rsa \
    neurodsp neurokit2 niseq nitime openneuro-py pactools \
    plotly pycrostates pyprep pyriemann python-picard sesameeg \
    sleepecg tensorpac yasa
