#!/bin/bash -ef

python -m pip install --upgrade "pip!=20.3.0" build
# This can be removed once dipy > 1.9.0 is released
python -m pip install --upgrade --progress-bar off \
    numpy scipy h5py
python -m pip install --pre --progress-bar off \
    --extra-index-url "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple" \
    "dipy>1.9"
python -m pip install --upgrade --progress-bar off \
    --only-binary "numpy,dipy,scipy,matplotlib,pandas,statsmodels" \
    -ve .[full,test,doc] "numpy>=2" \
    "git+https://github.com/pyvista/pyvista.git" \
    "git+https://github.com/sphinx-gallery/sphinx-gallery.git" \
    "git+https://github.com/mne-tools/mne-bids.git" \
    \
    alphaCSC autoreject bycycle conpy emd fooof meggie \
    mne-ari mne-bids-pipeline mne-faster mne-features \
    mne-icalabel mne-lsl mne-microstates mne-nirs mne-rsa \
    neurodsp neurokit2 niseq nitime openneuro-py pactools \
    plotly pycrostates pyprep pyriemann python-picard sesameeg \
    sleepecg tensorpac yasa
