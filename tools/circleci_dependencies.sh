#!/bin/bash -ef

python -m pip install --upgrade "pip!=20.3.0"
python -m pip install --upgrade --progress-bar off --only-binary "numpy,scipy,matplotlib,pandas,statsmodels" PyQt6 git+https://github.com/mne-tools/mne-qt-browser -ve .[full,test_base,doc]
