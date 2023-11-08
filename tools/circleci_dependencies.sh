#!/bin/bash -ef

python -m pip install --upgrade "pip!=20.3.0" build
python -m pip install --upgrade --progress-bar off --only-binary "numpy,scipy,matplotlib,pandas,statsmodels" PyQt6 git+https://github.com/mne-tools/mne-qt-browser -ve .[test_full,doc]
