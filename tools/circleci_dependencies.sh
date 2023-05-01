#!/bin/bash -ef

python -m pip install --upgrade "pip!=20.3.0" setuptools wheel
python -m pip install --upgrade --progress-bar off --only-binary "numpy,scipy,matplotlib,pandas,statsmodels" -r requirements.txt -r requirements_testing.txt -r requirements_doc.txt PyQt6 git+https://github.com/mne-tools/mne-qt-browser git+https://github.com/sphinx-gallery/sphinx-gallery -e .
