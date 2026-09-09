#!/bin/bash -ef

set -x
# Bootstrap uv and use it for all installs: its resolver and hardlink-based
# installs are dramatically faster than pip, and the venv is recreated each run
# so this "Get Python running" step reinstalls everything every time. uv's
# download cache lives in ~/.cache/uv (cached separately in the CircleCI config).
python -m pip install --upgrade "pip>=25.1" uv build
# rpy2 3.6.7 (or its deps) cause problems with our installed R version, so pin them
# also install colormath because it doesn't have a binary wheel
uv pip install --upgrade --only-binary=numpy,scipy \
    "rpy2==3.6.6" "rpy2-rinterface==3.6.5" "rpy2-robjects==3.6.4" mne-ari colormath
# TODO: PyVista pinned until the regression from 2026/09/09 is fixed
uv pip install --upgrade --overrides tools/circleci_uv_overrides.txt \
    -e .[full-pyside6] \
    --group=test \
    --group=doc-full \
    "mne-bids @ https://github.com/mne-tools/mne-bids/archive/refs/heads/main.zip" \
    "mne-qt-browser @ https://github.com/mne-tools/mne-qt-browser/archive/refs/heads/main.zip" \
    "pyvista @ https://github.com/pyvista/pyvista/archive/f7df2e1e5b14ea79f6cca1f50dd10cb8008a54a6.zip" \
    "pyvistaqt @ https://github.com/pyvista/pyvistaqt/archive/refs/heads/main.zip" \
    "sphinx-gallery @ https://github.com/sphinx-gallery/sphinx-gallery/archive/refs/heads/master.zip" \
    -r doc/sphinxext/related_software.txt
uv pip install --upgrade --no-deps \
    -r doc/sphinxext/related_software_nodeps.txt
