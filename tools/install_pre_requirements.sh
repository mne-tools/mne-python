#!/bin/bash

set -eo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PLATFORM=$(python -c 'import platform; print(platform.system())')

echo "Installing pip-pre dependencies on ${PLATFORM}"
STD_ARGS="--progress-bar off --upgrade --pre"
QT_BINDING="PySide6"

# Dependencies of scientific-python-nightly-wheels are installed here so that
# we can use strict --index-url (instead of --extra-index-url) below
set -x
python -m pip install $STD_ARGS pip setuptools packaging \
	threadpoolctl cycler fonttools kiwisolver pyparsing pillow python-dateutil \
	patsy pytz tzdata nibabel tqdm trx-python joblib numexpr \
	"$QT_BINDING!=6.9.1" \
	py-cpuinfo blosc2 hatchling "formulaic>=1.1.0"
python -m pip uninstall -yq numpy
python -m pip install $STD_ARGS --only-binary ":all:" --default-timeout=60 \
	--index-url "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple" \
	"numpy>=2.1.0.dev0" "scikit-learn>=1.6.dev0" "scipy>=1.15.0.dev0" \
	"matplotlib>=3.11.0.dev0" \
	"pandas>=3.0.0.dev0" \
	"dipy>=1.10.0.dev0" \
	"pyarrow>=20.0.0.dev0" \
	"tables>=3.10.3.dev0" \
	"h5py>=3.13.0"
# TODO: should have above: "statsmodels>=0.15.0.dev0"
# https://github.com/statsmodels/statsmodels/issues/9572

# No Numba because it forces an old NumPy version

pip install https://gitlab.com/obob/pymatreader/-/archive/master/pymatreader-master.zip

python -m pip install $STD_ARGS --only-binary ":all:" --extra-index-url "https://test.pypi.org/simple" "openmeeg>=2.6.0.dev4"

python -m pip install $STD_ARGS "git+https://github.com/nilearn/nilearn"

python -m pip install $STD_ARGS --only-binary ":all:" --extra-index-url "https://wheels.vtk.org" vtk
python -c "import vtk"

python -m pip install $STD_ARGS "git+https://github.com/pyvista/pyvista" trame trame-vtk trame-vuetify jupyter ipyevents ipympl

python -m pip install $STD_ARGS git+https://github.com/pierreablin/picard

pip install $STD_ARGS git+https://github.com/pyvista/pyvistaqt

pip install $STD_ARGS imageio-ffmpeg xlrd mffpy traitlets pybv eeglabio defusedxml antio

pip install $STD_ARGS git+https://github.com/mne-tools/mne-qt-browser

pip install $STD_ARGS git+https://github.com/mne-tools/mne-bids

pip install $STD_ARGS git+https://github.com/nipy/nibabel

pip install $STD_ARGS git+https://github.com/joblib/joblib

# Disable protection for Azure, see
# https://github.com/mne-tools/mne-python/pull/12609#issuecomment-2115639369
GIT_CLONE_PROTECTION_ACTIVE=false pip install $STD_ARGS git+https://github.com/the-siesta-group/edfio

pip install $STD_ARGS git+https://github.com/h5io/h5io

pip install $STD_ARGS git+https://github.com/BUNPC/pysnirf2

# Make sure we're on a NumPy 2.0 variant
python -c "import numpy as np; assert np.__version__[0] == '2', np.__version__"

# And that Qt works
${SCRIPT_DIR}/check_qt_import.sh "$QT_BINDING"
