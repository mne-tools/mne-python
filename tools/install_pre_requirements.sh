#!/bin/bash

set -eo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PLATFORM=$(python -c 'import platform; print(platform.system())')

echo "Installing pip-pre dependencies on ${PLATFORM}"
STD_ARGS="--progress-bar off --upgrade --pre"
QT_BINDING="PyQt6"

# Dependencies of scientific-python-nightly-wheels are installed here so that
# we can use strict --index-url (instead of --extra-index-url) below
echo "PyQt6 and scientific-python-nightly-wheels dependencies"
python -m pip install $STD_ARGS pip setuptools packaging \
	threadpoolctl cycler fonttools kiwisolver pyparsing pillow python-dateutil \
	patsy pytz tzdata nibabel tqdm trx-python joblib numexpr "$QT_BINDING" \
	py-cpuinfo blosc2 hatchling "formulaic>=1.1.0" \
	"h5py>=3.12.1"  # TODO move h5py back to nightlies when possible
echo "NumPy/SciPy/pandas etc."
python -m pip uninstall -yq numpy
python -m pip install $STD_ARGS --only-binary ":all:" --default-timeout=60 \
	--index-url "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple" \
	"numpy>=2.1.0.dev0" "scikit-learn>=1.6.dev0" "scipy>=1.15.0.dev0" \
	"matplotlib>=3.11.0.dev0" \
	"pandas>=3.0.0.dev0" \
	"dipy>=1.10.0.dev0" \
	"pyarrow>=20.0.0.dev0" \
	"tables>=3.10.3.dev0"
# TODO: should have above:  "h5py>=3.12.1"
# https://github.com/h5py/h5py/issues/2563
# TODO: should have above: "statsmodels>=0.15.0.dev0"
# https://github.com/statsmodels/statsmodels/issues/9572

# No Numba because it forces an old NumPy version

echo "pymatreader"
pip install https://gitlab.com/obob/pymatreader/-/archive/master/pymatreader-master.zip

echo "OpenMEEG"
python -m pip install $STD_ARGS --only-binary ":all:" --extra-index-url "https://test.pypi.org/simple" "openmeeg>=2.6.0.dev4"

echo "nilearn"
python -m pip install $STD_ARGS "git+https://github.com/nilearn/nilearn"

echo "VTK"
python -m pip install $STD_ARGS --only-binary ":all:" --extra-index-url "https://wheels.vtk.org" vtk
python -c "import vtk"

echo "PyVista"
python -m pip install $STD_ARGS "git+https://github.com/pyvista/pyvista" trame trame-vtk trame-vuetify jupyter ipyevents ipympl

echo "picard"
python -m pip install $STD_ARGS git+https://github.com/pierreablin/picard

echo "pyvistaqt"
pip install $STD_ARGS git+https://github.com/pyvista/pyvistaqt

echo "imageio-ffmpeg, xlrd, mffpy"
pip install $STD_ARGS imageio-ffmpeg xlrd mffpy traitlets pybv eeglabio defusedxml antio

echo "mne-qt-browser"
pip install $STD_ARGS git+https://github.com/mne-tools/mne-qt-browser

echo "mne-bids"
pip install $STD_ARGS git+https://github.com/mne-tools/mne-bids

echo "nibabel"
pip install $STD_ARGS git+https://github.com/nipy/nibabel

echo "joblib"
pip install $STD_ARGS git+https://github.com/joblib/joblib

echo "edfio"
# Disable protection for Azure, see
# https://github.com/mne-tools/mne-python/pull/12609#issuecomment-2115639369
GIT_CLONE_PROTECTION_ACTIVE=false pip install $STD_ARGS git+https://github.com/the-siesta-group/edfio

echo "h5io"
pip install $STD_ARGS git+https://github.com/h5io/h5io

echo "pysnirf2"
pip install $STD_ARGS git+https://github.com/BUNPC/pysnirf2

# Make sure we're on a NumPy 2.0 variant
echo "Checking NumPy version"
python -c "import numpy as np; assert np.__version__[0] == '2', np.__version__"

# And that Qt works
echo "Checking Qt"
${SCRIPT_DIR}/check_qt_import.sh "$QT_BINDING"
