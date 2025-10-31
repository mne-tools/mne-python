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
echo "::group::Prerequisites"
python -m pip install $STD_ARGS pip setuptools packaging \
	threadpoolctl cycler fonttools kiwisolver pyparsing pillow python-dateutil \
	patsy pytz tzdata nibabel tqdm trx-python joblib numexpr \
	"$QT_BINDING!=6.9.1" \
	py-cpuinfo blosc2 hatchling "formulaic>=1.1.0" \
	matplotlib
python -m pip uninstall -yq numpy
echo "::endgroup::"
echo "::group::Scientific Python Nightly Wheels"
python -m pip install $STD_ARGS --only-binary ":all:" --default-timeout=60 \
	--index-url "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple" \
	"numpy>=2.1.0.dev0" \
	"scipy>=1.15.0.dev0" \
	"scikit-learn>=1.6.dev0" \
	"pandas>=3.0.0.dev0" \
	"dipy>=1.10.0.dev0" \
	"tables>=3.10.3.dev0" \
	"statsmodels>=0.15.0.dev697" \
	"pyarrow>=22.0.0.dev0" \
	"matplotlib>=3.11.0.dev0" \
	"h5py>=3.13.0"
echo "::endgroup::"
# No Numba because it forces an old NumPy version

echo "::group::VTK"
python -m pip install $STD_ARGS --only-binary ":all:" --extra-index-url "https://wheels.vtk.org" vtk
python -c "import vtk"
echo "::endgroup::"

echo "::group::Everything else"
python -m pip install $STD_ARGS \
	"git+https://github.com/pyvista/pyvista" \
	"git+https://github.com/pyvista/pyvistaqt" \
	"git+https://github.com/nilearn/nilearn" \
	"git+https://github.com/pierreablin/picard" \
	"git+https://github.com/the-siesta-group/edfio" \
	https://gitlab.com/obob/pymatreader/-/archive/master/pymatreader-master.zip \
	git+https://github.com/mne-tools/mne-qt-browser \
	git+https://github.com/pyqtgraph/pyqtgraph \
	git+https://github.com/mne-tools/mne-bids \
	git+https://github.com/nipy/nibabel \
	git+https://github.com/joblib/joblib \
	git+https://github.com/h5io/h5io \
	git+https://github.com/BUNPC/pysnirf2 \
	git+https://github.com/the-siesta-group/edfio \
	git+https://github.com/python-quantities/python-quantities \
	trame trame-vtk trame-vuetify jupyter ipyevents ipympl openmeeg \
	imageio-ffmpeg xlrd mffpy traitlets pybv eeglabio defusedxml \
	antio curryreader
echo "::endgroup::"

echo "::group::Make sure we're on a NumPy 2.0 variant"
python -c "import numpy as np; assert np.__version__[0] == '2', np.__version__"
echo "::endgroup::"

echo "::group::Check Qt import"
${SCRIPT_DIR}/check_qt_import.sh "$QT_BINDING"
echo "::endgroup::"
