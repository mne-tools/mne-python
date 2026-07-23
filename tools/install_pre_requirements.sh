#!/bin/bash

set -eo pipefail

PLATFORM=$(python -c 'import platform; print(platform.system())')

echo "Installing pip-pre dependencies on ${PLATFORM}"
# Many deps below are pulled from GitHub/GitLab archives (codeload.github.com),
# which intermittently stalls mid-download and yields a fatal pip ReadTimeoutError.
# Give every pip call a longer socket timeout and extra retries so these
# transient network hiccups don't fail the whole job.
export PIP_DEFAULT_TIMEOUT=60
export PIP_RETRIES=10
STD_ARGS="--progress-bar off --upgrade --pre"
if [[ "$MNE_QT_BACKEND" == "" ]]; then
	MNE_QT_BACKEND="PySide6"
fi

# Dependencies of scientific-python-nightly-wheels are installed here so that
# we can use strict --index-url (instead of --extra-index-url) below
set -x
echo "::group::Prerequisites"
python -m pip install $STD_ARGS pip setuptools packaging \
	threadpoolctl cycler fonttools kiwisolver pyparsing pillow python-dateutil \
	patsy pytz tzdata nibabel tqdm trx-python joblib numexpr \
	"$MNE_QT_BACKEND!=6.9.1" \
	py-cpuinfo blosc2 hatchling "formulaic>=1.1.0" \
	scikit-learn tables
python -m pip uninstall -yq numpy
echo "::endgroup::"
echo "::group::Scientific Python Nightly Wheels"
python -m pip install $STD_ARGS --only-binary ":all:" \
	--index-url "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple" \
	"numpy>=2.5.0.dev0" \
	"scipy>=1.18.0.dev0" \
	"pandas>=3.1.0.dev0" \
	"dipy>=1.12.0.dev0" \
	"pyarrow>=22.0.0.dev0" \
	"matplotlib>=3.11.0.dev0" \
	"statsmodels>=0.15.0.dev0" \
	"h5py>=3.13.0"
# https://github.com/scikit-learn/scikit-learn/issues/34458
#	"scikit-learn>=1.9.dev0" \
# https://github.com/PyTables/PyTables/issues/1338
#	"tables>=3.10.3.dev0" \
echo "::endgroup::"
# No Numba because it forces an old NumPy version

echo "::group::VTK"
python -m pip install $STD_ARGS --only-binary ":all:" --extra-index-url "https://wheels.vtk.org" "vtk>=9.6.20260517.dev0,!=9.6.20260601,!=9.6.20260618"
python -c "import vtk"
echo "::endgroup::"

echo "::group::Everything else"
python -m pip install $STD_ARGS \
	"pyvista @ https://github.com/pyvista/pyvista/archive/refs/heads/main.zip" \
	"pyvistaqt @ https://github.com/larsoner/pyvistaqt/archive/refs/heads/qvtk-opengl-widget.zip" \
	"git+https://github.com/nilearn/nilearn" \
	"git+https://github.com/pierreablin/picard" \
	"git+https://github.com/the-siesta-group/edfio" \
	"https://gitlab.com/obob/pymatreader/-/archive/master/pymatreader-master.zip" \
	git+https://github.com/pyqtgraph/pyqtgraph \
	"mne-qt-browser @ https://github.com/mne-tools/mne-qt-browser/archive/refs/heads/main.zip" \
	"mne-bids @ https://github.com/mne-tools/mne-bids/archive/refs/heads/main.zip" \
	"nibabel @ https://github.com/nipy/nibabel/archive/refs/heads/master.zip" \
	git+https://github.com/joblib/joblib \
	git+https://github.com/h5io/h5io \
	git+https://github.com/BUNPC/pysnirf2 \
	git+https://github.com/the-siesta-group/edfio \
	trame trame-vtk "trame-vuetify!=3.2.3" trame-pyvista nest-asyncio2 jupyter ipyevents ipympl \
	openmeeg imageio-ffmpeg xlrd mffpy traitlets pybv eeglabio defusedxml antio curryreader \
	filelock
echo "::endgroup::"

echo "::group::Make sure we're on a NumPy 2.0 variant"
python -c "import numpy as np; assert np.__version__[0] == '2', np.__version__"
echo "::endgroup::"

echo "::group::Check Qt import"
curl https://raw.githubusercontent.com/mne-tools/mne-tools/main/tools/check_qt_import.sh -o check_qt_import.sh
./check_qt_import.sh "$MNE_QT_BACKEND"
echo "::endgroup::"
