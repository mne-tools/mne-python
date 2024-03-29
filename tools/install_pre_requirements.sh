#!/bin/bash

set -eo pipefail

PLATFORM=$(python -c 'import platform; print(platform.system())')

echo "Installing pip-pre dependencies on ${PLATFORM}"
STD_ARGS="--progress-bar off --upgrade --pre"

python -m pip install $STD_ARGS pip
echo "PyQt6"
# Now broken in latest release and in the pre release:
# pip install $STD_ARGS --only-binary ":all:" --default-timeout=60 --extra-index-url https://www.riverbankcomputing.com/pypi/simple "PyQt6!=6.6.1,!=6.6.2" "PyQt6-Qt6!=6.6.1,!=6.6.2"
python -m pip install $STD_ARGS --only-binary ":all:" --default-timeout=60 "PyQt6!=6.6.1,!=6.6.2" "PyQt6-Qt6!=6.6.1,!=6.6.2"
echo "NumPy/SciPy/pandas etc."
python -m pip uninstall -yq numpy
# No pyarrow yet https://github.com/apache/arrow/issues/40216
# No h5py (and thus dipy) yet until they improve/refactor thier wheel building infrastructure for Windows
OTHERS=""
if [[ "${PLATFORM}" == "Linux" ]]; then
	OTHERS="h5py dipy git+https://github.com/BUNPC/pysnirf2"
fi
python -m pip install $STD_ARGS --only-binary ":all:" --default-timeout=60 \
	--index-url "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple" \
	"numpy>=2.1.0.dev0" "scipy>=1.14.0.dev0" "scikit-learn>=1.5.dev0" \
	matplotlib statsmodels pandas \
	git+https://github.com/joblib/joblib@master \
	$OTHERS

# No python-picard (needs numexpr) until they update to NumPy 2.0 compat
# No Numba because it forces an old NumPy version

echo "OpenMEEG"
python -m pip install $STD_ARGS --only-binary ":all:" --extra-index-url "https://test.pypi.org/simple" "openmeeg>=2.6.0.dev4"

echo "nilearn"
python -m pip install $STD_ARGS git+https://github.com/nilearn/nilearn

echo "VTK"
python -m pip install $STD_ARGS --only-binary ":all:" --extra-index-url "https://wheels.vtk.org" vtk
python -c "import vtk"

echo "PyVista"
python -m pip install $STD_ARGS git+https://github.com/drammock/pyvista@numpy-2-compat

echo "pyvistaqt"
pip install $STD_ARGS git+https://github.com/pyvista/pyvistaqt

echo "imageio-ffmpeg, xlrd, mffpy"
pip install $STD_ARGS imageio-ffmpeg xlrd mffpy patsy pillow traitlets pybv eeglabio

echo "mne-qt-browser"
pip install $STD_ARGS git+https://github.com/mne-tools/mne-qt-browser

echo "nibabel"
pip install $STD_ARGS git+https://github.com/nipy/nibabel.git

echo "edfio"
pip install $STD_ARGS git+https://github.com/the-siesta-group/edfio

echo "h5io"
pip install $STD_ARGS git+https://github.com/h5io/h5io

# Make sure we're on a NumPy 2.0 variant
echo "Checking NumPy version"
python -c "import numpy as np; assert np.__version__[0] == '2', np.__version__"

# And that Qt works
echo "Checking Qt"
./tools/check_qt_import.sh PyQt6
