#!/bin/bash -ef

set -o pipefail

STD_ARGS="--progress-bar off --upgrade"
INSTALL_ARGS="-e"
INSTALL_KIND="test_extra,hdf5"
if [ ! -z "$CONDA_ENV" ]; then
	echo "Uninstalling MNE for CONDA_ENV=${CONDA_ENV}"
	conda remove -c conda-forge --force -yq mne-base
	python -m pip uninstall -y mne
	if [[ "${RUNNER_OS}" != "Windows" ]]; then
		INSTALL_ARGS=""
	fi
elif [ ! -z "$CONDA_DEPENDENCIES" ]; then
	echo "Using Mamba to install CONDA_DEPENDENCIES=${CONDA_DEPENDENCIES}"
	mamba install -y $CONDA_DEPENDENCIES
	# for compat_minimal and compat_old, we don't want to --upgrade
	STD_ARGS="--progress-bar off"
	INSTALL_KIND="test"
else
	echo "Install pip-pre dependencies"
	test "${MNE_CI_KIND}" == "pip-pre"
	STD_ARGS="$STD_ARGS --pre"
	python -m pip install $STD_ARGS pip
	echo "Numpy"
	pip uninstall -yq numpy
	echo "PyQt6"
	# Now broken in latest release and in the pre release:
	# pip install $STD_ARGS --only-binary ":all:" --default-timeout=60 --extra-index-url https://www.riverbankcomputing.com/pypi/simple "PyQt6!=6.6.1,!=6.6.2" "PyQt6-Qt6!=6.6.1,!=6.6.2"
	pip install $STD_ARGS --only-binary ":all:" --default-timeout=60 "PyQt6!=6.6.1,!=6.6.2" "PyQt6-Qt6!=6.6.1,!=6.6.2"
	echo "NumPy/SciPy/pandas etc."
	pip install $STD_ARGS --only-binary ":all:" --default-timeout=60 --extra-index-url "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple" "numpy>=2.0.0.dev0" "scipy>=1.12.0.dev0" "scikit-learn>=1.5.dev0" matplotlib pillow statsmodels pyarrow h5py
	# No pandas, dipy, openmeeg, python-picard (needs numexpr) until they update to NumPy 2.0 compat
	INSTALL_KIND="test_extra"
	# echo "dipy"
	# pip install $STD_ARGS --only-binary ":all:" --default-timeout=60 --extra-index-url "https://pypi.anaconda.org/scipy-wheels-nightly/simple" dipy
	# echo "OpenMEEG"
	# pip install $STD_ARGS --only-binary ":all:" --extra-index-url "https://test.pypi.org/simple" openmeeg
	# No Numba because it forces an old NumPy version
	echo "nilearn"
	pip install $STD_ARGS git+https://github.com/nilearn/nilearn
	echo "VTK"
	pip install $STD_ARGS --only-binary ":all:" --extra-index-url "https://wheels.vtk.org" vtk
	python -c "import vtk"
	echo "PyVista"
	pip install $STD_ARGS git+https://github.com/drammock/pyvista@numpy-2-compat
	echo "pyvistaqt"
	pip install $STD_ARGS git+https://github.com/pyvista/pyvistaqt
	echo "imageio-ffmpeg, xlrd, mffpy"
	pip install $STD_ARGS imageio-ffmpeg xlrd mffpy patsy traitlets pybv eeglabio
	echo "mne-qt-browser"
	pip install $STD_ARGS git+https://github.com/mne-tools/mne-qt-browser
	echo "nibabel with workaround"
	pip install $STD_ARGS git+https://github.com/nipy/nibabel.git
	echo "joblib"
	pip install $STD_ARGS git+https://github.com/joblib/joblib@master
	echo "edfio"
	pip install $STD_ARGS git+https://github.com/the-siesta-group/edfio
	# Make sure we're on a NumPy 2.0 variant
	python -c "import numpy as np; assert np.__version__[0] == '2', np.__version__"
fi
echo ""

echo "Installing test dependencies using pip"
python -m pip install $STD_ARGS $INSTALL_ARGS .[$INSTALL_KIND]
