#!/bin/bash -ef

STD_ARGS="--progress-bar off --upgrade"
EXTRA_ARGS=""
if [ ! -z "$CONDA_ENV" ]; then
	pip uninstall -yq mne
elif [ ! -z "$CONDA_DEPENDENCIES" ]; then
	conda install -y $CONDA_DEPENDENCIES
else
	# Changes here should also go in the interactive_test CircleCI job
	python -m pip install $STD_ARGS pip setuptools wheel
	echo "Numpy"
	pip uninstall -yq numpy
	echo "Date utils"
	# https://pip.pypa.io/en/latest/user_guide/#possible-ways-to-reduce-backtracking-occurring
	pip install $STD_ARGS --pre --only-binary ":all:" python-dateutil pytz joblib threadpoolctl six
	echo "PyQt6"
	# Broken as of 2022/09/20
	# pip install $STD_ARGS --pre --only-binary ":all:" --no-deps --extra-index-url https://www.riverbankcomputing.com/pypi/simple PyQt6 PyQt6-sip PyQt6-Qt6
	pip install $STD_ARGS --pre --only-binary ":all:" --no-deps PyQt6 PyQt6-sip PyQt6-Qt6
	echo "NumPy/SciPy/pandas etc."
	pip install $STD_ARGS --pre --only-binary ":all:" "matplotlib<3.7"  # gh-11332
	pip install $STD_ARGS --pre --only-binary ":all:" --no-deps  --default-timeout=60 -i "https://pypi.anaconda.org/scipy-wheels-nightly/simple" numpy scipy scikit-learn dipy pandas statsmodels
	pip install $STD_ARGS --pre --only-binary ":all:" --no-deps -f "https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com" h5py
	pip install $STD_ARGS --pre --only-binary ":all:" pillow
	# We don't install Numba here because it forces an old NumPy version
	echo "nilearn and openmeeg"
	pip install $STD_ARGS --pre git+https://github.com/nilearn/nilearn
	pip install $STD_ARGS --pre --only-binary ":all:" -i "https://test.pypi.org/simple" openmeeg
	echo "VTK"
	pip install $STD_ARGS --pre --only-binary ":all:" vtk
	python -c "import vtk"
	echo "PyVista"
	pip install --progress-bar off git+https://github.com/pyvista/pyvista
	echo "pyvistaqt"
	pip install --progress-bar off git+https://github.com/pyvista/pyvistaqt
	echo "imageio-ffmpeg, xlrd, mffpy, python-picard"
	pip install --progress-bar off --pre imageio-ffmpeg xlrd mffpy python-picard patsy
	if [ "$OSTYPE" == "darwin"* ]; then
	  echo "pyobjc-framework-Cocoa"
	  pip install --progress-bar off pyobjc-framework-Cocoa>=5.2.0
	fi
	echo "mne-qt-browser"
	pip install --progress-bar off git+https://github.com/mne-tools/mne-qt-browser
	EXTRA_ARGS="--pre"
fi
# for compat_minimal and compat_old, we don't want to --upgrade
if [ ! -z "$CONDA_DEPENDENCIES" ]; then
	pip install -r requirements_base.txt -r requirements_testing.txt
else
	pip install $STD_ARGS $EXTRA_ARGS -r requirements_base.txt -r requirements_testing.txt -r requirements_hdf5.txt
fi

if [ "${DEPS}" != "minimal" ]; then
	pip install $STD_ARGS $EXTRA_ARGS -r requirements_testing_extra.txt
fi
