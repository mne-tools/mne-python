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
	echo "PyQt5"
	pip install $STD_ARGS --pre --only-binary ":all:" --no-deps --extra-index-url https://www.riverbankcomputing.com/pypi/simple PyQt5 PyQt5-sip PyQt5-Qt5
	echo "NumPy/SciPy/pandas etc."
	# TODO: Currently missing dipy for 3.10 https://github.com/dipy/dipy/issues/2489
	# TODO: Revert this and Azure once https://github.com/numpy/numpy/pull/21054 is merged and a new wheel is out
	pip install $STD_ARGS --pre --only-binary ":all:" --no-deps -i "https://pypi.anaconda.org/scipy-wheels-nightly/simple" numpy scipy pandas scikit-learn statsmodels
	echo "H5py, pillow, matplotlib"
	pip install $STD_ARGS --pre --only-binary ":all:" --no-deps -f "https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com" h5py pillow matplotlib
	# We don't install Numba here because it forces an old NumPy version
	echo "nilearn"
	pip install $STD_ARGS --pre https://github.com/nilearn/nilearn/zipball/main
	echo "VTK"
	# Have to use our own version until VTK releases a 3.10 build
	wget -q https://osf.io/hjyvx/download -O vtk-9.1.20211213.dev0-cp310-cp310-linux_x86_64.whl
	pip install $STD_ARGS --pre --only-binary ":all:" vtk-9.1.20211213.dev0-cp310-cp310-linux_x86_64.whl
	echo "PyVista"
	pip install --progress-bar off https://github.com/pyvista/pyvista/zipball/main
	echo "pyvistaqt"
	pip install --progress-bar off https://github.com/pyvista/pyvistaqt/zipball/main
	echo "imageio-ffmpeg, xlrd, mffpy"
	pip install --progress-bar off --pre imageio-ffmpeg xlrd mffpy
	if [ "$OSTYPE" == "darwin"* ]; then
	  echo "pyobjc-framework-Cocoa"
	  pip install --progress-bar off pyobjc-framework-Cocoa>=5.2.0
	fi
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
