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
	# XXX: the wheels for PyQt6-sip 13.4 are not available yet from https://www.riverbankcomputing.com/pypi/simple
	# pip install $STD_ARGS --pre --only-binary ":all:" --no-deps --extra-index-url https://www.riverbankcomputing.com/pypi/simple PyQt6 PyQt6-sip PyQt6-Qt6
	pip install $STD_ARGS --only-binary ":all:" PyQt6 PyQt6-sip PyQt6-Qt6
	echo "NumPy/SciPy/pandas etc."
	pip install $STD_ARGS --pre --only-binary ":all:" --no-deps  --default-timeout=60 -i "https://pypi.anaconda.org/scipy-wheels-nightly/simple" numpy scipy scikit-learn dipy
	# as of 2022/08/29, pandas and statsmodels on scipy-wheels-nightly is broken for some reason (intermittent download issue?)
	pip install $STD_ARGS --pre --only-binary ":all:" pandas statsmodels
	echo "H5py, pillow, matplotlib"
	pip install $STD_ARGS --pre --only-binary ":all:" --no-deps -f "https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com" h5py pillow matplotlib
	# We don't install Numba here because it forces an old NumPy version
	echo "nilearn and openmeeg"
	pip install $STD_ARGS --pre https://github.com/nilearn/nilearn/zipball/main
	pip install $STD_ARGS --pre --only-binary ":all:" -i "https://test.pypi.org/simple" "openmeeg==2.6.0.dev1"
	echo "VTK"
	# Have to use our own version until VTK releases a 3.10 build
	wget -q https://osf.io/ajder/download -O vtk-9.1.20220406.dev0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
	pip install $STD_ARGS --pre --only-binary ":all:" vtk-9.1.20220406.dev0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
	python -c "import vtk"
	echo "PyVista"
	pip install --progress-bar off https://github.com/pyvista/pyvista/zipball/main
	echo "pyvistaqt"
	pip install --progress-bar off https://github.com/pyvista/pyvistaqt/zipball/main
	echo "imageio-ffmpeg, xlrd, mffpy, python-picard"
	pip install --progress-bar off --pre imageio-ffmpeg xlrd mffpy python-picard patsy
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
