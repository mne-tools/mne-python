#!/bin/bash -ef

STD_ARGS="--progress-bar off --upgrade"
if [ ! -z "$CONDA_ENV" ]; then
	echo "Uninstalling MNE for CONDA_ENV=${CONDA_ENV}"
	conda remove -c conda-forge --force -yq mne
	python -m pip uninstall -y mne
elif [ ! -z "$CONDA_DEPENDENCIES" ]; then
	echo "Using Mamba to install CONDA_DEPENDENCIES=${CONDA_DEPENDENCIES}"
	mamba install -y $CONDA_DEPENDENCIES
else
	echo "Install pip-pre dependencies"
	test "${MNE_CI_KIND}" == "pip-pre"
	STD_ARGS="$STD_ARGS --pre"
	python -m pip install $STD_ARGS pip setuptools wheel packaging
	echo "Numpy"
	pip uninstall -yq numpy
	echo "PyQt6"
	pip install $STD_ARGS --only-binary ":all:" --default-timeout=60 --extra-index-url https://www.riverbankcomputing.com/pypi/simple PyQt6
	echo "NumPy/SciPy/pandas etc."
	pip install $STD_ARGS --only-binary ":all:" --default-timeout=60 --extra-index-url "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple" "numpy>=2.0.0.dev0" scipy scikit-learn pandas matplotlib pillow statsmodels
	echo "dipy"
	pip install $STD_ARGS --only-binary ":all:" --default-timeout=60 --extra-index-url "https://pypi.anaconda.org/scipy-wheels-nightly/simple" dipy
	echo "H5py"
	pip install $STD_ARGS --only-binary ":all:" -f "https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com" h5py
	echo "OpenMEEG"
	pip install $STD_ARGS --only-binary ":all:" --extra-index-url "https://test.pypi.org/simple" openmeeg
	# No Numba because it forces an old NumPy version
	echo "nilearn and openmeeg"
	pip install $STD_ARGS git+https://github.com/nilearn/nilearn
	echo "VTK"
	pip install $STD_ARGS --only-binary ":all:" --extra-index-url "https://wheels.vtk.org" vtk
	python -c "import vtk"
	echo "PyVista"
	pip install $STD_ARGS git+https://github.com/pyvista/pyvista
	echo "pyvistaqt"
	pip install $STD_ARGS git+https://github.com/pyvista/pyvistaqt
	echo "imageio-ffmpeg, xlrd, mffpy, python-picard"
	pip install $STD_ARGS imageio-ffmpeg xlrd mffpy python-picard patsy
	echo "mne-qt-browser"
	pip install $STD_ARGS git+https://github.com/mne-tools/mne-qt-browser
	echo "nibabel with workaround"
	pip install $STD_ARGS git+https://github.com/mscheltienne/nibabel.git@np.sctypes
	echo "joblib"
	pip install $STD_ARGS git+https://github.com/joblib/joblib@master
	echo "EDFlib-Python"
	pip install $STD_ARGS git+https://gitlab.com/Teuniz/EDFlib-Python@master
fi
echo ""

# for compat_minimal and compat_old, we don't want to --upgrade
if [ ! -z "$CONDA_DEPENDENCIES" ]; then
	echo "Installing dependencies for conda"
	python -m pip install -r requirements_base.txt -r requirements_testing.txt
else
	echo "Installing dependencies using pip"
	python -m pip install $STD_ARGS -r requirements_base.txt -r requirements_testing.txt -r requirements_hdf5.txt
fi
echo ""

if [ "${DEPS}" != "minimal" ]; then
	echo "Installing non-minimal dependencies"
	python -m pip install $STD_ARGS -r requirements_testing_extra.txt
fi
