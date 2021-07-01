#!/bin/bash -ef

if [ ! -z "$CONDA_ENV" ]; then
	pip uninstall -yq mne
elif [ ! -z "$CONDA_DEPENDENCIES" ]; then
	conda install -y $CONDA_DEPENDENCIES
else # pip --pre 3.9 (missing dipy in pre)
	# Changes here should also go in the interactive_test CircleCI job
	python -m pip install --progress-bar off --upgrade "pip!=20.3.0" setuptools wheel
	pip uninstall -yq numpy
	pip install --progress-bar off --upgrade --pre --only-binary ":all:" python-dateutil pytz joblib threadpoolctl
	pip install --progress-bar off --upgrade --pre --only-binary ":all:" -i "https://pypi.anaconda.org/scipy-wheels-nightly/simple" --extra-index-url https://www.riverbankcomputing.com/pypi/simple numpy scipy pandas scikit-learn PyQt5
	pip install --progress-bar off --upgrade --pre --only-binary ":all:" -f "https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com" h5py pillow matplotlib
	pip install --progress-bar off --upgrade --pre --only-binary ":all:" numba llvmlite
	# built using vtk master branch on an Ubuntu 18.04.5 VM and uploaded to OSF:
	wget -q https://osf.io/kej3v/download -O vtk-9.0.20201117-cp39-cp39-linux_x86_64.whl
	pip install --progress-bar off vtk-9.0.20201117-cp39-cp39-linux_x86_64.whl
	echo "PyVista"
	pip install --progress-bar off https://github.com/pyvista/pyvista/zipball/main
	echo "pyvistaqt"
	pip install --progress-bar off https://github.com/pyvista/pyvistaqt/zipball/main
	echo "imageio-ffmpeg, xlrd, mffpy"
	pip install --progress-bar off --pre mayavi imageio-ffmpeg xlrd mffpy
fi
pip install --progress-bar off --upgrade -r requirements_testing.txt
if [ "${DEPS}" != "minimal" ]; then
	pip install --progress-bar off --upgrade -r requirements_testing_extra.txt
fi
