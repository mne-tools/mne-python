#!/bin/bash -ef

EXTRA_ARGS=""
if [ "${TEST_MODE}" == "pip" ]; then
	python -m pip install --upgrade pip setuptools wheel
	python -m pip install --upgrade --only-binary="numba,llvmlite,numpy,scipy,vtk" -r requirements.txt
elif [ "${TEST_MODE}" == "pip-pre" ]; then
	python -m pip install --progress-bar off --upgrade pip setuptools wheel packaging setuptools_scm
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --extra-index-url "https://www.riverbankcomputing.com/pypi/simple" PyQt6 PyQt6-sip PyQt6-Qt6
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --extra-index-url "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple" numpy scipy statsmodels pandas scikit-learn matplotlib
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --extra-index-url "https://pypi.anaconda.org/scipy-wheels-nightly/simple" dipy
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" -f "https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com" h5py
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --extra-index-url "https://wheels.vtk.org" vtk
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --extra-index-url "https://test.pypi.org/simple" openmeeg
	python -m pip install --progress-bar off git+https://github.com/pyvista/pyvista
	python -m pip install --progress-bar off git+https://github.com/pyvista/pyvistaqt
	python -m pip install --progress-bar off --upgrade --pre imageio-ffmpeg xlrd mffpy python-picard pillow
	EXTRA_ARGS="--pre"
	./tools/check_qt_import.sh PyQt6
else
	echo "Unknown run type ${TEST_MODE}"
	exit 1
fi
python -m pip install $EXTRA_ARGS .[test,hdf5]
