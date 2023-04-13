#!/bin/bash -ef

EXTRA_ARGS=""
if [ "${TEST_MODE}" == "pip" ]; then
	python -m pip install --upgrade pip setuptools wheel
	python -m pip install --upgrade --only-binary ":all:" numpy scipy vtk
	python -m pip install --upgrade --only-binary="numba,llvmlite" -r requirements.txt
	# This can be removed once PyVistaQt 0.6 is out (including https://github.com/pyvista/pyvistaqt/pull/127)
	python -m pip install --upgrade git+https://github.com/pyvista/pyvistaqt
elif [ "${TEST_MODE}" == "pip-pre" ]; then
	python -m pip install --progress-bar off --upgrade pip setuptools wheel
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" python-dateutil pytz joblib threadpoolctl six cycler kiwisolver pyparsing patsy
	# Broken as of 2022/09/20
	# python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --no-deps --extra-index-url https://www.riverbankcomputing.com/pypi/simple PyQt6 PyQt6-sip PyQt6-Qt6
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --no-deps PyQt6 PyQt6-sip PyQt6-Qt6
	# Wait for https://github.com/scipy/scipy/issues/17811
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --no-deps numpy
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --no-deps -i "https://pypi.anaconda.org/scipy-wheels-nightly/simple" scipy statsmodels pandas scikit-learn dipy matplotlib
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --no-deps -f "https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com" h5py
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --no-deps -i "https://test.pypi.org/simple" openmeeg
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --no-deps vtk
	python -m pip install --progress-bar off git+https://github.com/pyvista/pyvista
	python -m pip install --progress-bar off git+https://github.com/pyvista/pyvistaqt
	python -m pip install --progress-bar off --upgrade --pre imageio-ffmpeg xlrd mffpy python-picard patsy pillow
	EXTRA_ARGS="--pre"
	./tools/check_qt_import.sh PyQt6
else
	echo "Unknown run type ${TEST_MODE}"
	exit 1
fi
python -m pip install $EXTRA_ARGS .[test,hdf5]
