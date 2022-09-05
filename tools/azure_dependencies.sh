#!/bin/bash -ef

EXTRA_ARGS=""
if [ "${TEST_MODE}" == "pip" ]; then
	python -m pip install --upgrade pip setuptools wheel
	python -m pip install --upgrade --only-binary ":all:" numpy scipy vtk
	python -m pip install --upgrade --only-binary="numba,llvmlite" -r requirements.txt
	# This can be removed once PyVistaQt 0.6 is out (including https://github.com/pyvista/pyvistaqt/pull/127)
	python -m pip install --upgrade https://github.com/pyvista/pyvistaqt/zipball/main
elif [ "${TEST_MODE}" == "pip-pre" ]; then
	python -m pip install --progress-bar off --upgrade pip setuptools wheel
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" python-dateutil pytz joblib threadpoolctl six cycler kiwisolver pyparsing patsy
	# XXX: the wheels for PyQt6-sip 13.4 are not available yet from https://www.riverbankcomputing.com/pypi/simple
	# python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --no-deps --extra-index-url https://www.riverbankcomputing.com/pypi/simple PyQt6 PyQt6-sip PyQt6-Qt6
	python -m pip install --progress-bar off --upgrade --only-binary ":all:" PyQt6 PyQt6-sip PyQt6-Qt6
	# SciPy Windows build is missing from conda nightly builds
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --no-deps -i "https://pypi.anaconda.org/scipy-wheels-nightly/simple" numpy
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --no-deps scipy vtk
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --no-deps -i "https://test.pypi.org/simple" "openmeeg==2.6.0.dev1"
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --no-deps --default-timeout=60 -i "https://pypi.anaconda.org/scipy-wheels-nightly/simple" scikit-learn dipy
	# as of 2022/08/29, pandas and statsmodels on scipy-wheels-nightly is broken for some reason (intermittent download issue?)
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --no-deps pandas statsmodels
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --no-deps -f "https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com" h5py Pillow matplotlib
	python -m pip install --progress-bar off https://github.com/pyvista/pyvista/zipball/main
	python -m pip install --progress-bar off https://github.com/pyvista/pyvistaqt/zipball/main
	python -m pip install --progress-bar off imageio-ffmpeg xlrd mffpy python-picard patsy
	EXTRA_ARGS="--pre"
else
	echo "Unknown run type ${TEST_MODE}"
	exit 1
fi
python -m pip install $EXTRA_ARGS .[test,hdf5] codecov
