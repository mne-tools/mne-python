#!/bin/bash -ef

if [ "${TEST_MODE}" == "pip" ]; then
	python -m pip install --upgrade pip setuptools
	python -m pip install --upgrade --only-binary ":all:" numpy scipy vtk
	python -m pip install --upgrade --only-binary="numba,llvmlite" -r requirements.txt
elif [ "${TEST_MODE}" == "pip-pre" ]; then
	python -m pip install --progress-bar off --upgrade pip setuptools
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" python-dateutil pytz joblib threadpoolctl six cycler kiwisolver pyparsing patsy
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --no-deps --extra-index-url https://www.riverbankcomputing.com/pypi/simple PyQt5 PyQt5-sip PyQt5-Qt5
	# SciPy Windows build is missing from conda nightly builds, and statsmodels does not work
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --no-deps -i "https://pypi.anaconda.org/scipy-wheels-nightly/simple" numpy
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --no-deps scipy statsmodels
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --no-deps -i "https://pypi.anaconda.org/scipy-wheels-nightly/simple" pandas scikit-learn dipy
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" --no-deps -f "https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com" h5py Pillow matplotlib
	# we want --pre for VTK, but it breaks Azure as of 2021/06/14 (mayavi has problems with the 20210612 binary)
	python -m pip install --progress-bar off --upgrade --only-binary ":all" "vtk<=9.0.1"
	python -m pip install --progress-bar off https://github.com/pyvista/pyvista/zipball/main
	python -m pip install --progress-bar off https://github.com/pyvista/pyvistaqt/zipball/main
else
	echo "Unknown run type ${TEST_MODE}"
	exit 1
fi
python -m pip install -r requirements_testing.txt -r requirements_testing_extra.txt codecov
