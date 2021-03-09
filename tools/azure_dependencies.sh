#!/bin/bash -ef

if [ "${TEST_MODE}" == "pip" ]; then
	python -m pip install --upgrade pip setuptools
	python -m pip install --upgrade --only-binary ":all:" numpy scipy vtk
	python -m pip install --upgrade --only-binary="numba,llvmlite" -r requirements.txt
elif [ "${TEST_MODE}" == "pip-pre" ]; then
	python -m pip install --progress-bar off --upgrade pip setuptools
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" -i "https://pypi.anaconda.org/scipy-wheels-nightly/simple" --extra-index-url https://www.riverbankcomputing.com/pypi/simple numpy scipy pandas scikit-learn PyQt5
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" -f "https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com" h5py Pillow
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all" vtk
	python -m pip install --progress-bar off --upgrade --only-binary ":all" matplotlib
	python -m pip install --progress-bar off https://github.com/pyvista/pyvista/zipball/master
	python -m pip install --progress-bar off https://github.com/pyvista/pyvistaqt/zipball/master
	python -m pip install --progress-bar off --upgrade --only-binary="numba,llvmlite" -r requirements.txt
else
	echo "Unknown run type ${TEST_MODE}"
	exit 1
fi
python -m pip install -r requirements_testing.txt codecov
