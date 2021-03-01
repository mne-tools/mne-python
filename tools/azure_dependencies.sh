#!/bin/bash -ef

if [ "${TEST_MODE}" == "conda" ]; then
	conda update -n base -c defaults conda
	conda env update --name base --file environment.yml
	pip uninstall -yq mne
elif [ "${TEST_MODE}" == "pip" ]; then
	python -m pip install --upgrade pip setuptools
	python -m pip install --upgrade numpy scipy vtk
	python -m pip install --use-deprecated=legacy-resolver --only-binary="numba,llvmlite" -r requirements.txt
elif [ "${TEST_MODE}" == "pip-pre" ]; then
	python -m pip install --upgrade pip setuptools
	python -m pip install --use-deprecated=legacy-resolver --upgrade --pre --only-binary ":all:" -i "https://pypi.anaconda.org/scipy-wheels-nightly/simple" numpy
	python -m pip install --use-deprecated=legacy-resolver --upgrade --pre --only-binary ":all:" -f "https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com" scipy pandas scikit-learn h5py Pillow
	python -m pip install --use-deprecated=legacy-resolver --upgrade --pre --only-binary ":all:" matplotlib
	python -m pip install --upgrade --only-binary vtk vtk;
	python -m pip install https://github.com/pyvista/pyvista/zipball/master
	python -m pip install https://github.com/pyvista/pyvistaqt/zipball/master
	python -m pip install --use-deprecated=legacy-resolver --only-binary="numba,llvmlite" -r requirements.txt
else
	echo "Unknown run type ${TEST_MODE}"
	exit 1
fi
python -m pip install -r requirements_testing.txt codecov
