#!/bin/bash -ef

if [ ! -z "$CONDA_ENV" ]; then
	pip uninstall -yq mne
elif [ ! -z "$CONDA_DEPENDENCIES" ]; then
	conda install -y $CONDA_DEPENDENCIES
else # pip 3.9
	python -m pip install --upgrade pip setuptools wheel
	pip uninstall -yq numpy
	pip install --upgrade --pre --only-binary ":all:" python-dateutil pytz joblib threadpoolctl
	pip install --upgrade --pre --only-binary ":all:" -i "https://pypi.anaconda.org/scipy-wheels-nightly/simple" numpy scipy pandas scikit-learn
	pip install --upgrade --pre --only-binary ":all:" -f "https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com" matplotlib
	wget https://osf.io/kej3v/download -O vtk-9.0.20201117-cp39-cp39-linux_x86_64.whl
	pip install vtk-9.0.20201117-cp39-cp39-linux_x86_64.whl
	pip install https://github.com/pyvista/pyvista/zipball/master
	pip install https://github.com/pyvista/pyvistaqt/zipball/master
fi
pip install --upgrade -r requirements_testing.txt
if [ "${DEPS}" != "minimal" ]; then
	pip install nitime
fi
