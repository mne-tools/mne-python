#!/bin/bash -ef

if [ ! -z "$CONDA_ENV" ]; then
	conda env update --file $CONDA_ENV
	conda activate mne
	pip uninstall -yq mne
elif [ ! -z "$CONDA_DEPENDENCIES" ]; then
	conda install -y $CONDA_DEPENDENCIES
else # pip
	python -m pip install --upgrade pip setuptools wheel
	pip uninstall -yq numpy
	pip install -i "https://pypi.anaconda.org/scipy-wheels-nightly/simple" --pre "numpy!=1.20.0.dev0+20201111233731.0ffaaf8,!=1.20.0.dev0+20201111232921.0ffaaf8"
	pip install -f "https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com" scipy pandas scikit-learn matplotlib h5py Pillow
	pip install https://github.com/pyvista/pyvista/zipball/master
	pip install https://github.com/pyvista/pyvistaqt/zipball/master
fi
pip install --upgrade -r requirements_testing.txt
if [ "${DEPS}" != "minimal" ]; then
	pip install nitime
fi
