#!/bin/bash -ef

STD_ARGS="--progress-bar off --upgrade"
python -m pip install $STD_ARGS pip setuptools wheel packaging setuptools_scm
if [ "${TEST_MODE}" == "pip" ]; then
	python -m pip install --only-binary="numba,llvmlite,numpy,scipy,vtk" -e .[test,full]
elif [ "${TEST_MODE}" == "pip-pre" ]; then
	STD_ARGS="$STD_ARGS --pre"
	python -m pip install $STD_ARGS --only-binary ":all:" --extra-index-url "https://www.riverbankcomputing.com/pypi/simple" "PyQt6!=6.6.1" PyQt6-sip PyQt6-Qt6 "PyQt6-Qt6!=6.6.1"
	echo "Numpy etc."
	# See github_actions_dependencies.sh for comments
	# Until https://github.com/scipy/scipy/issues/19605 and
	# https://github.com/scipy/scipy/issues/19713 are resolved, we can't use the NumPy
	# 2.0 wheels :(
	python -m pip install $STD_ARGS --only-binary numpy scipy h5py
	python -m pip install $STD_ARGS --only-binary ":all:" --extra-index-url "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple" scikit-learn matplotlib statsmodels
	# echo "dipy"
	# python -m pip install $STD_ARGS --only-binary ":all:" --extra-index-url "https://pypi.anaconda.org/scipy-wheels-nightly/simple" dipy
	# echo "OpenMEEG"
	# pip install $STD_ARGS --only-binary ":all:" --extra-index-url "https://test.pypi.org/simple" openmeeg
	echo "vtk"
	python -m pip install $STD_ARGS --only-binary ":all:" --extra-index-url "https://wheels.vtk.org" vtk
	echo "nilearn"
	python -m pip install $STD_ARGS git+https://github.com/nilearn/nilearn
	echo "pyvista/pyvistaqt"
	python -m pip install --progress-bar off git+https://github.com/pyvista/pyvista
	python -m pip install --progress-bar off git+https://github.com/pyvista/pyvistaqt
	echo "misc"
	python -m pip install $STD_ARGS imageio-ffmpeg xlrd mffpy pillow traitlets pybv eeglabio
	echo "nibabel with workaround"
	python -m pip install --progress-bar off git+https://github.com/nipy/nibabel.git
	echo "joblib"
	python -m pip install --progress-bar off git+https://github.com/joblib/joblib@master
	echo "EDFlib-Python"
	python -m pip install $STD_ARGS git+https://github.com/the-siesta-group/edfio
	./tools/check_qt_import.sh PyQt6
	python -m pip install $STD_ARGS -e .[test]
else
	echo "Unknown run type ${TEST_MODE}"
	exit 1
fi
