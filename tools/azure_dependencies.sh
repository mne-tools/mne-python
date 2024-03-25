#!/bin/bash -ef

STD_ARGS="--progress-bar off --upgrade"
python -m pip install $STD_ARGS pip setuptools wheel packaging setuptools_scm
if [ "${TEST_MODE}" == "pip" ]; then
	python -m pip install --only-binary="numba,llvmlite,numpy,scipy,vtk" -e .[test,full]
elif [ "${TEST_MODE}" == "pip-pre" ]; then
	STD_ARGS="$STD_ARGS --pre"
	# python -m pip install $STD_ARGS --only-binary ":all:" --extra-index-url "https://www.riverbankcomputing.com/pypi/simple" "PyQt6!=6.6.1,!=6.6.2" PyQt6-sip PyQt6-Qt6 "PyQt6-Qt6!=6.6.1,!=6.6.2"
	python -m pip install $STD_ARGS --only-binary ":all:" "PyQt6!=6.6.1,!=6.6.2" PyQt6-sip PyQt6-Qt6 "PyQt6-Qt6!=6.6.1,!=6.6.2"
	echo "Numpy etc."
	# No pyarrow yet https://github.com/apache/arrow/issues/40216
	# No h5py (and thus dipy) yet until they improve/refactor thier wheel building infrastructure for Windows
	python -m pip install $STD_ARGS --only-binary ":all:" --default-timeout=60 --extra-index-url "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple" "numpy>=2.1.0.dev0" "scipy>=1.14.0.dev0" "scikit-learn>=1.5.dev0" matplotlib pillow statsmodels
	echo "OpenMEEG"
	pip install $STD_ARGS --only-binary ":all:" --extra-index-url "https://test.pypi.org/simple" "openmeeg>=2.6.0.dev4"
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
	echo "pysnirf2"
	python -m pip install $STD_ARGS git+https://github.com/BUNPC/pysnirf2
	./tools/check_qt_import.sh PyQt6
	python -m pip install $STD_ARGS -e .[test]
else
	echo "Unknown run type ${TEST_MODE}"
	exit 1
fi
