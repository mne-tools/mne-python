#!/bin/bash -ef

echo "Working around PyQt5 bugs"
# https://github.com/ContinuumIO/anaconda-issues/issues/9190#issuecomment-386508136
# https://github.com/golemfactory/golem/issues/1019
sudo apt update
sudo apt install libosmesa6 libglx-mesa0 libopengl0 libglx0 libdbus-1-3 \
	libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
	libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 libxcb-xinerama0 \
	graphviz optipng
sudo ln -s /usr/lib/x86_64-linux-gnu/libxcb-util.so.0 /usr/lib/x86_64-linux-gnu/libxcb-util.so.1

echo "Installing setuptools and sphinx"
python -m pip install --progress-bar off --upgrade "pip!=20.3.0" setuptools wheel
python -m pip install --upgrade --progress-bar off --pre sphinx
if [[ "$CIRCLE_JOB" == "interactive_test" ]]; then
	echo "Installing latest dependencies for interactive_test"
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" python-dateutil pytz joblib threadpoolctl
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" -i "https://pypi.anaconda.org/scipy-wheels-nightly/simple" --extra-index-url https://www.riverbankcomputing.com/pypi/simple numpy scipy pandas scikit-learn PyQt5
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" -f "https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com" h5py pillow matplotlib
	python -m pip install --progress-bar off --upgrade --pre --only-binary ":all:" numba llvmlite
	wget -q https://osf.io/kej3v/download -O vtk-9.0.20201117-cp39-cp39-linux_x86_64.whl
	python -m pip install --progress-bar off vtk-9.0.20201117-cp39-cp39-linux_x86_64.whl
	python -m pip install --progress-bar off https://github.com/pyvista/pyvista/zipball/main
	python -m pip install --progress-bar off https://github.com/pyvista/pyvistaqt/zipball/main
	python -m pip install --progress-bar off --upgrade -r requirements_testing.txt -r requirements_testing_extra.txt
	python -m pip install -e .
elif [[ "$CIRCLE_JOB" == "linkcheck"* ]]; then
	echo "Installing minimal linkcheck dependencies"
	python -m pip install --progress-bar off numpy scipy matplotlib pillow pytest
	python -m pip install -e .
	python -m pip install --progress-bar off -r requirements_doc.txt
else  # standard doc build
	echo "Installing doc build dependencies"
	python -m pip uninstall -y pydata-sphinx-theme
	python -m pip install --upgrade --progress-bar off -r requirements.txt -r requirements_testing.txt -r requirements_doc.txt
	python -m pip install --progress-bar off https://github.com/sphinx-gallery/sphinx-gallery/zipball/master https://github.com/pyvista/pyvista/zipball/main https://github.com/pyvista/pyvistaqt/zipball/main
	python -m pip uninstall -yq pysurfer mayavi
	python -m pip install -e .
fi
