#!/bin/bash -ef

python -m pip uninstall -y pydata-sphinx-theme
python -m pip install --user --upgrade --progress-bar off pip setuptools
python -m pip install --user --upgrade --progress-bar off --pre sphinx
python -m pip install --user --upgrade --progress-bar off -r requirements.txt -r requirements_testing.txt -r requirements_doc.txt
python -m pip install --user --progress-bar off https://github.com/pyvista/pyvista/zipball/master
python -m pip install --user --progress-bar off https://github.com/pyvista/pyvistaqt/zipball/master
python -m pip uninstall -yq pysurfer mayavi
python -m pip install --user -e .
