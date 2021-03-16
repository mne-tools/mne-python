#!/bin/bash -ef

# Work around PyQt5 bug
sudo ln -s /usr/lib/x86_64-linux-gnu/libxcb-util.so.0 /usr/lib/x86_64-linux-gnu/libxcb-util.so.1
python -m pip uninstall -y pydata-sphinx-theme
python -m pip install --user --upgrade --progress-bar off pip setuptools
python -m pip install --user --upgrade --progress-bar off --pre sphinx
python -m pip install --user --upgrade --progress-bar off -r requirements.txt -r requirements_testing.txt -r requirements_doc.txt
python -m pip install --user --progress-bar off https://github.com/pyvista/pyvista/zipball/master
python -m pip install --user --progress-bar off https://github.com/pyvista/pyvistaqt/zipball/master
python -m pip uninstall -yq pysurfer mayavi
python -m pip install --user -e .
