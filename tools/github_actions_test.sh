#!/bin/bash -ef

python setup.py build
python setup.py install
mne sys_info
python -c "import numpy; numpy.show_config()"
python -c 'import mne; mne.datasets.testing.data_path(verbose=True)';
echo "Print locale "
locale
echo "Other stuff"
USE_DIRS="mne/"
if [ "${TRAVIS_OS_NAME}" != "osx" ]; then
  CONDITION="not ultraslowtest"
else
  CONDITION="not slowtest"
fi
echo 'pytest -m "${CONDITION}" --tb=short --cov=mne --cov-report xml -vv ${USE_DIRS}'
pytest -m "${CONDITION}" --tb=short --cov=mne --cov-report xml -vv ${USE_DIRS}
