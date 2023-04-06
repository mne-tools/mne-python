#!/bin/bash -ef

echo "Installing setuptools and sphinx"
python -m pip install --upgrade "pip!=20.3.0"
python -m pip install --upgrade --progress-bar off setuptools wheel
python -m pip install --upgrade --progress-bar off --pre sphinx
if [[ "$CIRCLE_JOB" == "linkcheck"* ]]; then
	echo "Installing minimal linkcheck dependencies"
	python -m pip install --progress-bar off pillow pytest -r requirements_base.txt -r requirements_doc.txt
else  # standard doc build
	echo "Installing doc build dependencies"
	python -m pip install --upgrade --progress-bar off PyQt6
	python -m pip install --upgrade --progress-bar off --only-binary "numpy,scipy,matplotlib,pandas,statsmodels" -r requirements.txt -r requirements_testing.txt -r requirements_doc.txt
	python -m pip install --upgrade --progress-bar off git+https://github.com/mne-tools/mne-qt-browser git+https://github.com/sphinx-gallery/sphinx-gallery
	# deal with comparisons and escapes (https://app.circleci.com/pipelines/github/mne-tools/mne-python/9686/workflows/3fd32b47-3254-4812-8b9a-8bab0d646d18/jobs/32934)
	python -m pip install --upgrade --progress-bar off quantities
fi
python -m pip install -e .
