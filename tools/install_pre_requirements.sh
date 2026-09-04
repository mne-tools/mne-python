#!/bin/bash

set -eo pipefail

PLATFORM=$(python -c 'import platform; print(platform.system())')

echo "Installing pip-pre dependencies on ${PLATFORM}"
# uv rather than pip: it downloads in parallel and caches the wheels it builds
# for the git/archive deps below by resolved commit, so a warm UV_CACHE_DIR
# skips those builds entirely (~180 s -> ~1 s for the "Everything else" set).
python -m pip install --progress-bar off --upgrade "uv>=0.9"
export UV_SYSTEM_PYTHON=1  # CI gives us a bare interpreter, not a venv
export UV_NO_PROGRESS=1
# Many deps below are pulled from GitHub/GitLab archives (codeload.github.com),
# which intermittently stalls mid-download and yields a fatal read timeout.
export UV_HTTP_TIMEOUT=60
STD_ARGS="--upgrade --prerelease=allow"
if [[ "$MNE_QT_BACKEND" == "" ]]; then
	MNE_QT_BACKEND="PySide6"
fi

# Dependencies of scientific-python-nightly-wheels are installed here so that
# we can use strict --index-url (instead of --extra-index-url) below
set -x
echo "::group::Prerequisites"
uv pip install $STD_ARGS pip setuptools packaging \
	threadpoolctl cycler fonttools kiwisolver pyparsing pillow python-dateutil \
	patsy pytz tzdata nibabel tqdm trx-python joblib numexpr \
	"$MNE_QT_BACKEND!=6.9.1" \
	py-cpuinfo blosc2 hatchling "formulaic>=1.1.0" \
	scikit-learn tables
uv pip uninstall numpy
echo "::endgroup::"
echo "::group::Scientific Python Nightly Wheels"
uv pip install $STD_ARGS --only-binary ":all:" \
	--index-url "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple" \
	"numpy>=2.5.0.dev0" \
	"scipy>=1.18.0.dev0" \
	"pandas>=3.1.0.dev0" \
	"dipy>=1.12.0.dev0" \
	"pyarrow>=22.0.0.dev0" \
	"matplotlib>=3.11.0.dev0" \
	"statsmodels>=0.15.0.dev0" \
	"h5py>=3.13.0"
# https://github.com/scikit-learn/scikit-learn/issues/34458
#	"scikit-learn>=1.9.dev0" \
# https://github.com/PyTables/PyTables/issues/1338
#	"tables>=3.10.3.dev0" \
echo "::endgroup::"
# No Numba because it forces an old NumPy version

echo "::group::VTK"
# unsafe-best-match because uv's default stops at the first index carrying vtk (PyPI)
uv pip install $STD_ARGS --only-binary ":all:" --index-strategy unsafe-best-match --extra-index-url "https://wheels.vtk.org" "vtk>=9.6.20260517.dev0,!=9.6.20260601,!=9.6.20260618"
python -c "import vtk"
echo "::endgroup::"

# nilearn and edfio version via hatch-vcs and shipped no .git_archival.txt, so a
# codeload archive had no version to resolve and the build failed outright; the
# forks below exist only to carry that file. TODO: point both back at upstream
# main once those PRs merge (nilearn's clone alone was ~150 MB of the uv cache).
echo "::group::Everything else"
uv pip install $STD_ARGS \
	"pyvista @ https://github.com/pyvista/pyvista/archive/refs/heads/main.zip" \
	"pyvistaqt @ https://github.com/pyvista/pyvistaqt/archive/refs/heads/main.zip" \
	"nilearn @ https://github.com/larsoner/nilearn/archive/refs/heads/gitattr.zip" \
	"edfio @ https://github.com/larsoner/edfio/archive/refs/heads/gitattr.zip" \
	"python-picard @ https://github.com/pierreablin/picard/archive/refs/heads/master.zip" \
	"pymatreader @ https://gitlab.com/obob/pymatreader/-/archive/master/pymatreader-master.zip" \
	"pyqtgraph @ https://github.com/pyqtgraph/pyqtgraph/archive/refs/heads/master.zip" \
	"mne-qt-browser @ https://github.com/mne-tools/mne-qt-browser/archive/refs/heads/main.zip" \
	"mne-bids @ https://github.com/mne-tools/mne-bids/archive/refs/heads/main.zip" \
	"nibabel @ https://github.com/nipy/nibabel/archive/refs/heads/master.zip" \
	"nitime @ https://github.com/nipy/nitime/archive/refs/heads/master.zip" \
	"joblib @ https://github.com/joblib/joblib/archive/refs/heads/main.zip" \
	"h5io @ https://github.com/h5io/h5io/archive/refs/heads/main.zip" \
	"snirf @ https://github.com/BUNPC/pysnirf2/archive/refs/heads/main.zip" \
	trame trame-vtk "trame-vuetify!=3.2.3" trame-pyvista nest-asyncio2 jupyter ipyevents ipympl \
	openmeeg imageio-ffmpeg xlrd mffpy traitlets pybv eeglabio defusedxml antio curryreader \
	jamica filelock
echo "::endgroup::"

echo "::group::Make sure we're on a NumPy 2.0 variant"
python -c "import numpy as np; assert np.__version__[0] == '2', np.__version__"
echo "::endgroup::"

echo "::group::Check Qt import"
curl -fsSL https://raw.githubusercontent.com/mne-tools/mne-tools/main/tools/check_qt_import.sh | bash -s -- "$MNE_QT_BACKEND"
echo "::endgroup::"
