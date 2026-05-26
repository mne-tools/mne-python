#!/bin/bash -ef

set -eo pipefail

WANT_PYTHON_VERSION="$1"
if [[ -z "$WANT_PYTHON_VERSION" ]]; then
	echo "✕ ERROR: Missing required argument: want Python version (e.g., 3.10)"
	exit 1
fi

GOT_PYTHON=$(which python)
GOT_PYTHON_VERSION=$(python --version)
echo "Checking Python found at:"
echo "  \$(which python)     == ${GOT_PYTHON}"
echo "  \$(python --version) == ${GOT_PYTHON_VERSION}"
echo "for"
echo "  \$MNE_CI_KIND        == ${MNE_CI_KIND}"
if [[ "${MNE_CI_KIND}" == "conda" ]]; then
	WANT="micromamba/envs/mne"
elif [[ "${MNE_CI_KIND}" == "old" ]]; then
	WANT="mne-python/mne-python/.venv/bin"
elif [[ "${MNE_CI_KIND}" == "pip" ]] || [[ "${MNE_CI_KIND}" == "pip-pre" ]] || [[ "${MNE_CI_KIND}" == "minimal" ]]; then
	WANT="/hostedtoolcache/"
else
	echo "✕ ERROR: Unrecognized MNE_CI_KIND=${MNE_CI_KIND}"
	exit 1
fi
if [[ "${GOT_PYTHON}" != *"${WANT}"* ]]; then
	echo "✕ ERROR: Did not find \"${WANT}\" from PATH:"
	tr ':' '\n' <<< "$PATH"
	exit 1
else
	echo "☑ Found expected \"${WANT}\""
fi
if [[ "${GOT_PYTHON_VERSION}" != *"${WANT_PYTHON_VERSION}"* ]]; then
	echo "✕ ERROR: Did not find expected Python version \"${WANT_PYTHON_VERSION}\""
	exit 1
else
	echo "☑ Found expected Python version \"${WANT_PYTHON_VERSION}\""
fi