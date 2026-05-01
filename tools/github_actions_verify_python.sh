#!/bin/bash -ef

set -eo pipefail

GOT_PYTHON=$(which python)
echo "Checking Python found at:"
echo "  \$(which python) == ${GOT_PYTHON}"
echo "for"
echo "  \$MNE_CI_KIND   == ${MNE_CI_KIND}"
if [[ "${MNE_CI_KIND}" == "conda" ]] || [[ "${MNE_CI_KIND}" == "mamba" ]]; then
	WANT="micromamba/envs/mne"
elif [[ "${MNE_CI_KIND}" == "old" ]]; then
	WANT="mne-python/mne-python/.venv/bin"
elif [[ "${MNE_CI_KIND}" == "pip" ]] || [[ "${MNE_CI_KIND}" == "pip-pre" ]]; then
	WANT="hostedtoolcache/Python"
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
