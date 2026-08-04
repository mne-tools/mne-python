#!/bin/bash
set -eo pipefail

# Only measure coverage where it is cheap or uniquely valuable. On Python >= 3.14
# sys.monitoring is coverage's default core and can measure branches, making it
# nearly free. Below 3.14 coverage falls back to the C tracer (sys.monitoring
# cannot do branch coverage there), which roughly doubles Python-heavy test time,
# so we skip it -- except for the "minimal" and "old" kinds, which exercise code
# paths (missing optional dependencies, old pins) that no other job covers.
# The matrix Python isn't installed yet at this step, so key off $PYTHON_VERSION.
COV_ARGS="--cov=mne --cov-report=xml"
case "$MNE_CI_KIND" in
    minimal | old) ;;  # unique code paths, worth the slower C tracer
    *)
        case "$PYTHON_VERSION" in
            3.10 | 3.11 | 3.12 | 3.13) COV_ARGS="" ;;
        esac
        ;;
esac
echo "COV_ARGS=$COV_ARGS" | tee -a $GITHUB_ENV

# Number of pytest-xdist workers -- explicit ints (in the spirit of SciPy's CI)
# rather than "auto". macOS has  fewer cores and less RAM
if [[ "$CI_OS_NAME" == "macos"* ]]; then
    echo "PYTEST_XDIST_N=2" | tee -a $GITHUB_ENV
else
    echo "PYTEST_XDIST_N=4" | tee -a $GITHUB_ENV
fi

# old and minimal use conda
echo "::group::Setting pip env vars for $MNE_CI_KIND"
if [[ "$MNE_CI_KIND" == "pip"* ]]; then
    if [[ "$MNE_CI_KIND" == "pip-pre" ]]; then
        # We should test an eager import somewhere, might as well be here
        echo "EAGER_IMPORT=true" | tee -a $GITHUB_ENV
        # Make sure nothing unexpected is skipped
        echo "MNE_TEST_ALLOW_SKIP=.*(Requires (spm|brainstorm|misc) dataset|EAGER_IMPORT|CUDA not|Numba not|PySide6 causes segfaults|SCIPY_ARRAY_API).*" | tee -a $GITHUB_ENV
        echo "MNE_QT_BACKEND=PySide6" | tee -a $GITHUB_ENV
    elif [[ "$MNE_CI_KIND" == "pip" ]]; then
        if [[ "${RUNNER_OS}" == "macOS" ]]; then
            echo "MNE_TEST_ALLOW_SKIP=.*(Requires (spm|brainstorm|misc) dataset|SCIPY_ARRAY_API|FreeSurfer|MNE-C|CUDA not|macOS|PySide6 causes segfaults).*" | tee -a $GITHUB_ENV
        else
            echo "MNE_TEST_ALLOW_SKIP=.*(Requires (spm|brainstorm|misc) dataset|SCIPY_ARRAY_API|CUDA not|PySide6 causes segfaults).*" | tee -a $GITHUB_ENV
        fi
        echo "MNE_QT_BACKEND=PySide6" | tee -a $GITHUB_ENV
    elif [[ "$MNE_CI_KIND" == "pip-ft" ]]; then
        echo "No env vars to set"
    else
        echo "✕ ERROR: Unrecognized MNE_CI_KIND=${MNE_CI_KIND}"
        exit 1
    fi
elif [[ "$MNE_CI_KIND" == "minimal" ]]; then
    echo "MNE_QT_BACKEND=PySide6" | tee -a $GITHUB_ENV
elif [[ "$MNE_CI_KIND" == "old" ]]; then
    echo "MNE_IGNORE_WARNINGS_IN_TESTS=true" | tee -a $GITHUB_ENV
    echo "MNE_SKIP_NETWORK_TESTS=1" | tee -a $GITHUB_ENV
    echo "MNE_QT_BACKEND=PyQt6" | tee -a $GITHUB_ENV
elif [[ "$MNE_CI_KIND" == "conda" ]]; then
    echo "Setting conda env vars for $MNE_CI_KIND"
    echo "CONDA_ENV=environment.yml" | tee -a $GITHUB_ENV
    echo "MNE_LOGGING_LEVEL=warning" | tee -a $GITHUB_ENV
    echo "MNE_TEST_ALLOW_SKIP=.*(on conda|Requires (spm|brainstorm|misc) dataset|CUDA not|Flakey verbose behavior|PySide6 causes segfaults|SCIPY_ARRAY_API).*" | tee -a $GITHUB_ENV
    echo "MNE_QT_BACKEND=PySide6" | tee -a $GITHUB_ENV
else
    echo "✕ ERROR: Unrecognized MNE_CI_KIND=${MNE_CI_KIND}"
    exit 1
fi
if [[ "$CI_OS_NAME" == "windows"* ]]; then
    echo "MNE_IS_OSMESA=true" | tee -a $GITHUB_ENV
fi
echo "::endgroup::"
