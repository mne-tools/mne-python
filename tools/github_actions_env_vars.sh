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
            3.11 | 3.12 | 3.13) COV_ARGS="" ;;
        esac
        ;;
esac
echo "COV_ARGS=$COV_ARGS" | tee -a $GITHUB_ENV

# Where we do measure coverage, use the sys.monitoring core rather than the C
# tracer: it produces identical line and branch results for a fraction of the
# overhead. It needs Python >= 3.14 to measure branches (we always pass
# --cov-branch, see the pytest addopts), and coverage's free-threading support is
# still settling, so those jobs stay on the C tracer. When sysmon is unusable
# coverage falls back on its own, but only after a CoverageWarning that our
# warning filters would turn into an error, so don't ask for it there.
if [[ -n "$COV_ARGS" ]]; then
    case "$PYTHON_VERSION" in
        3.11 | 3.12 | 3.13 | *t) ;;
        *) echo "COVERAGE_CORE=sysmon" | tee -a $GITHUB_ENV ;;
    esac
fi

# Persist numba's on-disk JIT cache between runs (see the "Cache numba" step).
# Without it every xdist worker recompiles all ~30 jitted kernels from scratch:
# e.g. test_fit_chpi_quat_weighted takes 12 s cold and 0.3 s warm. The whole
# cache is only ~1 MB.
#
# numba keys its cache index on the host CPU model, so on CI -- where the runner
# pool hands out several different models -- a restored cache would usually miss.
# Compiling for a generic CPU makes the cached objects portable between them; the
# kernels are small enough that losing model-specific vectorization is nothing
# next to the compile time it saves.
echo "NUMBA_CACHE_DIR=$HOME/.cache/mne-numba" | tee -a $GITHUB_ENV
echo "NUMBA_CPU_NAME=generic" | tee -a $GITHUB_ENV
echo "NUMBA_CPU_FEATURES=" | tee -a $GITHUB_ENV

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
            echo "MNE_TEST_ALLOW_SKIP=.*(Requires (spm|brainstorm|misc) dataset|SCIPY_ARRAY_API|FreeSurfer|CUDA not|macOS|PySide6 causes segfaults).*" | tee -a $GITHUB_ENV
        else
            echo "MNE_TEST_ALLOW_SKIP=.*(Requires (spm|brainstorm|misc) dataset|SCIPY_ARRAY_API|CUDA not|PySide6 causes segfaults).*" | tee -a $GITHUB_ENV
        fi
        echo "MNE_QT_BACKEND=PySide6" | tee -a $GITHUB_ENV
    elif [[ "$MNE_CI_KIND" == "pip-ft" ]]; then
        echo "MNE_TEST_ALLOW_SKIP=.*(Requires (spm|brainstorm|misc|testing) dataset|Requires (MNE-C|FreeSurfer)|MNE_SKIP_NETWORK_TESTS|could not import|No module named|not installed|not available|has __version__|needs >=|[Nn]eeds [a-z0-9_.-]+|[Rr]equires [a-z0-9_.-]+$|[A-Za-z0-9_.-]+ (is )?required|fixed by [a-z0-9_.-]+ [0-9]|CUDA not|Numba not|SCIPY_ARRAY_API|[Aa]rray API|PySide6 causes segfaults).*" | tee -a $GITHUB_ENV
    else
        echo "::error::Unrecognized MNE_CI_KIND=${MNE_CI_KIND}"
        exit 1
    fi
elif [[ "$MNE_CI_KIND" == "minimal" ]]; then
    echo "MNE_TEST_ALLOW_SKIP=.*(Requires (spm|brainstorm|misc|testing) dataset|Requires (MNE-C|FreeSurfer)|MNE_SKIP_NETWORK_TESTS|could not import|No module named|not installed|not available|has __version__|needs >=|[Nn]eeds [a-z0-9_.-]+|[Rr]equires [a-z0-9_.-]+$|[A-Za-z0-9_.-]+ (is )?required|fixed by [a-z0-9_.-]+ [0-9]|CUDA not|Numba not|SCIPY_ARRAY_API|[Aa]rray API|PySide6 causes segfaults).*" | tee -a $GITHUB_ENV
    echo "MNE_QT_BACKEND=PySide6" | tee -a $GITHUB_ENV
elif [[ "$MNE_CI_KIND" == "old" ]]; then
    echo "MNE_IGNORE_WARNINGS_IN_TESTS=true" | tee -a $GITHUB_ENV
    echo "MNE_SKIP_NETWORK_TESTS=1" | tee -a $GITHUB_ENV
    echo "MNE_TEST_ALLOW_SKIP=.*(Requires (spm|brainstorm|misc|testing) dataset|Requires (MNE-C|FreeSurfer)|MNE_SKIP_NETWORK_TESTS|could not import|No module named|not installed|not available|has __version__|needs >=|[Nn]eeds [a-z0-9_.-]+|[Rr]equires [a-z0-9_.-]+$|[A-Za-z0-9_.-]+ (is )?required|fixed by [a-z0-9_.-]+ [0-9]|CUDA not|Numba not|SCIPY_ARRAY_API|[Aa]rray API|PySide6 causes segfaults).*" | tee -a $GITHUB_ENV
    echo "MNE_QT_BACKEND=PyQt6" | tee -a $GITHUB_ENV
elif [[ "$MNE_CI_KIND" == "conda" ]]; then
    echo "Setting conda env vars for $MNE_CI_KIND"
    echo "CONDA_ENV=environment.yml" | tee -a $GITHUB_ENV
    echo "MNE_LOGGING_LEVEL=warning" | tee -a $GITHUB_ENV
    echo "MNE_TEST_ALLOW_SKIP=.*(on conda|Requires (spm|brainstorm|misc) dataset|CUDA not|Flakey verbose behavior|PySide6 causes segfaults|SCIPY_ARRAY_API).*" | tee -a $GITHUB_ENV
    echo "MNE_QT_BACKEND=PySide6" | tee -a $GITHUB_ENV
else
    echo "::error::Unrecognized MNE_CI_KIND=${MNE_CI_KIND}"
    exit 1
fi
if [[ "$CI_OS_NAME" == "windows"* ]]; then
    echo "MNE_IS_OSMESA=true" | tee -a $GITHUB_ENV
fi
echo "::endgroup::"
