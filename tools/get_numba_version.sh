#!/bin/bash -e

set -o pipefail

# Numba's on-disk JIT cache (NUMBA_CACHE_DIR) is invalidated whenever numba or
# llvmlite is upgraded, so CI cache keys have to include their versions in a form
# usable in a cache key. Prints "none" when numba is not installed, which just
# means the cache never gets populated for that job.
NUMBA_VERSION=`python -c "
try:
    import llvmlite, numba
except Exception:
    print('none')
else:
    print(f'{numba.__version__}-{llvmlite.__version__}')
"`
if [ ! -z $GITHUB_ENV ]; then
	echo "NUMBA_VERSION="$NUMBA_VERSION | tee -a $GITHUB_ENV
elif [ ! -z $AZURE_CI ]; then
	echo "##vso[task.setvariable variable=numba_version]$NUMBA_VERSION"
else
	echo $NUMBA_VERSION
fi
