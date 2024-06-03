#!/usr/bin/env bash

set -eu

# Exit immediately if not running inside a Dev Container
if [ -z "${RUNNING_IN_DEV_CONTAINER+x}" ]; then
  echo -e "👋 Not running in a dev container, not auto-installing MNE-Python.\n"
  exit
fi

package_name="MNE-Python (dev)"
import_name="mne"

# Run the import test outside of the repository, so we don't accidentally import the
# `mne` directory from there. This is an annoyance caused by MNE-Python's not using a
# src/ layout.
orig_dir=$(pwd)
cd ~
if python -c "import $import_name" &> /dev/null; then
    echo -e "✅ $package_name is already installed.\n"
    cd "${orig_dir}"
else
    cd "${orig_dir}"
    echo -e "💡 $package_name is not installed. Installing now…\n"
    pip install -e ".[full-pyside6,dev,test_extra]"
    echo -e "\n✅ $package_name has been installed.\n"
    echo -e "💡 Installing pre-commit hooks…"
    pre-commit install --install-hooks
    echo -e "✅ pre-commit hooks installed.\n"
fi

echo -e "\n🚀 You're all set. Happy hacking!\n"
