#!/usr/bin/env bash

set -eu

# Exit immediately if not running inside a Dev Container
if [ -z "${RUNNING_IN_DEV_CONTAINER+x}" ]; then
  echo -e "👋 Not running in dev container, not installing MNE-Python (dev).\n"
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
    exit
else
    cd "${orig_dir}"
    code .devcontainer/Welcome.md
    echo -e "💡 $package_name is not installed. Installing now …\n"
    pipx run uv pip install -e ".[full-pyside6,dev,test_extra]"
    echo -e "\n✅ $package_name has been installed.\n"
    echo -e "💡 Installing pre-commit hooks …"
    pre-commit install --install-hooks
    echo -e "✅ pre-commit hooks installed.\n"
fi

echo -e "\n🚀 You're all set. Happy hacking!\n"
