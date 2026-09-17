"""Build-time config for the JupyterLite site's Pyodide lock.

Adds packages to the lock so Pyodide loads them at kernel initialization.
Packages Pyodide already curates keep their curated version.

On a dev build, MNE is installed by jupyterlite_setup_cell instead.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "sphinxext"))

from jupyterlite_lock_specs import (  # noqa: E402
    jupyterlite_package_names,
    jupyterlite_specs_to_lock,
    mne_pypi_spec,
)

c = get_config()  # noqa: F821

specs = jupyterlite_specs_to_lock()
prefetch = jupyterlite_package_names()

# a stable/maint build locks the real PyPI release instead of the dev wheel
# build_lite_wheel.py builds; see mne_pypi_spec for more info.
mne_spec = mne_pypi_spec()
if mne_spec is not None:
    specs.append(mne_spec)
    prefetch.append("mne")

c.PyodideLockAddon.enabled = True
c.PyodideLockAddon.specs = specs
c.PyodideLockAddon.prefetch_extra = prefetch
