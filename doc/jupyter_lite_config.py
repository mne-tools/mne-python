"""Build-time config for the JupyterLite site's Pyodide lock.

Adds MNE and the packages the notebooks import to the lock, so Pyodide loads
them at kernel initialization. Packages Pyodide already curates keep their
curated version.
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
)

c = get_config()  # noqa: F821

c.PyodideLockAddon.enabled = True
# the MNE wheel build_lite_wheel.py builds from this checkout (a tagged
# checkout gives the release): the solve takes it over PyPI, which would
# otherwise supply the latest release through mne-connectivity's mne>=1.6
c.PyodideLockAddon.wheels = ["pypi"]
c.PyodideLockAddon.specs = ["mne", *jupyterlite_specs_to_lock()]
c.PyodideLockAddon.prefetch_extra = ["mne", *jupyterlite_package_names()]
