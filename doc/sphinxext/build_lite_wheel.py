"""Build the MNE wheel for the JupyterLite browser kernel.

Run this once before building the docs, either in CI or locally::

    python doc/sphinxext/build_lite_wheel.py

The wheel is written to ``doc/pypi``, where the jupyterlite-pyodide-kernel
PipliteAddon discovers, copies and indexes it (adding it to ``pipliteUrls`` in
``jupyter-lite.json``), so the browser kernel installs the MNE the surrounding
pages are built from rather than the last release on PyPI. That is the
development version on ``main`` and that release's code on a ``maint/*``
branch, since the docs build from whichever branch it is running on. See
https://jupyterlite.readthedocs.io/en/latest/howto/pyodide/wheels.html

Both functions are importable, so a docs build can reuse a wheel that is already
present rather than building one on every invocation::

    from build_lite_wheel import build_wheel, find_wheels

    wheels = find_wheels() or build_wheel()
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPI_WHEELS_DIR = REPO_ROOT / "doc" / "pypi"


def find_wheels():
    """Return the MNE wheels already present in ``doc/pypi``.

    Returns
    -------
    wheels : list of Path
        Paths of the MNE wheels found, empty if there are none.
    """
    return sorted(PYPI_WHEELS_DIR.glob("mne-*.whl"))


def _latest_pypi_version():
    """Return the newest MNE version on PyPI, or None if it cannot be reached.

    Returns
    -------
    version : str | None
        The version string, or None if PyPI could not be queried.
    """
    # Broad on purpose: this only ever runs while raising, so a network problem
    # here must not replace the real error with a less useful one.
    try:
        url = "https://pypi.org/pypi/mne/json"
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.load(response)["info"]["version"]
    except Exception:
        return None


def build_wheel():
    """Build the MNE wheel for the browser kernel into ``doc/pypi``.

    Returns
    -------
    wheels : list of Path
        Paths of the MNE wheels that were built.
    """
    # This directory is the piplite index, so it should hold the wheel this
    # build produced and nothing else, including anything left behind by an
    # earlier build or a manual pip wheel. The version is left to hatch-vcs:
    # piplite serves this index exclusively rather than merging it with PyPI
    # (_query_package returns as soon as the package is found here), so a
    # development version has nothing to lose a resolution against.
    shutil.rmtree(PYPI_WHEELS_DIR, ignore_errors=True)
    PYPI_WHEELS_DIR.mkdir(parents=True, exist_ok=True)

    # The wheel is built from pyproject.toml as it stands: Pyodide 314 ships
    # matplotlib 3.10.8, scipy 1.18.0 and numpy 2.4.3, all of which satisfy the
    # minimums MNE declares, so none of them needs relaxing for the browser.
    # NB: build isolation is left ON (the default). MNE uses the hatchling build
    # backend, so pip must create an isolated build env to install
    # hatchling/hatch-vcs; --no-build-isolation fails with "Cannot import
    # 'hatchling.build'" on CI, where those build deps are not in the base
    # environment.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            REPO_ROOT,
            "--no-deps",
            "-w",
            PYPI_WHEELS_DIR,
        ],
        check=True,
    )

    # Fail loudly rather than silently letting the browser kernel fall back to
    # the released MNE from PyPI.
    wheels = find_wheels()
    if not wheels:
        latest = _latest_pypi_version()
        fallback = f"MNE {latest}" if latest else "the latest MNE release"
        raise RuntimeError(
            f"JupyterLite: no MNE wheel was built into {PYPI_WHEELS_DIR}; the "
            f"browser kernel would fall back to {fallback} from PyPI. Check the "
            "'pip wheel' output above."
        )
    return wheels


if __name__ == "__main__":
    # Reuse a wheel that is already there, so repeat `make html` runs do not
    # rebuild it. Remove doc/pypi (or `make clean`) to force a fresh one.
    existing = find_wheels()
    wheels = ", ".join(str(wheel) for wheel in (existing or build_wheel()))
    verb = "Reusing" if existing else "Built"
    print(f"[JupyterLite] {verb} MNE wheel(s) for the browser kernel: {wheels}")
