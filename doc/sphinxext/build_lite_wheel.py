"""Build the development MNE wheel for the JupyterLite browser kernel.

Run this once before building the docs, either in CI or locally::

    python doc/sphinxext/build_lite_wheel.py

The wheel is written to ``doc/pypi``, where the jupyterlite-pyodide-kernel
PipliteAddon discovers, copies and indexes it (adding it to ``pipliteUrls`` in
``jupyter-lite.json``), so the browser kernel installs the current development
MNE rather than the older release from PyPI. See
https://jupyterlite.readthedocs.io/en/latest/howto/pyodide/wheels.html

Both functions are importable, so a docs build can reuse a wheel that is already
present rather than building one on every invocation::

    from build_lite_wheel import build_wheel, find_wheels

    wheels = find_wheels() or build_wheel()
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import glob
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
PYPI_WHEELS_DIR = os.path.join(REPO_ROOT, "doc", "pypi")


def find_wheels():
    """Return the MNE wheels already present in ``doc/pypi``.

    Returns
    -------
    wheels : list of str
        Paths of the MNE wheels found, empty if there are none.
    """
    return glob.glob(os.path.join(PYPI_WHEELS_DIR, "mne-*.whl"))


def build_wheel():
    """Build the development MNE wheel into ``doc/pypi``.

    Returns
    -------
    wheels : list of str
        Paths of the MNE wheels that were built.
    """
    # Clean first so stale wheels from previous runs do not accumulate and
    # pollute the piplite all.json index.
    shutil.rmtree(PYPI_WHEELS_DIR, ignore_errors=True)
    os.makedirs(PYPI_WHEELS_DIR, exist_ok=True)

    # The wheel is built from pyproject.toml as it stands: Pyodide 314 ships
    # matplotlib 3.10.8, scipy 1.18.0 and numpy 2.4.3, all of which satisfy the
    # minimums MNE declares, so none of them needs relaxing for the browser.
    os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = "9999.0.1"
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
    # the older released MNE from PyPI.
    wheels = find_wheels()
    if not wheels:
        raise RuntimeError(
            f"JupyterLite: no MNE wheel was built into {PYPI_WHEELS_DIR!r}; the "
            "browser kernel would fall back to the released PyPI version. Check "
            "the 'pip wheel' output above."
        )
    return wheels


if __name__ == "__main__":
    print(f"[JupyterLite] Built MNE wheel(s) for the browser kernel: {build_wheel()}")
