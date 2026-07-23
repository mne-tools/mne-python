"""First-cut JupyterLite setup cell for the xeus-python kernel.

WIP / CI-PENDING -- this is the xeus counterpart of the Pyodide
``first_notebook_cell`` in ``conf.py``. It cannot be fully validated locally
(needs the real ``build_docs`` CI run); the assumptions marked ``CI-VERIFY``
below are the ones to confirm/iterate on first.

Why it is so much smaller than the Pyodide cell
------------------------------------------------
xeus-python is real CPython compiled to WebAssembly (NOT Pyodide), and the
packages are PRE-INSTALLED at build time from ``jupyterlite_environment.yml``.
So compared to the Pyodide setup cell this drops:

  * ``piplite.install(...)``  -> MNE is already installed
  * the ``lzma`` / ``multiprocessing`` mocks -> real CPython stdlib is present
  * the ``pyodide.http`` / ``requests`` patches and the ``js`` XHR fetch shims
    -> no Pyodide runtime to patch

What remains is just: point MNE's dataset loaders at the bundled data, block
accidental OSF downloads, and (still to port) the pyvista-js 3D shim.

Data model (CI-VERIFY)
----------------------
With ``XeusAddon.mount_jupyterlite_content=True`` the served JupyterLite content
is mounted into the kernel filesystem at ``/files``. So the curated MNE data
must be made available as JupyterLite content (a conf.py change: add the curated
``mne_data`` tree to ``jupyterlite_contents`` instead of only ``html_extra_path``).
The cell below resolves the data root by probing the likely mount locations so it
is robust to which mechanism ends up being used. Because every bundled file is
then present in the FS, NO runtime fetch is needed -- the loaders just return the
folder (much simpler than the Pyodide lazy-fetch shims).

conf.py wiring this expects (next step, not yet applied so the pyodide build
stays intact on this branch):

    from jupyterlite_xeus_setup_cell import XEUS_FIRST_NOTEBOOK_CELL
    jupyterlite_build_command_options = {
        "XeusAddon.environment_file": "jupyterlite_environment.yml",
        "XeusAddon.mount_jupyterlite_content": True,
    }
    sphinx_gallery_conf["first_notebook_cell"] = XEUS_FIRST_NOTEBOOK_CELL
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# The setup cell source, as a string (mirrors how conf.py stores the Pyodide one).
XEUS_FIRST_NOTEBOOK_CELL = r"""# 💡 Auto-added setup cell (xeus-python kernel).
# MNE and its dependencies are pre-installed in this kernel; this cell only
# points the dataset loaders at the pre-bundled data.
import os
from pathlib import Path as _Path
import mne

# --- locate the bundled MNE data (mounted from the served jupyterlite content) ---
# CI-VERIFY: confirm the actual mount path (/files with mount_jupyterlite_content).
_candidates = ["/files/mne_data", "/drive/mne_data", os.path.expanduser("~/mne_data")]
mne_data_path = next((p for p in _candidates if os.path.isdir(p)), "/files/mne_data")
os.makedirs(mne_data_path, exist_ok=True)
os.environ["MNE_DATA"] = mne_data_path

# Pre-create a valid empty config so MNE never hits a corrupt read.
_cfg = mne.get_config_path()
os.makedirs(os.path.dirname(_cfg), exist_ok=True)
if not os.path.exists(_cfg):
    with open(_cfg, "w") as _f:
        _f.write("{}")
mne.set_config("MNE_DATA", mne_data_path)
for _ds in ["SAMPLE", "TESTING", "SSVEP", "EEGBCI", "KILOWORD", "ERP_CORE", "MTRF"]:
    mne.set_config(f"MNE_DATASETS_{_ds}_PATH", mne_data_path)

# --- point dataset loaders at the pre-bundled folders (no OSF download) ---
# Everything is already on the filesystem, so these just return the folder --
# no lazy fetching needed (unlike the Pyodide build).
def _lite_folder(_name):
    def _data_path(*_a, **_kw):
        return _Path(mne_data_path) / _name
    return _data_path

mne.datasets.sample.data_path = _lite_folder("MNE-sample-data")
mne.datasets.kiloword.data_path = _lite_folder("MNE-kiloword-data")
mne.datasets.erp_core.data_path = _lite_folder("MNE-ERP-CORE-data")
mne.datasets.mtrf.data_path = _lite_folder("mTRF_1.5")

def _lite_eegbci_load_data(subject, runs, *_a, **_kw):
    _runs = [runs] if isinstance(runs, (int, float)) else list(runs)
    _subs = list(subject) if isinstance(subject, (list, tuple)) else [subject]
    _base = _Path(mne_data_path) / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"
    return [
        _base / f"S{int(s):03d}" / f"S{int(s):03d}R{int(r):02d}.edf"
        for s in _subs
        for r in _runs
    ]
mne.datasets.eegbci.load_data = _lite_eegbci_load_data

# Block accidental OSF downloads (data is pre-bundled or simply unavailable here).
import pooch
_orig_pooch_fetch = pooch.Pooch.fetch
def _lite_pooch_fetch(self, fname, processor=None, downloader=None):
    if "osf.io" in self.get_url(fname):
        raise RuntimeError(
            f"Cannot download {fname!r} in JupyterLite: open this notebook from "
            "mne.tools where the data is pre-bundled, or run it locally."
        )
    return _orig_pooch_fetch(self, fname, processor=processor, downloader=downloader)
pooch.Pooch.fetch = _lite_pooch_fetch

# TODO(CI-VERIFY): port the pyvista-js SourceEstimate.plot 3D shim from the
# Pyodide conf.py cell -- the approach (JS-native pyvista-js) is kernel-agnostic
# per the plan, only the import/interop path needs confirming under xeus.
"""
