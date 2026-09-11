# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# Notebook source, not a module: it installs packages with a top-level ``await``
# and imports them afterwards, so import placement rules do not apply. Keeping
# it a real file is what lets ruff lint and format the rest of it.
# ruff: noqa: E402, F704, I001
#
# Everything defined here lands in the notebook's namespace, so names are
# _-prefixed to stay out of the tutorials' way; ``mne_data_path`` is the one
# deliberate exception, since a reader may want it.

# --- JupyterLite setup cell -------------------------------------------------
# 💡 This cell is added to the start of every notebook. It installs MNE and
# patches the browser environment for Pyodide. Running this notebook locally?
# Delete this cell first: piplite only exists inside JupyterLite.

# === 1. Install ==============================================================
import piplite

# piplite (not micropip) prefers the development MNE wheel bundled with the docs
# over PyPI; keep_going reports a dependency with no wheel instead of aborting
await piplite.install(
    [
        "mne",
        "scikit-learn",
        "joblib",
        "pandas",
        "seaborn",
        "mne-connectivity",
        "nibabel",
        "pyvista-js",
        "pyxdf",
        "mffpy",
        "python-picard",
    ],
    keep_going=True,
)

# === 2. Pyodide compatibility, before MNE is imported ========================
import sys
import os
import inspect
import io
from pathlib import Path

# Route requests through the browser, for the pooch downloads that fetch from
# off-site (fetch_fsaverage and friends). A synchronous XMLHttpRequest reports
# the real HTTP status, which is what pooch's raise_for_status() needs.
import requests


def _pyodide_send(self, request, **kwargs):
    from js import XMLHttpRequest

    _xhr = XMLHttpRequest.new()
    _xhr.open(request.method or "GET", request.url, False)
    _xhr.responseType = "arraybuffer"
    _xhr.send()
    response = requests.Response()
    response.status_code = _xhr.status
    response.reason = _xhr.statusText
    response.url = request.url
    response.raw = io.BytesIO(bytes(_xhr.response.to_py()))
    return response


requests.Session.send = _pyodide_send

# === 3. Where the data comes from ===========================================
# The docs serve the data next to the pages (/mne_data/, via html_extra_path),
# and every file is fetched into the virtual filesystem on first use by the
# wrappers below. Pyodide may run in a web worker, where ``location`` exists
# but ``window`` does not.
import js

_base = str(js.location.href).split("/lite/")[0] + "/mne_data/"
mne_data_path = "/tmp/mne_data"
_mne_data_root = Path(mne_data_path)
_mne_data_root.mkdir(parents=True, exist_ok=True)
os.environ["MNE_DATA"] = mne_data_path

# an OSF download would fail on CORS or memory; say so instead
import pooch
from urllib.parse import urlparse

_orig_pooch_fetch = pooch.Pooch.fetch


def _pyodide_pooch_fetch(self, fname, processor=None, downloader=None):
    host = urlparse(self.get_url(fname)).hostname or ""
    if host == "osf.io" or host.endswith(".osf.io"):
        raise RuntimeError(
            f"Cannot download {fname!r} from OSF in JupyterLite: browser CORS "
            "policy and memory limits prevent large dataset downloads. Open this "
            "notebook from mne.tools, where the data is bundled, or run it locally."
        )
    return _orig_pooch_fetch(self, fname, processor=processor, downloader=downloader)


pooch.Pooch.fetch = _pyodide_pooch_fetch

# === 4. Fetch helpers =======================================================
import mne


def _lite_rel_to_data(fname):
    """Return ``fname`` relative to the data root, or None if it sits outside."""
    _p = Path(str(fname))
    if _p == _mne_data_root or not _p.is_relative_to(_mne_data_root):
        return None
    return _p.relative_to(_mne_data_root).as_posix()


def _lite_fetch_rel(rel):
    """Download one file (once) into the virtual filesystem and return its path.

    Synchronous, since it runs inside MNE readers, which cannot await; a
    blocking XHR may read binary in a web worker, where the kernel runs.
    """
    _dst = _mne_data_root / rel
    if not _dst.exists():
        from js import XMLHttpRequest

        _xhr = XMLHttpRequest.new()
        _xhr.open("GET", _base + rel, False)
        _xhr.responseType = "arraybuffer"
        _xhr.send()
        if _xhr.status != 200:
            raise FileNotFoundError(f"Could not fetch {rel} (HTTP {_xhr.status})")
        _dst.parent.mkdir(parents=True, exist_ok=True)
        _dst.write_bytes(bytes(_xhr.response.to_py()))
    return _dst


def _lite_fetch_if_under_mne_data(fname):
    """Fetch ``fname`` if we serve it, and hand it back either way."""
    _rel = _lite_rel_to_data(fname)
    if _rel is not None:
        _lite_fetch_rel(_rel)
    return fname


def _lite_fetch_optional(rels):
    """Fetch what is served among ``rels``, quietly skipping the rest.

    These are candidates MNE will choose between, or optional companions of a
    multi-file format; the reader raises its own error for a file it needed.
    """
    for _r in rels:
        try:
            _lite_fetch_rel(_r)
        except Exception:
            pass


def _lite_fetch_candidates(subject, subjects_dir, rel_paths):
    """Fetch ``rel_paths`` under ``<subjects_dir>/<subject>/``."""
    _rel = _lite_rel_to_data(subjects_dir if subjects_dir is not None else "")
    if subject and _rel is not None:
        _lite_fetch_optional(f"{_rel}/{subject}/{_p}" for _p in rel_paths)


def _lite_wrap_reader(module, name, siblings=None):
    """Wrap ``module.name`` to fetch its filename argument before it opens it.

    ``siblings`` maps a relative path to the other files that name implies (a
    BrainVision header's .eeg and .vmrk). The filename's keyword is read off
    the signature, since readers call it fname, filename or input_fname.
    """
    orig = getattr(module, name)
    arg = next(iter(inspect.signature(orig).parameters))

    def wrapped(*args, **kwargs):
        if arg in kwargs:  # normalize to positional
            args = (kwargs.pop(arg),) + args
        if args:
            _rel = _lite_rel_to_data(args[0])
            if _rel is not None:
                _lite_fetch_optional([_rel] + (siblings(_rel) if siblings else []))
        return orig(*args, **kwargs)

    setattr(module, name, wrapped)
    _lite_rebind(name, orig, wrapped)  # modules that imported it by name


def _lite_dir_reader(orig):
    """Wrap a reader of a folder, listed by the manifest conf.py leaves in it."""

    def _read(fname, *args, **kwargs):
        _rel = _lite_rel_to_data(fname)
        if _rel is not None:
            try:
                _names = _lite_fetch_rel(_rel + "/_lite_manifest.txt").read_text()
                _lite_fetch_optional(_rel + "/" + _n for _n in _names.split())
            except Exception as _e:
                print("[JupyterLite] could not fetch " + str(fname) + ": " + repr(_e))
        return orig(fname, *args, **kwargs)

    return _read


def _lite_rebind(name, old, new):
    """Point every MNE module that already imported ``old`` at ``new``."""
    for _m in list(sys.modules.values()):
        if (
            getattr(_m, "__name__", "").startswith("mne")
            and getattr(_m, name, None) is old
        ):
            setattr(_m, name, new)


# === 5. Where MNE looks for each dataset ====================================
# data_path() would download the archive from OSF; point it at the served
# folder instead. A probe file is fetched for datasets whose data is read by
# something other than an MNE reader (scipy for mtrf).
def _lite_dataset_path(folder, probe=None):
    def _data_path(*args, **kwargs):
        if probe is not None:
            _lite_fetch_rel(folder + "/" + probe)
        return _mne_data_root / folder

    return _data_path


for _ds, _folder, _probe in (
    ("sample", "MNE-sample-data", None),
    ("ssvep", "ssvep-example-data", None),
    ("misc", "MNE-misc-data", None),
    ("eyelink", "MNE-eyelink-data", None),
    ("fnirs_motor", "MNE-fNIRS-motor-data", None),
    ("phantom_kernel", "MNE-phantom-kernel-data", None),
    ("multimodal", "MNE-multimodal-data", None),
    ("kiloword", "MNE-kiloword-data", "kword_metadata-epo.fif"),
    ("mtrf", "mTRF_1.5", "speech_data.mat"),
):
    getattr(mne.datasets, _ds).data_path = _lite_dataset_path(_folder, _probe)
del _ds, _folder, _probe


def _lite_eegbci_load_data(subjects, runs, *args, **kwargs):  # by subject and run
    _runs = [runs] if isinstance(runs, (int, float)) else list(runs)
    _subjects = list(subjects) if isinstance(subjects, (list, tuple)) else [subjects]
    return [
        _lite_fetch_rel(
            f"MNE-eegbci-data/files/eegmmidb/1.0.0/S{_s:03d}/S{_s:03d}R{_r:02d}.edf"
        )
        for _s in _subjects
        for _r in _runs
    ]


mne.datasets.eegbci.load_data = _lite_eegbci_load_data

# === 6. Readers =============================================================
# Nearly every reader validates its filename with _check_fname(must_exist=True)
# before opening it, so one hook there fetches for all of them. The rest need
# one of three things: their own wrapper because they skip that check, the
# other files a single name implies, or a fetch before a filesystem probe
# (os.path.exists, glob) that no reader would ever trigger.
import mne.utils.check as mne_check

_orig_check_fname = mne_check._check_fname


def _lite_check_fname(
    fname, overwrite=False, must_exist=False, name="File", need_dir=False, **kwargs
):
    _rel = _lite_rel_to_data(fname) if must_exist else None
    if _rel is not None and need_dir:  # a served folder's files arrive on demand
        (_mne_data_root / _rel).mkdir(parents=True, exist_ok=True)
    elif _rel is not None:
        try:
            _lite_fetch_rel(_rel)
        except Exception:
            pass  # let MNE raise its own error for a missing file
    return _orig_check_fname(fname, overwrite, must_exist, name, need_dir, **kwargs)


mne_check._check_fname = _lite_check_fname
_lite_rebind("_check_fname", _orig_check_fname, _lite_check_fname)

import matplotlib.pyplot as plt  # the eyetracking heatmap reads its stimulus
import nibabel  # a few tutorials load an MRI themselves

for _module, _name, _siblings in (
    (plt, "imread", None),
    (nibabel, "load", None),
    (mne.io, "read_raw_eeglab", lambda rel: [rel.removesuffix(".set") + ".fdt"]),
    (
        mne.io,
        "read_raw_brainvision",
        lambda rel: [rel.removesuffix(".vhdr") + s for s in (".eeg", ".vmrk")],
    ),
    (
        mne,
        "read_source_estimate",
        lambda rel: [rel + s for s in ("-lh.stc", "-rh.stc")],
    ),
):
    _lite_wrap_reader(_module, _name, _siblings)
del _module, _name, _siblings
try:  # pyxdf has no wheel on every Pyodide build; only the XDF example needs it
    import pyxdf

    _lite_wrap_reader(pyxdf, "load_xdf")
except Exception:
    pass
mne.io.read_raw_nirx = _lite_dir_reader(mne.io.read_raw_nirx)  # a folder

# Filesystem probes: fetch the candidates first, in the order MNE tries them,
# then let it choose as it normally would. The viz modules bind these names at
# import, hence the rebinds.
import mne._freesurfer as mne_fs
import mne.surface as mne_surface
import mne.viz._3d  # noqa: F401

_orig_get_head_surface = mne_fs._get_head_surface
_orig_get_skull_surface = mne_fs._get_skull_surface
_orig_surface_head = mne_surface._get_head_surface
_orig_plot_bem = mne.viz.plot_bem


def _lite_get_head_surface(surf, subject, subjects_dir, bem=None, verbose=None):
    if surf in ("head-dense", "seghead"):
        _cands = [f"bem/{subject}-head-dense.fif", "surf/lh.seghead"]
    else:
        _cands = ["bem/outer_skin.surf", f"bem/{subject}-head.fif"]
    _lite_fetch_candidates(subject, subjects_dir, _cands)
    return _orig_get_head_surface(surf, subject, subjects_dir, bem=bem, verbose=verbose)


def _lite_get_skull_surface(surf, subject, subjects_dir, bem=None, verbose=None):
    _lite_fetch_candidates(subject, subjects_dir, [f"bem/{surf}_skull.surf"])
    return _orig_get_skull_surface(
        surf, subject, subjects_dir, bem=bem, verbose=verbose
    )


def _lite_surface_head_surface(
    subject, source, subjects_dir, on_defects, raise_error=True
):
    _srcs = [source] if isinstance(source, str) else list(source)
    _lite_fetch_candidates(
        subject, subjects_dir, [f"bem/{subject}-{_s}.fif" for _s in _srcs]
    )
    return _orig_surface_head(
        subject, source, subjects_dir, on_defects, raise_error=raise_error
    )


def _lite_plot_bem(subject=None, subjects_dir=None, *args, **kwargs):
    _want = ["bem/inner_skull.surf", "bem/outer_skull.surf", "bem/outer_skin.surf"]
    _want.append("mri/" + str(kwargs.get("mri", "T1.mgz")))
    _bs = kwargs.get("brain_surfaces")
    for _b in [_bs] if isinstance(_bs, str) else _bs or []:
        _want += [f"surf/lh.{_b}", f"surf/rh.{_b}"]
    _lite_fetch_candidates(subject, subjects_dir, _want)
    return _orig_plot_bem(subject, subjects_dir, *args, **kwargs)


mne_fs._get_head_surface = _lite_get_head_surface
_lite_rebind("_get_head_surface", _orig_get_head_surface, _lite_get_head_surface)
mne_fs._get_skull_surface = _lite_get_skull_surface
_lite_rebind("_get_skull_surface", _orig_get_skull_surface, _lite_get_skull_surface)
mne_surface._get_head_surface = _lite_surface_head_surface
mne.viz.plot_bem = _lite_plot_bem

# The logging tutorial reads a KIT file from inside the installed package,
# which the wheel leaves out, so stage the served copy where it looks.
import shutil

_orig_read_raw_kit = mne.io.read_raw_kit


def _lite_read_raw_kit(input_fname, *args, **kwargs):
    _p = Path(str(input_fname))
    if _p.name == "test.sqd" and not _p.exists():
        try:
            _p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_lite_fetch_rel("MNE-kit-testdata/test.sqd"), _p)
        except Exception as _e:
            print("[JupyterLite] could not stage test.sqd: " + repr(_e))
    return _orig_read_raw_kit(input_fname, *args, **kwargs)


mne.io.read_raw_kit = _lite_read_raw_kit

# === 7. What WebAssembly cannot do ==========================================
# No OS threads: the ProgressBar updater and tqdm's monitor only animate, so
# skip them (guarded, since both are private paths)
try:
    from mne.utils import progressbar

    progressbar._UpdateThread.start = lambda self: None
    progressbar._UpdateThread.join = lambda self, *args, **kwargs: None
except Exception:
    pass
try:
    import tqdm

    tqdm.tqdm.monitor_interval = 0
except Exception:
    pass

import IPython

IPython.get_ipython().run_line_magic("matplotlib", "inline")

# fig.show() warns on the inline Agg canvas, and a few tutorials call it
import matplotlib.figure as mpl_figure

mpl_figure.Figure.show = lambda self, *a, **k: None

# A plot that is also a cell's last expression returns its Figure, which Out[]
# would echo a second time after plt_show displayed it. Drop that echo for
# Figures and lists of them (guarded: a double render beats a broken cell).
try:
    _lite_dh = type(IPython.get_ipython().displayhook)
    _lite_dh_call = _lite_dh.__call__

    def _lite_displayhook(self, result=None):
        _figs = result if isinstance(result, (list, tuple)) else [result]
        if _figs and all(isinstance(_f, mpl_figure.Figure) for _f in _figs):
            result = None
        return _lite_dh_call(self, result)

    _lite_dh.__call__ = _lite_displayhook
except Exception:
    pass

# threadpoolctl 3.6.0 calls Pyodide's deprecated JsProxy.as_object_map(), which
# warns from mne.sys_info(); as_py_json() gives the same paths.
# TODO VERSION: fixed in joblib/threadpoolctl#201, drop once Pyodide bundles
# threadpoolctl >= 3.7.0
try:
    import threadpoolctl

    def _find_libraries_pyodide(self):
        from pyodide_js._module import LDSO

        for _fp in LDSO.loadedLibsByName.as_py_json():
            if Path(_fp).exists():
                self._make_controller_from_path(_fp)

    threadpoolctl.ThreadpoolController._find_libraries_pyodide = _find_libraries_pyodide
except Exception:
    pass

# === 8. 3D ==================================================================
# VTK has no WebAssembly build, so draw with pyvista-js (vtk.js) instead; see
# mne/viz/backends/_lite.py
try:
    mne.viz.set_3d_backend("jupyterlite_notebook")
except Exception as _e:
    print("[JupyterLite] could not select the pyvista-js renderer: " + repr(_e))
