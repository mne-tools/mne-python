# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# This file is notebook source rather than a module: it installs packages with
# a top-level ``await`` and imports them only afterwards, so the rules about
# import position, await position and import order do not apply to it. Ruff
# still lints and formats everything else here, which is the point of keeping
# it as a real file instead of a string.
# ruff: noqa: E402, F704, I001

# --- JupyterLite setup cell -------------------------------------------------
# 💡 This cell is automatically added to the start of each notebook.
# It installs MNE and patches the browser environment for Pyodide.
import piplite

# Use piplite (not micropip) so the locally-built development MNE wheel
# bundled into the JupyterLite build is preferred over the older PyPI
# release;
# piplite checks the local index first and falls back to PyPI for deps.
# keep_going=True so a dependency with no pure-Python wheel is reported
# at the end rather than aborting the whole install on the first one.
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

import sys
import os
import io

# lzma: try real stdlib first (Pyodide ships it); only mock if absent. The
# import has to be attempted rather than probed with find_spec, because the
# mock below is only installed when it actually fails.
try:
    import lzma  # noqa: F401
except ImportError:

    class _LZMAFile:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def write(self, d):
            pass

        def read(self, n=-1):
            return b""

        def close(self):
            pass

    class _MockLZMA:
        LZMAError = Exception
        LZMAFile = _LZMAFile
        FORMAT_XZ = 1
        FORMAT_ALONE = 2

        def __getattr__(self, name):
            return object

    import sys as _sys

    _sys.modules["lzma"] = _MockLZMA()

# Mock multiprocessing — missing in Pyodide but imported by joblib
from unittest.mock import MagicMock

if "multiprocessing" not in sys.modules:
    m = MagicMock()
    m.cpu_count.return_value = 1
    sys.modules["multiprocessing"] = m
    sys.modules["multiprocessing.util"] = m.util
    sys.modules["multiprocessing.pool"] = m.pool

# Patch requests so pooch can fetch files already on /drive/mne_data.
# open_url works for both text and binary in Pyodide >= 0.21.
import requests
import pyodide

orig_send = requests.Session.send


def pyodide_send(self, request, **kwargs):
    try:
        buf = pyodide.http.open_url(request.url)
        content = buf.getvalue() if hasattr(buf, "getvalue") else buf.read()
        if isinstance(content, str):
            content = content.encode("utf-8")
    except Exception as e:
        print(f"open_url failed for {request.url}: {e}")
        return orig_send(self, request, **kwargs)
    response = requests.Response()
    response.status_code = 200
    response.url = request.url
    response.raw = io.BytesIO(content)
    return response


requests.Session.send = pyodide_send

# /drive/ in Pyodide requires Cross-Origin-Isolation headers
# (COOP/COEP) which many static servers (e.g. CircleCI artifacts)
# do not send. Fetch the data over HTTP into /tmp/mne_data instead
# — same-origin, no CORS. The data is served at the docs root
# (/mne_data/...) via Sphinx html_extra_path.
# Pyodide may run in a web worker (no `window`); `location` exists
# in both the main thread and workers, so use it to find the docs
# root by splitting on '/lite/'.
import pyodide.http as _phttp
import js as _js

try:
    _page = str(_js.location.href)
except Exception:
    _page = str(_js.window.location.href)
_base = _page.split("/lite/")[0] + "/mne_data/"
mne_data_path = "/tmp/mne_data"
_sample_dir = mne_data_path + "/MNE-sample-data"
# Eager 'core': small, commonly-used sample files fetched once at
# notebook start. The heavy files (raw / filt raw / ernoise / fwd /
# inv / src, ~360 MB total) are intentionally omitted here -- they are
# fetched lazily on first read via the reader shims below, so each
# notebook only downloads the sample files it actually uses.
_sample_files = [
    "version.txt",
    "MEG/sample/sample_audvis_raw-eve.fif",
    "MEG/sample/sample_audvis_filt-0-40_raw-eve.fif",
    "MEG/sample/sample_audvis_ecg-proj.fif",
    "MEG/sample/sample_audvis-cov.fif",
    "MEG/sample/sample_audvis-ave.fif",
    "MEG/sample/sample_audvis-no-filter-ave.fif",
    "MEG/sample/sample_audvis_raw-trans.fif",
    "MEG/sample/sample_audvis-shrunk-cov.fif",
    "MEG/sample/sample_audvis-meg-lh.stc",
    "MEG/sample/sample_audvis-meg-rh.stc",
    "subjects/sample/mri/T1.mgz",
    "subjects/sample/surf/rh.pial",
    "subjects/sample/surf/lh.pial",
    "subjects/sample/surf/rh.white",
    "subjects/sample/surf/lh.white",
    "subjects/sample/label/lh.aparc.annot",
    "subjects/sample/label/rh.aparc.annot",
    "SSS/sss_cal_mgh.dat",
    "SSS/ct_sparse_mgh.fif",
]
print("Fetching MNE sample data (once per session)...")
for _f in _sample_files:
    _dst = _sample_dir + "/" + _f
    if os.path.exists(_dst):
        continue
    _url = _base + "MNE-sample-data/" + _f
    try:
        _r = await _phttp.pyfetch(_url)
        if _r.status != 200:
            print(f"  HTTP {_r.status} for {_url}")
            continue
        _d = await _r.bytes()
        if _d[:4] == b"<!DO" or _d[:5] == b"<html":
            print(f"  skipped {_f} (server returned HTML)")
            continue
        os.makedirs(os.path.dirname(_dst), exist_ok=True)
        with open(_dst, "wb") as _fh:
            _fh.write(_d)
    except Exception as _e:
        print(f"  failed to fetch {_f}: {_e}")
os.makedirs(mne_data_path, exist_ok=True)
os.environ["MNE_DATA"] = mne_data_path
os.environ["MNE_DATASETS_SAMPLE_PATH"] = mne_data_path

# Block pooch from attempting large OSF downloads in the browser.
# The required files are either pre-injected or unavailable.
import pooch

orig_pooch_fetch = pooch.Pooch.fetch


def pyodide_pooch_fetch(self, fname, processor=None, downloader=None):
    url = self.get_url(fname)
    if "osf.io" in url or "files.osf.io" in url:
        raise RuntimeError(
            f"Cannot download {fname!r} from OSF in JupyterLite: "
            "browser CORS policy and memory limits prevent large "
            "dataset downloads. Open this notebook from mne.tools "
            "where sample data is pre-bundled, or run it locally."
        )
    return orig_pooch_fetch(self, fname, processor=processor, downloader=downloader)


pooch.Pooch.fetch = pyodide_pooch_fetch

# Import MNE and finalize setup.
import mne

# Pre-create a valid empty config file so MNE never hits a corrupt read.
_cfg = mne.get_config_path()
os.makedirs(os.path.dirname(_cfg), exist_ok=True)
if not os.path.exists(_cfg):
    with open(_cfg, "w") as _f:
        _f.write("{}")
mne.set_config("MNE_DATA", mne_data_path)
for ds in ["SAMPLE", "TESTING", "SSVEP", "EEGBCI", "SOMATO", "BRAINSTORM"]:
    mne.set_config(f"MNE_DATASETS_{ds}_PATH", mne_data_path)

# Bypass pooch's archive check: data_path() normally looks for the
# .tar.gz archive, not just the extracted folder. Return the folder
# directly so pooch never tries to download from OSF. Return a Path
# (not a str) since tutorials use the / operator on the result.
from pathlib import Path as _Path

_sample_path = _Path(_sample_dir)


def _lite_sample_data_path(*_a, **_kw):
    return _sample_path


mne.datasets.sample.data_path = _lite_sample_data_path


# Several non-sample datasets are each used by only a couple of
# notebooks (kiloword/erp_core for Epochs 30 & 40; mtrf/eegbci for the
# decoding examples), so fetch them LAZILY — only when their
# data_path()/load_data() is called — to avoid taxing every other
# notebook's setup. Pyodide runs in a web worker here, where a
# synchronous XHR may set responseType='arraybuffer', letting a sync
# data_path() read binary.
def _lite_fetch_rel(_rel):
    _dst = mne_data_path + "/" + _rel
    if not os.path.exists(_dst):
        from js import XMLHttpRequest

        _xhr = XMLHttpRequest.new()
        _xhr.open("GET", _base + _rel, False)
        _xhr.responseType = "arraybuffer"
        _xhr.send()
        if _xhr.status != 200:
            raise FileNotFoundError(f"Could not fetch {_rel} (HTTP {_xhr.status})")
        os.makedirs(os.path.dirname(_dst), exist_ok=True)
        with open(_dst, "wb") as _fh:
            _fh.write(bytes(_xhr.response.to_py()))
    return _dst


def _lite_lazy_fetch(_folder, _fname):
    _lite_fetch_rel(_folder + "/" + _fname)
    return _Path(mne_data_path + "/" + _folder)


def _lite_kiloword_data_path(*_a, **_kw):
    return _lite_lazy_fetch("MNE-kiloword-data", "kword_metadata-epo.fif")


mne.datasets.kiloword.data_path = _lite_kiloword_data_path


def _lite_erp_core_data_path(*_a, **_kw):
    return _lite_lazy_fetch(
        "MNE-ERP-CORE-data", "ERP-CORE_Subject-001_Task-Flankers_eeg.fif"
    )


mne.datasets.erp_core.data_path = _lite_erp_core_data_path


def _lite_mtrf_data_path(*_a, **_kw):
    return _lite_lazy_fetch("mTRF_1.5", "speech_data.mat")


mne.datasets.mtrf.data_path = _lite_mtrf_data_path


# testing hands back the folder and lets the shimmed readers pull
# individual files, so a notebook that wants the EEGLAB recording does
# not also drag down the 39 MB movement raw.
def _lite_testing_data_path(*_a, **_kw):
    return _Path(mne_data_path + "/MNE-testing-data")


mne.datasets.testing.data_path = _lite_testing_data_path


# Same again for the datasets behind a single example each. Only the
# files those examples read are served, and the shimmed readers below
# pull them individually.
def _lite_folder_data_path(_folder):
    def _data_path(*_a, **_kw):
        return _Path(mne_data_path + "/" + _folder)

    return _data_path


for _ds, _folder in (
    ("ssvep", "ssvep-example-data"),
    ("misc", "MNE-misc-data"),
    ("eyelink", "MNE-eyelink-data"),
    ("fnirs_motor", "MNE-fNIRS-motor-data"),
    ("refmeg_noise", "MNE-refmeg-noise-data"),
    ("phantom_kernel", "MNE-phantom-kernel-data"),
    ("multimodal", "MNE-multimodal-data"),
):
    getattr(mne.datasets, _ds).data_path = _lite_folder_data_path(_folder)


def _lite_eegbci_load_data(subject, runs, *_a, **_kw):
    _runs = [runs] if isinstance(runs, (int, float)) else list(runs)
    _subjects = list(subject) if isinstance(subject, (list, tuple)) else [subject]
    _out = []
    for _s in _subjects:
        for _r in _runs:
            _rel = (
                "MNE-eegbci-data/files/eegmmidb/1.0.0/"
                f"S{int(_s):03d}/S{int(_s):03d}R{int(_r):02d}.edf"
            )
            _out.append(_Path(_lite_fetch_rel(_rel)))
    return _out


mne.datasets.eegbci.load_data = _lite_eegbci_load_data


# Some MNE-sample-data files (e.g. the fixed-orientation forward/
# inverse used by the point-spread tutorial) aren't in the eager
# _sample_files list above because only one or two notebooks need
# them. Rather than hand-listing every such file, lazily fetch any
# sample-data path the first time read_forward_solution/
# read_inverse_operator is asked to open it.
def _lite_fetch_if_under_mne_data(fname):
    _p = str(fname)
    if _p.startswith(mne_data_path + "/"):
        _lite_fetch_rel(_p[len(mne_data_path) + 1 :])
    return fname


# Most readers just need their file pulled down before MNE opens it.
# One wrapper, driven by the table further below; readers that need
# more than this (a sibling file, a chain of candidates) keep their
# own shim.
def _lite_wrap_reader(_mods, _name, _arg):
    _orig = getattr(_mods[0], _name)

    def _wrapped(*_a, **_kw):
        if _a:
            _a = (_lite_fetch_if_under_mne_data(_a[0]),) + _a[1:]
        elif _arg in _kw:
            # positionally, as the hand-written shims did
            _a = (_lite_fetch_if_under_mne_data(_kw.pop(_arg)),)
        return _orig(*_a, **_kw)

    for _m in _mods:
        setattr(_m, _name, _wrapped)


# Lazily fetch the heavy sample raw / source-space files only when a
# notebook actually reads them (same pattern as the fwd/inv shims
# above), instead of pulling the whole sample set up front.
# Nearly every MNE reader validates its filename through
# _check_fname(must_exist=True) before opening it, so hooking that one
# function covers read_info, read_evokeds, read_cov, read_label and the
# rest without a wrapper each. Failures stay silent here so MNE still
# raises its own, clearer error for a file that genuinely is missing.
import mne.utils.check as _mne_check

_orig_check_fname = _mne_check._check_fname


def _lite_check_fname(fname, overwrite=False, must_exist=False, *_a, **_kw):
    if must_exist:
        try:
            _lite_fetch_if_under_mne_data(fname)
        except Exception:
            pass
    return _orig_check_fname(fname, overwrite, must_exist, *_a, **_kw)


_mne_check._check_fname = _lite_check_fname
# modules that imported it before now hold their own reference; ones
# loaded later (mne lazy-loads most of itself) pick up the patch
for _m in list(sys.modules.values()):
    if (
        getattr(_m, "__name__", "").startswith("mne")
        and getattr(_m, "_check_fname", None) is _orig_check_fname
    ):
        _m._check_fname = _lite_check_fname
# read_label, read_epochs and read_raw_edf open their file directly
# rather than validating it first, so the hook above never sees them
# an EEGLAB .set keeps its samples in a sibling .fdt, so fetch both
_orig_read_raw_eeglab = mne.io.read_raw_eeglab


def _lite_read_raw_eeglab(input_fname, *_a, **_kw):
    _p = str(input_fname)
    if _p.startswith(mne_data_path + "/"):
        for _cand in (_p, _p[:-4] + ".fdt"):
            try:
                _lite_fetch_rel(_cand[len(mne_data_path) + 1 :])
            except Exception:
                pass
    return _orig_read_raw_eeglab(input_fname, *_a, **_kw)


mne.io.read_raw_eeglab = _lite_read_raw_eeglab


# read_raw_nirx and read_raw_egi open a folder, so there is no single
# name to fetch; conf.py leaves a listing next to the copy.
def _lite_fetch_dir(_rel):
    _manifest = _lite_fetch_rel(_rel + "/_lite_manifest.txt")
    with open(_manifest) as _fh:
        _names = [_n.strip() for _n in _fh if _n.strip()]
    for _name in _names:
        # one unreachable member must not abandon the rest of the
        # recording; the reader complains if it needed that file
        try:
            _lite_fetch_rel(_rel + "/" + _name)
        except Exception as _e:
            print("[JupyterLite] skipped " + _name + ": " + repr(_e))
    return mne_data_path + "/" + _rel


def _lite_dir_reader(_orig):
    def _read(fname, *_a, **_kw):
        _p = str(fname)
        if _p.startswith(mne_data_path + "/"):
            try:
                _lite_fetch_dir(_p[len(mne_data_path) + 1 :])
            except Exception as _e:
                print("[JupyterLite] could not fetch " + _p + ": " + repr(_e))
        return _orig(fname, *_a, **_kw)

    return _read


mne.io.read_raw_nirx = _lite_dir_reader(mne.io.read_raw_nirx)
mne.io.read_raw_egi = _lite_dir_reader(mne.io.read_raw_egi)
# the logging tutorial reads a KIT file from inside the installed
# package; the wheel excludes mne/**/tests, so stage the served copy
# into the path the tutorial builds rather than editing the tutorial
import shutil as _shutil

_orig_read_raw_kit = mne.io.read_raw_kit


def _lite_read_raw_kit(input_fname, *_a, **_kw):
    _p = str(input_fname)
    if _p.endswith("test.sqd") and not os.path.exists(_p):
        try:
            _staged = _lite_fetch_rel("MNE-kit-testdata/test.sqd")
            os.makedirs(os.path.dirname(_p), exist_ok=True)
            _shutil.copyfile(_staged, _p)
        except Exception as _e:
            print("[JupyterLite] could not stage test.sqd: " + repr(_e))
    return _orig_read_raw_kit(input_fname, *_a, **_kw)


mne.io.read_raw_kit = _lite_read_raw_kit
# a BrainVision .vhdr is a text header pointing at a .eeg and a .vmrk
_orig_read_raw_brainvision = mne.io.read_raw_brainvision


def _lite_read_raw_brainvision(vhdr_fname, *_a, **_kw):
    _p = str(vhdr_fname)
    if _p.startswith(mne_data_path + "/"):
        _stem = _p[:-5] if _p.endswith(".vhdr") else _p
        for _cand in (_p, _stem + ".eeg", _stem + ".vmrk"):
            try:
                _lite_fetch_rel(_cand[len(mne_data_path) + 1 :])
            except Exception:
                pass
    return _orig_read_raw_brainvision(vhdr_fname, *_a, **_kw)


mne.io.read_raw_brainvision = _lite_read_raw_brainvision
# eyelink .asc recordings are single files
# the heatmap example draws its stimulus straight through pyplot, and
# read_xdf goes through pyxdf -- neither is an MNE reader, so shim the
# two entry points as well
import matplotlib.pyplot as _plt

_orig_imread = _plt.imread


def _lite_imread(fname, *_a, **_kw):
    return _orig_imread(_lite_fetch_if_under_mne_data(fname), *_a, **_kw)


_plt.imread = _lite_imread
try:
    import pyxdf as _pyxdf

    _orig_load_xdf = _pyxdf.load_xdf

    def _lite_load_xdf(fname, *_a, **_kw):
        return _orig_load_xdf(_lite_fetch_if_under_mne_data(fname), *_a, **_kw)

    _pyxdf.load_xdf = _lite_load_xdf
except Exception:
    pass
# The readers that only need the fetch. Two of them are bound on a
# private alias as well as the public one, so both are listed.
import mne.minimum_norm as _mne_minv
import mne.chpi as _mne_chpi

for _mods, _name, _arg in (
    ((mne,), "read_forward_solution", "fname"),
    ((_mne_minv, mne.minimum_norm), "read_inverse_operator", "fname"),
    ((mne.io,), "read_raw_fif", "fname"),
    ((mne.io,), "read_raw", "fname"),
    ((mne,), "read_source_spaces", "fname"),
    ((mne,), "read_label", "filename"),
    ((mne,), "read_epochs", "fname"),
    ((mne.io,), "read_raw_edf", "input_fname"),
    ((mne,), "read_bem_solution", "fname"),
    ((mne,), "read_events", "fname"),
    ((mne.io,), "read_raw_eyelink", "fname"),
    ((_mne_chpi, mne.chpi), "read_head_pos", "fname"),
):
    _lite_wrap_reader(_mods, _name, _arg)
# read_source_estimate is handed the stem of a .stc pair, so fetch
# both hemispheres before letting MNE resolve the name itself.
_orig_read_source_estimate = mne.read_source_estimate


def _lite_read_source_estimate(fname, *_a, **_kw):
    _p = str(fname)
    if _p.startswith(mne_data_path + "/"):
        for _suf in ("", "-lh.stc", "-rh.stc"):
            try:
                _lite_fetch_rel(_p[len(mne_data_path) + 1 :] + _suf)
            except Exception:
                pass
    return _orig_read_source_estimate(fname, *_a, **_kw)


mne.read_source_estimate = _lite_read_source_estimate
# plot_alignment locates its head surface by probing the filesystem
# with os.path.exists before any reader runs, so a reader shim never
# fires. Fetch the candidates first and let MNE choose as it normally
# would. Several viz modules bind the name at import time, so rebind
# it wherever the original landed instead of in one known place.
import mne._freesurfer as _mne_fs

_orig_get_head_surface = _mne_fs._get_head_surface


def _lite_get_head_surface(surf, subject, subjects_dir, bem=None, verbose=None):
    _sd = str(subjects_dir) if subjects_dir is not None else ""
    if subject and _sd.startswith(mne_data_path + "/"):
        _rel = _sd[len(mne_data_path) + 1 :] + "/" + str(subject)
        if surf in ("head-dense", "seghead"):
            _cands = ["bem/" + str(subject) + "-head-dense.fif", "surf/lh.seghead"]
        else:
            # same order MNE tries, so the browser picks the same
            # surface the rendered docs did
            _cands = ["bem/outer_skin.surf", "bem/" + str(subject) + "-head.fif"]
        for _c in _cands:
            try:
                _lite_fetch_rel(_rel + "/" + _c)
            except Exception:
                pass
    return _orig_get_head_surface(surf, subject, subjects_dir, bem=bem, verbose=verbose)


_mne_fs._get_head_surface = _lite_get_head_surface
# import the 3D module first so the sweep below is guaranteed to see
# it; anything imported later picks the patched name up on its own.
import mne.viz._3d  # noqa: F401

for _m in list(sys.modules.values()):
    if (
        getattr(_m, "__name__", "").startswith("mne")
        and getattr(_m, "_get_head_surface", None) is _orig_get_head_surface
    ):
        _m._get_head_surface = _lite_get_head_surface
# same story for the skull surfaces, which _check_fname insists
# already exist on disk
_orig_get_skull_surface = _mne_fs._get_skull_surface


def _lite_get_skull_surface(surf, subject, subjects_dir, bem=None, verbose=None):
    _sd = str(subjects_dir) if subjects_dir is not None else ""
    if subject and _sd.startswith(mne_data_path + "/"):
        try:
            _lite_fetch_rel(
                _sd[len(mne_data_path) + 1 :]
                + "/"
                + str(subject)
                + "/bem/"
                + surf
                + "_skull.surf"
            )
        except Exception:
            pass
    return _orig_get_skull_surface(
        surf, subject, subjects_dir, bem=bem, verbose=verbose
    )


_mne_fs._get_skull_surface = _lite_get_skull_surface
for _m in list(sys.modules.values()):
    if (
        getattr(_m, "__name__", "").startswith("mne")
        and getattr(_m, "_get_skull_surface", None) is _orig_get_skull_surface
    ):
        _m._get_skull_surface = _lite_get_skull_surface
# dig_mri_distances reaches a second, unrelated _get_head_surface, the
# one in mne/surface.py: it takes a list of candidate sources and
# probes bem/ with os.path.exists and glob, raising if the directory
# is absent, so the candidates have to land before it runs.
import mne.surface as _mne_surface

_orig_surface_head = _mne_surface._get_head_surface


def _lite_surface_head_surface(
    subject, source, subjects_dir, on_defects, raise_error=True
):
    _sd = str(subjects_dir) if subjects_dir is not None else ""
    if subject and _sd.startswith(mne_data_path + "/"):
        _rel = _sd[len(mne_data_path) + 1 :] + "/" + str(subject)
        _srcs = [source] if isinstance(source, str) else list(source)
        for _s in _srcs:
            try:
                _lite_fetch_rel(_rel + "/bem/" + str(subject) + "-" + _s + ".fif")
            except Exception:
                pass
    return _orig_surface_head(
        subject, source, subjects_dir, on_defects, raise_error=raise_error
    )


_mne_surface._get_head_surface = _lite_surface_head_surface
# plot_bem globs bem/*.surf and requires the bem directory to exist,
# so pull its three contours (plus the MRI it draws them on) down
# first; fetching creates the directory as a side effect.
_orig_plot_bem = mne.viz.plot_bem


def _lite_plot_bem(subject=None, subjects_dir=None, *_a, **_kw):
    _sd = str(subjects_dir) if subjects_dir is not None else ""
    if subject and _sd.startswith(mne_data_path + "/"):
        _rel = _sd[len(mne_data_path) + 1 :] + "/" + str(subject)
        _want = [
            "bem/inner_skull.surf",
            "bem/outer_skull.surf",
            "bem/outer_skin.surf",
            "mri/" + str(_kw.get("mri", "T1.mgz")),
        ]
        _bs = _kw.get("brain_surfaces")
        if _bs is not None:
            _bs = [_bs] if isinstance(_bs, str) else list(_bs)
            for _b in _bs:
                _want += ["surf/lh." + _b, "surf/rh." + _b]
        for _c in _want:
            try:
                _lite_fetch_rel(_rel + "/" + _c)
            except Exception:
                pass
    return _orig_plot_bem(subject, subjects_dir, *_a, **_kw)


mne.viz.plot_bem = _lite_plot_bem


# EXPERIMENTAL 3D: MNE's normal Brain/VTK stack can't load in WASM, so
# route SourceEstimate.plot() through pyvista-js (vtk.js) instead.
# pyvista-js (0.15) has no scalar colormap in its renderer, so we
# approximate MNE's Brain look with solid-colored meshes: a two-tone
# curvature base (light gyri + dark sulci) plus many thin 'hot' bands
# for the activation, on a black background with even scene lighting.
# Static, one time point, no time slider yet. Fully guarded — any
# failure prints a message so the notebook completes. Returns a stub
# 'brain' whose methods (add_foci/add_text/show_view/...) are safe
# no-ops, so tutorials that call brain.add_foci(...) after plot() work.
class _LiteBrain:
    def screenshot(self, *_a, **_kw):
        import numpy as _np

        return _np.zeros((2, 2, 3), dtype="uint8")

    def __getattr__(self, _name):
        return lambda *_a, **_kw: None


def _lite_stc_plot(self, *_a, **_kw):
    try:
        import numpy as _np
        import nibabel as _nib
        from scipy.spatial import cKDTree as _KDTree
        from matplotlib import colormaps as _cmaps
        import pyvista_js as _pv

        _subj = (
            _kw.get("subject")
            or (_a[0] if _a and isinstance(_a[0], str) else None)
            or "sample"
        )
        _sdir = _kw.get("subjects_dir")
        _sdir = (
            str(_sdir)
            if _sdir is not None
            else mne_data_path + "/MNE-sample-data/subjects"
        )
        # surfaces are fetched relative to the served mne_data root, so
        # derive that from subjects_dir rather than assuming sample --
        # a dataset may keep its FreeSurfer subjects under its own folder.
        _rel_sdir = (
            _sdir[len(mne_data_path) + 1 :]
            if _sdir.startswith(mne_data_path + "/")
            else "MNE-sample-data/subjects"
        )
        _init = _kw.get("initial_time", None)
        if _init is None:
            _ti = int(_np.argmax(_np.abs(self.data).mean(0)))
        else:
            _ti = int(_np.argmin(_np.abs(self.times - _init)))
        _hot = _cmaps["hot"]
        _N = 10

        def _flat(_t):
            return _np.hstack(
                [_np.full((len(_t), 1), 3, dtype=_np.int64), _t.astype(_np.int64)]
            ).ravel()

        def _sub(_pts, _tris, _mask, _lift=0.0, _cen=None):
            _sel = _tris[_mask]
            if len(_sel) == 0:
                return None
            _u, _iv = _np.unique(_sel, return_inverse=True)
            _p = _pts[_u]
            if _lift and _cen is not None:
                _p = _cen + (_p - _cen) * (1.0 + _lift)
            return _p, _iv.reshape(-1, 3)

        _plotter = _pv.Plotter()
        _plotter.background_color = "black"
        # even lighting so the surface isn't black when rotated
        for _lp in (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ):
            _plotter.add_light(
                _pv.Light(
                    position=(300.0 * _lp[0], 300.0 * _lp[1], 300.0 * _lp[2]),
                    focal_point=(0.0, 0.0, 0.0),
                    intensity=0.4,
                )
            )
        _nlh = len(self.vertices[0])
        _hemis = (("lh", 0, self.vertices[0]), ("rh", 1, self.vertices[1]))
        for _h, _hi, _vno in _hemis:
            if len(_vno) == 0:
                continue
            _pre = _rel_sdir + "/" + _subj + "/surf/" + _h
            _lite_fetch_rel(_pre + ".inflated")
            _lite_fetch_rel(_pre + ".curv")
            _bpath = _sdir + "/" + _subj + "/surf/" + _h
            _rr, _tris = mne.read_surface(_bpath + ".inflated")
            _cv = _nib.freesurfer.read_morph_data(_bpath + ".curv")
            _hdata = self.data[:_nlh] if _hi == 0 else self.data[_nlh:]
            # color each surface vertex from the nearest ACTIVE source
            # within a small radius, so single-vertex (point) sources
            # show as visible blobs and dense sources fill in as usual
            _sv = _hdata[:, _ti].astype(float)
            _act = _sv != 0
            _scal = _np.zeros(len(_rr))
            if _act.any():
                _atree = _KDTree(_rr[_vno][_act])
                _ad, _ai = _atree.query(_rr)
                _scal = _np.where(_ad <= 12.0, _sv[_act][_ai], 0.0)
            # offset hemispheres along x so they do not overlap
            _off = -60.0 if _h == "lh" else 60.0
            _pts = _np.round(_rr, 2)
            _pts[:, 0] = _pts[:, 0] + _off
            _cen = _pts.mean(0)
            # curvature base: light gyri (curv<0) + dark sulci (curv>=0)
            _fc = _cv[_tris].mean(1)
            for _cm, _col in (
                (_fc < 0, (0.68, 0.68, 0.68)),
                (_fc >= 0, (0.38, 0.38, 0.38)),
            ):
                _s = _sub(_pts, _tris, _cm)
                if _s is not None:
                    _plotter.add_mesh(
                        _pv.PolyData(points=_s[0], faces=_flat(_s[1])),
                        color=_col,
                        smooth_shading=True,
                    )
            # activation as a smooth hot gradient in N value bands,
            # each lifted 2% off the surface to avoid z-fighting
            _fv = _scal[_tris].mean(1)
            _p90 = _np.percentile(_scal, 90.0)
            _fmax = float(_scal.max())
            # keep the background gray: for sparse point sources the
            # 90th pct is ~0 (most of the brain is zero), which would
            # paint everything, so fall back to a fraction of the max.
            _fmin = _p90 if _p90 > _fmax * 0.05 else _fmax * 0.4
            if _fmax > _fmin:
                _edges = _np.linspace(_fmin, _fmax, _N + 1)
                for _i in range(_N):
                    if _i < _N - 1:
                        _m = (_fv >= _edges[_i]) & (_fv < _edges[_i + 1])
                    else:
                        _m = _fv >= _edges[_i]
                    if int(_m.sum()) == 0:
                        continue
                    _rgb = _hot(0.25 + 0.41 * (_i / (_N - 1)))
                    _col = (float(_rgb[0]), float(_rgb[1]), float(_rgb[2]))
                    _s = _sub(_pts, _tris, _m, 0.02, _cen)
                    if _s is not None:
                        _plotter.add_mesh(
                            _pv.PolyData(points=_s[0], faces=_flat(_s[1])),
                            color=_col,
                            smooth_shading=True,
                        )
        # Open on the lateral profile (camera along the medial-lateral
        # X axis, superior up), like native MNE, instead of vtk.js's
        # default anterior/face-on view. Guarded so a missing
        # view_vector never costs us the render.
        try:
            _plotter.view_vector((-1.0, 0.0, 0.0), viewup=(0.0, 0.0, 1.0))
        except Exception:
            pass
        _plotter.show()
    except Exception as _e:
        print("[JupyterLite] pyvista-js 3D render unavailable: " + repr(_e))
    return _LiteBrain()


mne.SourceEstimate.plot = _lite_stc_plot

# Pyodide/WASM has no OS threads, so MNE's ProgressBar background
# updater thread (used by the ProgressBar context manager, e.g. in
# permutation cluster tests) crashes with 'can't start new thread'.
# That thread only animates a cosmetic bar — the computation runs on
# the main thread and __exit__ writes the final state — so no-op its
# start/join. Only affects notebooks that use it; results are unchanged.
try:
    from mne.utils import progressbar as _mpb

    _mpb._UpdateThread.start = lambda self: None
    _mpb._UpdateThread.join = lambda self, *_a, **_kw: None
except Exception:
    pass
# tqdm also spawns its own monitor thread, which likewise can't start in
# WASM and emits a TqdmMonitorWarning. Setting monitor_interval=0 before
# any bar is created skips that thread entirely (bars still display).
try:
    import tqdm as _tqdm

    _tqdm.tqdm.monitor_interval = 0
except Exception:
    pass

# Switch matplotlib to inline so figures render in the notebook.
import IPython

IPython.get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt

# Silence the spurious 'FigureCanvasAgg is non-interactive' warning
# at its source. MNE's plt_show calls fig.show() (the inline backend
# isn't detected as 'agg'), and the inline Agg canvas warns. Patching
# viz.utils.plt_show is not enough: other modules did
# `from .utils import plt_show` and hold their own reference. Every
# path resolves fig.show on the class at call time, so a no-op here
# silences it everywhere. Figures still render via the inline backend.
import matplotlib.figure as _mfig

_mfig.Figure.show = lambda self, *a, **k: None
import importlib

viz_utils = importlib.import_module("mne.viz.utils")


# Also display+close via IPython for paths that call plt_show
# directly, so figures render exactly once.
def pyodide_plt_show(show=True, fig=None, **kwargs):
    if not show:
        return
    import IPython.display

    _f = fig if fig is not None else plt.gcf()
    IPython.display.display(_f)
    plt.close(_f)


viz_utils.plt_show = pyodide_plt_show


# EXPERIMENTAL 3D: plot_sparse_source_estimates builds its 3D renderer
# BEFORE the time-course figure, so in WASM the whole call dies and the
# notebook loses both halves. Rebuild it here: the same glass brain from
# the source space and a marker per active dipole via pyvista-js, plus
# the matplotlib time courses (which are the quantitative half). Same
# approach as the SourceEstimate.plot shim above.
def _lite_plot_sparse_source_estimates(
    src,
    stcs,
    colors=None,
    linewidth=2,
    fontsize=18,
    bgcolor=(0.05, 0, 0.1),
    opacity=0.2,
    brain_color=(0.7,) * 3,
    show=True,
    high_resolution=False,
    fig_name=None,
    fig_number=None,
    labels=None,
    modes=("cone", "sphere"),
    scale_factors=(1, 0.6),
    **kwargs,
):
    import numpy as _np
    from itertools import cycle as _cycle
    from matplotlib.colors import to_rgb as _to_rgb

    if not isinstance(stcs, list):
        stcs = [stcs]
    _lhp = src[0]["rr"]
    _pts = _np.r_[_lhp, src[1]["rr"]] * 170
    _nrm = _np.r_[src[0]["nn"], src[1]["nn"]]
    # use_tris is the decimated mesh and can be None on some source
    # spaces; fall back to the full tris in that case.
    _lt = src[0]["tris"] if high_resolution else src[0]["use_tris"]
    _rt = src[1]["tris"] if high_resolution else src[1]["use_tris"]
    if _lt is None or _rt is None:
        _lt, _rt = src[0]["tris"], src[1]["tris"]
    _faces = _np.r_[_lt, len(_lhp) + _rt]
    _vertnos = [_np.r_[_s.lh_vertno, len(_lhp) + _s.rh_vertno] for _s in stcs]
    _uniq = _np.unique(_np.concatenate(_vertnos).ravel())
    # --- time courses -------------------------------------------------
    _fig = plt.figure(fig_number, layout="constrained")
    _fig.clf()
    _ax = _fig.add_subplot(111)
    _cyc = _cycle(
        colors
        if colors is not None
        else plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    _marks = []
    for _v in _uniq:
        _ind = [_k for _k, _vn in enumerate(_vertnos) if _v in _vn]
        _c = next(_cyc)
        _marks.append((int(_v), _to_rgb(_c), len(_ind) > 1))
        for _k in _ind:
            _m = _vertnos[_k] == _v
            _ax.plot(
                1e3 * stcs[_k].times,
                1e9 * stcs[_k].data[_m].ravel(),
                c=_c,
                linewidth=linewidth,
            )
    _ax.set_xlabel("Time (ms)", fontsize=fontsize)
    _ax.set_ylabel("Source amplitude (nAm)", fontsize=fontsize)
    if fig_name is not None:
        _ax.set_title(fig_name)
    pyodide_plt_show(show)
    # --- glass brain + dipole markers ---------------------------------
    try:
        import pyvista_js as _pv

        _plotter = _pv.Plotter()
        _plotter.background_color = tuple(
            float(min(max(_x, 0.0), 1.0)) for _x in bgcolor
        )
        for _lp in (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ):
            _plotter.add_light(
                _pv.Light(
                    position=(300.0 * _lp[0], 300.0 * _lp[1], 300.0 * _lp[2]),
                    focal_point=(0.0, 0.0, 0.0),
                    intensity=0.4,
                )
            )
        _flat_faces = _np.hstack(
            [_np.full((len(_faces), 1), 3, dtype=_np.int32), _faces.astype(_np.int32)]
        ).ravel()
        _plotter.add_mesh(
            _pv.PolyData(points=_pts.astype(_np.float32), faces=_flat_faces),
            color=tuple(float(_x) for _x in brain_color),
            opacity=float(opacity),
            smooth_shading=True,
        )
        for _v, _col, _common in _marks:
            _sf = float(scale_factors[1] if _common else scale_factors[0])
            _mode = modes[1] if _common else modes[0]
            _xyz = tuple(float(_q) for _q in _pts[_v])
            if _mode == "sphere":
                _glyph = _pv.Sphere(radius=_sf, center=_xyz)
            else:
                _glyph = _pv.Cone(
                    center=_xyz,
                    direction=tuple(float(_q) for _q in _nrm[_v]),
                    height=2.0 * _sf,
                    radius=_sf,
                )
            _plotter.add_mesh(_glyph, color=_col, smooth_shading=True)
        try:
            _plotter.view_vector((-1.0, 0.0, 0.0), viewup=(0.0, 0.0, 1.0))
        except Exception:
            pass
        _plotter.show()
    except Exception as _e:
        print("[JupyterLite] pyvista-js glass brain unavailable: " + repr(_e))


mne.viz.plot_sparse_source_estimates = _lite_plot_sparse_source_estimates

# Each MNE plot is rendered once by pyodide_plt_show above (display()).
# When a plot call is also a cell's last expression, the method returns
# the Figure, which Jupyter echoes a SECOND time as the Out[] result
# (the duplicate seen below inline plots). Drop that redundant echo for
# Figures (and pure lists of Figures, e.g. ica.plot_properties) so each
# plot appears exactly once. Non-figure results (numbers, DataFrames,
# reprs) are untouched, and raw matplotlib figures never shown still
# render via the inline backend's end-of-cell flush, so nothing hides.
# Wrapped in try/except (like the patches below): if anything about
# the displayhook is unexpected, silently keep the current behavior
# (harmless double render) rather than breaking the setup cell.
try:
    _lite_dh = type(IPython.get_ipython().displayhook)
    if not getattr(_lite_dh, "_lite_no_fig_echo", False):
        _lite_dh_call = _lite_dh.__call__

        def _lite_displayhook(self, result=None):
            if isinstance(result, _mfig.Figure):
                result = None
            elif (
                isinstance(result, (list, tuple))
                and result
                and all(isinstance(_x, _mfig.Figure) for _x in result)
            ):
                result = None
            return _lite_dh_call(self, result)

        _lite_dh.__call__ = _lite_displayhook
        _lite_dh._lite_no_fig_echo = True
except Exception:
    pass

# Real fix (not a warnings filter) for the threadpoolctl Pyodide
# RuntimeWarning seen via mne.sys_info(): threadpoolctl 3.6.0 (latest
# release) still calls the deprecated Pyodide JsProxy.as_object_map().
# Pyodide's own message says to use as_py_json() instead; both yield the
# same library filepaths, so we swap the call at its source. This removes
# the deprecated API usage entirely, so the warning is never emitted.
# The upstream fix is already merged (joblib/threadpoolctl#201) but
# unreleased; Pyodide bundles the released 3.6.0 wheel. DROP THIS PATCH
# once threadpoolctl 3.7.0 is released and Pyodide bundles it.
try:
    import os as _os
    import threadpoolctl as _tpc

    def _find_libraries_pyodide(self):
        from pyodide_js._module import LDSO

        for _fp in LDSO.loadedLibsByName.as_py_json():
            if _os.path.exists(_fp):
                self._make_controller_from_path(_fp)

    _tpc.ThreadpoolController._find_libraries_pyodide = _find_libraries_pyodide
except Exception:
    pass
