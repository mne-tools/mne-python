# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# This file is notebook source rather than a module: it installs packages with
# a top-level ``await`` and imports them only afterwards, so the rules about
# import position, await position and import order do not apply to it. Ruff
# still lints and formats everything else here, which is the point of keeping
# it as a real file instead of a string.
# ruff: noqa: E402, F704, I001

# Naming: everything this cell defines lands in the notebook's own namespace,
# so anything it invents is _-prefixed and cannot shadow a variable the
# tutorial goes on to use. Module imports are left plain: a tutorial importing
# the same module binds the same object, so there is nothing to protect.
# `mne_data_path` is the deliberate exception, since a reader may want it.

# --- JupyterLite setup cell -------------------------------------------------
# 💡 This cell is automatically added to the start of each notebook.
# It installs MNE and patches the browser environment for Pyodide.
# Downloading this notebook to run it locally? Delete this cell first:
# piplite exists only inside JupyterLite, and a local MNE needs none of
# the patches below.
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

# Mock multiprocessing — missing in Pyodide but imported by joblib
from unittest.mock import MagicMock

if "multiprocessing" not in sys.modules:
    m = MagicMock()
    m.cpu_count.return_value = 1
    sys.modules["multiprocessing"] = m
    sys.modules["multiprocessing.util"] = m.util
    sys.modules["multiprocessing.pool"] = m.pool

# Route requests through pyodide.http so the downloads that still go through
# pooch work in the browser. The one that matters is fetch_infant_template
# (25_automated_coreg), which reaches pooch.retrieve -> pooch.HTTPDownloader
# -> requests, and whose files live on github.com rather than OSF. open_url
# handles both text and binary in Pyodide >= 0.21.
import requests
import pyodide

_orig_send = requests.Session.send


def _pyodide_send(self, request, **kwargs):
    try:
        buf = pyodide.http.open_url(request.url)
        content = buf.getvalue() if hasattr(buf, "getvalue") else buf.read()
        if isinstance(content, str):
            content = content.encode("utf-8")
    except Exception as e:
        print(f"open_url failed for {request.url}: {e}")
        return _orig_send(self, request, **kwargs)
    response = requests.Response()
    response.status_code = 200
    response.url = request.url
    response.raw = io.BytesIO(content)
    return response


requests.Session.send = _pyodide_send

# /drive/ in Pyodide requires Cross-Origin-Isolation headers
# (COOP/COEP) which many static servers (e.g. CircleCI artifacts)
# do not send. Fetch the data over HTTP into /tmp/mne_data instead
# — same-origin, no CORS. The data is served at the docs root
# (/mne_data/...) via Sphinx html_extra_path.
# Pyodide may run in a web worker (no `window`); `location` exists
# in both the main thread and workers, so use it to find the docs
# root by splitting on '/lite/'.
import pyodide.http
import js

try:
    _page = str(js.location.href)
except Exception:
    _page = str(js.window.location.href)
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
        _r = await pyodide.http.pyfetch(_url)
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

# Turn an OSF download into a readable error rather than an opaque CORS or
# out-of-memory failure. This covers Pooch.fetch, which is the path
# mne/datasets/_fetch.py uses for every packaged dataset; the handful of
# callers that use pooch.retrieve directly all point at other hosts.
import pooch
from urllib.parse import urlparse

_orig_pooch_fetch = pooch.Pooch.fetch


def _pyodide_pooch_fetch(self, fname, processor=None, downloader=None):
    url = self.get_url(fname)
    # Compare the host rather than searching the whole URL: "osf.io" can turn
    # up legitimately elsewhere in one (a query string, a path), and a
    # substring test would refuse those downloads too.
    host = urlparse(url).hostname or ""
    if host == "osf.io" or host.endswith(".osf.io"):
        raise RuntimeError(
            f"Cannot download {fname!r} from OSF in JupyterLite: "
            "browser CORS policy and memory limits prevent large "
            "dataset downloads. Open this notebook from mne.tools "
            "where sample data is pre-bundled, or run it locally."
        )
    return _orig_pooch_fetch(self, fname, processor=processor, downloader=downloader)


pooch.Pooch.fetch = _pyodide_pooch_fetch

# Import MNE and finalize setup.
import mne

# Pre-create a valid empty config file so MNE never hits a corrupt read.
_cfg = mne.get_config_path()
os.makedirs(os.path.dirname(_cfg), exist_ok=True)
if not os.path.exists(_cfg):
    with open(_cfg, "w") as _f:
        _f.write("{}")
mne.set_config("MNE_DATA", mne_data_path)
for _ds in ["SAMPLE", "TESTING", "SSVEP", "EEGBCI", "SOMATO", "BRAINSTORM"]:
    mne.set_config(f"MNE_DATASETS_{_ds}_PATH", mne_data_path)
del _ds

# Bypass pooch's archive check: data_path() normally looks for the
# .tar.gz archive, not just the extracted folder. Return the folder
# directly so pooch never tries to download from OSF. Return a Path
# (not a str) since tutorials use the / operator on the result.
from pathlib import Path

_mne_data_root = Path(mne_data_path)


def _lite_data_path(rel):
    """Return ``rel`` resolved under the data root."""
    return _mne_data_root / rel


def _lite_rel_to_data(fname):
    """Return ``fname`` relative to the data root, or None if it sits outside.

    Replaces the ``startswith(mne_data_path + "/")`` plus manual slicing this
    file used to repeat at every reader shim.
    """
    _p = Path(str(fname))
    if _p == _mne_data_root or not _p.is_relative_to(_mne_data_root):
        return None
    return _p.relative_to(_mne_data_root).as_posix()


_sample_path = Path(_sample_dir)


def _lite_sample_data_path(*args, **kwargs):
    return _sample_path


mne.datasets.sample.data_path = _lite_sample_data_path


# Several non-sample datasets are each used by only a couple of
# notebooks (kiloword/erp_core for Epochs 30 & 40; mtrf/eegbci for the
# decoding examples), so fetch them LAZILY — only when their
# data_path()/load_data() is called — to avoid taxing every other
# notebook's setup. Pyodide runs in a web worker here, where a
# synchronous XHR may set responseType='arraybuffer', letting a sync
# data_path() read binary.
def _lite_fetch_rel(rel):
    _dst = _lite_data_path(rel)
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


def _lite_lazy_fetch(_folder, _fname):
    _lite_fetch_rel(_folder + "/" + _fname)
    return _lite_data_path(_folder)


def _lite_kiloword_data_path(*args, **kwargs):
    return _lite_lazy_fetch("MNE-kiloword-data", "kword_metadata-epo.fif")


mne.datasets.kiloword.data_path = _lite_kiloword_data_path


def _lite_erp_core_data_path(*args, **kwargs):
    return _lite_lazy_fetch(
        "MNE-ERP-CORE-data", "ERP-CORE_Subject-001_Task-Flankers_eeg.fif"
    )


mne.datasets.erp_core.data_path = _lite_erp_core_data_path


def _lite_mtrf_data_path(*args, **kwargs):
    return _lite_lazy_fetch("mTRF_1.5", "speech_data.mat")


mne.datasets.mtrf.data_path = _lite_mtrf_data_path


# testing hands back the folder and lets the shimmed readers pull
# individual files, so a notebook that wants the EEGLAB recording does
# not also drag down the 39 MB movement raw.
def _lite_testing_data_path(*args, **kwargs):
    return _lite_data_path("MNE-testing-data")


mne.datasets.testing.data_path = _lite_testing_data_path


# Same again for the datasets behind a single example each. Only the
# files those examples read are served, and the shimmed readers below
# pull them individually.
def _lite_folder_data_path(_folder):
    def _data_path(*args, **kwargs):
        return _lite_data_path(_folder)

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


def _lite_eegbci_load_data(subject, runs, *args, **kwargs):
    _runs = [runs] if isinstance(runs, (int, float)) else list(runs)
    _subjects = list(subject) if isinstance(subject, (list, tuple)) else [subject]
    _out = []
    for _s in _subjects:
        for _r in _runs:
            _rel = (
                "MNE-eegbci-data/files/eegmmidb/1.0.0/"
                f"S{int(_s):03d}/S{int(_s):03d}R{int(_r):02d}.edf"
            )
            _out.append(_lite_fetch_rel(_rel))
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
    if _lite_rel_to_data(_p) is not None:
        _lite_fetch_rel(_lite_rel_to_data(_p))
    return fname


# Reader overrides.
#
# Nothing is on disk here. The data is served over HTTP next to the docs, so
# a file has to be in the virtual filesystem by the time a reader opens it.
#
# There IS one general hook: nearly every MNE reader validates its filename
# through _check_fname(must_exist=True) first, so patching that one function
# (further down) covers read_info, read_evokeds, read_cov, read_label and the
# rest with no wrapper each. Three kinds of caller escape it, and those are
# what the wrappers below are for:
#
#   1. one filename that means several files. read_raw_brainvision is handed
#      only the .vhdr, opens it, reads the names of its .eeg and .vmrk out of
#      it, and opens those -- by which point we are inside the reader and it
#      is too late to fetch. Same shape for EEGLAB (.set + .fdt), a .stc stem
#      (lh + rh) and the formats that are a directory rather than a file.
#   2. code that probes instead of opening. _get_head_surface calls
#      os.path.exists before any reader runs, so a fetch-on-open hook never
#      fires for it.
#   3. readers that open their file without validating it first.
#
# The ones in group 3 need nothing but the fetch, so they are driven by the
# table further down rather than a shim each.
#
# `module` is where the name is bound, `name` is the function and `arg` is the
# keyword its filename arrives under when it is not passed positionally.
def _lite_wrap_reader(module, name, arg):
    orig = getattr(module, name)

    def wrapped(*args, **kwargs):
        if args:
            args = (_lite_fetch_if_under_mne_data(args[0]),) + args[1:]
        elif arg in kwargs:
            # positionally, as the hand-written shims did
            args = (_lite_fetch_if_under_mne_data(kwargs.pop(arg)),)
        return orig(*args, **kwargs)

    setattr(module, name, wrapped)


# Lazily fetch the heavy sample raw / source-space files only when a
# notebook actually reads them (same pattern as the fwd/inv shims
# above), instead of pulling the whole sample set up front.
# Nearly every MNE reader validates its filename through
# _check_fname(must_exist=True) before opening it, so hooking that one
# function covers read_info, read_evokeds, read_cov, read_label and the
# rest without a wrapper each. Failures stay silent here so MNE still
# raises its own, clearer error for a file that genuinely is missing.
import mne.utils.check as mne_check

_orig_check_fname = mne_check._check_fname


def _lite_check_fname(fname, overwrite=False, must_exist=False, *args, **kwargs):
    if must_exist:
        try:
            _lite_fetch_if_under_mne_data(fname)
        except Exception:
            pass
    return _orig_check_fname(fname, overwrite, must_exist, *args, **kwargs)


mne_check._check_fname = _lite_check_fname
# modules that imported it before now hold their own reference; ones
# loaded later (mne lazy-loads most of itself) pick up the patch
for _m in list(sys.modules.values()):
    if (
        getattr(_m, "__name__", "").startswith("mne")
        and getattr(_m, "_check_fname", None) is _orig_check_fname
    ):
        _m._check_fname = _lite_check_fname
# Below are the readers the _check_fname hook cannot serve on its own,
# because one filename implies more than one file.
#
# An EEGLAB .set keeps its samples in a sibling .fdt, so fetch both.
_orig_read_raw_eeglab = mne.io.read_raw_eeglab


def _lite_read_raw_eeglab(input_fname, *args, **kwargs):
    _p = str(input_fname)
    if _lite_rel_to_data(_p) is not None:
        for _cand in (_p, _p[:-4] + ".fdt"):
            try:
                _lite_fetch_rel(_lite_rel_to_data(_cand))
            except Exception:
                pass
    return _orig_read_raw_eeglab(input_fname, *args, **kwargs)


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
    return _lite_data_path(_rel)


def _lite_dir_reader(_orig):
    def _read(fname, *args, **kwargs):
        _p = str(fname)
        if _lite_rel_to_data(_p) is not None:
            try:
                _lite_fetch_dir(_lite_rel_to_data(_p))
            except Exception as _e:
                print("[JupyterLite] could not fetch " + _p + ": " + repr(_e))
        return _orig(fname, *args, **kwargs)

    return _read


mne.io.read_raw_nirx = _lite_dir_reader(mne.io.read_raw_nirx)
mne.io.read_raw_egi = _lite_dir_reader(mne.io.read_raw_egi)
# the logging tutorial reads a KIT file from inside the installed
# package; the wheel excludes mne/**/tests, so stage the served copy
# into the path the tutorial builds rather than editing the tutorial
import shutil

_orig_read_raw_kit = mne.io.read_raw_kit


def _lite_read_raw_kit(input_fname, *args, **kwargs):
    _p = str(input_fname)
    if _p.endswith("test.sqd") and not os.path.exists(_p):
        try:
            _staged = _lite_fetch_rel("MNE-kit-testdata/test.sqd")
            os.makedirs(os.path.dirname(_p), exist_ok=True)
            shutil.copyfile(_staged, _p)
        except Exception as _e:
            print("[JupyterLite] could not stage test.sqd: " + repr(_e))
    return _orig_read_raw_kit(input_fname, *args, **kwargs)


mne.io.read_raw_kit = _lite_read_raw_kit
# a BrainVision .vhdr is a text header pointing at a .eeg and a .vmrk
_orig_read_raw_brainvision = mne.io.read_raw_brainvision


def _lite_read_raw_brainvision(vhdr_fname, *args, **kwargs):
    _p = str(vhdr_fname)
    if _lite_rel_to_data(_p) is not None:
        _stem = _p[:-5] if _p.endswith(".vhdr") else _p
        for _cand in (_p, _stem + ".eeg", _stem + ".vmrk"):
            try:
                _lite_fetch_rel(_lite_rel_to_data(_cand))
            except Exception:
                pass
    return _orig_read_raw_brainvision(vhdr_fname, *args, **kwargs)


mne.io.read_raw_brainvision = _lite_read_raw_brainvision
# eyelink .asc recordings are single files
# the heatmap example draws its stimulus straight through pyplot, and
# read_xdf goes through pyxdf -- neither is an MNE reader, so shim the
# two entry points as well
import matplotlib.pyplot as plt

_orig_imread = plt.imread


def _lite_imread(fname, *args, **kwargs):
    return _orig_imread(_lite_fetch_if_under_mne_data(fname), *args, **kwargs)


plt.imread = _lite_imread
try:
    import pyxdf as _pyxdf

    _orig_load_xdf = _pyxdf.load_xdf

    def _lite_load_xdf(fname, *args, **kwargs):
        return _orig_load_xdf(_lite_fetch_if_under_mne_data(fname), *args, **kwargs)

    _pyxdf.load_xdf = _lite_load_xdf
except Exception:
    pass
# The tier-one table (see "Reader overrides" above for why this exists).
# Each row is one reader that needs nothing but its file fetched first:
#   module  where the name is bound
#   name    the function to wrap there
#   arg     the keyword its filename arrives under, for calls that pass it
#           by name rather than positionally
for _module, _name, _arg in (
    (mne, "read_forward_solution", "fname"),
    (mne.minimum_norm, "read_inverse_operator", "fname"),
    (mne.io, "read_raw_fif", "fname"),
    (mne.io, "read_raw", "fname"),
    (mne, "read_source_spaces", "fname"),
    (mne, "read_label", "filename"),
    (mne, "read_epochs", "fname"),
    (mne.io, "read_raw_edf", "input_fname"),
    (mne, "read_bem_solution", "fname"),
    (mne, "read_events", "fname"),
    (mne.io, "read_raw_eyelink", "fname"),
    (mne.chpi, "read_head_pos", "fname"),
):
    _lite_wrap_reader(_module, _name, _arg)
# read_source_estimate is handed the stem of a .stc pair, so fetch
# both hemispheres before letting MNE resolve the name itself.
_orig_read_source_estimate = mne.read_source_estimate


def _lite_read_source_estimate(fname, *args, **kwargs):
    _p = str(fname)
    if _lite_rel_to_data(_p) is not None:
        for _suf in ("", "-lh.stc", "-rh.stc"):
            try:
                _lite_fetch_rel(_lite_rel_to_data(_p) + _suf)
            except Exception:
                pass
    return _orig_read_source_estimate(fname, *args, **kwargs)


mne.read_source_estimate = _lite_read_source_estimate
# plot_alignment locates its head surface by probing the filesystem
# with os.path.exists before any reader runs, so a reader shim never
# fires. Fetch the candidates first and let MNE choose as it normally
# would. Several viz modules bind the name at import time, so rebind
# it wherever the original landed instead of in one known place.
import mne._freesurfer as mne_fs

_orig_get_head_surface = mne_fs._get_head_surface


def _lite_get_head_surface(surf, subject, subjects_dir, bem=None, verbose=None):
    _sd = str(subjects_dir) if subjects_dir is not None else ""
    if subject and _lite_rel_to_data(_sd) is not None:
        _rel = _lite_rel_to_data(_sd) + "/" + str(subject)
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


mne_fs._get_head_surface = _lite_get_head_surface
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
_orig_get_skull_surface = mne_fs._get_skull_surface


def _lite_get_skull_surface(surf, subject, subjects_dir, bem=None, verbose=None):
    _sd = str(subjects_dir) if subjects_dir is not None else ""
    if subject and _lite_rel_to_data(_sd) is not None:
        try:
            _lite_fetch_rel(
                _lite_rel_to_data(_sd)
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


mne_fs._get_skull_surface = _lite_get_skull_surface
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
import mne.surface as mne_surface

_orig_surface_head = mne_surface._get_head_surface


def _lite_surface_head_surface(
    subject, source, subjects_dir, on_defects, raise_error=True
):
    _sd = str(subjects_dir) if subjects_dir is not None else ""
    if subject and _lite_rel_to_data(_sd) is not None:
        _rel = _lite_rel_to_data(_sd) + "/" + str(subject)
        _srcs = [source] if isinstance(source, str) else list(source)
        for _s in _srcs:
            try:
                _lite_fetch_rel(_rel + "/bem/" + str(subject) + "-" + _s + ".fif")
            except Exception:
                pass
    return _orig_surface_head(
        subject, source, subjects_dir, on_defects, raise_error=raise_error
    )


mne_surface._get_head_surface = _lite_surface_head_surface
# plot_bem globs bem/*.surf and requires the bem directory to exist,
# so pull its three contours (plus the MRI it draws them on) down
# first; fetching creates the directory as a side effect.
_orig_plot_bem = mne.viz.plot_bem


def _lite_plot_bem(subject=None, subjects_dir=None, *args, **kwargs):
    _sd = str(subjects_dir) if subjects_dir is not None else ""
    if subject and _lite_rel_to_data(_sd) is not None:
        _rel = _lite_rel_to_data(_sd) + "/" + str(subject)
        _want = [
            "bem/inner_skull.surf",
            "bem/outer_skull.surf",
            "bem/outer_skin.surf",
            "mri/" + str(kwargs.get("mri", "T1.mgz")),
        ]
        _bs = kwargs.get("brain_surfaces")
        if _bs is not None:
            _bs = [_bs] if isinstance(_bs, str) else list(_bs)
            for _b in _bs:
                _want += ["surf/lh." + _b, "surf/rh." + _b]
        for _c in _want:
            try:
                _lite_fetch_rel(_rel + "/" + _c)
            except Exception:
                pass
    return _orig_plot_bem(subject, subjects_dir, *args, **kwargs)


mne.viz.plot_bem = _lite_plot_bem


# Pyodide/WASM has no OS threads, so MNE's ProgressBar background
# updater thread (used by the ProgressBar context manager, e.g. in
# permutation cluster tests) crashes with 'can't start new thread'.
# That thread only animates a cosmetic bar — the computation runs on
# the main thread and __exit__ writes the final state — so no-op its
# start/join. Only affects notebooks that use it; results are unchanged.
try:
    from mne.utils import progressbar as _mpb

    _mpb._UpdateThread.start = lambda self: None
    _mpb._UpdateThread.join = lambda self, *args, **kwargs: None
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

# Silence the spurious 'FigureCanvasAgg is non-interactive' warning
# at its source. MNE's plt_show calls fig.show() (the inline backend
# isn't detected as 'agg'), and the inline Agg canvas warns. Patching
# viz.utils.plt_show is not enough: other modules did
# `from .utils import plt_show` and hold their own reference. Every
# path resolves fig.show on the class at call time, so a no-op here
# silences it everywhere. Figures still render via the inline backend.
import matplotlib.figure as mpl_figure

mpl_figure.Figure.show = lambda self, *a, **k: None
import importlib

_viz_utils = importlib.import_module("mne.viz.utils")


# Also display+close via IPython for paths that call plt_show
# directly, so figures render exactly once.
def _pyodide_plt_show(show=True, fig=None, **kwargs):
    if not show:
        return
    import IPython.display

    _f = fig if fig is not None else plt.gcf()
    IPython.display.display(_f)
    plt.close(_f)


_viz_utils.plt_show = _pyodide_plt_show
