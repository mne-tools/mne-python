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

# Layout, in the order the sections appear below:
#   1. install MNE into the browser kernel
#   2. patch what Pyodide lacks, before MNE is imported
#   3. work out where the data is served and copy the core of it in
#   4. import MNE and point its datasets at that copy
#   5. define the fetch helpers everything after this point uses
#   6. tell MNE where each dataset lives
#   7. wrap the readers, in three groups by how much work each needs
#   8. stub out what WebAssembly cannot do

# --- JupyterLite setup cell -------------------------------------------------
# 💡 This cell is automatically added to the start of each notebook.
# It installs MNE and patches the browser environment for Pyodide.
# Downloading this notebook to run it locally? Delete this cell first:
# piplite exists only inside JupyterLite, and a local MNE needs none of
# the patches below.

# === 1. Install ==============================================================
import piplite

# Use piplite (not micropip) so the locally-built development MNE wheel
# bundled into the JupyterLite build is preferred over the older PyPI
# release: piplite checks the local index first and falls back to PyPI
# for dependencies.
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

# === 2. Pyodide compatibility, before MNE is imported ========================
import sys
import os
import io
import inspect
from pathlib import Path

# Mock multiprocessing — missing in Pyodide but imported by joblib
from unittest.mock import MagicMock

if "multiprocessing" not in sys.modules:
    _mp = MagicMock()
    _mp.cpu_count.return_value = 1
    sys.modules["multiprocessing"] = _mp
    sys.modules["multiprocessing.util"] = _mp.util
    sys.modules["multiprocessing.pool"] = _mp.pool

# Route requests over the browser's own transport so the downloads that still
# go through pooch work here. That path is pooch.retrieve ->
# pooch.HTTPDownloader -> requests, taken by the fetchers whose files live off
# the docs site (fetch_fsaverage, fetch_infant_template and the parcellation
# ones) and so are not in the copy html_extra_path serves. Every notebook that
# calls one is on JUPYTERLITE_EXCLUDE today, so nothing reaches this on the
# badged pages; it stays because that list is the only thing keeping it that
# way, and a notebook added to the gallery tomorrow would otherwise fail here
# with a Pyodide socket error rather than a real HTTP one.
#
# XMLHttpRequest rather than pyodide.http.open_url, and the same blocking call
# _lite_fetch_rel uses below: open_url reports no status, so a 404 page came
# back looking like a successful 200 and pooch wrote the error page to disk,
# only failing later on a confusing hash mismatch. XHR gives the real status,
# which is what pooch's raise_for_status() needs. Nothing is caught here: the
# browser is the only transport available, so a failure has no fallback worth
# taking and the real error is more useful than a substituted one.
import requests

_orig_send = requests.Session.send


def _pyodide_send(self, request, **kwargs):
    from js import XMLHttpRequest

    _xhr = XMLHttpRequest.new()
    _xhr.open(request.method or "GET", request.url, False)
    _xhr.responseType = "arraybuffer"
    _xhr.send()
    response = requests.Response()
    response.status_code = _xhr.status
    response.url = request.url
    response.raw = io.BytesIO(bytes(_xhr.response.to_py()))
    return response


requests.Session.send = _pyodide_send

# === 3. Where the data comes from ===========================================
# /drive/ in Pyodide requires Cross-Origin-Isolation headers
# (COOP/COEP) which many static servers (e.g. CircleCI artifacts)
# do not send. Fetch the data over HTTP into /tmp/mne_data instead:
# same-origin, no CORS. The data is served at the docs root
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
_mne_data_root = Path(mne_data_path)
_sample_dir = _mne_data_root / "MNE-sample-data"
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
# These are served from the same origin as this page, so if the page loaded,
# the server is up: a miss here means the docs build did not stage the file,
# not that the network is flaky. Several of them (the SSS calibration pair,
# the surfaces read through nibabel) have no lazy path either, so a miss would
# otherwise surface as a confusing error many cells later. Collect every
# failure and raise once, naming them all, since one staging bug usually drops
# more than one file.
# print, not a logger: this cell runs in the browser kernel, not in the Sphinx
# process, so its output is simply what the notebook reader sees.
print("Fetching MNE sample data (once per session)...")
_missing = []
for _f in _sample_files:
    _dst = _sample_dir / _f
    if _dst.exists():
        continue
    _url = _base + "MNE-sample-data/" + _f
    try:
        _r = await pyodide.http.pyfetch(_url)
        if _r.status != 200:
            _missing.append(f"{_f} (HTTP {_r.status})")
            continue
        _d = await _r.bytes()
        # a static server answers a missing path with its 404 page and a 200
        # status, so the body is the only way to tell the two apart
        if _d[:4] == b"<!DO" or _d[:5] == b"<html":
            _missing.append(f"{_f} (server returned HTML, so it is not there)")
            continue
        _dst.parent.mkdir(parents=True, exist_ok=True)
        _dst.write_bytes(_d)
    except Exception as _e:
        _missing.append(f"{_f} ({_e!r})")
if _missing:
    raise RuntimeError(
        "The JupyterLite build did not stage these sample files under "
        + _base
        + ":\n  "
        + "\n  ".join(_missing)
    )
_mne_data_root.mkdir(parents=True, exist_ok=True)
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

# === 4. Import MNE ==========================================================
import mne

# Pre-create a valid empty config file so MNE never hits a corrupt read.
_cfg = Path(mne.get_config_path())
_cfg.parent.mkdir(parents=True, exist_ok=True)
if not _cfg.exists():
    _cfg.write_text("{}")
mne.set_config("MNE_DATA", mne_data_path)
for _ds in ["SAMPLE", "TESTING", "SSVEP", "EEGBCI", "SOMATO", "BRAINSTORM"]:
    mne.set_config(f"MNE_DATASETS_{_ds}_PATH", mne_data_path)
del _ds

# === 5. Fetch helpers =======================================================
# Nothing is on disk in this kernel. The datasets are served over HTTP next to
# the docs, and MNE's readers all assume a real filesystem, so every file has
# to be written into the virtual filesystem *before* the code that opens it
# runs. Section 3 did that for a small eager core; everything else arrives on
# demand, through these helpers.
#
# Every download funnels through _lite_fetch_rel, which takes a path relative
# to the data root and is a no-op once the file is there. The two above it
# translate between the three ways a path shows up in MNE: absolute (what a
# reader is handed), relative to the data root (what the server wants), and a
# dataset folder (what data_path() returns). The rest build on those.


def _lite_data_path(rel):
    """Return ``rel`` resolved under the data root."""
    return _mne_data_root / rel


def _lite_rel_to_data(fname):
    """Return ``fname`` relative to the data root, or None if it sits outside.

    Everything below uses this to decide whether a path is ours to fetch, so it
    has to reject lookalikes: /tmp/mne_data_other is not under /tmp/mne_data.
    """
    _p = Path(str(fname))
    if _p == _mne_data_root or not _p.is_relative_to(_mne_data_root):
        return None
    return _p.relative_to(_mne_data_root).as_posix()


def _lite_fetch_rel(rel):
    """Download one file into the virtual filesystem and return its path.

    Synchronous on purpose: it is called from inside MNE readers, which cannot
    await. Pyodide runs in a web worker here, where a blocking XHR may set
    responseType='arraybuffer', which is what lets a sync call read binary.
    """
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


def _lite_fetch_if_under_mne_data(fname):
    """Fetch ``fname`` if it names something we serve, then hand it back.

    The identity return is what lets this wrap a reader argument in place.
    """
    _rel = _lite_rel_to_data(fname)
    if _rel is not None:
        _lite_fetch_rel(_rel)
    return fname


def _lite_fetch_dir(rel):
    """Fetch a directory-shaped recording using the manifest served with it.

    read_raw_nirx and read_raw_egi are handed a folder rather than a file, so
    there is no single name to fetch; conf.py leaves a listing next to the copy.
    """
    _manifest = _lite_fetch_rel(rel + "/_lite_manifest.txt")
    _names = [_n.strip() for _n in _manifest.read_text().splitlines() if _n.strip()]
    for _name in _names:
        # one unreachable member must not abandon the rest of the
        # recording; the reader complains if it needed that file
        try:
            _lite_fetch_rel(rel + "/" + _name)
        except Exception as _e:
            print("[JupyterLite] skipped " + _name + ": " + repr(_e))
    return _lite_data_path(rel)


def _lite_fetch_optional(rels):
    """Fetch each of ``rels``, tolerating the ones that are not served.

    This is the only place a failed fetch is allowed to pass quietly, and it is
    used only where a miss is an expected outcome rather than a fault: MNE is
    being offered a list of candidates it will choose between (group B), or the
    optional companions of a multi-file format (group C). If the file really
    was needed, the reader that opens it raises its own, clearer error. A file
    that must exist is fetched through _lite_fetch_rel directly, which raises.
    """
    for _r in rels:
        try:
            _lite_fetch_rel(_r)
        except Exception:
            pass


def _lite_fetch_candidates(subject, subjects_dir, rel_paths):
    """Fetch ``rel_paths`` under ``<subjects_dir>/<subject>/`` (group B)."""
    _rel = _lite_rel_to_data(subjects_dir if subjects_dir is not None else "")
    if not subject or _rel is None:
        return
    _lite_fetch_optional(f"{_rel}/{subject}/{_p}" for _p in rel_paths)


def _lite_dataset_path(folder, probe=None):
    """Build a ``data_path()`` that returns ``folder`` under the data root.

    With ``probe``, the named file is fetched when data_path() is called. That
    is what covers mtrf, whose .mat is read by scipy rather than by an MNE
    reader, so nothing downstream would otherwise fetch it.
    """

    def _data_path(*args, **kwargs):
        if probe is not None:
            _lite_fetch_rel(folder + "/" + probe)
        return _lite_data_path(folder)

    return _data_path


def _lite_wrap_reader(module, name):
    """Wrap ``module.name`` so its filename argument is fetched before it opens.

    The keyword to intercept is read off the wrapped function rather than
    listed by hand: the readers below disagree about whether it is ``fname``,
    ``filename`` or ``input_fname``, and a name written out here that drifted
    from the real one would silently stop fetching for keyword callers.
    """
    orig = getattr(module, name)
    arg = next(iter(inspect.signature(orig).parameters))

    def wrapped(*args, **kwargs):
        if args:
            args = (_lite_fetch_if_under_mne_data(args[0]),) + args[1:]
        elif arg in kwargs:
            # move it to a positional argument, since it is no longer in kwargs
            args = (_lite_fetch_if_under_mne_data(kwargs.pop(arg)),)
        return orig(*args, **kwargs)

    setattr(module, name, wrapped)


def _lite_dir_reader(orig):
    """Wrap a reader that is handed a folder rather than a file."""

    def _read(fname, *args, **kwargs):
        _rel = _lite_rel_to_data(fname)
        if _rel is not None:
            try:
                _lite_fetch_dir(_rel)
            except Exception as _e:
                print("[JupyterLite] could not fetch " + str(fname) + ": " + repr(_e))
        return orig(fname, *args, **kwargs)

    return _read


def _lite_rebind(name, old, new):
    """Point every module that already imported ``old`` at ``new``.

    MNE lazy-loads most of itself, so a module that ran ``from x import f``
    before this cell holds its own reference and would not see the patch.
    Modules imported afterwards pick it up on their own.
    """
    for _m in list(sys.modules.values()):
        if (
            getattr(_m, "__name__", "").startswith("mne")
            and getattr(_m, name, None) is old
        ):
            setattr(_m, name, new)


# === 6. Where MNE looks for each dataset ====================================
# data_path() normally checks for the .tar.gz archive, not just the extracted
# folder, and would try to download from OSF when it does not find one. Point
# each dataset at its folder under the data root instead. The ones with a probe
# file are used by only a couple of notebooks each, so nothing is fetched until
# their data_path() is actually called.
for _ds, _folder, _probe in (
    ("sample", "MNE-sample-data", None),
    # testing hands back the folder and lets the shimmed readers pull
    # individual files, so a notebook that wants the EEGLAB recording does
    # not also drag down the 39 MB movement raw.
    ("testing", "MNE-testing-data", None),
    # datasets behind a single example each; only the files those examples
    # read are served, and the readers below pull them individually
    ("ssvep", "ssvep-example-data", None),
    ("misc", "MNE-misc-data", None),
    ("eyelink", "MNE-eyelink-data", None),
    ("fnirs_motor", "MNE-fNIRS-motor-data", None),
    ("refmeg_noise", "MNE-refmeg-noise-data", None),
    ("phantom_kernel", "MNE-phantom-kernel-data", None),
    ("multimodal", "MNE-multimodal-data", None),
    # kiloword/erp_core for Epochs 30 & 40, mtrf for the decoding examples
    ("kiloword", "MNE-kiloword-data", "kword_metadata-epo.fif"),
    ("erp_core", "MNE-ERP-CORE-data", "ERP-CORE_Subject-001_Task-Flankers_eeg.fif"),
    ("mtrf", "mTRF_1.5", "speech_data.mat"),
):
    getattr(mne.datasets, _ds).data_path = _lite_dataset_path(_folder, _probe)
del _ds, _folder, _probe


# eegbci is addressed by subject and run rather than by path, so it needs its
# own shim rather than a row in the table above.
def _lite_eegbci_load_data(subjects, runs, *args, **kwargs):
    # the parameter is `subjects`, matching MNE: 35_eeg_no_mri calls it by
    # keyword, so a shim spelled `subject` would raise TypeError there
    _runs = [runs] if isinstance(runs, (int, float)) else list(runs)
    _subjects = list(subjects) if isinstance(subjects, (list, tuple)) else [subjects]
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

# === 7. Reader overrides ====================================================
# MNE functions need one of three treatments here, depending on how much they
# do before the file is actually opened.
#
#   A. reads one file, and validates the name first. Nearly every MNE reader
#      calls _check_fname(must_exist=True) before opening anything, so patching
#      that single function covers read_info, read_evokeds, read_cov,
#      read_label and the rest at once. A handful skip the validation, and are
#      listed in a table instead. Nothing else is needed for this group.
#   B. probes the filesystem before any reader runs. _get_head_surface calls
#      os.path.exists, plot_bem globs bem/*.surf, so a fetch-on-open hook never
#      fires. The candidates have to be on disk before the probe.
#   C. one filename that means several files. read_raw_brainvision is handed
#      only the .vhdr, opens it, reads the names of its .eeg and .vmrk out of
#      it, and opens those, by which point we are inside the reader and it is
#      too late to fetch. Same shape for EEGLAB (.set + .fdt), a .stc stem
#      (lh + rh), and the formats that are a directory rather than a file.

# --- A. reads one file ------------------------------------------------------
# The general hook. Failures stay silent here so MNE still raises its own,
# clearer error for a file that genuinely is missing.
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
_lite_rebind("_check_fname", _orig_check_fname, _lite_check_fname)
# The readers that open their file without validating it first, so the hook
# above never sees them. Each needs nothing but its file fetched.
for _module, _name in (
    (mne, "read_forward_solution"),
    (mne.minimum_norm, "read_inverse_operator"),
    (mne.io, "read_raw_fif"),
    (mne.io, "read_raw"),
    (mne, "read_source_spaces"),
    (mne, "read_label"),
    (mne, "read_epochs"),
    (mne.io, "read_raw_edf"),
    (mne, "read_bem_solution"),
    (mne, "read_events"),
    (mne.io, "read_raw_eyelink"),
    (mne.chpi, "read_head_pos"),
):
    _lite_wrap_reader(_module, _name)
del _module, _name
# The eyetracking heatmap example draws its stimulus straight through pyplot,
# and read_xdf goes through pyxdf. Neither is an MNE reader, but both take a
# path we serve, so they get the same treatment.
import matplotlib.pyplot as plt

_orig_imread = plt.imread


def _lite_imread(fname, *args, **kwargs):
    return _orig_imread(_lite_fetch_if_under_mne_data(fname), *args, **kwargs)


plt.imread = _lite_imread
# guarded: pyxdf has no pure-Python wheel on every Pyodide build, and only the
# XDF example needs it
try:
    import pyxdf

    _orig_load_xdf = pyxdf.load_xdf

    def _lite_load_xdf(fname, *args, **kwargs):
        return _orig_load_xdf(_lite_fetch_if_under_mne_data(fname), *args, **kwargs)

    pyxdf.load_xdf = _lite_load_xdf
except Exception:
    pass

# --- B. probes the filesystem first -----------------------------------------
# plot_alignment locates its head surface with os.path.exists before any reader
# runs. Fetch the candidates first and let MNE choose as it normally would.
# Several viz modules bind the name at import time, so rebind it wherever the
# original landed rather than in one known place.
import mne._freesurfer as mne_fs

_orig_get_head_surface = mne_fs._get_head_surface


def _lite_get_head_surface(surf, subject, subjects_dir, bem=None, verbose=None):
    if surf in ("head-dense", "seghead"):
        _cands = [f"bem/{subject}-head-dense.fif", "surf/lh.seghead"]
    else:
        # same order MNE tries, so the browser picks the same
        # surface the rendered docs did
        _cands = ["bem/outer_skin.surf", f"bem/{subject}-head.fif"]
    _lite_fetch_candidates(subject, subjects_dir, _cands)
    return _orig_get_head_surface(surf, subject, subjects_dir, bem=bem, verbose=verbose)


mne_fs._get_head_surface = _lite_get_head_surface
# import the 3D module first so the rebind is guaranteed to see it;
# anything imported later picks the patched name up on its own.
import mne.viz._3d  # noqa: F401

_lite_rebind("_get_head_surface", _orig_get_head_surface, _lite_get_head_surface)
# same story for the skull surfaces, which _check_fname insists
# already exist on disk
_orig_get_skull_surface = mne_fs._get_skull_surface


def _lite_get_skull_surface(surf, subject, subjects_dir, bem=None, verbose=None):
    _lite_fetch_candidates(subject, subjects_dir, [f"bem/{surf}_skull.surf"])
    return _orig_get_skull_surface(
        surf, subject, subjects_dir, bem=bem, verbose=verbose
    )


mne_fs._get_skull_surface = _lite_get_skull_surface
_lite_rebind("_get_skull_surface", _orig_get_skull_surface, _lite_get_skull_surface)
# dig_mri_distances reaches a second, unrelated _get_head_surface, the
# one in mne/surface.py: it takes a list of candidate sources and
# probes bem/ with os.path.exists and glob, raising if the directory
# is absent, so the candidates have to land before it runs.
import mne.surface as mne_surface

_orig_surface_head = mne_surface._get_head_surface


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


# no _lite_rebind for this one: unlike the _freesurfer function above, nothing
# outside mne/surface.py imports it by name, so patching the module is enough.
mne_surface._get_head_surface = _lite_surface_head_surface
# plot_bem globs bem/*.surf and requires the bem directory to exist,
# so pull its three contours (plus the MRI it draws them on) down
# first; fetching creates the directory as a side effect.
_orig_plot_bem = mne.viz.plot_bem


def _lite_plot_bem(subject=None, subjects_dir=None, *args, **kwargs):
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
            _want += [f"surf/lh.{_b}", f"surf/rh.{_b}"]
    _lite_fetch_candidates(subject, subjects_dir, _want)
    return _orig_plot_bem(subject, subjects_dir, *args, **kwargs)


mne.viz.plot_bem = _lite_plot_bem

# --- C. one filename, several files -----------------------------------------
# An EEGLAB .set keeps its samples in a sibling .fdt, so fetch both.
_orig_read_raw_eeglab = mne.io.read_raw_eeglab


def _lite_read_raw_eeglab(input_fname, *args, **kwargs):
    _rel = _lite_rel_to_data(input_fname)
    if _rel is not None:
        _lite_fetch_optional((_rel, _rel[:-4] + ".fdt"))
    return _orig_read_raw_eeglab(input_fname, *args, **kwargs)


mne.io.read_raw_eeglab = _lite_read_raw_eeglab
# a BrainVision .vhdr is a text header pointing at a .eeg and a .vmrk
_orig_read_raw_brainvision = mne.io.read_raw_brainvision


def _lite_read_raw_brainvision(vhdr_fname, *args, **kwargs):
    _rel = _lite_rel_to_data(vhdr_fname)
    if _rel is not None:
        _stem = _rel[:-5] if _rel.endswith(".vhdr") else _rel
        _lite_fetch_optional((_rel, _stem + ".eeg", _stem + ".vmrk"))
    return _orig_read_raw_brainvision(vhdr_fname, *args, **kwargs)


mne.io.read_raw_brainvision = _lite_read_raw_brainvision
# read_source_estimate is handed the stem of a .stc pair, so fetch
# both hemispheres before letting MNE resolve the name itself.
_orig_read_source_estimate = mne.read_source_estimate


def _lite_read_source_estimate(fname, *args, **kwargs):
    _rel = _lite_rel_to_data(fname)
    if _rel is not None:
        _lite_fetch_optional(_rel + _suf for _suf in ("", "-lh.stc", "-rh.stc"))
    return _orig_read_source_estimate(fname, *args, **kwargs)


mne.read_source_estimate = _lite_read_source_estimate
# read_raw_nirx and read_raw_egi open a folder, listed by its manifest
mne.io.read_raw_nirx = _lite_dir_reader(mne.io.read_raw_nirx)
mne.io.read_raw_egi = _lite_dir_reader(mne.io.read_raw_egi)
# the odd one out in this group: the name is enough, but it points inside the
# installed package rather than at the data root. The logging tutorial builds
# a path into mne/**/tests, which the wheel excludes, so copy the served file
# to where the tutorial expects it rather than editing the tutorial.
import shutil

_orig_read_raw_kit = mne.io.read_raw_kit


def _lite_read_raw_kit(input_fname, *args, **kwargs):
    _p = Path(str(input_fname))
    if _p.name == "test.sqd" and not _p.exists():
        try:
            _staged = _lite_fetch_rel("MNE-kit-testdata/test.sqd")
            _p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_staged, _p)
        except Exception as _e:
            print("[JupyterLite] could not stage test.sqd: " + repr(_e))
    return _orig_read_raw_kit(input_fname, *args, **kwargs)


mne.io.read_raw_kit = _lite_read_raw_kit

# === 8. What WebAssembly cannot do ==========================================
# Pyodide/WASM has no OS threads, so MNE's ProgressBar background
# updater thread (used by the ProgressBar context manager, e.g. in
# permutation cluster tests) crashes with 'can't start new thread'.
# That thread only animates a cosmetic bar: the computation runs on
# the main thread and __exit__ writes the final state, so no-op its
# start/join. Only affects notebooks that use it; results are unchanged.
# Guarded because this is a private MNE path: if it is ever renamed, losing a
# cosmetic patch is better than failing every notebook at the setup cell.
try:
    from mne.utils import progressbar

    progressbar._UpdateThread.start = lambda self: None
    progressbar._UpdateThread.join = lambda self, *args, **kwargs: None
except Exception:
    pass
# tqdm also spawns its own monitor thread, which likewise can't start in
# WASM and emits a TqdmMonitorWarning. Setting monitor_interval=0 before
# any bar is created skips that thread entirely (bars still display).
# Guarded because tqdm is a transitive dependency that may not be installed.
try:
    import tqdm

    tqdm.tqdm.monitor_interval = 0
except Exception:
    pass

# Switch matplotlib to inline so figures render in the notebook.
import IPython

IPython.get_ipython().run_line_magic("matplotlib", "inline")

# Silence the spurious 'FigureCanvasAgg is non-interactive' warning that the
# inline Agg canvas raises from fig.show(). MNE's own plt_show no longer
# triggers it (gh-14076 taught it to call plt.show() on inline backends), but
# tutorials still call fig.show() directly -- 50_ssvep does it four times, and
# 10_background_stats once -- and those warn. Every path resolves fig.show on
# the class at call time, so a no-op here covers them all. Figures still
# render via the inline backend.
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
