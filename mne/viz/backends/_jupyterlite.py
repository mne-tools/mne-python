"""Adapt MNE to the Pyodide kernel the JupyterLite docs run in.

The docs prepend a cell to every notebook that installs MNE and calls
:func:`setup_notebook`. Pyodide has no OS threads, no VTK, and a virtual
filesystem the datasets are not in, so this fetches the files the docs serve
next to the pages over HTTP on first use, points the dataset fetchers at those
served folders, stubs what WebAssembly cannot do, and selects the vtk.js
renderer (see ``_lite.py``). Nothing here runs outside that kernel.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import functools
import inspect
import io
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

from ..._freesurfer import _get_head_surface, _get_skull_surface, read_talxfm
from ...surface import _get_head_surface as _surface_head_surface
from ...utils import progressbar
from ...utils.check import _check_fname

_base = None  # URL the docs serve the data at, set by setup_notebook
_data_root = None  # where the fetched files land
_orig = dict()  # the functions patched below that are imported lazily, by name


# /tmp here is the Pyodide in-browser virtual filesystem, not a shared host dir
def setup_notebook(data_path="/tmp/mne_data"):  # nosec B108
    """Patch MNE for the browser kernel; the docs' first cell calls this.

    Parameters
    ----------
    data_path : str
        Where fetched data files are written; ``MNE_DATA`` is set to it.
    """
    global _base, _data_root
    if _data_root is not None:  # the cell ran again: the wrappers below would
        return  # otherwise wrap their own wrapped selves and recurse
    import js  # only exists inside Pyodide

    # The docs serve the data next to the pages (/mne_data/, via
    # html_extra_path). Pyodide may run in a web worker, where ``location``
    # exists but ``window`` does not.
    _base = str(js.location.href).split("/lite/")[0] + "/mne_data/"
    _data_root = Path(data_path)
    _data_root.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = data_path
    _patch_downloads()
    _patch_datasets()
    _patch_readers()
    _patch_wasm()
    # VTK has no WebAssembly build, so draw with pyvista-js (vtk.js) instead
    from ...viz import set_3d_backend

    try:
        set_3d_backend("jupyterlite_notebook")
    except Exception as exc:
        print(f"[JupyterLite] could not select the pyvista-js renderer: {exc!r}")


# --- Fetching ----------------------------------------------------------------
def _rel_to_data(fname):
    """Return ``fname`` relative to the data root, or None if it sits outside."""
    path = Path(str(fname))
    if path == _data_root or not path.is_relative_to(_data_root):
        return None
    return path.relative_to(_data_root).as_posix()


def _fetch_rel(rel):
    """Download one file (once) into the virtual filesystem and return its path.

    Synchronous, since it runs inside MNE readers, which cannot await; a
    blocking XHR may read binary in a web worker, where the kernel runs.
    """
    dst = _data_root / rel
    if not dst.exists():
        from js import XMLHttpRequest

        xhr = XMLHttpRequest.new()
        xhr.open("GET", _base + rel, False)
        xhr.responseType = "arraybuffer"
        xhr.send()
        if xhr.status != 200:
            raise FileNotFoundError(f"Could not fetch {rel} (HTTP {xhr.status})")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(bytes(xhr.response.to_py()))
    return dst


def _fetch_optional(rels):
    """Fetch what is served among ``rels``, quietly skipping the rest.

    These are candidates MNE will choose between, or optional companions of a
    multi-file format; the reader raises its own error for a file it needed.
    """
    for rel in rels:
        try:
            _fetch_rel(rel)
        except Exception:
            pass


def _fetch_candidates(subject, subjects_dir, rel_paths):
    """Fetch ``rel_paths`` under ``<subjects_dir>/<subject>/``."""
    rel = _rel_to_data(subjects_dir if subjects_dir is not None else "")
    if subject and rel is not None:
        _fetch_optional(f"{rel}/{subject}/{p}" for p in rel_paths)


def _rebind(name, old, new):
    """Replace ``old`` with ``new`` in every MNE module that holds it as ``name``.

    That includes the module defining it, so later importers get ``new`` too,
    but not this one, whose copies are the originals the wrappers call.
    """
    for mod in list(sys.modules.values()):
        if (
            getattr(mod, "__name__", "").startswith("mne")
            and mod is not sys.modules[__name__]
            and getattr(mod, name, None) is old
        ):
            setattr(mod, name, new)


def _wrap_reader(orig, siblings=None, module=None):
    """Wrap a reader to fetch its filename argument before it opens it.

    ``siblings`` maps a relative path to the other files that name implies (a
    BrainVision header's .eeg and .vmrk). The filename's keyword is read off
    the signature, since readers call it fname, filename or input_fname. The
    wrapper replaces ``orig`` where it is defined (or on ``module``) and in
    every MNE module that imported it.
    """
    name = orig.__name__
    arg = next(iter(inspect.signature(orig).parameters))

    def wrapped(*args, **kwargs):
        if arg in kwargs:  # normalize to positional
            args = (kwargs.pop(arg),) + args
        if args:
            rel = _rel_to_data(args[0])
            if rel is not None:
                _fetch_optional([rel] + (siblings(rel) if siblings else []))
        return orig(*args, **kwargs)

    setattr(sys.modules[orig.__module__] if module is None else module, name, wrapped)
    _rebind(name, orig, wrapped)


def _dir_reader(orig):
    """Wrap a reader of a folder, listed by the manifest the docs leave in it."""

    def read(fname, *args, **kwargs):
        rel = _rel_to_data(fname)
        if rel is not None:
            try:
                names = _fetch_rel(rel + "/_lite_manifest.txt").read_text()
                _fetch_optional(rel + "/" + name for name in names.split())
            except Exception as exc:
                print(f"[JupyterLite] could not fetch {fname}: {exc!r}")
        return orig(fname, *args, **kwargs)

    return read


# --- Downloads MNE makes itself ----------------------------------------------
def _pyodide_send(self, request, **kwargs):
    """Send a requests.Request through the browser.

    For the pooch downloads that fetch from off-site (fetch_fsaverage and
    friends). A synchronous XMLHttpRequest reports the real HTTP status, which
    is what pooch's raise_for_status() needs.
    """
    import requests
    from js import XMLHttpRequest

    xhr = XMLHttpRequest.new()
    xhr.open(request.method or "GET", request.url, False)
    xhr.responseType = "arraybuffer"
    xhr.send()
    response = requests.Response()
    response.status_code = xhr.status
    response.reason = xhr.statusText
    response.url = request.url
    response.raw = io.BytesIO(bytes(xhr.response.to_py()))
    return response


def _pooch_fetch(self, fname, processor=None, downloader=None):
    """Refuse an OSF download with a reason rather than a CORS or memory error."""
    host = urlparse(self.get_url(fname)).hostname or ""
    if host == "osf.io" or host.endswith(".osf.io"):
        raise RuntimeError(
            f"Cannot download {fname!r} from OSF in JupyterLite: browser CORS "
            "policy and memory limits prevent large dataset downloads. Open this "
            "notebook from mne.tools, where the data is bundled, or run it locally."
        )
    return _orig["pooch_fetch"](self, fname, processor=processor, downloader=downloader)


def _patch_downloads():
    import pooch
    import requests

    requests.Session.send = _pyodide_send
    _orig["pooch_fetch"] = pooch.Pooch.fetch
    pooch.Pooch.fetch = _pooch_fetch


# --- Where MNE looks for each dataset ----------------------------------------
def _served_data_path(folder, probe, *args, **kwargs):
    """Stand in for a dataset's data_path(), which would download from OSF.

    A probe file is fetched for datasets whose data is read by something other
    than an MNE reader (scipy for mtrf).
    """
    if probe is not None:
        _fetch_rel(folder + "/" + probe)
    return _data_root / folder


def _eegbci_load_data(subjects, runs, *args, **kwargs):
    """Fetch EEGBCI runs by subject and run number."""
    runs = [runs] if isinstance(runs, int | float) else list(runs)
    subjects = list(subjects) if isinstance(subjects, list | tuple) else [subjects]
    return [
        _fetch_rel(
            f"MNE-eegbci-data/files/eegmmidb/1.0.0/S{s:03d}/S{s:03d}R{r:02d}.edf"
        )
        for s in subjects
        for r in runs
    ]


def _patch_datasets():
    from ...datasets import (
        eegbci,
        eyelink,
        fnirs_motor,
        kiloword,
        misc,
        mtrf,
        multimodal,
        phantom_kernel,
        sample,
        ssvep,
    )

    for dataset, folder, probe in (
        (sample, "MNE-sample-data", None),
        (ssvep, "ssvep-example-data", None),
        (misc, "MNE-misc-data", None),
        (eyelink, "MNE-eyelink-data", None),
        (fnirs_motor, "MNE-fNIRS-motor-data", None),
        (phantom_kernel, "MNE-phantom-kernel-data", None),
        (multimodal, "MNE-multimodal-data", None),
        (kiloword, "MNE-kiloword-data", "kword_metadata-epo.fif"),
        (mtrf, "mTRF_1.5", "speech_data.mat"),
    ):
        dataset.data_path = functools.partial(_served_data_path, folder, probe)
    eegbci.load_data = _eegbci_load_data


# --- Readers -----------------------------------------------------------------
# Nearly every reader validates its filename with _check_fname(must_exist=True)
# before opening it, so one hook there fetches for all of them. The rest need
# one of three things: their own wrapper because they skip that check, the
# other files a single name implies, or a fetch before a filesystem probe
# (os.path.exists, glob) that no reader would ever trigger.
def _check_fname_fetching(
    fname, overwrite=False, must_exist=False, name="File", need_dir=False, **kwargs
):
    rel = _rel_to_data(fname) if must_exist else None
    if rel is not None and need_dir:  # a served folder's files arrive on demand
        (_data_root / rel).mkdir(parents=True, exist_ok=True)
    elif rel is not None:
        try:
            _fetch_rel(rel)
        except Exception:
            pass  # let MNE raise its own error for a missing file
    return _check_fname(fname, overwrite, must_exist, name, need_dir, **kwargs)


# Filesystem probes: fetch the candidates first, in the order MNE tries them,
# then let MNE choose as it normally would.
def _get_head_surface_fetching(surf, subject, subjects_dir, bem=None, verbose=None):
    if surf in ("head-dense", "seghead"):
        cands = [f"bem/{subject}-head-dense.fif", "surf/lh.seghead"]
    else:
        cands = ["bem/outer_skin.surf", f"bem/{subject}-head.fif"]
    _fetch_candidates(subject, subjects_dir, cands)
    return _get_head_surface(surf, subject, subjects_dir, bem=bem, verbose=verbose)


def _get_skull_surface_fetching(surf, subject, subjects_dir, bem=None, verbose=None):
    _fetch_candidates(subject, subjects_dir, [f"bem/{surf}_skull.surf"])
    return _get_skull_surface(surf, subject, subjects_dir, bem=bem, verbose=verbose)


def _surface_head_surface_fetching(
    subject, source, subjects_dir, on_defects, raise_error=True
):
    srcs = [source] if isinstance(source, str) else list(source)
    _fetch_candidates(subject, subjects_dir, [f"bem/{subject}-{s}.fif" for s in srcs])
    return _surface_head_surface(
        subject, source, subjects_dir, on_defects, raise_error=raise_error
    )


def _read_talxfm_fetching(subject, subjects_dir=None, verbose=None):
    # in the order MNE probes them: the docs serve T1.mgz, not orig.mgz
    want = ["mri/orig.mgz", "mri/T1.mgz", "mri/transforms/talairach.xfm"]
    _fetch_candidates(subject, subjects_dir, want)
    return read_talxfm(subject, subjects_dir, verbose=verbose)


def _plot_bem_fetching(subject=None, subjects_dir=None, *args, **kwargs):
    want = ["bem/inner_skull.surf", "bem/outer_skull.surf", "bem/outer_skin.surf"]
    want.append("mri/" + str(kwargs.get("mri", "T1.mgz")))
    surfs = kwargs.get("brain_surfaces")
    for surf in [surfs] if isinstance(surfs, str) else surfs or []:
        want += [f"surf/lh.{surf}", f"surf/rh.{surf}"]
    _fetch_candidates(subject, subjects_dir, want)
    return _orig["plot_bem"](subject, subjects_dir, *args, **kwargs)


def _read_raw_kit_fetching(input_fname, *args, **kwargs):
    """Stage the served copy of the package's KIT test file where it is looked for.

    The logging tutorial reads it from inside the installed package, which the
    wheel leaves out.
    """
    path = Path(str(input_fname))
    if path.name == "test.sqd" and not path.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_fetch_rel("MNE-kit-testdata/test.sqd"), path)
        except Exception as exc:
            print(f"[JupyterLite] could not stage test.sqd: {exc!r}")
    return _orig["read_raw_kit"](input_fname, *args, **kwargs)


def _patch_readers():
    import matplotlib.pyplot as plt  # the eyetracking heatmap reads its stimulus
    import nibabel  # a few tutorials load an MRI themselves

    from ...io import read_raw_brainvision, read_raw_eeglab, read_raw_kit, read_raw_nirx
    from ...source_estimate import read_source_estimate
    from ...viz import plot_bem

    _rebind("_check_fname", _check_fname, _check_fname_fetching)
    _wrap_reader(plt.imread)
    _wrap_reader(nibabel.load, module=nibabel)
    _wrap_reader(read_raw_eeglab, lambda rel: [rel.removesuffix(".set") + ".fdt"])
    _wrap_reader(
        read_raw_brainvision,
        lambda rel: [rel.removesuffix(".vhdr") + s for s in (".eeg", ".vmrk")],
    )
    _wrap_reader(
        read_source_estimate, lambda rel: [rel + s for s in ("-lh.stc", "-rh.stc")]
    )
    try:  # pyxdf has no wheel on every Pyodide build; only the XDF example needs it
        import pyxdf

        _wrap_reader(pyxdf.load_xdf, module=pyxdf)
    except Exception:
        pass
    _rebind("read_raw_nirx", read_raw_nirx, _dir_reader(read_raw_nirx))  # a folder
    _rebind("_get_head_surface", _get_head_surface, _get_head_surface_fetching)
    _rebind("_get_skull_surface", _get_skull_surface, _get_skull_surface_fetching)
    _rebind("_get_head_surface", _surface_head_surface, _surface_head_surface_fetching)
    _rebind("read_talxfm", read_talxfm, _read_talxfm_fetching)
    _orig["plot_bem"] = plot_bem
    _rebind("plot_bem", plot_bem, _plot_bem_fetching)
    _orig["read_raw_kit"] = read_raw_kit
    _rebind("read_raw_kit", read_raw_kit, _read_raw_kit_fetching)


# --- What WebAssembly cannot do ----------------------------------------------
def _displayhook(self, result=None):
    """Drop the echo of a Figure that plt_show already displayed.

    A plot that is also a cell's last expression returns its Figure, which
    Out[] would otherwise render a second time.
    """
    from matplotlib.figure import Figure

    figs = result if isinstance(result, list | tuple) else [result]
    if figs and all(isinstance(fig, Figure) for fig in figs):
        result = None
    return _orig["displayhook"](self, result)


def _find_libraries_pyodide(self):
    from pyodide_js._module import LDSO

    for path in LDSO.loadedLibsByName.as_py_json():
        if Path(path).exists():
            self._make_controller_from_path(path)


def _patch_wasm():
    from matplotlib.figure import Figure

    # No OS threads: the ProgressBar updater and tqdm's monitor only animate
    progressbar._UpdateThread.start = lambda self: None
    progressbar._UpdateThread.join = lambda self, *args, **kwargs: None
    try:
        import tqdm

        tqdm.tqdm.monitor_interval = 0
    except Exception:
        pass
    # fig.show() warns on the inline Agg canvas, and a few tutorials call it
    Figure.show = lambda self, *args, **kwargs: None
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        ipython.run_line_magic("matplotlib", "inline")
        # guarded: a double render beats a broken cell
        _orig["displayhook"] = type(ipython.displayhook).__call__
        type(ipython.displayhook).__call__ = _displayhook
    except Exception:
        pass
    # threadpoolctl 3.6.0 calls Pyodide's deprecated JsProxy.as_object_map(),
    # which warns from mne.sys_info(); as_py_json() gives the same paths.
    # TODO VERSION: fixed in joblib/threadpoolctl#201, drop once Pyodide
    # bundles threadpoolctl >= 3.7.0
    try:
        import threadpoolctl

        threadpoolctl.ThreadpoolController._find_libraries_pyodide = (
            _find_libraries_pyodide
        )
    except Exception:
        pass
