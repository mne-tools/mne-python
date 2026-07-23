"""JupyterLite setup cell for the xeus-python kernel.

This is the xeus counterpart of the Pyodide ``first_notebook_cell`` in
``conf.py``.

How it differs from the Pyodide cell
------------------------------------
xeus-python is real CPython compiled to WebAssembly (NOT Pyodide), and the
packages are PRE-INSTALLED at build time from ``jupyterlite_environment.yml``.
So compared to the Pyodide setup cell this drops:

  * ``piplite.install(...)``  -> MNE is already installed
  * the ``lzma`` / ``multiprocessing`` mocks -> real CPython stdlib is present
  * the ``pyodide.http`` / ``requests`` patches -> no Pyodide runtime to patch
  * the ``threadpoolctl`` ``as_object_map()`` patch -> Pyodide-only API

The WASM-level workarounds that are not Pyodide-specific (no OS threads for
MNE's ProgressBar and tqdm's monitor, inline matplotlib, single-render
displayhook) are kept, since they apply to any WASM kernel.

Data model
----------
Same as the Pyodide build: the curated data is served at the docs root under
``/mne_data/`` (via Sphinx ``html_extra_path``, see ``conf.py``) and fetched
over HTTP into the kernel's own filesystem at ``/tmp/mne_data``.

The fetch is a *synchronous* ``XMLHttpRequest`` with
``responseType='arraybuffer'``. JupyterLite kernels run in a web worker, where
that combination is allowed (the spec only forbids setting ``responseType`` on
a synchronous request when the global is a ``Window``). This matters because it
needs neither cross-origin isolation (COOP/COEP) nor a service worker, and
static hosts such as the CircleCI artifact server provide neither.

Being synchronous is what lets a plain ``data_path()`` / ``read_raw_fif()``
call fetch on demand, so each notebook only downloads the files it actually
touches instead of the whole sample dataset up front.

The one xeus-specific difference is how JavaScript is reached. Pyodide exposes
a ``js`` module whose proxies carry a ``.to_py()`` method; xeus exposes
``pyjs``, where the equivalents are the module-level ``pyjs.new()`` and
``pyjs.to_py()`` (a ``pyjs`` ``JsValue`` has no ``.to_py()`` method, and the
bundled ``pyjs.pyodide_polyfill`` deliberately provides only ``to_js``).

3D rendering
------------
The ``SourceEstimate.plot`` shim that routes through pyvista-js (vtk.js) is
carried over from the Pyodide cell unchanged -- it is plain numpy/nibabel/
scipy/matplotlib plus ``pyvista_js``, with no kernel-specific code. It does
require ``pyvista-js`` to be installed, see ``jupyterlite_environment.yml``.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# The setup cell source, as a string (mirrors how conf.py stores the Pyodide one).
XEUS_FIRST_NOTEBOOK_CELL = r"""# 💡 Auto-added setup cell (xeus-python kernel).
# MNE and its dependencies are pre-installed in this kernel; this cell only
# wires up the sample data, which is fetched on demand from the docs server.
import os
from pathlib import Path as _Path
import pyjs

# --- locate the docs root ---------------------------------------------------
# The data is served at the docs root (/mne_data/...) via html_extra_path.
# `location` exists both in the main thread and in a web worker, so use it to
# find the docs root by splitting the page URL on '/lite/'.
_base = str(pyjs.js.location.href).split('/lite/')[0] + '/mne_data/'
mne_data_path = '/tmp/mne_data'
_sample_dir = mne_data_path + '/MNE-sample-data'


def _lite_bytes(_resp):
    # JS ArrayBuffer -> Python bytes. pyjs converts an ArrayBuffer via
    # Uint8Array into an object supporting the buffer protocol. Note this is
    # the module-level pyjs.to_py(); unlike Pyodide, a pyjs JsValue has no
    # .to_py() method.
    return bytes(pyjs.to_py(_resp))


def _lite_fetch_rel(_rel):
    # Fetch one file, relative to the served mne_data root, and cache it.
    _dst = mne_data_path + '/' + _rel
    if not os.path.exists(_dst):
        _xhr = pyjs.new(pyjs.js.XMLHttpRequest)
        _xhr.open('GET', _base + _rel, False)  # False -> synchronous
        _xhr.responseType = 'arraybuffer'
        _xhr.send()
        if int(_xhr.status) != 200:
            raise FileNotFoundError(
                f'Could not fetch {_rel} (HTTP {int(_xhr.status)})'
            )
        _data = _lite_bytes(_xhr.response)
        # A static server answers a missing path with an HTML error page.
        if _data[:4] == b'<!DO' or _data[:5] == b'<html':
            raise FileNotFoundError(f'Could not fetch {_rel} (got HTML)')
        os.makedirs(os.path.dirname(_dst), exist_ok=True)
        with open(_dst, 'wb') as _fh:
            _fh.write(_data)
    return _dst


def _lite_lazy_fetch(_folder, _fname):
    _lite_fetch_rel(_folder + '/' + _fname)
    return _Path(mne_data_path + '/' + _folder)


# --- eager 'core' sample files ----------------------------------------------
# Small, commonly-used files fetched once at notebook start. The heavy ones
# (raw / filt raw / ernoise / fwd / inv / src, ~360 MB total) are deliberately
# absent here -- the reader shims below fetch them on first read, so each
# notebook only downloads what it actually uses.
_sample_files = [
    'version.txt',
    'MEG/sample/sample_audvis_raw-eve.fif',
    'MEG/sample/sample_audvis_filt-0-40_raw-eve.fif',
    'MEG/sample/sample_audvis_ecg-proj.fif',
    'MEG/sample/sample_audvis-cov.fif',
    'MEG/sample/sample_audvis-ave.fif',
    'MEG/sample/sample_audvis-no-filter-ave.fif',
    'MEG/sample/sample_audvis_raw-trans.fif',
    'MEG/sample/sample_audvis-shrunk-cov.fif',
    'MEG/sample/sample_audvis-meg-lh.stc',
    'MEG/sample/sample_audvis-meg-rh.stc',
    'subjects/sample/mri/T1.mgz',
    'subjects/sample/surf/rh.pial',
    'subjects/sample/surf/lh.pial',
    'subjects/sample/surf/rh.white',
    'subjects/sample/surf/lh.white',
    'subjects/sample/label/lh.aparc.annot',
    'subjects/sample/label/rh.aparc.annot',
]
print('Fetching MNE sample data (once per session)...')
for _f in _sample_files:
    try:
        _lite_fetch_rel('MNE-sample-data/' + _f)
    except Exception as _e:
        print(f'  failed to fetch {_f}: {_e}')

os.makedirs(mne_data_path, exist_ok=True)
os.environ['MNE_DATA'] = mne_data_path
os.environ['MNE_DATASETS_SAMPLE_PATH'] = mne_data_path

# Block pooch from attempting large OSF downloads in the browser: the required
# files are either fetched above or unavailable here.
import pooch
_orig_pooch_fetch = pooch.Pooch.fetch


def _lite_pooch_fetch(self, fname, processor=None, downloader=None):
    if 'osf.io' in self.get_url(fname):
        raise RuntimeError(
            f'Cannot download {fname!r} from OSF in JupyterLite: browser CORS '
            'policy and memory limits prevent large dataset downloads. Open '
            'this notebook from mne.tools where the sample data is served, or '
            'run it locally.'
        )
    return _orig_pooch_fetch(
        self, fname, processor=processor, downloader=downloader
    )


pooch.Pooch.fetch = _lite_pooch_fetch

import mne

# Pre-create a valid empty config file so MNE never hits a corrupt read.
_cfg = mne.get_config_path()
os.makedirs(os.path.dirname(_cfg), exist_ok=True)
if not os.path.exists(_cfg):
    with open(_cfg, 'w') as _f:
        _f.write('{}')
mne.set_config('MNE_DATA', mne_data_path)
for _ds in ['SAMPLE', 'TESTING', 'SSVEP', 'EEGBCI', 'SOMATO', 'BRAINSTORM']:
    mne.set_config(f'MNE_DATASETS_{_ds}_PATH', mne_data_path)

# --- dataset loaders --------------------------------------------------------
# Bypass pooch's archive check: data_path() normally looks for the .tar.gz
# archive, not just the extracted folder. Return the folder directly (as a
# Path, since tutorials use the / operator on the result).
_sample_path = _Path(_sample_dir)


def _lite_sample_data_path(*_a, **_kw):
    return _sample_path


mne.datasets.sample.data_path = _lite_sample_data_path


# Several non-sample datasets are each used by only a couple of notebooks
# (kiloword/erp_core for Epochs 30 & 40; mtrf/eegbci for the decoding
# examples), so fetch them lazily -- only when their data_path()/load_data()
# is called -- to avoid taxing every other notebook's setup.
def _lite_kiloword_data_path(*_a, **_kw):
    return _lite_lazy_fetch('MNE-kiloword-data', 'kword_metadata-epo.fif')


mne.datasets.kiloword.data_path = _lite_kiloword_data_path


def _lite_erp_core_data_path(*_a, **_kw):
    return _lite_lazy_fetch(
        'MNE-ERP-CORE-data', 'ERP-CORE_Subject-001_Task-Flankers_eeg.fif'
    )


mne.datasets.erp_core.data_path = _lite_erp_core_data_path


def _lite_mtrf_data_path(*_a, **_kw):
    return _lite_lazy_fetch('mTRF_1.5', 'speech_data.mat')


mne.datasets.mtrf.data_path = _lite_mtrf_data_path


def _lite_eegbci_load_data(subject, runs, *_a, **_kw):
    _runs = [runs] if isinstance(runs, (int, float)) else list(runs)
    _subjects = list(subject) if isinstance(subject, (list, tuple)) else [subject]
    _out = []
    for _s in _subjects:
        for _r in _runs:
            _rel = (
                'MNE-eegbci-data/files/eegmmidb/1.0.0/'
                f'S{int(_s):03d}/S{int(_s):03d}R{int(_r):02d}.edf'
            )
            _out.append(_Path(_lite_fetch_rel(_rel)))
    return _out


mne.datasets.eegbci.load_data = _lite_eegbci_load_data


# --- lazy reader shims ------------------------------------------------------
# Rather than hand-listing every sample file that only one or two notebooks
# need, fetch any sample-data path the first time a reader is asked to open it.
def _lite_fetch_if_under_mne_data(fname):
    _p = str(fname)
    if _p.startswith(mne_data_path + '/'):
        _lite_fetch_rel(_p[len(mne_data_path) + 1:])
    return fname


_orig_read_forward_solution = mne.read_forward_solution


def _lite_read_forward_solution(fname, *_a, **_kw):
    return _orig_read_forward_solution(
        _lite_fetch_if_under_mne_data(fname), *_a, **_kw
    )


mne.read_forward_solution = _lite_read_forward_solution

import mne.minimum_norm as _mne_minv

_orig_read_inverse_operator = _mne_minv.read_inverse_operator


def _lite_read_inverse_operator(fname, *_a, **_kw):
    return _orig_read_inverse_operator(
        _lite_fetch_if_under_mne_data(fname), *_a, **_kw
    )


_mne_minv.read_inverse_operator = _lite_read_inverse_operator
mne.minimum_norm.read_inverse_operator = _lite_read_inverse_operator

_orig_read_raw_fif = mne.io.read_raw_fif


def _lite_read_raw_fif(fname, *_a, **_kw):
    return _orig_read_raw_fif(_lite_fetch_if_under_mne_data(fname), *_a, **_kw)


mne.io.read_raw_fif = _lite_read_raw_fif

# Tutorials that use the generic dispatcher (e.g. 70_report,
# 14_quality_control_report) bypass read_raw_fif, so shim it too.
_orig_read_raw = mne.io.read_raw


def _lite_read_raw(fname, *_a, **_kw):
    return _orig_read_raw(_lite_fetch_if_under_mne_data(fname), *_a, **_kw)


mne.io.read_raw = _lite_read_raw

_orig_read_source_spaces = mne.read_source_spaces


def _lite_read_source_spaces(fname, *_a, **_kw):
    return _orig_read_source_spaces(
        _lite_fetch_if_under_mne_data(fname), *_a, **_kw
    )


mne.read_source_spaces = _lite_read_source_spaces

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
        return _np.zeros((2, 2, 3), dtype='uint8')
    def __getattr__(self, _name):
        return lambda *_a, **_kw: None
def _lite_stc_plot(self, *_a, **_kw):
    try:
        import numpy as _np
        import nibabel as _nib
        from scipy.spatial import cKDTree as _KDTree
        from matplotlib import colormaps as _cmaps
        import pyvista_js as _pv
        _subj = (_kw.get('subject')
                 or (_a[0] if _a and isinstance(_a[0], str) else None)
                 or 'sample')
        _sdir = _kw.get('subjects_dir')
        _sdir = (str(_sdir) if _sdir is not None else
                 mne_data_path + '/MNE-sample-data/subjects')
        _init = _kw.get('initial_time', None)
        if _init is None:
            _ti = int(_np.argmax(_np.abs(self.data).mean(0)))
        else:
            _ti = int(_np.argmin(_np.abs(self.times - _init)))
        _hot = _cmaps['hot']
        _N = 10
        # xeus-python's WASM heap is capped at 2 GiB (see bin/xpython.js:
        # WebAssembly.Memory maximum=32768 pages), and this runs at the end
        # of a notebook that is already holding raw/epochs/stc. Every band
        # below is a separate mesh, so keep the geometry in the narrowest
        # dtypes VTK accepts: int32 face indices and float32 points. That
        # halves the two biggest arrays and does not change the render.
        def _flat(_t):
            return _np.hstack([
                _np.full((len(_t), 1), 3, dtype=_np.int32),
                _t.astype(_np.int32)]).ravel()
        def _sub(_pts, _tris, _mask, _lift=0.0, _cen=None):
            _sel = _tris[_mask]
            if len(_sel) == 0:
                return None
            _u, _iv = _np.unique(_sel, return_inverse=True)
            _p = _pts[_u]
            if _lift and _cen is not None:
                _p = _cen + (_p - _cen) * (1.0 + _lift)
            return _p.astype(_np.float32, copy=False), _iv.reshape(-1, 3)
        _plotter = _pv.Plotter()
        _plotter.background_color = 'black'
        # even lighting so the surface isn't black when rotated
        for _lp in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                    (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            _plotter.add_light(_pv.Light(
                position=(300.0 * _lp[0], 300.0 * _lp[1],
                          300.0 * _lp[2]),
                focal_point=(0.0, 0.0, 0.0), intensity=0.4))
        _nlh = len(self.vertices[0])
        _hemis = (('lh', 0, self.vertices[0]),
                  ('rh', 1, self.vertices[1]))
        for _h, _hi, _vno in _hemis:
            if len(_vno) == 0:
                continue
            _pre = ('MNE-sample-data/subjects/' + _subj +
                    '/surf/' + _h)
            _lite_fetch_rel(_pre + '.inflated')
            _lite_fetch_rel(_pre + '.curv')
            _bpath = _sdir + '/' + _subj + '/surf/' + _h
            _rr, _tris = mne.read_surface(_bpath + '.inflated')
            _cv = _nib.freesurfer.read_morph_data(_bpath + '.curv')
            _hdata = self.data[:_nlh] if _hi == 0 else self.data[_nlh:]
            # color each surface vertex from the nearest ACTIVE source
            # within a small radius, so single-vertex (point) sources
            # show as visible blobs and dense sources fill in as usual
            _sv = _hdata[:, _ti].astype(_np.float32)
            _act = _sv != 0
            _scal = _np.zeros(len(_rr), dtype=_np.float32)
            if _act.any():
                _atree = _KDTree(_rr[_vno][_act])
                _ad, _ai = _atree.query(_rr)
                _scal = _np.where(_ad <= 12.0, _sv[_act][_ai], 0.0)
                # the tree and its query results are the largest temporaries
                # here; drop them before building any meshes
                del _atree, _ad, _ai
            # offset hemispheres along x so they do not overlap
            _off = -60.0 if _h == 'lh' else 60.0
            _pts = _np.round(_rr, 2).astype(_np.float32, copy=False)
            _pts[:, 0] = _pts[:, 0] + _off
            _cen = _pts.mean(0)
            del _rr
            # curvature base: light gyri (curv<0) + dark sulci (curv>=0)
            _fc = _cv[_tris].astype(_np.float32, copy=False).mean(1)
            del _cv
            for _cm, _col in (
                    (_fc < 0, (0.68, 0.68, 0.68)),
                    (_fc >= 0, (0.38, 0.38, 0.38))):
                _s = _sub(_pts, _tris, _cm)
                if _s is not None:
                    _plotter.add_mesh(
                        _pv.PolyData(points=_s[0], faces=_flat(_s[1])),
                        color=_col, smooth_shading=True)
            del _fc
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
                    _col = (float(_rgb[0]), float(_rgb[1]),
                            float(_rgb[2]))
                    _s = _sub(_pts, _tris, _m, 0.02, _cen)
                    if _s is not None:
                        _plotter.add_mesh(
                            _pv.PolyData(points=_s[0],
                                         faces=_flat(_s[1])),
                            color=_col, smooth_shading=True)
        # Open on the lateral profile (camera along the medial-lateral
        # X axis, superior up), like native MNE, instead of vtk.js's
        # default anterior/face-on view. Guarded so a missing
        # view_vector never costs us the render.
        try:
            _plotter.view_vector((-1.0, 0.0, 0.0),
                                 viewup=(0.0, 0.0, 1.0))
        except Exception:
            pass
        _plotter.show()
    except Exception as _e:
        print('[JupyterLite] pyvista-js 3D render unavailable: '
              + repr(_e))
    return _LiteBrain()
mne.SourceEstimate.plot = _lite_stc_plot

# WASM has no OS threads, so MNE's ProgressBar background
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
IPython.get_ipython().run_line_magic('matplotlib', 'inline')
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
viz_utils = importlib.import_module('mne.viz.utils')
# Also display+close via IPython for paths that call plt_show
# directly, so figures render exactly once.
def _lite_plt_show(show=True, fig=None, **kwargs):
    if not show:
        return
    import IPython.display
    _f = fig if fig is not None else plt.gcf()
    IPython.display.display(_f)
    plt.close(_f)
viz_utils.plt_show = _lite_plt_show

# Each MNE plot is rendered once by _lite_plt_show above (display()).
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
    if not getattr(_lite_dh, '_lite_no_fig_echo', False):
        _lite_dh_call = _lite_dh.__call__
        def _lite_displayhook(self, result=None):
            if isinstance(result, _mfig.Figure):
                result = None
            elif (isinstance(result, (list, tuple)) and result
                  and all(isinstance(_x, _mfig.Figure) for _x in result)):
                result = None
            return _lite_dh_call(self, result)
        _lite_dh.__call__ = _lite_displayhook
        _lite_dh._lite_no_fig_echo = True
except Exception:
    pass
"""
