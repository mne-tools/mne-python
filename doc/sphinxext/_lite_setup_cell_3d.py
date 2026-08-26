# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# The experimental part of the browser setup, kept apart from the rest so the
# solid ground and the shifting ground are easy to tell apart. Everything here
# stands in for MNE's Brain/VTK stack, which has no WebAssembly build, and is
# the part most likely to be dropped as pyvista-js gains features upstream.
# Appended after the base cell, which it depends on: the second block below
# uses the matplotlib-inline shim that cell installs.
# This runs as a continuation of the base cell, in the same namespace, so it
# reads names that cell defined (mne, plt, the fetch helpers) rather than
# importing them again; F821 is off for that reason, not to hide typos.
# ruff: noqa: E402, F704, F821, I001

# --- JupyterLite setup cell, 3D -----------------------------------------------
# EXPERIMENTAL 3D: MNE's normal Brain/VTK stack can't load in WASM, so
# route SourceEstimate.plot() through pyvista-js (vtk.js) instead.
# pyvista-js (0.15) has no scalar colormap in its renderer, so we
# approximate MNE's Brain look with solid-colored meshes: a two-tone
# curvature base (light gyri + dark sulci) plus many thin 'hot' bands
# for the activation, on a black background with even scene lighting.
# Static, one time point, no time slider yet.
#
# A failed render prints and lets the notebook carry on, which is the opposite
# of how the data fetch in the base cell behaves. That is deliberate: a file
# missing there means the docs build is broken and should say so, while this
# whole shim stands in for a stack with no WebAssembly build at all, so failing
# hard would take out every 3D notebook rather than report one bug.
#
# The stub 'brain' it returns makes the decorating calls (add_foci/add_text/
# show_view/...) no-ops so the rest of the notebook still runs. screenshot() is
# the one exception, and it raises: see below.
# Say once per session that what the browser draws is not what the rendered
# docs show, so a reader comparing the two is not left guessing. pyvista-js has
# no scalar colormap, so activation arrives as discrete solid bands rather than
# a continuous scale, there is no colorbar or time slider, and the hemispheres
# are drawn side by side rather than in anatomical position.
_lite_3d_noted = False


def _lite_note_3d_approximation():
    global _lite_3d_noted
    if _lite_3d_noted:
        return
    _lite_3d_noted = True
    print(
        "[JupyterLite] 3D drawn with pyvista-js: activation is shown as solid "
        "colour bands at a single time point, with the hemispheres side by "
        "side and no colorbar. The figure in the rendered docs is MNE's full "
        "Brain view and will not look the same."
    )


class _LiteBrain:
    def screenshot(self, *args, **kwargs):
        # No blank array here. vtk.js draws into a browser canvas that Python
        # cannot read back, so there is no image to return, and handing back a
        # blank one is worse than failing: 10_publication_figure crops its
        # screenshot and shows before/after, so it would publish two black
        # squares as though they were the real thing. A notebook that needs a
        # screenshot belongs in JUPYTERLITE_EXCLUDE instead.
        raise NotImplementedError(
            "brain.screenshot() is not available in JupyterLite: the vtk.js "
            "renderer draws to a browser canvas that Python cannot read back. "
            "Run this notebook locally to capture the scene."
        )

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def _lite_stc_plot(self, *args, **kwargs):
    try:
        import numpy as _np
        import nibabel as _nib
        from scipy.spatial import cKDTree as _KDTree
        from matplotlib import colormaps as _cmaps
        import pyvista_js as _pv

        _subj = (
            kwargs.get("subject")
            or (args[0] if args and isinstance(args[0], str) else None)
            or "sample"
        )
        _sdir = kwargs.get("subjects_dir")
        # kept as a str on both branches: it is concatenated below
        _sdir = (
            str(_sdir)
            if _sdir is not None
            else str(_lite_data_path("MNE-sample-data/subjects"))
        )
        # surfaces are fetched relative to the served mne_data root, so
        # derive that from subjects_dir rather than assuming sample --
        # a dataset may keep its FreeSurfer subjects under its own folder.
        _rel_sdir = (
            _lite_rel_to_data(_sdir)
            if _lite_rel_to_data(_sdir) is not None
            else "MNE-sample-data/subjects"
        )
        _init = kwargs.get("initial_time", None)
        if _init is None:
            _ti = int(_np.argmax(_np.abs(self.data).mean(0)))
        else:
            _ti = int(_np.argmin(_np.abs(self.times - _init)))
        _hot = _cmaps["hot"]
        # Tuned against the inflated FreeSurfer surfaces MNE ships, whose
        # coordinates are in mm.
        _N = 10  # activation value bands
        _BLOB_MM = 12.0  # colour a surface vertex from an active source within
        # this radius, so a single-vertex source reads as a blob and not a dot
        _HEMI_MM = 60.0  # push the hemispheres apart so they do not overlap
        _LIFT = 0.02  # raise each band off the surface to avoid z-fighting
        _HOT_LO, _HOT_HI = 0.25, 0.66  # slice of 'hot' to use; its ends are
        # near-black and near-white, which read as background here
        _SPARSE_P90 = 0.05  # below this fraction of the max the 90th pct means
        _SPARSE_FLOOR = 0.4  # the data is sparse, so threshold on the max

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
                _scal = _np.where(_ad <= _BLOB_MM, _sv[_act][_ai], 0.0)
            # offset hemispheres along x so they do not overlap
            _off = -_HEMI_MM if _h == "lh" else _HEMI_MM
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
            _fmin = _p90 if _p90 > _fmax * _SPARSE_P90 else _fmax * _SPARSE_FLOOR
            if _fmax > _fmin:
                _edges = _np.linspace(_fmin, _fmax, _N + 1)
                for _i in range(_N):
                    if _i < _N - 1:
                        _m = (_fv >= _edges[_i]) & (_fv < _edges[_i + 1])
                    else:
                        _m = _fv >= _edges[_i]
                    if int(_m.sum()) == 0:
                        continue
                    _rgb = _hot(_HOT_LO + (_HOT_HI - _HOT_LO) * (_i / (_N - 1)))
                    _col = (float(_rgb[0]), float(_rgb[1]), float(_rgb[2]))
                    _s = _sub(_pts, _tris, _m, _LIFT, _cen)
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
        _lite_note_3d_approximation()
    except Exception as _e:
        print("[JupyterLite] pyvista-js 3D render unavailable: " + repr(_e))
    return _LiteBrain()


mne.SourceEstimate.plot = _lite_stc_plot


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
    _pyodide_plt_show(show)
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
        _lite_note_3d_approximation()
    except Exception as _e:
        print("[JupyterLite] pyvista-js glass brain unavailable: " + repr(_e))


mne.viz.plot_sparse_source_estimates = _lite_plot_sparse_source_estimates

# Each MNE plot is rendered once by _pyodide_plt_show above (display()).
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
            if isinstance(result, mpl_figure.Figure):
                result = None
            elif (
                isinstance(result, (list, tuple))
                and result
                and all(isinstance(_x, mpl_figure.Figure) for _x in result)
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
    import threadpoolctl

    def _find_libraries_pyodide(self):
        from pyodide_js._module import LDSO

        for _fp in LDSO.loadedLibsByName.as_py_json():
            if Path(_fp).exists():
                self._make_controller_from_path(_fp)

    threadpoolctl.ThreadpoolController._find_libraries_pyodide = _find_libraries_pyodide
except Exception:
    pass
