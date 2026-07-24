"""A pyvista-js drawing backend for MNE's 3D renderer, for JupyterLite.

MNE's 3D functions (``plot_alignment``, ``plot_bem``, ``plot_sparse_source_estimates``,
``SourceSpaces.plot``, ...) all build their figure the same way: they do their own
geometry and coordinate-frame work in numpy, then hand the result to a renderer
obtained from ``mne.viz.backends.renderer._get_renderer``. Only that last step needs
VTK, and VTK cannot load in WebAssembly.

So instead of reimplementing those functions one by one, this module supplies a
renderer that draws with pyvista-js (vtk.js) and patches the factory. MNE then does
all of the transform math itself -- which matters, because getting a head/MRI/device
transform subtly wrong produces a plausible-looking picture with the sensors in the
wrong place, and several of these tutorials are specifically *about* coordinate
alignment.

What is supported: meshes, surfaces, spheres, tubes and glyphs -- enough for the
static figures the docs render. What is not: the interactive ``Brain`` time viewer,
which additionally needs dock widgets and toolbars, and scalar colormaps, which
pyvista-js 0.15 does not have (scalars fall back to a solid color).

The source is kept as a string because it has to run inside the browser kernel; see
``first_notebook_cell`` in ``conf.py``.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

LITE_RENDERER_CELL = r'''
# --- pyvista-js drawing backend for MNE's 3D renderer -----------------------
# Patches mne.viz.backends.renderer._get_renderer so MNE keeps doing its own
# geometry and coordinate-frame work and only the drawing is replaced.
class _LiteRenderer:
    """Minimal MNE 3D renderer backed by pyvista-js."""

    def __init__(self, *args, **kwargs):
        import numpy as _np
        import pyvista_js as _pv
        self._np = _np
        self._pv = _pv
        self.plotter = _pv.Plotter()
        _bg = kwargs.get("bgcolor", kwargs.get("background_color", "black"))
        try:
            self.plotter.background_color = self._rgb(_bg)
        except Exception:
            pass
        # even lighting, so a surface is not black when rotated
        for _lp in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                    (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            try:
                self.plotter.add_light(_pv.Light(
                    position=(300.0 * _lp[0], 300.0 * _lp[1], 300.0 * _lp[2]),
                    focal_point=(0.0, 0.0, 0.0), intensity=0.4))
            except Exception:
                pass

    # -- helpers ------------------------------------------------------------
    def _rgb(self, color):
        """Return an (r, g, b) 0-1 tuple; pyvista-js rejects hex strings."""
        if color is None:
            return (0.5, 0.5, 0.5)
        from matplotlib.colors import to_rgb as _to_rgb
        if isinstance(color, str):
            return _to_rgb(color)
        _c = self._np.asarray(color, dtype=float).ravel()[:3]
        if _c.size < 3:
            return (0.5, 0.5, 0.5)
        if _c.max() > 1.0:  # 0-255 form
            _c = _c / 255.0
        return tuple(float(min(max(_v, 0.0), 1.0)) for _v in _c)

    def _faces(self, tris):
        _np = self._np
        _t = _np.asarray(tris, dtype=_np.int32).reshape(-1, 3)
        return _np.hstack([
            _np.full((len(_t), 1), 3, dtype=_np.int32), _t]).ravel()

    def _glyph_template(self, kind, radius=None, height=None, center=None,
                        resolution=None, **kwargs):
        """Return (rr, tris) for instanced_mesh, oriented along +x.

        pyvista-js's Sphere/Cylinder are parametric primitives with no
        triangle list, so build the templates here. These are markers a few
        millimetres across, so keep them low-poly -- every instance is a
        separate mesh and the WASM heap is not large.
        """
        _np = self._np
        if kind == "sphere":
            _r = 0.5 if radius is None else float(radius)
            rr = _np.array([[_r, 0, 0], [-_r, 0, 0], [0, _r, 0],
                            [0, -_r, 0], [0, 0, _r], [0, 0, -_r]], float)
            tris = _np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                              [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]], int)
            return rr, tris
        # cylinder along +x, matching _cylinder_geom's convention
        _r = 0.1 if radius is None else float(radius)
        _h = 1.0 if height is None else float(height)
        _n = 8 if not resolution else max(3, int(resolution) // 2)
        _c = _np.zeros(3) if center is None else _np.asarray(center, float)
        _ang = _np.linspace(0.0, 2 * _np.pi, _n, endpoint=False)
        _ring = _np.column_stack([_np.zeros(_n), _r * _np.cos(_ang),
                                  _r * _np.sin(_ang)])
        _back = _ring + _np.array([-_h / 2.0, 0, 0])
        _front = _ring + _np.array([_h / 2.0, 0, 0])
        rr = _np.vstack([_back, _front,
                         [[-_h / 2.0, 0, 0]], [[_h / 2.0, 0, 0]]]) + _c
        tris = []
        for _i in range(_n):
            _j = (_i + 1) % _n
            tris += [[_i, _j, _n + _j], [_i, _n + _j, _n + _i]]   # wall
            tris += [[2 * _n, _j, _i]]                            # back cap
            tris += [[2 * _n + 1, _n + _i, _n + _j]]              # front cap
        return rr, _np.asarray(tris, int)

    def _add(self, points, tris, color, opacity=1.0):
        """Draw a mesh and return MNE's (actor, mesh) pair."""
        _np = self._np
        _pd = self._pv.PolyData(
            points=_np.asarray(points, dtype=_np.float32),
            faces=self._faces(tris))
        _actor = self.plotter.add_mesh(
            _pd, color=self._rgb(color), opacity=float(opacity),
            smooth_shading=True)
        return _actor, _pd

    # -- drawing ------------------------------------------------------------
    def mesh(self, x, y, z, triangles, color=None, opacity=1.0, *args, **kwargs):
        _np = self._np
        _pts = _np.column_stack([_np.asarray(x).ravel(),
                                 _np.asarray(y).ravel(),
                                 _np.asarray(z).ravel()])
        return self._add(_pts, triangles, color, opacity)

    def surface(self, surface, color=None, opacity=1.0, *args, **kwargs):
        return self._add(surface["rr"], surface["tris"], color, opacity)

    def sphere(self, center, color=None, scale=1.0, opacity=1.0,
               resolution=8, backface_culling=False, radius=None, **kwargs):
        _np = self._np
        _c = _np.atleast_2d(_np.asarray(center, dtype=float))
        _r = float(radius if radius is not None else scale)
        _actor = _mesh = None
        for _p in _c:
            _mesh = self._pv.Sphere(
                radius=_r, center=tuple(float(_q) for _q in _p[:3]))
            _actor = self.plotter.add_mesh(
                _mesh, color=self._rgb(color), opacity=float(opacity),
                smooth_shading=True)
        return _actor, _mesh

    def tube(self, origin, destination, radius=0.001, color=None, *args, **kwargs):
        _np = self._np
        _o = _np.atleast_2d(_np.asarray(origin, dtype=float))
        _d = _np.atleast_2d(_np.asarray(destination, dtype=float))
        _actor = _mesh = None
        for _a, _b in zip(_o, _d):
            _vec = _b[:3] - _a[:3]
            _len = float(_np.linalg.norm(_vec))
            if _len == 0.0:
                continue
            _mesh = self._pv.Cylinder(
                center=tuple(float(_q) for _q in (_a[:3] + _b[:3]) / 2.0),
                direction=tuple(float(_q) for _q in _vec / _len),
                radius=float(radius), height=_len)
            _actor = self.plotter.add_mesh(
                _mesh, color=self._rgb(color), smooth_shading=True)
        return _actor, _mesh

    def quiver3d(self, x, y, z, u, v, w, color=None, scale=1.0, mode="arrow",
                 opacity=1.0, *args, **kwargs):
        _np = self._np
        _x, _y, _z = (_np.atleast_1d(_np.asarray(_q, dtype=float))
                      for _q in (x, y, z))
        _u, _v, _w = (_np.atleast_1d(_np.asarray(_q, dtype=float))
                      for _q in (u, v, w))
        _s = float(_np.asarray(scale).ravel()[0]) if _np.size(scale) else 1.0
        _actor = _g = None
        for _i in range(len(_x)):
            _ctr = (float(_x[_i]), float(_y[_i]), float(_z[_i]))
            _dir = (float(_u[_i % len(_u)]), float(_v[_i % len(_v)]),
                    float(_w[_i % len(_w)]))
            if _np.linalg.norm(_dir) == 0.0:
                _dir = (0.0, 0.0, 1.0)
            if mode == "sphere":
                _g = self._pv.Sphere(radius=_s / 2.0, center=_ctr)
            elif mode in ("cylinder", "oct"):
                _g = self._pv.Cylinder(center=_ctr, direction=_dir,
                                       radius=_s / 4.0, height=_s)
            else:  # arrow / cone / 2darrow
                _g = self._pv.Cone(center=_ctr, direction=_dir,
                                   height=_s, radius=_s / 2.0)
            _actor = self.plotter.add_mesh(
                _g, color=self._rgb(color), opacity=float(opacity),
                smooth_shading=True)
        return _actor, _g

    def instanced_mesh(self, rr, tris, positions, quats=None, colors=None,
                       scales=None, opacity=1.0, *args, **kwargs):
        # one copy of the template per position; rotate with MNE's own
        # quaternion helper so oriented glyphs (EEG cylinders) point the way
        # MNE intended rather than all along +x.
        _np = self._np
        _rr = _np.asarray(rr, dtype=float)
        _pos = _np.atleast_2d(_np.asarray(positions, dtype=float))
        _quats = None if quats is None else _np.atleast_2d(
            _np.asarray(quats, dtype=float))
        _rot = None
        if _quats is not None:
            try:
                from mne.transforms import quat_to_rot as _q2r
                _rot = _q2r(_quats)
            except Exception:
                _rot = None
        _out = (None, None)
        for _i, _p in enumerate(_pos):
            _s = 1.0
            if scales is not None:
                _sa = _np.atleast_1d(_np.asarray(scales, dtype=float))
                _s = float(_sa[_i % len(_sa)])
            _col = colors
            if colors is not None and _np.ndim(colors) > 1:
                _col = _np.asarray(colors)[_i % len(colors)]
            _pts = _rr * _s
            if _rot is not None:
                _pts = _pts @ _np.asarray(_rot[_i % len(_rot)]).T
            _out = self._add(_pts + _p[:3], tris, _col, opacity)
        return _out

    # -- things the static docs do not need ---------------------------------
    def contour(self, *args, **kwargs):
        # pyvista-js 0.15 has no scalar contouring; callers unpack a pair
        return None, None

    def text2d(self, *args, **kwargs):
        return None

    def text3d(self, *args, **kwargs):
        return None

    def scalarbar(self, *args, **kwargs):
        return None

    def legend(self, *args, **kwargs):
        return None

    def subplot(self, *args, **kwargs):
        return None

    def set_interaction(self, *args, **kwargs):
        return None

    def remove_mesh(self, *args, **kwargs):
        return None

    def project(self, xyz, ch_names):
        return self._np.asarray(xyz, dtype=float)[:, :2]

    def screenshot(self, mode="rgb", filename=None, **kwargs):
        return self._np.zeros((2, 2, 3), dtype="uint8")

    def close(self):
        return None

    def _update(self, *args, **kwargs):
        return None

    def _process_events(self, *args, **kwargs):
        return None

    def _enable_time_interaction(self, *args, **kwargs):
        # the figures are static here; there is no time slider to wire up
        return None

    def _window_close_connect(self, *args, **kwargs):
        return None

    def _window_set_cursor(self, *args, **kwargs):
        return None

    def get_camera(self, *args, **kwargs):
        return (0.0, 0.0, 1.0, (0.0, 0.0, 0.0), 0.0)

    def set_camera(self, azimuth=None, elevation=None, distance=None,
                   focalpoint=None, roll=None, *args, **kwargs):
        # pyvista-js has no azimuth/elevation camera; approximate the common
        # views and otherwise leave the default.
        try:
            if azimuth is None:
                return None
            _a = float(azimuth) % 360.0
            if 45 <= _a < 135:
                _vec = (0.0, -1.0, 0.0)
            elif 135 <= _a < 225:
                _vec = (1.0, 0.0, 0.0)
            elif 225 <= _a < 315:
                _vec = (0.0, 1.0, 0.0)
            else:
                _vec = (-1.0, 0.0, 0.0)
            self.plotter.view_vector(_vec, viewup=(0.0, 0.0, 1.0))
        except Exception:
            pass
        return None

    def scene(self):
        return self.plotter

    def show(self):
        try:
            self.plotter.show()
        except Exception as _e:
            print("[JupyterLite] pyvista-js render failed: " + repr(_e))
        return None


def _lite_get_renderer(*args, **kwargs):
    return _LiteRenderer(*args, **kwargs)


try:
    import mne.viz.backends.renderer as _mne_rend
    _mne_rend._get_renderer = _lite_get_renderer
except Exception as _e:
    print("[JupyterLite] could not install the pyvista-js renderer: " + repr(_e))
'''
