"""
A pyvista-js drawing backend for MNE's 3D renderer.

MNE's 3D functions (``plot_alignment``, ``plot_bem``,
``plot_sparse_source_estimates``, ``SourceSpaces.plot``, ...) all build their
figure the same way: they do their own geometry and coordinate-frame work in
numpy, then hand the result to a renderer obtained from
:func:`mne.viz.backends.renderer._get_renderer`. Only that last step needs VTK,
and VTK cannot load in WebAssembly.

Rather than reimplement those functions one by one, this module supplies a
renderer that draws with `pyvista-js <https://github.com/tkoyama010/pyvista-js>`__
(vtk.js) and, via :func:`_activate`, patches that factory along with the
``renderer.backend`` global that ``set_3d_view`` and the other scene-level
helpers read directly. MNE keeps doing all of the transform math itself, which
matters because getting a head/MRI/device transform subtly wrong produces a
plausible-looking picture with the sensors in the wrong place.

Supported: meshes, surfaces, spheres, tubes and glyphs, which covers the static
figures the documentation renders. Not supported: the interactive
:class:`mne.viz.Brain` time viewer, which additionally needs dock widgets and a
time slider, and scalar colormaps, which pyvista-js 0.15 does not implement
(scalars fall back to a solid color).

Importing this module needs pyvista-js, the same way importing ``_pyvista``
needs VTK, but has no other effect: :func:`_activate` is what puts this
renderer in front of MNE's own.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pyvista_js as pv

from ...transforms import _find_vector_rotation, _sph_to_cart, quat_to_rot
from ._abstract import _AbstractRenderer
from ._utils import _vtk_faces


def _lite_set_view(plotter, azimuth=None, elevation=None):
    """Point a plotter along the requested azimuth and elevation."""
    if azimuth is None and elevation is None:
        return None
    phi = np.deg2rad(90.0 if azimuth is None else azimuth)
    theta = np.deg2rad(90.0 if elevation is None else elevation)
    position = _sph_to_cart(np.array([[1.0, phi, theta]]))[0]
    # view up flips near the poles, matching the 5/175 threshold _set_3d_view
    # uses, because there the view plane normal runs parallel to the camera
    if 5.0 <= abs(np.rad2deg(theta)) <= 175.0:
        viewup = (0.0, 0.0, 1.0)
    else:
        viewup = (0.0, 1.0, 0.0)
    plotter.view_vector(-position, viewup=viewup)
    return None


# Every scene a notebook drew used to stay live for the kernel's lifetime,
# because the close helpers on _LiteBackend were no-ops. Track the plotters
# weakly -- so they stay collectable -- and give close_all something to free.
_lite_live_plotters = []


def _lite_release_plotter(plotter, close=True):
    """Hand back a plotter's meshes, JS arrays and GPU buffers.

    ``clear()`` empties the actor list, which is where the geometry is held,
    so that is what frees the memory. ``close=False`` additionally says not to
    tear the render window down -- what trimming an older scene wants, since
    the notebook has already drawn it. pyvista-js 0.15 implements neither
    ``deep_clean`` nor ``close``, so today the two paths do the same thing;
    the flag keeps the intent right if that changes.
    """
    import gc as _gc

    if plotter is None:
        return None
    for _i in range(len(_lite_live_plotters) - 1, -1, -1):
        _p = _lite_live_plotters[_i]()
        if _p is None or _p is plotter:
            del _lite_live_plotters[_i]
    # pyvista-js is someone else's surface, so use whichever teardown of these
    # it actually implements
    _names = ("clear", "deep_clean", "close") if close else ("clear", "deep_clean")
    for _name in _names:
        _fn = getattr(plotter, _name, None)
        if _fn is not None:
            try:
                _fn()
            except Exception:
                pass
    _gc.collect()
    return None


# Each live scene holds its meshes in the WASM heap, a copy of them in JS and
# a set of GPU buffers. Nothing in a notebook calls close_3d_figure, so without
# a cap they all stay: 20_source_alignment builds six, which is enough to run
# the tab out of memory. Keep the newest few and give the rest their geometry
# back as new ones arrive -- scrolling back shows an empty canvas, which is a
# far better outcome than losing the page.
_LITE_MAX_LIVE_SCENES = 2


def _lite_trim_live_plotters():
    """Release everything but the most recent scenes."""
    while len(_lite_live_plotters) > _LITE_MAX_LIVE_SCENES:
        _p = _lite_live_plotters[0]()
        if _p is None:
            _lite_live_plotters.pop(0)
        else:
            # also drops it from the registry, so this terminates
            _lite_release_plotter(_p, close=False)
    return None


class _LiteRenderer(_AbstractRenderer):
    """Minimal MNE 3D renderer backed by pyvista-js."""

    # Its own kind rather than "notebook": the desktop notebook backend shares
    # a kernel with the page but still has VTK, a filesystem and OS threads,
    # and none of those are here. The one place that would care,
    # mne/gui/_coreg.py, never reaches its `_kind != "notebook"` branch in the
    # browser, because _configure_dock asks the renderer for ten _dock_add_*
    # methods this one does not have and fails first.
    _kind = "jupyterlite_notebook"

    def __init__(self, *args, **kwargs):
        # plot_alignment(fig=...) and plot_dipole_locations(fig=...) composite
        # into a scene the notebook already made, so draw into that plotter
        # rather than opening a second one and splitting the picture in two.
        # plot_alignment passes it positionally and create_3d_figure by name,
        # and `fig` is _PyVistaRenderer's first argument, so accept both.
        _fig = args[0] if args else kwargs.get("fig", None)
        if _fig is not None and hasattr(_fig, "add_mesh"):
            self.plotter = _fig
            return
        self.plotter = pv.Plotter()
        import weakref as _weakref

        _lite_live_plotters.append(_weakref.ref(self.plotter))
        # trim AFTER appending, so the scene being built is never the one freed
        _lite_trim_live_plotters()
        _bg = kwargs.get("bgcolor", kwargs.get("background_color", "black"))
        try:
            self.plotter.background_color = self._rgb(_bg)
        except Exception:
            pass
        # even lighting, so a surface is not black when rotated
        for _lp in (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ):
            try:
                self.plotter.add_light(
                    pv.Light(
                        position=(300.0 * _lp[0], 300.0 * _lp[1], 300.0 * _lp[2]),
                        focal_point=(0.0, 0.0, 0.0),
                        intensity=0.4,
                    )
                )
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
        _c = np.asarray(color, dtype=float).ravel()[:3]
        if _c.size < 3:
            return (0.5, 0.5, 0.5)
        if _c.max() > 1.0:  # 0-255 form
            _c = _c / 255.0
        return tuple(float(min(max(_v, 0.0), 1.0)) for _v in _c)

    def _subdivide(self, rr, tris):
        """One level of midpoint subdivision, sharing the new edge vertices."""
        _rr = [tuple(_v) for _v in np.asarray(rr, dtype=float)]
        _mid = {}
        _out = []
        for _a, _b, _c in np.asarray(tris, dtype=int):
            _m = []
            for _p, _q in ((_a, _b), (_b, _c), (_c, _a)):
                _k = (min(int(_p), int(_q)), max(int(_p), int(_q)))
                if _k not in _mid:
                    _mid[_k] = len(_rr)
                    _rr.append(tuple((np.asarray(_rr[_p]) + np.asarray(_rr[_q])) / 2.0))
                _m.append(_mid[_k])
            _ab, _bc, _ca = _m
            _out += [[_a, _ab, _ca], [_ab, _b, _bc], [_ca, _bc, _c], [_ab, _bc, _ca]]
        return np.asarray(_rr, dtype=float), np.asarray(_out, dtype=int)

    def _glyph_template(
        self, kind, radius=None, height=None, center=None, resolution=None, **kwargs
    ):
        """Return (rr, tris) for a glyph template, oriented along +x.

        pyvista-js's Sphere/Cylinder are parametric primitives with no
        triangle list, so build the templates here. ``_tile`` then stamps one
        of these at every position and merges the result, which is what keeps
        these cheap -- the copies share a single mesh and a single actor.

        Sizes follow the templates ``_pyvista.py`` hands the glyph filter, so
        the browser draws the markers at the size the rendered docs do.
        """
        if kind in ("sphere", "oct"):
            _r = 0.5 if radius is None else float(radius)
            rr = np.array(
                [
                    [1.0, 0, 0],
                    [-1.0, 0, 0],
                    [0, 1.0, 0],
                    [0, -1.0, 0],
                    [0, 0, 1.0],
                    [0, 0, -1.0],
                ],
                float,
            )
            tris = np.array(
                [
                    [0, 2, 4],
                    [2, 1, 4],
                    [1, 3, 4],
                    [3, 0, 4],
                    [2, 0, 5],
                    [1, 2, 5],
                    [3, 1, 5],
                    [0, 3, 5],
                ],
                int,
            )
            # "oct" is an octahedron on purpose -- that is what _pyvista.py
            # hands the glyph filter. A "sphere" has to look round, though:
            # fiducials and dig points are drawn with it, so subdivide onto the
            # unit sphere to land near the reference's 8x8 sphere (58 verts).
            if kind == "sphere":
                for _ in range(2):
                    rr, tris = self._subdivide(rr, tris)
                    rr /= np.linalg.norm(rr, axis=1)[:, None]
            return rr * _r, tris
        if kind == "cone":
            # apex along +x so the glyph filter's orientation applies, matching
            # pyvista.Cone(center=(0.5, 0, 0)): base at x=0, apex at x=height
            _r = 0.15 if radius is None else float(radius)
            _h = 1.0 if height is None else float(height)
            _n = 8 if not resolution else max(3, int(resolution) // 2)
            _ang = np.linspace(0.0, 2 * np.pi, _n, endpoint=False)
            _ring = np.column_stack(
                [np.zeros(_n), _r * np.cos(_ang), _r * np.sin(_ang)]
            )
            rr = np.vstack([_ring, [[_h, 0, 0]], [[0.0, 0, 0]]])
            tris = []
            for _i in range(_n):
                _j = (_i + 1) % _n
                tris += [[_i, _j, _n], [_n + 1, _j, _i]]  # side, base
            return rr, np.asarray(tris, int)
        # cylinder along +x, matching _cylinder_geom's convention
        _r = 0.1 if radius is None else float(radius)
        _h = 1.0 if height is None else float(height)
        _n = 8 if not resolution else max(3, int(resolution) // 2)
        _c = np.zeros(3) if center is None else np.asarray(center, float)
        _ang = np.linspace(0.0, 2 * np.pi, _n, endpoint=False)
        _ring = np.column_stack([np.zeros(_n), _r * np.cos(_ang), _r * np.sin(_ang)])
        _back = _ring + np.array([-_h / 2.0, 0, 0])
        _front = _ring + np.array([_h / 2.0, 0, 0])
        rr = np.vstack([_back, _front, [[-_h / 2.0, 0, 0]], [[_h / 2.0, 0, 0]]]) + _c
        tris = []
        for _i in range(_n):
            _j = (_i + 1) % _n
            tris += [[_i, _j, _n + _j], [_i, _n + _j, _n + _i]]  # wall
            tris += [[2 * _n, _j, _i]]  # back cap
            tris += [[2 * _n + 1, _n + _i, _n + _j]]  # front cap
        return rr, np.asarray(tris, int)

    def _add(self, points, tris, color, opacity=1.0):
        """Draw a mesh and return MNE's (actor, mesh) pair.

        ``opacity=None`` means "renderer default" in MNE's renderer API, which
        for PyVista reaches ``add_mesh(opacity=None)`` and draws opaque. Every
        drawing method here funnels through this, so translating it once covers
        all of them.
        """
        _pd = pv.PolyData(
            points=np.asarray(points, dtype=np.float32), faces=_vtk_faces(tris)
        )
        _actor = self.plotter.add_mesh(
            _pd,
            color=self._rgb(color),
            opacity=1.0 if opacity is None else float(opacity),
            smooth_shading=True,
        )
        return _actor, _pd

    def _rots_from_dirs(self, dirs):
        """Rotations carrying +x onto each direction, as the glyphs assume."""
        _x = np.array([1.0, 0.0, 0.0])
        return np.asarray([_find_vector_rotation(_x, _d) for _d in dirs], dtype=float)

    def _tile(self, rr, tris, positions, scales=None, rots=None, axis_scales=None):
        """Stamp one template mesh at many positions as a single mesh.

        ``_pyvista.py`` hands its template to VTK's glyph filter, which bakes
        every copy into one mesh and adds it once. Doing this per position
        instead means an oct-6 source space becomes 8196 meshes and 8196
        actors, which is enough to run the browser tab out of memory.
        """
        _rr = np.asarray(rr, dtype=float)
        _tris = np.asarray(tris, dtype=int)
        _pos = np.atleast_2d(np.asarray(positions, dtype=float))[:, :3]
        _n = len(_pos)
        _pts = np.repeat(_rr[None, :, :], _n, axis=0)
        if axis_scales is not None:
            # tubes span a given length without fattening, so scale the
            # template's axis alone
            _ax = np.atleast_1d(np.asarray(axis_scales, dtype=float))
            _pts[:, :, 0] *= _ax[np.arange(_n) % len(_ax)][:, None]
        if scales is not None:
            _sa = np.atleast_1d(np.asarray(scales, dtype=float))
            _pts *= _sa[np.arange(_n) % len(_sa)][:, None, None]
        if rots is not None:
            _ra = np.asarray(rots, dtype=float)
            _pts = np.einsum("nij,nkj->nki", _ra[np.arange(_n) % len(_ra)], _pts)
        _pts += _pos[:, None, :]
        _off = (np.arange(_n) * len(_rr))[:, None, None]
        return (_pts.reshape(-1, 3), (_tris[None, :, :] + _off).reshape(-1, 3))

    # -- drawing ------------------------------------------------------------
    def mesh(self, x, y, z, triangles, color=None, opacity=1.0, *args, **kwargs):
        _pts = np.column_stack(
            [np.asarray(x).ravel(), np.asarray(y).ravel(), np.asarray(z).ravel()]
        )
        return self._add(_pts, triangles, color, opacity)

    def surface(self, surface, color=None, opacity=1.0, *args, **kwargs):
        return self._add(surface["rr"], surface["tris"], color, opacity)

    def sphere(
        self,
        center,
        color=None,
        scale=1.0,
        opacity=1.0,
        resolution=8,
        backface_culling=False,
        radius=None,
        **kwargs,
    ):
        _c = np.atleast_2d(np.asarray(center, dtype=float))
        if not len(_c):
            return None, None
        _r = float(radius if radius is not None else scale)
        _rr, _tris = self._glyph_template("sphere", radius=_r, resolution=resolution)
        _pts, _faces = self._tile(_rr, _tris, _c)
        return self._add(_pts, _faces, color, opacity)

    def tube(self, origin, destination, radius=0.001, color=None, *args, **kwargs):
        _o = np.atleast_2d(np.asarray(origin, dtype=float))[:, :3]
        _d = np.atleast_2d(np.asarray(destination, dtype=float))[:, :3]
        _n = min(len(_o), len(_d))
        if not _n:
            return None, None
        _vec = _d[:_n] - _o[:_n]
        _len = np.linalg.norm(_vec, axis=1)
        _keep = _len > 0
        if not _keep.any():
            return None, None
        _vec, _len = _vec[_keep], _len[_keep]
        _ctr = (_o[:_n][_keep] + _d[:_n][_keep]) / 2.0
        # one unit-height template stretched to each segment, merged into a
        # single mesh rather than a cylinder primitive per segment
        _rr, _tris = self._glyph_template("cylinder", radius=float(radius), height=1.0)
        _pts, _faces = self._tile(
            _rr,
            _tris,
            _ctr,
            rots=self._rots_from_dirs(_vec / _len[:, None]),
            axis_scales=_len,
        )
        return self._add(_pts, _faces, color, kwargs.get("opacity", 1.0))

    def quiver3d(
        self,
        x,
        y,
        z,
        u,
        v,
        w,
        color=None,
        scale=1.0,
        mode="arrow",
        opacity=1.0,
        *,
        glyph_height=None,
        glyph_center=None,
        glyph_resolution=None,
        glyph_radius=0.15,
        solid_transform=None,
        **kwargs,
    ):
        """Draw one merged glyph mesh, the way the glyph filter would.

        ``_pyvista.py`` builds a template, lets VTK's glyph filter bake a copy
        at every point into one mesh, and adds that once. Drawing a primitive
        per point instead is what made ``20_source_alignment`` -- an oct-6
        source space, so 8196 glyphs, twice -- exhaust the browser tab.
        """
        _x, _y, _z = (np.atleast_1d(np.asarray(_q, dtype=float)) for _q in (x, y, z))
        _ctr = np.column_stack([_x, _y, _z])
        _n = len(_ctr)
        if not _n:
            return None, None
        _s = float(np.asarray(scale).ravel()[0]) if np.size(scale) else 1.0
        _i = np.arange(_n)
        _u, _v, _w = (np.atleast_1d(np.asarray(_q, dtype=float)) for _q in (u, v, w))
        _dirs = np.column_stack([_u[_i % len(_u)], _v[_i % len(_v)], _w[_i % len(_w)]])
        _norm = np.linalg.norm(_dirs, axis=1)
        _flat = _norm == 0
        _dirs[_flat] = (1.0, 0.0, 0.0)
        _norm[_flat] = 1.0
        _dirs = _dirs / _norm[:, None]
        # the same templates _pyvista.py feeds the filter; `scale` then plays
        # the part its `factor` does
        if mode == "oct":
            # vtkPlatonicSolidSource puts its octahedron on the unit
            # circumsphere, and the MRI fiducials get their real size from
            # solid_transform (mri_fid_scale, 5 mm) rather than from `scale`
            _kind, _tkw = "oct", dict(radius=1.0)
        elif mode == "sphere":
            _kind, _tkw = "sphere", dict(radius=0.5)
        elif mode == "cylinder":
            _kind = "cylinder"
            _tkw = dict(
                radius=glyph_radius,
                height=glyph_height,
                center=glyph_center,
                resolution=glyph_resolution,
            )
        else:  # arrow / cone / 2darrow
            _kind = "cone"
            _tkw = dict(
                radius=glyph_radius, height=glyph_height, resolution=glyph_resolution
            )
        _rr, _tris = self._glyph_template(_kind, **_tkw)
        if solid_transform is not None:
            # _pyvista.py transforms the template before glyphing, and this is
            # where the fiducial markers get their size and 45 deg roll
            _st = np.asarray(solid_transform, dtype=float)
            _rr = _rr @ _st[:3, :3].T + _st[:3, 3]
        _rots = None if mode in ("sphere", "oct") else self._rots_from_dirs(_dirs)
        _pts, _faces = self._tile(_rr, _tris, _ctr, scales=_s, rots=_rots)
        return self._add(_pts, _faces, color, opacity)

    def instanced_mesh(
        self,
        rr,
        tris,
        positions,
        quats=None,
        colors=None,
        scales=None,
        opacity=1.0,
        *args,
        **kwargs,
    ):
        """Stamp the template at every position, merged per distinct color.

        Rotate with MNE's own quaternion helper so oriented glyphs (EEG
        cylinders) point the way MNE intended rather than all along +x.
        pyvista-js has no per-vertex color, so instances are grouped by the
        color they asked for and each group becomes one mesh -- a handful of
        actors for a sensor array instead of one per sensor.
        """
        _pos = np.atleast_2d(np.asarray(positions, dtype=float))[:, :3]
        _n = len(_pos)
        if not _n:
            return None, None
        _rot = None
        if quats is not None:
            _rot = np.asarray(
                quat_to_rot(np.atleast_2d(np.asarray(quats, dtype=float))), dtype=float
            )
        _idx = np.arange(_n)
        if colors is not None and np.ndim(colors) > 1:
            _ca = np.asarray(colors)
            _uniq, _inv = np.unique(_ca[_idx % len(_ca)], axis=0, return_inverse=True)
            _inv = np.asarray(_inv).ravel()
            _groups = [(_uniq[_k], _idx[_inv == _k]) for _k in range(len(_uniq))]
        else:
            _groups = [(colors, _idx)]
        _out = (None, None)
        for _col, _sel in _groups:
            _sc = None
            if scales is not None:
                _sa = np.atleast_1d(np.asarray(scales, dtype=float))
                _sc = _sa[_sel % len(_sa)]
            _rt = None if _rot is None else _rot[_sel % len(_rot)]
            _pts, _faces = self._tile(rr, tris, _pos[_sel], scales=_sc, rots=_rt)
            _out = self._add(_pts, _faces, _col, opacity)
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
        raise NotImplementedError(
            "Projecting 3D positions onto the scene is not supported in the "
            "browser: it has to return a _Projection built from the render "
            "window, and pyvista-js does not expose one."
        )

    def screenshot(self, mode="rgb", filename=None, **kwargs):
        raise NotImplementedError(
            "Taking a screenshot is not supported in the browser: vtk.js draws "
            "to a live canvas and pyvista-js cannot read it back as an array."
        )

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
        # Same order as _get_3d_view: roll, distance, azimuth, elevation and
        # then the focalpoint. Brain unpacks positions 3 and 4 as the angles,
        # so the focalpoint has to be the last element rather than the fourth.
        return (0.0, 1.0, 0.0, 0.0, np.zeros(3))

    def set_camera(
        self,
        azimuth=None,
        elevation=None,
        distance=None,
        focalpoint=None,
        roll=None,
        *args,
        **kwargs,
    ):
        return _lite_set_view(self.plotter, azimuth, elevation)

    @property
    def figure(self):
        """The scene, under the name the tutorials reach for.

        ``_PyVistaRenderer`` hands out one object as both ``.figure`` and
        ``.scene()``; ``20_source_alignment`` builds a renderer itself with
        ``create_3d_figure(scene=False)`` and then passes ``renderer.figure``
        to ``set_3d_view``, so the two have to stay the same thing here too.
        """
        return self.plotter

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


class _LiteBackend:
    """Stand-in for the module MNE imports into ``renderer.backend``.

    ``set_3d_view``, ``set_3d_title`` and the ``close_*`` helpers are module-level
    functions that reach for that global directly instead of going through
    ``_get_renderer``, so replacing the factory alone leaves them calling into
    ``None``.  The figure they are handed is the pyvista-js plotter that
    ``_LiteRenderer.scene`` returns.
    """

    def _set_3d_view(
        self,
        figure,
        azimuth=None,
        elevation=None,
        focalpoint=None,
        distance=None,
        roll=None,
    ):
        return _lite_set_view(figure, azimuth, elevation)

    def _set_3d_title(
        self, figure, title, size=40, color="white", position="upper_left"
    ):
        return None

    def _close_3d_figure(self, figure):
        _lite_release_plotter(figure)
        return None

    def _close_all(self):
        # the registry holds weak references, so deref before releasing --
        # handing the ref itself to _lite_release_plotter matches nothing and
        # never shortens the list
        while _lite_live_plotters:
            _p = _lite_live_plotters[-1]()
            if _p is None:
                _lite_live_plotters.pop()
            else:
                _lite_release_plotter(_p)
        return None


_LITE_SAVED = {}


def _activate():
    """Install this renderer as the one MNE draws with.

    Replaces the ``_get_renderer`` factory and the ``renderer.backend`` global
    that ``set_3d_view``, ``set_3d_title`` and the ``close_*`` helpers read
    directly, so that patching the factory alone does not leave them calling
    into ``None``. Returns the previous state, which :func:`_deactivate` puts
    back.
    """
    from . import renderer as _mne_rend

    if not _LITE_SAVED:
        _LITE_SAVED.update(
            _get_renderer=_mne_rend._get_renderer,
            backend=_mne_rend.backend,
            MNE_3D_BACKEND=_mne_rend.MNE_3D_BACKEND,
        )
    _mne_rend._get_renderer = _lite_get_renderer
    _mne_rend.backend = _LiteBackend()
    # Naming a backend keeps _get_3d_backend() from walking VALID_3D_BACKENDS and
    # importing _qt, which would overwrite the stub above on its way to failing.
    _mne_rend.MNE_3D_BACKEND = "notebook"
    return dict(_LITE_SAVED)


def _deactivate():
    """Undo :func:`_activate`, restoring whatever MNE was drawing with before."""
    if not _LITE_SAVED:
        return None
    from . import renderer as _mne_rend

    for _name, _value in _LITE_SAVED.items():
        setattr(_mne_rend, _name, _value)
    _LITE_SAVED.clear()
    return None
