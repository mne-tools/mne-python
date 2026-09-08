"""A pyvista-js (vtk.js) drawing backend for MNE's 3D renderer.

MNE's 3D functions do their geometry and coordinate-frame work in numpy and only
hand the result to a renderer, so swapping that last step is enough to draw in
a browser kernel, where VTK cannot load. Selected with
``mne.viz.set_3d_backend("jupyterlite_notebook")``.

Supported: meshes, surfaces, spheres, tubes and glyphs, which covers the static
figures. Not supported: :class:`mne.viz.Brain` (needs dock widgets), scalars,
colormaps and contours (the vtk.js template builds no lookup table, so every
mesh is one solid color), and figure size (pyvista-js writes a 600x400 canvas).
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import weakref
from contextlib import nullcontext

import numpy as np
import pyvista_js as pv
from matplotlib.colors import to_rgb

from ...surface import _tessellate_sphere
from ...transforms import (
    _cart_to_sph,
    _find_vector_rotation,
    _sph_to_cart,
    quat_to_rot,
)
from ...utils import _check_option, _validate_type
from ._abstract import Figure3D, _AbstractRenderer
from ._utils import ALLOWED_QUIVER_MODES, _vtk_faces

# vtk.js places text in normalized window coordinates; PyVista takes these names
_TITLE_POSITIONS = {
    "lower_left": (0.05, 0.05),
    "lower_right": (0.65, 0.05),
    "upper_left": (0.05, 0.90),
    "upper_right": (0.65, 0.90),
}
_DEFAULT_COLOR = (0.5, 0.5, 0.5)  # what color=None draws as
# Nothing in a notebook closes figures, and each live scene holds its meshes in
# the WASM heap, JS and GPU buffers (20_source_alignment makes six, enough to run
# the tab out of memory), so only the newest few stay live
_LITE_MAX_LIVE_SCENES = 2
_lite_live_plotters = []  # weakrefs, so scenes stay collectable


def _rgb(color):
    """Return the (r, g, b) tuple pyvista-js takes; it cannot parse hex itself."""
    return _DEFAULT_COLOR if color is None else to_rgb(color)


def _lite_unsupported(what):
    raise NotImplementedError(f"{what} is not supported in the browser.")


def _lite_add_text(plotter, text, position, size, color):
    actor = pv.Text(str(text), position=tuple(float(coord) for coord in position))
    actor.prop.font_size = int(size)
    actor.prop.color = _rgb(color)
    plotter.add_text(actor)
    return actor


def _lite_view_angles(plotter):
    """Return the (azimuth, elevation) in degrees the plotter looks from, or None.

    The view is kept as ``view_vector``, a camera position that vtk.js aims at
    the origin and then frames with ``resetCamera()``, rather than as a camera
    object, which would need the distance that MNE mostly passes as None.
    """
    view_vector = plotter._renderer._view_vector  # pyvista-js 0.15
    if view_vector is None:  # nothing set yet, so vtk.js chooses
        return None
    _, phi, theta = _cart_to_sph(np.asarray(view_vector, float)[np.newaxis])[0]
    return float(np.rad2deg(phi)) % 360, float(np.rad2deg(theta)) % 180


def _lite_set_view(plotter, azimuth=None, elevation=None):
    """Point the plotter, keeping the angle not given as _pyvista._set_3d_view does."""
    if azimuth is None and elevation is None:
        return
    current = _lite_view_angles(plotter) or (90.0, 90.0)  # plot_alignment's view
    phi = np.deg2rad(current[0] if azimuth is None else azimuth)
    theta = np.deg2rad(current[1] if elevation is None else elevation)
    # view up flips near the poles, matching _set_3d_view
    up = (
        (0.0, 0.0, 1.0)
        if elevation is None or 5 <= abs(elevation) <= 175
        else (0.0, 1.0, 0.0)
    )
    plotter.view_vector(
        tuple(_sph_to_cart(np.array([[1.0, phi, theta]]))[0]), viewup=up
    )


def _lite_get_view(plotter):
    """Return (roll, distance, azimuth, elevation, focalpoint) as _get_3d_view does."""
    azimuth, elevation = _lite_view_angles(plotter) or (0.0, 0.0)
    return (0.0, 1.0, azimuth, elevation, np.zeros(3))


def _lite_release_plotter(plotter):
    """Drop a plotter's geometry and forget it; clear() is all the teardown there is."""
    _lite_live_plotters[:] = [
        ref for ref in _lite_live_plotters if ref() not in (None, plotter)
    ]
    if plotter is not None:
        plotter.clear()


def _lite_opacity(color, opacity):
    """Fold RGBA alpha into the opacity, since each mesh here is one solid color.

    _PyVistaRenderer draws instance colors as RGBA scalars, and the alpha is
    how plot_alignment fades the MEG coils.
    """
    opacity = 1.0 if opacity is None else float(opacity)
    if not isinstance(color, str) and np.shape(color) == (4,):
        opacity *= float(color[3])
    return opacity


def _lite_revolve(profile, n_side):
    """Return (rr, tris) of rings of ``radius`` at ``x`` about +x, joined and capped."""
    angles = np.linspace(0.0, 2 * np.pi, n_side, endpoint=False)
    rr = [
        np.column_stack([np.full(n_side, x), r * np.cos(angles), r * np.sin(angles)])
        for x, r in profile
    ]
    idx = np.arange(n_side)
    nxt = (idx + 1) % n_side
    tris = []
    for a in range(0, (len(profile) - 1) * n_side, n_side):
        b = a + n_side
        tris += [np.c_[a + idx, a + nxt, b + nxt], np.c_[a + idx, b + nxt, b + idx]]
    n_points = len(profile) * n_side
    ends = ((0, profile[0], True), (n_points - n_side, profile[-1], False))
    for ring, (x, r), flip in ends:
        if r > 0:  # cap it, wound to face outward
            rr.append([[x, 0.0, 0.0]])
            cap = np.c_[np.full(n_side, n_points), ring + idx, ring + nxt]
            tris.append(cap[:, ::-1] if flip else cap)
            n_points += 1
    return np.vstack(rr), np.vstack(tris).astype(int)


class _LiteFigure(Figure3D):
    """pyvista-js-based 3D figure; ``.plotter`` is the pyvista-js plotter."""

    def __init__(self):
        pass

    def _init(self, plotter):
        self._plotter = plotter
        return self


class _LiteRenderer(_AbstractRenderer):
    """MNE 3D renderer backed by pyvista-js."""

    # not "notebook": that backend has VTK, a filesystem and OS threads
    _kind = "jupyterlite_notebook"

    def __init__(
        self,
        fig=None,
        size=(600, 600),
        bgcolor="black",
        *,
        name=None,
        show=False,
        shape=(1, 1),
        notebook=None,
        smooth_shading=True,
        splash=False,
        multi_samples=None,
    ):
        # _PyVistaRenderer's signature, but size, shape, name and show cannot
        # be honored: the canvas is fixed and written only when show() runs
        _validate_type(fig, (None, _LiteFigure), "fig")
        if fig is not None:  # plot_alignment(fig=...) composites into it
            self._figure = fig
            return
        self._figure = _LiteFigure()._init(pv.Plotter())
        _lite_live_plotters.append(weakref.ref(self.plotter))
        while len(_lite_live_plotters) > _LITE_MAX_LIVE_SCENES:
            _lite_release_plotter(_lite_live_plotters[0]())
        self.plotter.background_color = _rgb(bgcolor)
        # one light per axis direction, since a vtk.js light only lights what
        # faces it, each dim enough that a surface facing two does not blow out
        for position in np.vstack([np.eye(3), -np.eye(3)]) * 300.0:
            self.plotter.add_light(
                pv.Light(
                    position=tuple(position), focal_point=(0.0,) * 3, intensity=0.4
                )
            )

    @property
    def plotter(self):
        return self._figure.plotter

    @property
    def figure(self):
        return self._figure  # 20_source_alignment passes this to set_3d_view

    def scene(self):
        return self._figure

    def show(self):
        self.plotter.show()

    # -- geometry -----------------------------------------------------------
    def _glyph_template(
        self, kind, radius=None, height=None, center=None, resolution=None
    ):
        """Return (rr, tris) of a glyph along +x, sized like _pyvista.py's templates."""
        # half of VTK's side count: these get stamped at every sensor
        n_side = 8 if resolution is None else max(3, int(resolution) // 2)
        if kind in ("sphere", "oct"):
            # level 1 is the octahedron vtkPlatonicSolidSource draws; level 3
            # is round enough for dig points at 66 vertices
            rr, tris = _tessellate_sphere(1 if kind == "oct" else 3)
            return rr * (0.5 if radius is None else radius), tris
        if kind == "arrow":  # vtkArrowSource: 0.03 shaft under a 0.1 tip from 0.65
            return _lite_revolve([(0, 0.03), (0.65, 0.03), (0.65, 0.1), (1, 0)], 12)
        height = 1.0 if height is None else float(height)
        if kind == "cone":  # base at the position, apex along +x
            return _lite_revolve(
                [(0, 0.15 if radius is None else radius), (height, 0)], n_side
            )
        assert kind == "cylinder", kind
        radius = 0.1 if radius is None else float(radius)
        rr, tris = _lite_revolve([(-height / 2, radius), (height / 2, radius)], n_side)
        if center is not None:
            # _cylinder_geom builds along y and turns 90 degrees about z, so the
            # (EEG electrode) offset it is given lands at (-cy, cx, cz)
            rr = rr + np.array([-center[1], center[0], center[2]], float)
        return rr, tris

    def _tile(self, rr, tris, positions, scales=None, rots=None, axis_scales=None):
        """Stamp a template at every position into one mesh, as VTK's glyph filter does.

        One mesh per position runs the tab out of memory for an oct-6 source
        space. ``axis_scales`` stretches the template along x alone (tubes).
        """
        rr = np.asarray(rr, float)
        positions = np.atleast_2d(np.asarray(positions, float))[:, :3]
        n_pos = len(positions)
        idx = np.arange(n_pos)
        points = np.repeat(rr[np.newaxis], n_pos, axis=0)
        if axis_scales is not None:
            axis_scales = np.atleast_1d(np.asarray(axis_scales, float))
            points[:, :, 0] *= axis_scales[idx % len(axis_scales)][:, np.newaxis]
        if scales is not None:
            scales = np.atleast_1d(np.asarray(scales, float))
            points *= scales[idx % len(scales)][:, np.newaxis, np.newaxis]
        if rots is not None:
            points = np.einsum(
                "nij,nkj->nki", np.asarray(rots)[idx % len(rots)], points
            )
        points += positions[:, np.newaxis]
        offsets = (idx * len(rr))[:, np.newaxis, np.newaxis]
        tris = np.asarray(tris, int)[np.newaxis] + offsets
        return points.reshape(-1, 3), tris.reshape(-1, 3)

    def _add(self, points, tris, color, opacity=1.0):
        """Draw one solid-color mesh and return MNE's (actor, mesh) pair."""
        # float32 halves the WASM cost, and vtk.js is single precision anyway;
        # the faces go over flat because vtk.js reads one VTK cell array
        mesh = pv.PolyData(
            points=np.asarray(points, np.float32), faces=_vtk_faces(tris).ravel()
        )
        actor = self.plotter.add_mesh(
            mesh,
            color=_rgb(color),
            opacity=1.0 if opacity is None else float(opacity),
            smooth_shading=True,
        )
        return actor, mesh

    # -- drawing ------------------------------------------------------------
    # The signatures follow _PyVistaRenderer's so positional calls bind alike;
    # scalars, colormaps, culling, normals and names are accepted and ignored.
    def mesh(
        self,
        x,
        y,
        z,
        triangles,
        color,
        opacity=1.0,
        *,
        backface_culling=False,
        scalars=None,
        colormap=None,
        vmin=None,
        vmax=None,
        interpolate_before_map=True,
        representation="surface",
        line_width=1.0,
        normals=None,
        name=None,
        **kwargs,
    ):
        points = np.column_stack([np.ravel(x), np.ravel(y), np.ravel(z)])
        return self._add(points, triangles, color, opacity)

    def surface(
        self,
        surface,
        color=None,
        opacity=1.0,
        vmin=None,
        vmax=None,
        colormap=None,
        normalized_colormap=False,
        scalars=None,
        backface_culling=False,
        *,
        name=None,
    ):
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
    ):
        # _pyvista.py glyphs a radius-0.5 sphere by `scale`, or `radius` by 1
        center = np.atleast_2d(np.asarray(center, float))
        if not len(center):
            return None, None
        rr, tris = self._glyph_template(
            "sphere", radius=0.5 * scale if radius is None else radius
        )
        return self.instanced_mesh(rr, tris, center, colors=color, opacity=opacity)

    def tube(
        self,
        origin,
        destination,
        radius=0.001,
        color="white",
        scalars=None,
        vmin=None,
        vmax=None,
        colormap="RdBu",
        normalized_colormap=False,
        reverse_lut=False,
        opacity=None,
    ):
        origin = np.atleast_2d(np.asarray(origin, float))[:, :3]
        destination = np.atleast_2d(np.asarray(destination, float))[:, :3]
        vec = destination - origin
        length = np.linalg.norm(vec, axis=1)
        keep = length > 0
        if not keep.any():
            return None, None
        vec, length, origin = vec[keep], length[keep], origin[keep]
        # a unit cylinder stretched along its axis to each segment
        rr, tris = self._glyph_template("cylinder", radius=radius, resolution=20)
        rots = _find_vector_rotation(np.array([1.0, 0, 0]), vec / length[:, np.newaxis])
        points, faces = self._tile(
            rr, tris, origin + vec / 2, rots=rots, axis_scales=length
        )
        return self._add(points, faces, color, opacity)

    def quiver3d(
        self,
        x,
        y,
        z,
        u,
        v,
        w,
        color,
        scale,
        mode,
        *,
        glyph_height=None,
        glyph_center=None,
        glyph_resolution=None,
        opacity=1.0,
        scale_mode="none",
        scalars=None,
        colormap=None,
        backface_culling=False,
        glyph_radius=0.15,
        solid_transform=None,
        clim=None,
    ):
        _check_option("mode", mode, ALLOWED_QUIVER_MODES)
        _check_option("scale_mode", scale_mode, ("none", "scalar", "vector"))
        x, y, z, u, v, w = (np.ravel(np.asarray(q, float)) for q in (x, y, z, u, v, w))
        n_pos = len(x)
        if not n_pos:
            return None, None
        idx = np.arange(n_pos)
        dirs = np.column_stack([q[idx % len(q)] for q in (u, v, w)])
        norms = np.linalg.norm(dirs, axis=1)
        dirs[norms == 0] = (1.0, 0.0, 0.0)
        dirs /= np.where(norms == 0, 1.0, norms)[:, np.newaxis]
        # _pyvista.py sizes "arrow" by the scalars (plot_alignment's axes rely
        # on it) and "2darrow" by the vector, whatever scale_mode says
        scale_mode = {"arrow": "scalar", "2darrow": "vector"}.get(mode, scale_mode)
        if scale_mode == "scalar":
            values = np.ones(n_pos) if scalars is None else np.ravel(scalars)
            sizes = scale * np.asarray(values, float)[idx % len(values)]
        else:
            sizes = scale * norms if scale_mode == "vector" else float(scale)
        # the templates _pyvista.py feeds the glyph filter, with `scale` as its
        # factor; the unit "oct" is sized by solid_transform (MRI fiducials),
        # and "2darrow" borrows the 3D arrow since vtk.js has no 2D glyphs
        template_kw = dict(
            oct=dict(radius=1.0),
            sphere=dict(radius=0.5),
            cylinder=dict(
                radius=glyph_radius,
                height=glyph_height,
                center=glyph_center,
                resolution=glyph_resolution,
            ),
            cone=dict(
                radius=glyph_radius, height=glyph_height, resolution=glyph_resolution
            ),
        ).get(mode, dict())
        rr, tris = self._glyph_template(
            "arrow" if mode == "2darrow" else mode, **template_kw
        )
        if solid_transform is not None:
            solid_transform = np.asarray(solid_transform, float)
            rr = rr @ solid_transform[:3, :3].T + solid_transform[:3, 3]
        # spheres look the same however turned, and the fiducial "oct" is
        # always asked for along +x, so neither needs rotations
        rots = None
        if mode not in ("sphere", "oct"):
            rots = _find_vector_rotation(np.array([1.0, 0, 0]), dirs)
        points, faces = self._tile(rr, tris, np.column_stack([x, y, z]), sizes, rots)
        return self._add(points, faces, color, opacity)

    def instanced_mesh(
        self,
        rr,
        tris,
        positions,
        quats=None,
        colors=None,
        scales=None,
        opacity=1.0,
        backface_culling=False,
        *,
        name=None,
    ):
        """Stamp the template at every position, one merged mesh per distinct color.

        vtk.js cannot color per instance, so several colors give a list of
        actors. The cloud handed back second is what _3d.py writes channel
        names onto, and is always one object.
        """
        positions = np.atleast_2d(np.asarray(positions, float))[:, :3]
        n_pos = len(positions)
        cloud = pv.PolyData(points=positions)
        cloud.field_data = dict()
        if not n_pos:
            return None, cloud
        rots = None
        if quats is not None:
            quats = np.atleast_2d(np.asarray(quats, float))
            assert quats.shape[-1] == 3, quats.shape  # (w, x, y, z) would misread
            rots = quat_to_rot(quats)
        idx = np.arange(n_pos)
        if colors is not None and np.ndim(colors) > 1:
            colors = np.asarray(colors, float)[idx % len(colors)]
            uniq, inverse = np.unique(colors, axis=0, return_inverse=True)
            groups = [(uniq[k], idx[np.ravel(inverse) == k]) for k in range(len(uniq))]
        else:
            groups = [(colors, idx)]
        actors = list()
        for color, sel in groups:
            group_scales = None
            if scales is not None:
                scales = np.atleast_1d(np.asarray(scales, float))
                group_scales = scales[sel % len(scales)]
            group_rots = None if rots is None else rots[sel % len(rots)]
            points, faces = self._tile(
                rr, tris, positions[sel], group_scales, group_rots
            )
            actors.append(
                self._add(points, faces, color, _lite_opacity(color, opacity))[0]
            )
        return (actors[0] if len(actors) == 1 else actors), cloud

    def text2d(
        self,
        x_window,
        y_window,
        text,
        size=14,
        color="white",
        justification=None,
        font_file=None,
    ):
        if justification is not None or font_file is not None:
            _lite_unsupported("Justified text and custom fonts")
        return _lite_add_text(self.plotter, text, (x_window, y_window), size, color)

    def remove_mesh(self, mesh_data):
        # the renderer keeps the actor dict and the plotter a dict pointing at
        # it, so drop both; instanced_mesh hands back one actor per color
        actor, _ = mesh_data
        actors = actor if isinstance(actor, list) else [actor]
        plotter = self.plotter
        plotter.actors[:] = [a for a in plotter.actors if a["actor"] not in actors]
        plotter._renderer.actors[:] = [
            a for a in plotter._renderer.actors if a not in actors
        ]

    # -- nothing to do in a browser -----------------------------------------
    def set_interaction(self, interaction):
        pass  # vtk.js ships one trackball style

    def _update(self):
        pass  # the page paints after the cell finishes

    def _window_close_connect(self, func, *, after=True):
        pass  # an output cell has no close event

    def text3d(self, x, y, z, text, scale, color="white"):
        pass  # no camera-facing 3D text, so sensors go unlabeled

    def close(self):
        _lite_release_plotter(self.plotter)

    # -- things pyvista-js cannot do ----------------------------------------
    def contour(self, *args, **kwargs):
        _lite_unsupported("Drawing contours")  # one color would mislead

    def scalarbar(self, *args, **kwargs):
        _lite_unsupported("Drawing a scalar bar")

    def legend(self, *args, **kwargs):
        _lite_unsupported("Drawing a legend")

    def subplot(self, *args, **kwargs):
        _lite_unsupported("Subplots")

    def _process_events(self, *args, **kwargs):
        _lite_unsupported("Draining the event loop")  # the page runs it

    def _window_set_cursor(self, *args, **kwargs):
        _lite_unsupported("Setting the cursor")

    def _enable_time_interaction(self, *args, **kwargs):
        _lite_unsupported("The time slider")  # needs dock widgets

    def project(self, xyz, ch_names):
        _lite_unsupported("Projecting 3D positions onto the scene")

    def screenshot(self, mode="rgb", filename=None):
        return _take_3d_screenshot(self._figure, mode=mode, filename=filename)

    # -- camera -------------------------------------------------------------
    def get_camera(self, *, rigid=None):
        return _lite_get_view(self.plotter)

    def set_camera(
        self,
        azimuth=None,
        elevation=None,
        distance=None,
        focalpoint=None,
        roll=None,
        *,
        rigid=None,
        update=True,
    ):
        # distance, focalpoint and roll go unused: vtk.js frames the scene
        _lite_set_view(self.plotter, azimuth, elevation)


# -- the module surface renderer.py expects of a 3D backend -----------------
_Renderer = _LiteRenderer
_testing_context = nullcontext  # nothing draws differently under test


def _set_3d_view(
    figure,
    azimuth=None,
    elevation=None,
    focalpoint=None,
    distance=None,
    roll=None,
    rigid=None,
    update=True,
):
    _lite_set_view(figure.plotter, azimuth, elevation)


def _set_3d_title(figure, title, size=16, *, color="white", position="upper_left"):
    if isinstance(position, str):
        _check_option("position", position, sorted(_TITLE_POSITIONS))
        position = _TITLE_POSITIONS[position]
    return _lite_add_text(figure.plotter, title, position, size, color)


def _check_3d_figure(figure):
    _validate_type(figure, _LiteFigure, "figure")


def _take_3d_screenshot(figure, mode="rgb", filename=None):
    # pyvista-js can, but only by driving a headless browser from outside
    _lite_unsupported("Taking a screenshot")


def _clear_3d_figure(figure):
    figure.plotter.clear()


def _close_3d_figure(figure):
    _lite_release_plotter(figure.plotter)  # there is no window to close


def _close_all():
    while _lite_live_plotters:
        _lite_release_plotter(_lite_live_plotters[-1]())
