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
(vtk.js). It is a 3D backend like ``_qt`` and ``_notebook`` are, selected with
``mne.viz.set_3d_backend("jupyterlite")``, which is all a browser kernel has to
do. MNE keeps doing all of the transform math itself, which matters because
getting a head/MRI/device transform subtly wrong produces a plausible-looking
picture with the sensors in the wrong place.

Supported: meshes, surfaces, spheres, tubes and glyphs, which covers the static
figures the documentation renders. Not supported: the interactive
:class:`mne.viz.Brain` time viewer, which additionally needs dock widgets and a
time slider, and scalar colormaps: ``Plotter.add_mesh`` takes ``scalars`` and
``cmap`` and writes them into the scene, but the vtk.js template it renders
through builds no lookup table and never reads them, so a mesh carrying
scalars draws in a solid color.

Importing this module needs pyvista-js, the same way importing ``_pyvista``
needs VTK.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import gc
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
from ...utils import _check_option
from ._abstract import _AbstractRenderer
from ._utils import _vtk_faces

# vtk.js positions text in normalized window coordinates, where PyVista takes
# the names MNE's set_3d_title passes through
_TITLE_POSITIONS = {
    "lower_left": (0.05, 0.05),
    "lower_right": (0.65, 0.05),
    "upper_left": (0.05, 0.90),
    "upper_right": (0.65, 0.90),
}

# what MNE means by ``color=None``: "whatever the renderer draws by default",
# which for PyVista is resolved inside add_mesh and has to be picked here
_DEFAULT_COLOR = (0.5, 0.5, 0.5)


def _rgb(color):
    """Return an (r, g, b) 0-1 tuple, the only color form pyvista-js takes.

    Its own parser knows fourteen color names and rejects hex strings, so do
    the conversion here. ``to_rgb`` covers every form MNE hands a renderer --
    color names, hex, ``"C0"`` and 0-1 ``(r, g, b[, a])`` sequences -- and
    raises on anything else, which is what should happen.
    """
    return _DEFAULT_COLOR if color is None else to_rgb(color)


def _lite_add_text(plotter, text, position, size, color):
    """Add a vtk.js 2D text actor and return it, the way MNE's text2d does."""
    actor = pv.Text(str(text), position=tuple(float(coord) for coord in position))
    actor.prop.font_size = int(size)
    actor.prop.color = _rgb(color)
    plotter.add_text(actor)
    return actor


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


def _lite_get_view(plotter):
    """Return the direction a plotter is looking along, as _get_3d_view orders it.

    ``view_vector`` stores that direction on the plotter's renderer and leaves
    ``Plotter.camera`` unset, which is deliberate: the vtk.js side then calls
    ``resetCamera()`` and frames the scene, whereas a camera object would have
    to carry a distance, and MNE asks for ``distance=None`` more often than not.
    The cost is that ``Plotter.camera_position`` reads that unset camera and so
    stays ``None``; the direction below is the scene's actual camera state, and
    inverting it recovers the angles :func:`_lite_set_view` applied.
    """
    view_vector = plotter._renderer._view_vector  # pyvista-js 0.15
    if view_vector is None:  # no view set yet, so vtk.js is choosing one
        return (0.0, 1.0, 0.0, 0.0, np.zeros(3))
    _, phi, theta = _cart_to_sph(-np.asarray(view_vector, float)[np.newaxis])[0]
    # roll is 0 because the only camera roll _lite_set_view applies is the view
    # up flip at the poles; the distance and focalpoint are the ones the vtk.js
    # resetCamera() gives this path, a unit direction aimed at the origin
    return (
        0.0,
        1.0,
        float(np.rad2deg(phi)) % 360,
        float(np.rad2deg(theta)) % 180,
        np.zeros(3),
    )


# Every scene a notebook drew used to stay live for the kernel's lifetime,
# because the close helpers were no-ops. Track the plotters weakly -- so they
# stay collectable -- and give close_all something to free.
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
    if plotter is None:
        return None
    for idx in range(len(_lite_live_plotters) - 1, -1, -1):
        live = _lite_live_plotters[idx]()
        if live is None or live is plotter:
            del _lite_live_plotters[idx]
    # pyvista-js is someone else's surface, so use whichever teardown of these
    # it actually implements
    names = ("clear", "deep_clean", "close") if close else ("clear", "deep_clean")
    for name in names:
        teardown = getattr(plotter, name, None)
        if teardown is not None:
            teardown()
    gc.collect()
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
        oldest = _lite_live_plotters[0]()
        if oldest is None:
            _lite_live_plotters.pop(0)
        else:
            # also drops it from the registry, so this terminates
            _lite_release_plotter(oldest, close=False)
    return None


class _LiteRenderer(_AbstractRenderer):
    """Minimal MNE 3D renderer backed by pyvista-js."""

    # Its own kind rather than "notebook": the desktop notebook backend shares
    # a kernel with the page but still has VTK, a filesystem and OS threads,
    # and none of those are here. The one place that would care,
    # mne/gui/_coreg.py, never reaches its `_kind != "notebook"` branch in the
    # browser, because _configure_dock asks the renderer for a dock and toolbar
    # API this one does not implement and fails first.
    _kind = "jupyterlite_notebook"

    def __init__(self, fig=None, size=(600, 600), bgcolor="black", **kwargs):
        # plot_alignment(fig=...) and plot_dipole_locations(fig=...) composite
        # into a scene the notebook already made, so draw into that plotter
        # rather than opening a second one and splitting the picture in two.
        if fig is not None:
            self.plotter = fig
            return
        self.plotter = pv.Plotter()
        _lite_live_plotters.append(weakref.ref(self.plotter))
        # trim after appending, so the new scene counts towards the cap and
        # _LITE_MAX_LIVE_SCENES is the number that actually stays live
        _lite_trim_live_plotters()
        self.plotter.background_color = _rgb(bgcolor)
        # even lighting, so a surface is not black when rotated
        for direction in (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ):
            self.plotter.add_light(
                pv.Light(
                    position=tuple(300.0 * coord for coord in direction),
                    focal_point=(0.0, 0.0, 0.0),
                    intensity=0.4,
                )
            )

    # -- helpers ------------------------------------------------------------
    def _glyph_template(
        self, kind, radius=None, height=None, center=None, resolution=None, **kwargs
    ):
        """Return (rr, tris) for a glyph template, oriented along +x.

        pyvista-js's Sphere/Cylinder are parametric primitives with no
        triangle list, so the templates come from MNE's own tessellation or are
        built here. ``_tile`` then stamps one of these at every position and
        merges the result, which is what keeps these cheap -- the copies share
        a single mesh and a single actor.

        Sizes follow the templates ``_pyvista.py`` hands the glyph filter, so
        the browser draws the markers at the size the rendered docs do.
        """
        if kind in ("sphere", "oct"):
            scale = 0.5 if radius is None else float(radius)
            # "oct" is an octahedron on purpose -- that is what _pyvista.py
            # hands the glyph filter, and level 1 is the unit octahedron
            # vtkPlatonicSolidSource draws. A "sphere" has to look round,
            # though: fiducials and dig points are drawn with it, so level 3
            # subdivides that octahedron onto the unit sphere at 66 vertices,
            # near the reference's 8x8 sphere (58).
            rr, tris = _tessellate_sphere(1 if kind == "oct" else 3)
            return rr * scale, tris
        if kind == "arrow":
            # vtkArrowSource, which is what _pyvista.py glyphs mode="arrow"
            # with: a cylindrical shaft carrying a cone tip, together spanning
            # 0 to 1 along +x. Its defaults, not glyph_radius, set the widths.
            # both halves are placed here rather than through the cylinder's
            # `center`, which is read in _cylinder_geom's pre-rotation frame
            shaft_rr, shaft_tris = self._glyph_template(
                "cylinder", radius=0.03, height=0.65, resolution=12
            )
            shaft_rr = shaft_rr + np.array([0.325, 0, 0])
            tip_rr, tip_tris = self._glyph_template(
                "cone", radius=0.1, height=0.35, resolution=12
            )
            tip_rr = tip_rr + np.array([0.65, 0, 0])
            return (
                np.vstack([shaft_rr, tip_rr]),
                np.vstack([shaft_tris, tip_tris + len(shaft_rr)]),
            )
        if kind == "cone":
            # apex along +x so the glyph filter's orientation applies, matching
            # pyvista.Cone(center=(0.5, 0, 0)): base at x=0, apex at x=height
            rad = 0.15 if radius is None else float(radius)
            hgt = 1.0 if height is None else float(height)
            n_side = 8 if not resolution else max(3, int(resolution) // 2)
            angles = np.linspace(0.0, 2 * np.pi, n_side, endpoint=False)
            ring = np.column_stack(
                [np.zeros(n_side), rad * np.cos(angles), rad * np.sin(angles)]
            )
            rr = np.vstack([ring, [[hgt, 0, 0]], [[0.0, 0, 0]]])
            tris = []
            for this in range(n_side):
                nxt = (this + 1) % n_side
                tris += [[this, nxt, n_side], [n_side + 1, nxt, this]]  # side, base
            return rr, np.asarray(tris, int)
        # cylinder along +x, matching _cylinder_geom's convention
        rad = 0.1 if radius is None else float(radius)
        hgt = 1.0 if height is None else float(height)
        # half the sides VTK would use: _tile stamps this template at every
        # sensor, so the side count multiplies straight into the WASM heap and
        # a 16-sided EEG cylinder is smooth enough at the size it draws
        n_side = 8 if not resolution else max(3, int(resolution) // 2)
        # _cylinder_geom builds the cylinder along y and turns it 90 degrees
        # about z to point it along x, which carries the center round with it:
        # (cx, cy, cz) lands at (-cy, cx, cz). _3d.py gives the EEG electrode
        # offset in that pre-turn frame, so turn it here too, or the cylinders
        # sit beside their sensors instead of standing on them.
        if center is None:
            offset = np.zeros(3)
        else:
            center = np.asarray(center, dtype=float)
            offset = np.array([-center[1], center[0], center[2]])
        angles = np.linspace(0.0, 2 * np.pi, n_side, endpoint=False)
        ring = np.column_stack(
            [np.zeros(n_side), rad * np.cos(angles), rad * np.sin(angles)]
        )
        back = ring + np.array([-hgt / 2.0, 0, 0])
        front = ring + np.array([hgt / 2.0, 0, 0])
        rr = (
            np.vstack([back, front, [[-hgt / 2.0, 0, 0]], [[hgt / 2.0, 0, 0]]]) + offset
        )
        tris = []
        for this in range(n_side):
            nxt = (this + 1) % n_side
            tris += [[this, nxt, n_side + nxt], [this, n_side + nxt, n_side + this]]
            tris += [[2 * n_side, nxt, this]]  # back cap
            tris += [[2 * n_side + 1, n_side + this, n_side + nxt]]  # front cap
        return rr, np.asarray(tris, int)

    def _add(self, points, tris, color, opacity=1.0):
        """Draw a mesh and return MNE's (actor, mesh) pair.

        ``opacity=None`` means "renderer default" in MNE's renderer API, which
        for PyVista reaches ``add_mesh(opacity=None)`` and draws opaque. Every
        drawing method here funnels through this, so translating it once covers
        all of them.
        """
        mesh = pv.PolyData(
            points=np.asarray(points, dtype=np.float32), faces=_vtk_faces(tris)
        )
        actor = self.plotter.add_mesh(
            mesh,
            color=_rgb(color),
            opacity=1.0 if opacity is None else float(opacity),
            smooth_shading=True,
        )
        return actor, mesh

    def _rots_from_dirs(self, dirs):
        """Rotations carrying +x onto each direction, as the glyphs assume."""
        x_axis = np.array([1.0, 0.0, 0.0])
        return np.asarray([_find_vector_rotation(x_axis, d) for d in dirs], dtype=float)

    def _tile(self, rr, tris, positions, scales=None, rots=None, axis_scales=None):
        """Stamp one template mesh at many positions as a single mesh.

        ``_pyvista.py`` hands its template to VTK's glyph filter, which bakes
        every copy into one mesh and adds it once. Doing this per position
        instead means an oct-6 source space becomes 8196 meshes and 8196
        actors, which is enough to run the browser tab out of memory.

        This is the shared step behind every glyph method here, including
        :meth:`instanced_mesh`; it takes rotation matrices because that is what
        ``quiver3d`` and ``tube`` already have, and ``instanced_mesh`` converts
        its quaternions once on the way in.
        """
        rr = np.asarray(rr, dtype=float)
        tris = np.asarray(tris, dtype=int)
        positions = np.atleast_2d(np.asarray(positions, dtype=float))[:, :3]
        n_pos = len(positions)
        points = np.repeat(rr[None, :, :], n_pos, axis=0)
        if axis_scales is not None:
            # tubes span a given length without fattening, so scale the
            # template's axis alone
            axis_scales = np.atleast_1d(np.asarray(axis_scales, dtype=float))
            points[:, :, 0] *= axis_scales[np.arange(n_pos) % len(axis_scales)][:, None]
        if scales is not None:
            scales = np.atleast_1d(np.asarray(scales, dtype=float))
            points *= scales[np.arange(n_pos) % len(scales)][:, None, None]
        if rots is not None:
            rots = np.asarray(rots, dtype=float)
            points = np.einsum(
                "nij,nkj->nki", rots[np.arange(n_pos) % len(rots)], points
            )
        points += positions[:, None, :]
        offsets = (np.arange(n_pos) * len(rr))[:, None, None]
        return (points.reshape(-1, 3), (tris[None, :, :] + offsets).reshape(-1, 3))

    # -- drawing ------------------------------------------------------------
    # Three arguments below are named rather than swallowed by **kwargs, because
    # MNE passes them on the paths the docs render and vtk.js cannot honour any
    # of them: it has no backface culling (its actor style only covers
    # representation, shading and edges), it recomputes normals itself with
    # vtkPolyDataNormals rather than taking an array, and it has no actor
    # registry to look a `name` up in later. The visible cost is that a
    # transparent surface shows its own inside.
    def mesh(
        self,
        x,
        y,
        z,
        triangles,
        color=None,
        opacity=1.0,
        backface_culling=False,
        normals=None,
        *args,
        **kwargs,
    ):
        points = np.column_stack(
            [np.asarray(x).ravel(), np.asarray(y).ravel(), np.asarray(z).ravel()]
        )
        return self._add(points, triangles, color, opacity)

    def surface(
        self,
        surface,
        color=None,
        opacity=1.0,
        backface_culling=False,
        name=None,
        *args,
        **kwargs,
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
        **kwargs,
    ):
        center = np.atleast_2d(np.asarray(center, dtype=float))
        if not len(center):
            return None, None
        # _pyvista.py glyphs a radius-0.5 sphere by `scale`, or a
        # radius-`radius` sphere by 1, so the drawn radius is half of `scale`
        rr, tris = self._glyph_template(
            "sphere", radius=0.5 * float(scale) if radius is None else float(radius)
        )
        # one template stamped at every center, which is exactly instanced_mesh
        # without orientations or per-instance colors
        return self.instanced_mesh(rr, tris, center, colors=color, opacity=opacity)

    def tube(
        self,
        origin,
        destination,
        radius=0.001,
        color=None,
        opacity=1.0,
        *args,
        **kwargs,
    ):
        origin = np.atleast_2d(np.asarray(origin, dtype=float))[:, :3]
        destination = np.atleast_2d(np.asarray(destination, dtype=float))[:, :3]
        n_seg = min(len(origin), len(destination))
        if not n_seg:
            return None, None
        vec = destination[:n_seg] - origin[:n_seg]
        length = np.linalg.norm(vec, axis=1)
        keep = length > 0
        if not keep.any():
            return None, None
        vec, length = vec[keep], length[keep]
        centers = (origin[:n_seg][keep] + destination[:n_seg][keep]) / 2.0
        # one unit-height template stretched to each segment, merged into a
        # single mesh rather than a cylinder primitive per segment. This cannot
        # go through instanced_mesh: the stretch is along the template's axis
        # only, and instanced_mesh scales every instance isotropically.
        rr, tris = self._glyph_template(
            "cylinder", radius=float(radius), height=1.0, resolution=20
        )  # 20 is _PyVistaRenderer.tube_n_sides, halved on the way in
        points, faces = self._tile(
            rr,
            tris,
            centers,
            rots=self._rots_from_dirs(vec / length[:, None]),
            axis_scales=length,
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
        color=None,
        scale=1.0,
        mode="arrow",
        opacity=1.0,
        *,
        scale_mode="none",
        scalars=None,
        glyph_height=None,
        glyph_center=None,
        glyph_resolution=None,
        glyph_radius=0.15,
        solid_transform=None,
        backface_culling=False,
        **kwargs,
    ):
        """Draw one merged glyph mesh, the way the glyph filter would.

        ``_pyvista.py`` builds a template, lets VTK's glyph filter bake a copy
        at every point into one mesh, and adds that once. Drawing a primitive
        per point instead is what made ``20_source_alignment`` -- an oct-6
        source space, so 8196 glyphs, twice -- exhaust the browser tab.
        """
        x, y, z = (np.atleast_1d(np.asarray(q, dtype=float)) for q in (x, y, z))
        centers = np.column_stack([x, y, z])
        n_pos = len(centers)
        if not n_pos:
            return None, None
        factor = float(np.asarray(scale).ravel()[0]) if np.size(scale) else 1.0
        idx = np.arange(n_pos)
        u, v, w = (np.atleast_1d(np.asarray(q, dtype=float)) for q in (u, v, w))
        dirs = np.column_stack([u[idx % len(u)], v[idx % len(v)], w[idx % len(w)]])
        norms = np.linalg.norm(dirs, axis=1)
        flat = norms == 0
        dirs[flat] = (1.0, 0.0, 0.0)
        dirs = dirs / np.where(flat, 1.0, norms)[:, None]
        # per-glyph size, matching what _pyvista.py hands the glyph filter: it
        # sends "arrow" through _glyph, whose own default scales by the scalars,
        # and "2darrow" through _arrow_glyph, which scales by the vector; every
        # other mode gets whatever scale_mode asks for. plot_alignment's
        # show_axes draws its three axis arrows at 1/3, 2/3 and full length this
        # way, so ignoring it would make the coordinate frames wrong.
        _check_option("scale_mode", scale_mode, ("none", "scalar", "vector"))
        if mode == "arrow":
            scale_mode = "scalar"
        elif mode == "2darrow":
            scale_mode = "vector"
        if scale_mode == "scalar":
            values = (
                np.ones(n_pos)
                if scalars is None
                else np.atleast_1d(np.asarray(scalars, dtype=float)).ravel()
            )
            sizes = factor * values[idx % len(values)]
        elif scale_mode == "vector":
            sizes = factor * norms
        else:
            sizes = factor
        # the same templates _pyvista.py feeds the filter; `scale` then plays
        # the part its `factor` does
        if mode == "oct":
            # vtkPlatonicSolidSource puts its octahedron on the unit
            # circumsphere, and the MRI fiducials get their real size from
            # solid_transform (mri_fid_scale, 5 mm) rather than from `scale`
            kind, template_kw = "oct", dict(radius=1.0)
        elif mode == "sphere":
            kind, template_kw = "sphere", dict(radius=0.5)
        elif mode == "cylinder":
            kind = "cylinder"
            template_kw = dict(
                radius=glyph_radius,
                height=glyph_height,
                center=glyph_center,
                resolution=glyph_resolution,
            )
        elif mode == "cone":
            kind = "cone"
            template_kw = dict(
                radius=glyph_radius, height=glyph_height, resolution=glyph_resolution
            )
        else:  # arrow / 2darrow, both of which vtk draws with a shaft and a tip
            kind, template_kw = "arrow", dict()
        rr, tris = self._glyph_template(kind, **template_kw)
        if solid_transform is not None:
            # _pyvista.py transforms the template before glyphing, and this is
            # where the fiducial markers get their size and 45 deg roll
            solid_transform = np.asarray(solid_transform, dtype=float)
            rr = rr @ solid_transform[:3, :3].T + solid_transform[:3, 3]
        rots = None if mode in ("sphere", "oct") else self._rots_from_dirs(dirs)
        points, faces = self._tile(rr, tris, centers, scales=sizes, rots=rots)
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
        name=None,
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
        positions = np.atleast_2d(np.asarray(positions, dtype=float))[:, :3]
        n_pos = len(positions)
        if not n_pos:
            return None, None
        rots = None
        if quats is not None:
            rots = np.asarray(
                quat_to_rot(np.atleast_2d(np.asarray(quats, dtype=float))), dtype=float
            )
        idx = np.arange(n_pos)
        if colors is not None and np.ndim(colors) > 1:
            colors = np.asarray(colors)
            uniq, inverse = np.unique(
                colors[idx % len(colors)], axis=0, return_inverse=True
            )
            inverse = np.asarray(inverse).ravel()
            groups = [(uniq[k], idx[inverse == k]) for k in range(len(uniq))]
        else:
            groups = [(colors, idx)]
        out = (None, None)
        for color, sel in groups:
            group_scales = None
            if scales is not None:
                group_scales = np.atleast_1d(np.asarray(scales, dtype=float))
                group_scales = group_scales[sel % len(group_scales)]
            group_rots = None if rots is None else rots[sel % len(rots)]
            points, faces = self._tile(
                rr, tris, positions[sel], scales=group_scales, rots=group_rots
            )
            out = self._add(points, faces, color, opacity)
        return out

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
        """Draw text over the scene, in normalized window coordinates."""
        if justification is not None or font_file is not None:
            raise NotImplementedError(
                "Justified text and custom fonts are not supported in the "
                "browser: vtk.js draws text at a point, in the page's font."
            )
        return _lite_add_text(self.plotter, text, (x_window, y_window), size, color)

    # -- nothing to do in a browser -----------------------------------------
    # These are reached while drawing the figures the docs render, and there is
    # genuinely nothing for them to do here, so each says why rather than
    # raising and taking a working page down with it.
    def set_interaction(self, *args, **kwargs):
        # plot_alignment sets this unconditionally (mne/viz/_3d.py); vtk.js
        # ships one trackball style and no way to swap it
        return None

    def _update(self, *args, **kwargs):
        # plot_alignment calls this to force a repaint of a window already up;
        # the browser paints from the JS side, after the cell has finished
        return None

    def _window_close_connect(self, *args, **kwargs):
        # mne/viz/ui_events.py asks to be told when the window closes, and a
        # canvas in an output cell has no close event to connect to
        return None

    def text3d(self, *args, **kwargs):
        # plot_alignment(show_channel_names=True) labels each sensor, which
        # needs a follow-the-camera 3D text actor. pyvista-js 0.15 has only
        # Text, positioned in normalized window coordinates, and projecting the
        # sensor positions into those is exactly what `project` cannot do, so
        # the sensors are drawn without their labels.
        return None

    def close(self):
        _lite_release_plotter(self.plotter)
        return None

    # -- things pyvista-js cannot do ----------------------------------------
    def contour(self, *args, **kwargs):
        # PolyData.contour can extract the isolines -- it marches triangles in
        # JS at render time -- but with no lookup table they would all draw in
        # one color, and plot_evoked_field asks for a single set spanning
        # -vmax to +vmax. Drawing a field map whose positive and negative lines
        # look identical is worse than not drawing it.
        raise NotImplementedError(
            "Drawing contours is not supported in the browser: every line would "
            "come out the same color, which for a field map is misleading."
        )

    def scalarbar(self, *args, **kwargs):
        # Plotter.add_scalar_bar records the request on the Python side, but
        # nothing reaches the scene: the vtk.js template draws no scalar bar,
        # and there is no colormap for one to label anyway
        raise NotImplementedError(
            "Drawing a scalar bar is not supported in the browser: vtk.js draws "
            "no scalar bar and nothing here is colored by scalars."
        )

    def legend(self, *args, **kwargs):
        # only mne coreg asks for one, and pyvista-js has no add_legend
        raise NotImplementedError(
            "Drawing a legend is not supported in the browser: pyvista-js does "
            "not have one."
        )

    def subplot(self, *args, **kwargs):
        # a pyvista-js Plotter renders a single vtk.js view into one canvas
        raise NotImplementedError(
            "Subplots are not supported in the browser: a scene is one canvas."
        )

    def remove_mesh(self, *args, **kwargs):
        # add_mesh hands back a dict describing the actor rather than a handle
        # the plotter can look up again
        raise NotImplementedError(
            "Removing a mesh is not supported in the browser: pyvista-js draws "
            "the scene as a whole and cannot take one mesh back out of it."
        )

    def _process_events(self, *args, **kwargs):
        # the kernel and the canvas are different threads here, and the Pyodide
        # worker cannot drive the page's event loop
        raise NotImplementedError(
            "Draining the event loop is not supported in the browser: the page "
            "runs the loop, not the kernel."
        )

    def _window_set_cursor(self, *args, **kwargs):
        # pyvista-js has no window object to set a cursor on
        raise NotImplementedError(
            "Setting the cursor is not supported in the browser: the scene is a "
            "canvas in an output cell, not a window."
        )

    def _enable_time_interaction(self, *args, **kwargs):
        # inheriting renderer._TimeInteraction would not help: it builds the
        # slider out of _dock_add_slider and the rest of the dock and toolbar
        # API, none of which is implemented here. The _Ipy* mixins that do
        # implement it live in _notebook.py, which imports _pyvista at module
        # level and so cannot be imported at all without VTK.
        raise NotImplementedError(
            "The time slider is not supported in the browser: it needs dock "
            "widgets, which this backend does not draw."
        )

    def project(self, xyz, ch_names):
        # a _Projection needs the render window's coordinate transform, and
        # pyvista-js does not expose a render window
        raise NotImplementedError(
            "Projecting 3D positions onto the scene is not supported in the browser."
        )

    def screenshot(self, mode="rgb", filename=None, **kwargs):
        # Plotter.screenshot does exist, but it renders the scene by driving a
        # headless browser with Playwright from outside the page, which is not
        # something the page can do to itself
        raise NotImplementedError(
            "Taking a screenshot is not supported in the browser: vtk.js draws "
            "to a live canvas that cannot be read back as an array."
        )

    # -- camera and scene ---------------------------------------------------
    def get_camera(self, *args, **kwargs):
        # Same order as _get_3d_view: roll, distance, azimuth, elevation and
        # then the focalpoint. Brain unpacks positions 3 and 4 as the angles,
        # so the focalpoint has to be the last element rather than the fourth.
        return _lite_get_view(self.plotter)

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
        # distance, focalpoint and roll go unused: vtk.js frames the scene with
        # resetCamera() on this path, which is what lets it handle the
        # distance=None that MNE asks for far more often. See _lite_get_view.
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
        self.plotter.show()
        return None


# -- the module surface renderer.py expects of a 3D backend -----------------
# set_3d_view, set_3d_title and the close_* helpers reach for these on the
# ``renderer.backend`` global rather than going through _get_renderer, and the
# figure they hand over is the pyvista-js plotter _LiteRenderer.scene returns.
_Renderer = _LiteRenderer

# nothing here draws differently under test, the way an on-screen window does
_testing_context = nullcontext


def _set_3d_view(
    figure, azimuth=None, elevation=None, focalpoint=None, distance=None, roll=None
):
    return _lite_set_view(figure, azimuth, elevation)


def _set_3d_title(figure, title, size=16, *, color="white", position="upper_left"):
    if isinstance(position, str):
        _check_option("position", position, sorted(_TITLE_POSITIONS))
        position = _TITLE_POSITIONS[position]
    return _lite_add_text(figure, title, position, size, color)


def _clear_3d_figure(figure):
    # close=False is already the "give the geometry back but keep the scene"
    # path, which is what clearing means
    _lite_release_plotter(figure, close=False)
    return None


def _close_3d_figure(figure):
    _lite_release_plotter(figure)
    return None


def _close_all():
    # the registry holds weak references, so deref before releasing -- handing
    # the ref itself to _lite_release_plotter matches nothing and never
    # shortens the list
    while _lite_live_plotters:
        plotter = _lite_live_plotters[-1]()
        if plotter is None:
            _lite_live_plotters.pop()
        else:
            _lite_release_plotter(plotter)
    return None
