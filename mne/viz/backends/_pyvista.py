"""
Core visualization operations based on PyVista.

Actual implementation of _Renderer and _Projection classes.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import platform
import re
import warnings
from contextlib import contextmanager
from inspect import signature

import numpy as np
import pyvista
from pyvista import Line, Plotter, PolyData, close_all
from pyvista.plotting.plotter import _ALL_PLOTTERS
from pyvistaqt import BackgroundPlotter
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import VTK_UNSIGNED_CHAR, vtkCommand, vtkLookupTable
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import vtkGlyph3D
from vtkmodules.vtkFiltersGeneral import vtkMarchingContourFilter
from vtkmodules.vtkFiltersHybrid import vtkPolyDataSilhouette
from vtkmodules.vtkFiltersSources import (
    vtkArrowSource,
    vtkCylinderSource,
    vtkGlyphSource2D,
)
from vtkmodules.vtkImagingCore import vtkImageReslice
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCellPicker,
    vtkColorTransferFunction,
    vtkCoordinate,
    vtkDataSetMapper,
    vtkGlyph3DMapper,
    vtkMapper,
    vtkPolyDataMapper,
    vtkVolume,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper

from ...fixes import _compare_version
from ...surface import _vtk_smooth
from ...transforms import _cart_to_sph, _sph_to_cart, apply_trans
from ...utils import (
    _check_option,
    _require_version,
    _validate_type,
    warn,
)
from ._abstract import Figure3D, _AbstractRenderer
from ._utils import (
    ALLOWED_QUIVER_MODES,
    _alpha_blend_background,
    _get_colormap_from_array,
    _init_mne_qtapp,
)

try:
    from vtkmodules.vtkFiltersGeneral import vtkTransformFilter
except ImportError:  # TODO VERSION VTK 9.7+
    from vtkmodules.vtkFiltersGeneral import (
        vtkTransformPolyDataFilter as vtkTransformFilter,
    )

_FIGURES = dict()


class PyVistaFigure(Figure3D):
    """PyVista-based 3D Figure.

    .. note:: This class should not be instantiated directly via
              ``mne.viz.PyVistaFigure(...)``. Instead, use
              :func:`mne.viz.create_3d_figure`.

    See Also
    --------
    mne.viz.create_3d_figure
    """

    def __init__(self):
        pass

    def _init(
        self,
        plotter=None,
        show=False,
        title="MNE-Python 3D Figure",
        size=(600, 600),
        shape=(1, 1),
        background_color="black",
        smooth_shading=True,
        off_screen=False,
        notebook=False,
        splash=False,
    ):
        self._plotter = plotter
        self.display = None
        self.background_color = background_color
        self.smooth_shading = smooth_shading
        self.notebook = notebook
        self.title = title
        self.splash = splash

        self.store = dict()
        self.store["window_size"] = size
        self.store["shape"] = shape
        self.store["off_screen"] = off_screen
        self.store["border"] = False
        self.store["line_smoothing"] = True
        self.store["polygon_smoothing"] = True
        self.store["point_smoothing"] = True

        if not self.notebook:
            self.store["show"] = show
            self.store["title"] = title
            self.store["auto_update"] = False
            self.store["menu_bar"] = False
            self.store["toolbar"] = False
            self.store["update_app_icon"] = False
            self._plotter_class = _SafeBackgroundPlotter
            if "app_window_class" in signature(BackgroundPlotter).parameters:
                from ._qt import _MNEMainWindow

                self.store["app_window_class"] = _MNEMainWindow
        else:
            self._plotter_class = Plotter

        self._nrows, self._ncols = self.store["shape"]

    def _build(self):
        if self.plotter is None:
            if not self.notebook:
                out = _init_mne_qtapp(enable_icon=True, splash=self.splash)
                # replace it with the Qt object
                if self.splash:
                    self.splash = out[1]
                    app = out[0]
                else:
                    app = out
                self.store["app"] = app
            plotter = self._plotter_class(**self.store)
            plotter.background_color = self.background_color
            self._plotter = plotter
        # TODO: This breaks trame "client" backend
        if self.plotter.iren is not None:
            self.plotter.iren.initialize()
        _process_events(self.plotter)
        _process_events(self.plotter)
        return self.plotter

    def _is_active(self):
        return hasattr(self.plotter, "ren_win")


class _Projection:
    """Class storing projection information.

    Attributes
    ----------
    xy : array
        Result of 2d projection of 3d data.
    pts : None
        Scene sensors handle.
    """

    def __init__(self, *, xy, pts, plotter):
        """Store input projection information into attributes."""
        self.xy = xy
        self.pts = pts
        self.plotter = plotter

    def visible(self, state):
        """Modify visibility attribute of the sensors."""
        self.pts.SetVisibility(state)
        self.plotter.render()


class _PyVistaRenderer(_AbstractRenderer):
    """Class managing rendering scene.

    Attributes
    ----------
    plotter: Plotter
        Main PyVista access point.
    name: str
        Name of the window.
    """

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
        from .._3d import _get_3d_option

        # TODO VERSION change whenever PyVista min gets updated:
        _require_version("pyvista", "use 3D rendering", "0.43")
        multi_samples = _get_3d_option("multi_samples")
        # multi_samples > 1 is broken on macOS + Intel Iris + volume rendering
        if platform.system() == "Darwin":
            multi_samples = 1
        figure = PyVistaFigure()
        figure._init(
            show=show,
            title=name,
            size=size,
            shape=shape,
            background_color=bgcolor,
            notebook=notebook,
            smooth_shading=smooth_shading,
            splash=splash,
        )
        self.font_family = "arial"
        self.tube_n_sides = 20
        self.antialias = _get_3d_option("antialias")
        self.depth_peeling = _get_3d_option("depth_peeling")
        self.multi_samples = multi_samples
        self.smooth_shading = smooth_shading
        if isinstance(fig, int):
            saved_fig = _FIGURES.get(fig)
            # Restore only active plotter
            if saved_fig is not None and saved_fig._is_active():
                self.figure = saved_fig
            else:
                self.figure = figure
                _FIGURES[fig] = self.figure
        elif fig is None:
            self.figure = figure
        else:
            self.figure = fig

        # Enable off_screen if sphinx-gallery or testing
        if pyvista.OFF_SCREEN:
            self.figure.store["off_screen"] = True

        # pyvista theme may enable depth peeling by default so
        # we disable it initially to better control the value afterwards
        with _disabled_depth_peeling():
            self.plotter = self.figure._build()
        self._hide_axes()
        self._toggle_antialias()
        self._enable_depth_peeling()
        self._picker = vtkCellPicker()

        # FIX: https://github.com/pyvista/pyvistaqt/pull/68
        if not hasattr(self.plotter, "iren"):
            self.plotter.iren = None

        self.update_lighting()

    @property
    def _all_plotters(self):
        if self.figure.plotter is not None:
            return [self.figure.plotter]
        else:
            return list()

    @property
    def _all_renderers(self):
        if self.figure.plotter is not None:
            return self.figure.plotter.renderers
        else:
            return list()

    def _hide_axes(self):
        for renderer in self._all_renderers:
            renderer.hide_axes()

    def _update(self):
        for plotter in self._all_plotters:
            plotter.update()

    def _index_to_loc(self, idx):
        _ncols = self.figure._ncols
        row = idx // _ncols
        col = idx % _ncols
        return (row, col)

    def _loc_to_index(self, loc):
        _ncols = self.figure._ncols
        return loc[0] * _ncols + loc[1]

    def subplot(self, x, y):
        x = np.max([0, np.min([x, self.figure._nrows - 1])])
        y = np.max([0, np.min([y, self.figure._ncols - 1])])
        self.plotter.subplot(x, y)

    def scene(self):
        return self.figure

    def update_lighting(self):
        # Inspired from Mayavi's version of Raymond Maple 3-lights illumination
        # below and centered, left and above, right and above
        az_el_in = ((0, -45, 0.7), (-60, 30, 0.7), (60, 30, 0.7))
        for renderer in self._all_renderers:
            renderer.remove_all_lights()
            for azimuth, elevation, intensity in az_el_in:
                light = pyvista.Light(
                    position=_to_pos(azimuth, elevation),
                    color="white",
                    light_type="camera light",
                    intensity=intensity,
                )
                renderer.add_light(light)

    def set_interaction(self, interaction):
        if not hasattr(self.plotter, "iren") or self.plotter.iren is None:
            return
        if interaction == "rubber_band_2d":
            for renderer in self._all_renderers:
                renderer.enable_parallel_projection()
            self.plotter.enable_rubber_band_2d_style()
        else:
            for renderer in self._all_renderers:
                renderer.disable_parallel_projection()
            kwargs = dict()
            if interaction == "terrain":
                kwargs["mouse_wheel_zooms"] = True
            getattr(self.plotter, f"enable_{interaction}_style")(**kwargs)

    def legend(self, labels, size=0.1, face="triangle", loc="upper left"):
        return self.plotter.add_legend(labels, size=(size, size), face=face, loc=loc)

    def polydata(
        self,
        mesh,
        color=None,
        opacity=1.0,
        normals=None,
        backface_culling=False,
        scalars=None,
        colormap=None,
        vmin=None,
        vmax=None,
        interpolate_before_map=True,
        representation="surface",
        line_width=1.0,
        *,
        name=None,
        **kwargs,
    ):
        from matplotlib.colors import to_rgba_array

        rgba = False
        if color is not None:
            # See if we need to convert or not
            check_color = to_rgba_array(color)
            if len(check_color) == mesh.n_points:
                scalars = (check_color * 255).astype("ubyte")
                color = None
                rgba = True
        if isinstance(colormap, np.ndarray):
            if colormap.dtype == np.uint8:
                colormap = colormap.astype(np.float64) / 255.0
            from matplotlib.colors import ListedColormap

            colormap = ListedColormap(colormap)
        if normals is not None:
            mesh.point_data["Normals"] = normals
            mesh.GetPointData().SetActiveNormals("Normals")
        else:
            _compute_normals(mesh)
        smooth_shading = self.smooth_shading
        if representation == "wireframe":
            smooth_shading = False  # never use smooth shading for wf
        rgba = kwargs.pop("rgba", rgba)
        actor = _add_mesh(
            plotter=self.plotter,
            mesh=mesh,
            name=name,
            color=color,
            scalars=scalars,
            edge_color=color,
            opacity=opacity,
            cmap=colormap,
            backface_culling=backface_culling,
            rng=[vmin, vmax],
            show_scalar_bar=False,
            rgba=rgba,
            smooth_shading=smooth_shading,
            interpolate_before_map=interpolate_before_map,
            style=representation,
            line_width=line_width,
            **kwargs,
        )

        return actor, mesh

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
        vertices = np.c_[x, y, z].astype(float)
        triangles = np.c_[np.full(len(triangles), 3), triangles]
        mesh = PolyData(vertices, triangles)
        return self.polydata(
            mesh=mesh,
            color=color,
            opacity=opacity,
            normals=normals,
            backface_culling=backface_culling,
            scalars=scalars,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
            interpolate_before_map=interpolate_before_map,
            representation=representation,
            line_width=line_width,
            name=name,
            **kwargs,
        )

    def contour(
        self,
        surface,
        scalars,
        contours,
        width=1.0,
        opacity=1.0,
        vmin=None,
        vmax=None,
        colormap=None,
        normalized_colormap=False,
        kind="line",
        color=None,
    ):
        if colormap is not None:
            colormap = _get_colormap_from_array(colormap, normalized_colormap)
        vertices = np.array(surface["rr"])
        triangles = np.array(surface["tris"])
        n_triangles = len(triangles)
        triangles = np.c_[np.full(n_triangles, 3), triangles]
        mesh = PolyData(vertices, triangles)
        mesh.point_data["scalars"] = scalars
        contour = mesh.contour(isosurfaces=contours)
        line_width = width
        if kind == "tube":
            contour = contour.tube(radius=width, n_sides=self.tube_n_sides)
            line_width = 1.0
        actor = _add_mesh(
            plotter=self.plotter,
            mesh=contour,
            show_scalar_bar=False,
            line_width=line_width,
            color=color,
            rng=[vmin, vmax],
            cmap=colormap,
            opacity=opacity,
            smooth_shading=self.smooth_shading,
        )
        return actor, contour

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
        normals = surface.get("nn", None)
        vertices = np.array(surface["rr"])
        triangles = np.array(surface["tris"])
        triangles = np.c_[np.full(len(triangles), 3), triangles]
        mesh = PolyData(vertices, triangles)
        colormap = _get_colormap_from_array(colormap, normalized_colormap)
        if scalars is not None:
            mesh.point_data["scalars"] = scalars
        return self.polydata(
            mesh=mesh,
            color=color,
            opacity=opacity,
            normals=normals,
            backface_culling=backface_culling,
            scalars=scalars,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
            name=name,
        )

    def sphere(
        self,
        center,
        color,
        scale,
        opacity=1.0,
        resolution=8,
        backface_culling=False,
        radius=None,
    ):
        factor = 1.0 if radius is not None else scale
        center = np.array(center, dtype=float)
        if len(center) == 0:
            return None, None
        _check_option("center.ndim", center.ndim, (1, 2))
        _check_option("center.shape[-1]", center.shape[-1], (3,))
        geom = pyvista.Sphere(
            radius=0.5 if radius is None else radius,
            theta_resolution=resolution,
            phi_resolution=resolution,
        )
        mesh = PolyData(center)
        glyph = mesh.glyph(orient=False, scale=False, factor=factor, geom=geom)
        actor = _add_mesh(
            self.plotter,
            mesh=glyph,
            color=color,
            opacity=opacity,
            backface_culling=backface_culling,
            smooth_shading=self.smooth_shading,
        )
        return actor, glyph

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
        cmap = _get_colormap_from_array(colormap, normalized_colormap)
        for pointa, pointb in zip(origin, destination):
            line = Line(pointa, pointb)
            if scalars is not None:
                line.point_data["scalars"] = scalars[0, :]
                scalars = "scalars"
                color = None
            else:
                scalars = None
            tube = line.tube(radius=radius, n_sides=self.tube_n_sides)
            actor = _add_mesh(
                plotter=self.plotter,
                mesh=tube,
                scalars=scalars,
                flip_scalars=reverse_lut,
                rng=[vmin, vmax],
                color=color,
                show_scalar_bar=False,
                cmap=cmap,
                smooth_shading=self.smooth_shading,
                opacity=opacity,
            )
        return actor, tube

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
        _validate_type(scale_mode, str, "scale_mode")
        scale_map = dict(none=False, scalar="scalars", vector="vec")
        _check_option("scale_mode", scale_mode, list(scale_map))
        factor = scale
        vectors = np.c_[u, v, w]
        points = np.vstack(np.c_[x, y, z]).astype(float)
        n_points = len(points)
        grid = PolyData(points)
        if scalars is None:
            scalars = np.ones((n_points,))
            mesh_scalars = None
        else:
            mesh_scalars = "scalars"
        grid.point_data["scalars"] = np.array(scalars, float)
        grid.point_data["vec"] = vectors
        if mode == "2darrow":
            return _arrow_glyph(grid, factor), grid
        elif mode == "arrow":
            alg = _glyph(grid, orient="vec", scalars="scalars", factor=factor)
            mesh = pyvista.wrap(alg.GetOutput())
        else:
            if mode == "cone":
                geom = pyvista.Cone(center=(0.5, 0, 0), radius=glyph_radius)
            elif mode == "cylinder":
                geom = _cylinder_geom(
                    radius=glyph_radius,
                    height=glyph_height,
                    center=glyph_center,
                    resolution=glyph_resolution,
                )
            elif mode == "oct":
                geom = pyvista.PlatonicSolid(kind="octahedron")
                if solid_transform is not None:
                    assert solid_transform.shape == (4, 4)
                    geom = geom.transform(
                        solid_transform.astype(np.float64), inplace=True
                    )
            else:
                assert mode == "sphere", mode  # guaranteed above
                geom = pyvista.Sphere(theta_resolution=8, phi_resolution=8)
            mesh = grid.glyph(
                orient="vec",
                scale=scale_map[scale_mode],
                factor=factor,
                geom=geom,
            )
        actor = _add_mesh(
            self.plotter,
            mesh=mesh,
            color=color,
            opacity=opacity,
            scalars=mesh_scalars if colormap is not None else None,
            colormap=colormap,
            show_scalar_bar=False,
            backface_culling=backface_culling,
            clim=clim,
        )
        return actor, mesh

    # quiver3d (above) and instanced_mesh (below) split along principled lines:
    # quiver3d bakes a static, merged glyph mesh with direction-vector
    # orientation and scalar/colormap coloring (arrows and friends), while
    # instanced_mesh GPU-instances one template with per-instance, updatable
    # quaternion orientation and RGBA coloring (sensors, MEG coils).
    def instanced_mesh(
        self,
        rr,
        tris,
        positions,
        quats,
        colors,
        scales=None,
        opacity=1.0,
        backface_culling=False,
        *,
        name=None,
    ):
        faces = np.c_[np.full(len(tris), 3), tris]
        geom = PolyData(np.asarray(rr, float), faces)
        _compute_normals(geom)

        cloud = PolyData(np.asarray(positions, float).copy())
        cloud.point_data["orientation"] = _quat_to_vtk_wxyz(np.asarray(quats, float))
        cloud.point_data["colors"] = (np.asarray(colors, float) * 255).astype(np.uint8)

        mapper = vtkGlyph3DMapper()
        mapper.SetInputData(cloud)
        mapper.SetSourceData(geom)
        mapper.SetOrientationArray("orientation")
        mapper.SetOrientationModeToQuaternion()
        if scales is None:
            # size is baked into rr (e.g. MEG coils); no per-instance scaling
            mapper.ScalingOff()
        else:
            cloud.point_data["scale"] = np.asarray(scales, float)
            mapper.SetScaleArray("scale")
            mapper.SetScaleModeToScaleByMagnitude()
            mapper.SetScaleFactor(1.0)
            mapper.ScalingOn()
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray("colors")
        mapper.SetColorModeToDirectScalars()

        actor = self._actor(mapper)
        prop = actor.GetProperty()
        prop.SetOpacity(opacity)
        prop.SetBackfaceCulling(backface_culling)
        if self.smooth_shading and "Normals" in geom.point_data:
            prop.SetInterpolationToPhong()
        self.plotter.add_actor(
            actor, name=name, render=False, reset_camera=False, pickable=True
        )
        return actor, cloud

    def _glyph_template(self, kind, **kwargs):
        """Return (rr, tris) for a standard template mesh for instanced_mesh.

        ``kind`` is ``"sphere"`` (unit-diameter, i.e. radius 0.5) or
        ``"cylinder"`` (see ``_cylinder_geom`` for ``**kwargs``). The template
        is oriented along +x so per-instance quaternions can point it anywhere.
        """
        if kind == "sphere":
            geom = pyvista.Sphere(radius=0.5, theta_resolution=8, phi_resolution=8)
        else:
            assert kind == "cylinder", kind
            geom = pyvista.wrap(_cylinder_geom(**kwargs))
        geom = geom.triangulate()
        rr = np.asarray(geom.points, float)
        tris = np.asarray(geom.faces).reshape(-1, 4)[:, 1:]
        return rr, tris

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
        size = 14 if size is None else size
        position = (x_window, y_window)
        actor = self.plotter.add_text(
            text=text,
            position=position,
            font_size=size,
            color=color,
            viewport=True,
            font_file=font_file,
        )
        if isinstance(justification, str):
            if justification == "left":
                actor.GetTextProperty().SetJustificationToLeft()
            elif justification == "center":
                actor.GetTextProperty().SetJustificationToCentered()
            elif justification == "right":
                actor.GetTextProperty().SetJustificationToRight()
            else:
                raise ValueError(
                    "Expected values for `justification` are `left`, `center` or "
                    f"`right` but got {justification} instead."
                )
        _hide_testing_actor(actor)
        return actor

    def text3d(self, x, y, z, text, scale, color="white"):
        actor = self.plotter.add_point_labels(
            points=np.array([x, y, z]).astype(float),
            labels=[text],
            point_size=scale,
            text_color=color,
            font_family=self.font_family,
            name=text,
            shape_opacity=0,
            always_visible=True,
        )
        _hide_testing_actor(actor)
        return actor

    def scalarbar(
        self,
        source,
        color="white",
        title=None,
        n_labels=4,
        bgcolor=None,
        **extra_kwargs,
    ):
        if isinstance(source, vtkMapper):
            mapper = source
        elif isinstance(source, vtkActor):
            mapper = source.GetMapper()
        else:
            mapper = None
        kwargs = dict(
            color=color,
            title=_truncate_scalar_bar_title(title),
            n_labels=n_labels,
            use_opacity=False,
            n_colors=256,
            position_x=0.15,
            position_y=0.05,
            width=0.7,
            height=0.10,
            shadow=False,
            bold=False,
            label_font_size=16,
            font_family=self.font_family,
            background_color=bgcolor,
            mapper=mapper,
        )
        kwargs.update(extra_kwargs)
        actor = self.plotter.add_scalar_bar(**kwargs)
        actor.SetTextPad(10)
        _hide_testing_actor(actor)
        tick_actor = self._add_scalarbar_ticks(actor, kwargs["n_labels"])
        return actor, tick_actor

    def _add_scalarbar_ticks(self, bar_actor, n_labels):
        from vtkmodules.vtkRenderingAnnotation import vtkAxisActor2D

        axis = vtkAxisActor2D()
        axis.GetPositionCoordinate().SetCoordinateSystemToDisplay()
        axis.GetPosition2Coordinate().SetCoordinateSystemToDisplay()
        axis.SetNumberOfLabels(n_labels)
        # otherwise VTK rounds the tick count to "nice" values, desyncing the
        # marks from the scalar bar's own label positions
        axis.SetAdjustLabels(False)
        axis.SetTickLength(5)
        axis.SetLabelVisibility(False)
        axis.SetTitleVisibility(False)
        axis.SetAxisVisibility(False)  # only the tick marks, no connecting line
        axis.SetTickVisibility(True)
        axis.GetProperty().SetColor(*bar_actor.GetLabelTextProperty().GetColor())

        def reposition(caller, event):
            self.reposition_scalarbar_ticks(bar_actor, axis)

        self.plotter.iren.add_observer(vtkCommand.RenderEvent, reposition)
        self.plotter.renderer.AddActor(axis)
        _hide_testing_actor(axis)
        return axis

    def set_scalarbar_title(self, bar_actor, title):
        bar_actor.SetTitle(_truncate_scalar_bar_title(title))

    def reposition_scalarbar_ticks(self, bar_actor, tick_actor):
        rect = [0, 0, 0, 0]
        bar_actor.GetScalarBarRect(rect, self.plotter.renderer)
        x0, y0, width, height = rect
        horizontal = bar_actor.GetOrientation() == 0
        inset_low, inset_high = 4, 22
        if horizontal:
            tick_actor.GetPositionCoordinate().SetValue(x0 + inset_low, y0 + height)
            tick_actor.GetPosition2Coordinate().SetValue(
                x0 + width - inset_high, y0 + height
            )
        else:
            tick_actor.GetPositionCoordinate().SetValue(x0 + width, y0 + inset_low)
            tick_actor.GetPosition2Coordinate().SetValue(
                x0 + width, y0 + height - inset_high
            )

    def show(self):
        self.plotter.show()

    def close(self):
        _close_3d_figure(figure=self.figure)

    def get_camera(self, *, rigid=None):
        return _get_3d_view(self.figure, rigid=rigid)

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
        _set_3d_view(
            self.figure,
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            focalpoint=focalpoint,
            roll=roll,
            rigid=rigid,
            update=update,
        )

    def screenshot(self, mode="rgb", filename=None):
        return _take_3d_screenshot(figure=self.figure, mode=mode, filename=filename)

    def project(self, xyz, ch_names):
        xy = _3d_to_2d(self.plotter, xyz)
        xy = dict(zip(ch_names, xy))
        # pts = self.fig.children[-1]
        pts = self.plotter.renderer.GetActors().GetLastItem()

        return _Projection(xy=xy, pts=pts, plotter=self.plotter)

    def _enable_depth_peeling(self):
        for plotter in self._all_plotters:
            if self.depth_peeling:
                plotter.enable_depth_peeling()
            else:
                plotter.disable_depth_peeling()

    def _toggle_antialias(self):
        """Enable it everywhere except on systems with problematic OpenGL."""
        # MESA can't seem to handle MSAA and depth peeling simultaneously, see
        # https://github.com/pyvista/pyvista/issues/4867
        bad_system = _is_osmesa(self.plotter)
        for plotter in self._all_plotters:
            if bad_system or not self.antialias:
                plotter.disable_anti_aliasing()
            else:
                if not bad_system:
                    plotter.enable_anti_aliasing(
                        aa_type="msaa",
                        multi_samples=self.multi_samples,
                    )

    def remove_mesh(self, mesh_data):
        actor, _ = mesh_data
        self.plotter.remove_actor(actor)

    @contextmanager
    def _disabled_interaction(self):
        if not self.plotter.renderer.GetInteractive():
            yield
        else:
            self.plotter.disable()
            try:
                yield
            finally:
                self.plotter.enable()

    def _actor(self, mapper=None):
        actor = vtkActor()
        if mapper is not None:
            actor.SetMapper(mapper)
        _hide_testing_actor(actor)
        return actor

    def _process_events(self):
        for plotter in self._all_plotters:
            _process_events(plotter)

    def _update_picking_callback(
        self, on_mouse_move, on_button_press, on_button_release, on_pick
    ):
        add_obs = self.plotter.iren.add_observer
        add_obs(vtkCommand.RenderEvent, on_mouse_move)
        add_obs(vtkCommand.LeftButtonPressEvent, on_button_press)
        add_obs(vtkCommand.EndInteractionEvent, on_button_release)
        self._picker.AddObserver(vtkCommand.EndPickEvent, on_pick)
        self._picker.SetVolumeOpacityIsovalue(0.0)

    def _set_colormap_range(
        self, actor, ctable, scalar_bar, rng=None, background_color=None, fmt=None
    ):
        if rng is not None:
            mapper = actor.GetMapper()
            mapper.SetScalarRange(*rng)
            lut = mapper.GetLookupTable()
            lut.SetTable(numpy_to_vtk(ctable))
        if scalar_bar is not None:
            lut = scalar_bar.GetLookupTable()
            if background_color is not None:
                background_color = np.array(background_color) * 255
                ctable = _alpha_blend_background(ctable, background_color)
            lut.SetTable(numpy_to_vtk(ctable, array_type=VTK_UNSIGNED_CHAR))
            lut.SetRange(*rng)
            if fmt is not None:
                scalar_bar.SetLabelFormat(fmt)

    def _set_volume_range(self, volume, ctable, alpha, scalar_bar, rng, fmt=None):
        color_tf = vtkColorTransferFunction()
        opacity_tf = vtkPiecewiseFunction()
        for loc, color in zip(np.linspace(*rng, num=len(ctable)), ctable):
            color_tf.AddRGBPoint(loc, *(color[:-1] / 255.0))
            opacity_tf.AddPoint(loc, color[-1] * alpha / 255.0)
        color_tf.ClampingOn()
        opacity_tf.ClampingOn()
        prop = volume.GetProperty()
        prop.SetColor(color_tf)
        prop.SetScalarOpacity(opacity_tf)
        if scalar_bar is not None:
            lut = vtkLookupTable()
            lut.SetRange(*rng)
            lut.SetTable(numpy_to_vtk(ctable))
            scalar_bar.SetLookupTable(lut)
            if fmt is not None:
                scalar_bar.SetLabelFormat(fmt)

    def _sphere(self, center, color, radius):
        mesh = pyvista.Sphere(
            radius=radius, center=center, theta_resolution=8, phi_resolution=8
        )
        actor = _add_mesh(self.plotter, mesh=mesh, color=color)
        return actor, mesh

    def _volume(
        self,
        dimensions,
        origin,
        spacing,
        scalars,
        surface_alpha,
        resolution,
        blending,
        center,
        interpolation="linear",
    ):
        # Note: this method is used by mne-gui-addons, so we should be mindful of
        # backwards compatibility when changing it.

        # Now we can actually construct the visualization
        grid = pyvista.ImageData(
            dimensions=dimensions,
            spacing=spacing,
            origin=origin,
        )
        del origin, spacing, dimensions
        grid.point_data["values"] = scalars
        assert interpolation in ("nearest", "linear"), interpolation

        # Add contour of enclosed volume (use GetOutput instead of
        # GetOutputPort below to avoid updating)

        if surface_alpha > 0:
            grid_surface = vtkMarchingContourFilter()
            grid_surface.ComputeNormalsOn()
            grid_surface.ComputeScalarsOff()
            grid_surface.SetInputData(grid)
            grid_surface.SetValue(0, 0.1)
            grid_surface.Update()
            grid_smooth = _vtk_smooth(grid_surface.GetOutput(), 0.75, verbose=False)
            grid_mesh = vtkPolyDataMapper()
            grid_mesh.SetInputData(grid_smooth)
        else:
            grid_mesh = None

        mapper = vtkSmartVolumeMapper()
        interp_map_meth = "SetInterpolationModeTo"
        interp_map_meth += dict(nearest="NearestNeighbor", linear="Linear")[
            interpolation
        ]
        interp_prop_meth = "SetInterpolationTypeTo"
        interp_prop_meth += dict(nearest="Nearest", linear="Linear")[interpolation]
        del interpolation
        if resolution is None:  # native
            mapper.SetScalarModeToUsePointData()
            mapper.SetInputDataObject(grid)
        else:
            upsampler = vtkImageReslice()
            getattr(upsampler, interp_map_meth)()
            upsampler.SetOutputOrigin(grid.origin)
            upsampler.SetOutputSpacing(*([resolution] * 3))
            upsampler.SetInputDataObject(grid)
            mapper.SetInputConnection(upsampler.GetOutputPort())
        getattr(mapper, interp_map_meth)()
        # Additive, AverageIntensity, and Composite might also be reasonable
        blend_meth = "SetBlendModeTo"
        blend_meth += dict(composite="Composite", mip="MaximumIntensity")[blending]
        getattr(mapper, blend_meth)()
        volume_pos = vtkVolume()
        volume_pos.SetMapper(mapper)
        dist = grid.length / np.mean(grid.dimensions)
        volume_pos_prop = volume_pos.GetProperty()
        volume_pos_prop.SetScalarOpacityUnitDistance(dist)
        volume_pos_prop.ShadeOn()
        getattr(volume_pos_prop, interp_prop_meth)()
        if center is not None and blending == "mip":
            # We need to create a minimum intensity projection for the neg half
            mapper_neg = vtkSmartVolumeMapper()
            if resolution is None:  # native
                mapper_neg.SetScalarModeToUsePointData()
                mapper_neg.SetInputDataObject(grid)
            else:
                mapper_neg.SetInputConnection(upsampler.GetOutputPort())
            mapper_neg.SetBlendModeToMinimumIntensity()
            volume_neg = vtkVolume()
            volume_neg.SetMapper(mapper_neg)
            volume_neg_prop = volume_neg.GetProperty()
            volume_neg_prop.SetScalarOpacityUnitDistance(dist)
            volume_neg_prop.ShadeOn()
            getattr(volume_neg_prop, interp_prop_meth)()
        else:
            volume_neg = None
        return grid, grid_mesh, volume_pos, volume_neg

    def _silhouette(self, mesh, color=None, line_width=None, alpha=None, decimate=None):
        mesh = mesh.decimate(decimate) if decimate is not None else mesh
        silhouette_filter = vtkPolyDataSilhouette()
        silhouette_filter.SetInputData(mesh)
        silhouette_filter.SetCamera(self.plotter.renderer.GetActiveCamera())
        silhouette_filter.SetEnableFeatureAngle(0)
        silhouette_mapper = vtkPolyDataMapper()
        silhouette_mapper.SetInputConnection(silhouette_filter.GetOutputPort())
        actor, prop = self.plotter.add_actor(
            silhouette_mapper,
            culling=False,
            pickable=False,
            reset_camera=False,
            render=False,
        )
        if color is not None:
            prop.SetColor(*color)
        if alpha is not None:
            prop.SetOpacity(alpha)
        if line_width is not None:
            prop.SetLineWidth(line_width)
        _hide_testing_actor(actor)
        return actor


def _compute_normals(mesh):
    """Patch PyVista compute_normals."""
    if "Normals" not in mesh.point_data:
        mesh.compute_normals(
            cell_normals=False,
            consistent_normals=False,
            non_manifold_traversal=False,
            inplace=True,
        )


def _quat_to_vtk_wxyz(quat):
    """Convert MNE's (..., 3) unit quaternions to VTK-order (w, x, y, z)."""
    assert quat.ndim >= 2 and quat.shape[-1] == 3, quat.shape
    w = np.sqrt(np.clip(1.0 - (quat * quat).sum(-1), 0.0, 1.0))
    return np.concatenate([w[..., np.newaxis], quat], axis=-1)


def _add_mesh(plotter, **kwargs):
    """Patch PyVista add_mesh."""
    mesh = kwargs.get("mesh")
    if "smooth_shading" in kwargs:
        smooth_shading = kwargs.pop("smooth_shading")
    else:
        smooth_shading = True
    # disable rendering pass for add_mesh, render()
    # is called in show()
    if "render" not in kwargs:
        kwargs["render"] = False
    if "reset_camera" not in kwargs:
        kwargs["reset_camera"] = False
    actor = plotter.add_mesh(**kwargs)
    if smooth_shading and "Normals" in mesh.point_data:
        prop = actor.GetProperty()
        prop.SetInterpolationToPhong()
    _hide_testing_actor(actor)
    return actor


def _hide_testing_actor(actor):
    from . import renderer

    if renderer.MNE_3D_BACKEND_TESTING:
        actor.SetVisibility(False)


def _truncate_scalar_bar_title(title, max_chars=20):
    if title is None or len(title) <= max_chars:
        return title
    return title[: max_chars - 1] + "…"


def _to_pos(azimuth, elevation):
    theta = azimuth * np.pi / 180.0
    phi = (90.0 - elevation) * np.pi / 180.0
    x = np.sin(theta) * np.sin(phi)
    y = np.cos(phi)
    z = np.cos(theta) * np.sin(phi)
    return x, y, z


def _3d_to_2d(plotter, xyz):
    # https://vtk.org/Wiki/VTK/Examples/Cxx/Utilities/Coordinate
    coordinate = vtkCoordinate()
    coordinate.SetCoordinateSystemToWorld()
    xy = list()
    for coord in xyz:
        coordinate.SetValue(*coord)
        xy.append(coordinate.GetComputedLocalDisplayValue(plotter.renderer))
    xy = np.array(xy, float).reshape(-1, 2)  # in case it's empty
    return xy


def _close_all():
    close_all()
    _FIGURES.clear()


def _get_user_camera_direction(plotter, rigid):
    position, focalpoint = np.array(plotter.camera_position[:2], float)
    if rigid is not None:
        position = apply_trans(rigid, position, move=False)
        focalpoint = apply_trans(rigid, focalpoint, move=False)
    return tuple(_cart_to_sph(position - focalpoint)[0])


def _get_3d_view(figure, *, rigid=None):
    focalpoint = np.array(figure.plotter.camera_position[1], float)
    _, phi, theta = _get_user_camera_direction(figure.plotter, rigid)
    azimuth, elevation = np.rad2deg(phi) % 360, np.rad2deg(theta) % 180
    return (
        figure.plotter.camera.roll,
        figure.plotter.camera.distance,
        azimuth,
        elevation,
        focalpoint,
    )


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
    # Only compute bounds if we need to
    bounds = None
    if isinstance(focalpoint, str) or isinstance(distance, str):
        bounds = np.array(figure.plotter.renderer.ComputeVisiblePropBounds(), float)

    # camera slides along the vector defined from camera position to focal point until
    # all of the actors can be seen (quoting PyVista's docs)
    # Figure out our current parameters in the transformed space
    _, phi, theta = _get_user_camera_direction(figure.plotter, rigid)

    # focalpoint: if 'auto', we use the center of mass of the visible
    # bounds, if None, we use the existing camera focal point otherwise
    # we use the values given by the user
    if isinstance(focalpoint, str):
        _check_option("focalpoint", focalpoint, ("auto",), extra="when a string")
        focalpoint = (bounds[1::2] + bounds[::2]) * 0.5
    elif focalpoint is None:
        focalpoint = figure.plotter.camera_position[1]
    focalpoint = np.array(focalpoint, float)  # in real-world coords
    if distance is None:
        distance = figure.plotter.camera.distance
    elif isinstance(distance, str):
        _check_option("distance", distance, ("auto",), extra="when a string")
        distance = max(bounds[1::2] - bounds[::2]) * 2.0
    distance = float(distance)

    if azimuth is not None:
        phi = np.deg2rad(azimuth)
    if elevation is not None:
        theta = np.deg2rad(elevation)

    # Now calculate the view_up vector of the camera.  If the view up is
    # close to the 'z' axis, the view plane normal is parallel to the
    # camera which is unacceptable, so we use a different view up.
    if elevation is None or 5.0 <= abs(elevation) <= 175.0:
        view_up = [0, 0, 1]
    else:
        view_up = [0, 1, 0]

    position = _sph_to_cart([distance, phi, theta])[0]

    # restore to the original frame
    if rigid is not None:
        rigid_inv = np.linalg.inv(rigid)
        position = apply_trans(rigid_inv, position, move=False)
        view_up = apply_trans(rigid_inv, view_up, move=False)
    figure.plotter.camera_position = [position, focalpoint, view_up]
    if roll is not None:
        figure.plotter.camera.roll = roll

    if update:
        figure.plotter.update()
        _process_events(figure.plotter)


def _set_3d_title(figure, title, size=16, *, color="white", position="upper_left"):
    handle = figure.plotter.add_text(
        title,
        font_size=size,
        color=color,
        position=position,
        name="title",
    )
    figure.plotter.update()
    _process_events(figure.plotter)
    return handle


def _check_3d_figure(figure):
    _validate_type(figure, PyVistaFigure, "figure")


def _close_3d_figure(figure):
    # copy the plotter locally because figure.plotter is modified
    plotter = figure.plotter
    # close the window
    plotter.close()  # additional cleaning following signal_close
    _process_events(plotter)
    # free memory and deregister from the scraper
    plotter.deep_clean()  # remove internal references
    _ALL_PLOTTERS.pop(plotter._id_name, None)
    _process_events(plotter)


def _take_3d_screenshot(figure, mode="rgb", filename=None):
    _process_events(figure.plotter)
    return figure.plotter.screenshot(
        transparent_background=(mode == "rgba"), filename=filename
    )


def _process_events(plotter):
    if hasattr(plotter, "app"):
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("ignore", "constrained_layout")
            plotter.app.processEvents()


def _add_camera_callback(camera, callback):
    return camera.AddObserver(vtkCommand.ModifiedEvent, callback)


def _cylinder_geom(radius=None, height=None, center=None, resolution=None):
    """Build a cylinder vtkPolyData with its axis along +x.

    vtkCylinderSource's axis is along y, so we rotate 90 degrees about z to
    match the arrow/cone convention of pointing along x (the axis that
    vtkGlyph3D/vtkGlyph3DMapper orient toward the per-instance vector).
    """
    source = vtkCylinderSource()
    if radius is not None:
        source.SetRadius(radius)
    if height is not None:
        source.SetHeight(height)
    if center is not None:
        source.SetCenter(center)
    if resolution is not None:
        source.SetResolution(resolution)
    source.Update()
    tr = vtkTransform()
    tr.RotateWXYZ(90, 0, 0, 1)
    trp = vtkTransformFilter()
    trp.SetInputData(source.GetOutput())
    trp.SetTransform(tr)
    trp.Update()
    return trp.GetOutput()


def _arrow_glyph(grid, factor):
    glyph = vtkGlyphSource2D()
    glyph.SetGlyphTypeToArrow()
    glyph.FilledOff()
    glyph.Update()

    # fix position
    tr = vtkTransform()
    tr.Translate(0.5, 0.0, 0.0)
    trp = vtkTransformFilter()
    trp.SetInputConnection(glyph.GetOutputPort())
    trp.SetTransform(tr)
    trp.Update()

    alg = _glyph(
        grid,
        scale_mode="vector",
        scalars=False,
        orient="vec",
        factor=factor,
        geom=trp.GetOutputPort(),
    )
    mapper = vtkDataSetMapper()
    mapper.SetInputConnection(alg.GetOutputPort())
    return mapper


def _glyph(
    dataset,
    *,
    scale_mode="scalar",
    orient=True,
    scalars=True,
    factor=1.0,
    geom=None,
    absolute=False,
    clamping=False,
    rng=None,
):
    if geom is None:
        arrow = vtkArrowSource()
        arrow.Update()
        geom = arrow.GetOutputPort()
    alg = vtkGlyph3D()
    alg.SetSourceConnection(geom)
    if isinstance(scalars, str):
        dataset.active_scalars_name = scalars
    if isinstance(orient, str):
        dataset.active_vectors_name = orient
        orient = True
    if scale_mode == "scalar":
        alg.SetScaleModeToScaleByScalar()
    elif scale_mode == "vector":
        alg.SetScaleModeToScaleByVector()
    else:
        alg.SetScaleModeToDataScalingOff()
    if rng is not None:
        alg.SetRange(rng)
    alg.SetOrient(orient)
    alg.SetInputData(dataset)
    alg.SetScaleFactor(factor)
    alg.SetClamping(clamping)
    alg.Update()
    return alg


@contextmanager
def _disabled_depth_peeling():
    depth_peeling = pyvista.global_theme.depth_peeling
    depth_peeling_enabled = depth_peeling["enabled"]
    depth_peeling["enabled"] = False
    try:
        yield
    finally:
        depth_peeling["enabled"] = depth_peeling_enabled


def _is_osmesa(plotter):
    # MESA (could use GPUInfo / _get_gpu_info here, but it takes
    # > 700 ms to make a new window + report capabilities!)
    # CircleCI's is: "Mesa 20.0.8 via llvmpipe (LLVM 10.0.0, 256 bits)"
    # and a working Nouveau is: "Mesa 24.2.3-1ubuntu1 via NVE6"
    if platform.system() == "Darwin":  # segfaults on macOS sometimes
        return False
    if os.getenv("MNE_IS_OSMESA", "").lower() == "true":
        return True
    gpu_info_full = plotter.ren_win.ReportCapabilities()
    gpu_info = re.findall(
        "OpenGL (?:version|renderer) string:(.+)\n",
        gpu_info_full,
    )
    gpu_info = " ".join(gpu_info).lower()
    is_osmesa = "mesa" in gpu_info.split()
    if is_osmesa:
        # Try to warn if it's ancient
        version = re.findall("mesa ([0-9.]+)[ -].*", gpu_info) or re.findall(
            "OpenGL version string: .* Mesa ([0-9.]+)\n", gpu_info_full
        )
        if version:
            version = version[0]
            if _compare_version(version, "<", "18.3.6"):
                warn(
                    f"Mesa version {version} is too old for translucent 3D "
                    "surface rendering, consider upgrading to 18.3.6 or "
                    "later."
                )
        is_osmesa = "llvmpipe" in gpu_info
    return is_osmesa


class _SafeBackgroundPlotter(BackgroundPlotter):
    # https://github.com/pyvista/pyvistaqt/pull/258
    def __del__(self) -> None:  # pragma: no cover
        """Delete the qt plotter."""
        self.close()
