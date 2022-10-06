"""
Core visualization operations based on PyVista.

Actual implementation of _Renderer and _Projection classes.
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#
# License: Simplified BSD

from contextlib import contextmanager
from inspect import signature
import os
import re
import sys
import warnings

import numpy as np

from ._abstract import _AbstractRenderer, Figure3D
from ._utils import (_get_colormap_from_array, _alpha_blend_background,
                     ALLOWED_QUIVER_MODES, _init_mne_qtapp)
from ...fixes import _point_data, _cell_data, _compare_version
from ...transforms import apply_trans
from ...utils import (copy_base_doc_to_subclass_doc, _check_option,
                      _require_version, _validate_type)


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pyvista
    from pyvista import Plotter, PolyData, Line, close_all, UnstructuredGrid
    try:
        from pyvistaqt import BackgroundPlotter  # noqa
    except ImportError:
        from pyvista import BackgroundPlotter
    from pyvista.plotting.plotting import _ALL_PLOTTERS

from vtkmodules.vtkCommonCore import (
    vtkCommand, vtkLookupTable, VTK_UNSIGNED_CHAR)
from vtkmodules.vtkCommonDataModel import VTK_VERTEX, vtkPiecewiseFunction
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import vtkCellDataToPointData, vtkGlyph3D
from vtkmodules.vtkFiltersGeneral import (
    vtkTransformPolyDataFilter, vtkMarchingContourFilter)
from vtkmodules.vtkFiltersHybrid import vtkPolyDataSilhouette
from vtkmodules.vtkFiltersSources import (
    vtkSphereSource, vtkConeSource, vtkCylinderSource, vtkArrowSource,
    vtkPlatonicSolidSource, vtkGlyphSource2D)
from vtkmodules.vtkImagingCore import vtkImageReslice
from vtkmodules.vtkRenderingCore import (
    vtkMapper, vtkActor, vtkCellPicker, vtkColorTransferFunction,
    vtkPolyDataMapper, vtkVolume, vtkCoordinate, vtkDataSetMapper)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
from vtkmodules.util.numpy_support import numpy_to_vtk
try:
    from vtkmodules.vtkCommonCore import VTK_VERSION
except Exception:  # some bad versions of VTK
    VTK_VERSION = '9.0'

VTK9 = _compare_version(VTK_VERSION, '>=', '9.0')


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

    def _init(self, plotter=None, show=False, title='PyVista Scene',
              size=(600, 600), shape=(1, 1), background_color='black',
              smooth_shading=True, off_screen=False, notebook=False,
              splash=False, multi_samples=None):
        self._plotter = plotter
        self.display = None
        self.background_color = background_color
        self.smooth_shading = smooth_shading
        self.notebook = notebook
        self.title = title
        self.splash = splash

        self.store = dict()
        self.store['window_size'] = size
        self.store['shape'] = shape
        self.store['off_screen'] = off_screen
        self.store['border'] = False
        self.store['multi_samples'] = multi_samples

        if not self.notebook:
            self.store['show'] = show
            self.store['title'] = title
            self.store['auto_update'] = False
            self.store['menu_bar'] = False
            self.store['toolbar'] = False
            self.store['update_app_icon'] = False
            self._plotter_class = BackgroundPlotter
            if 'app_window_class' in signature(BackgroundPlotter).parameters:
                from ._qt import _MNEMainWindow
                self.store['app_window_class'] = _MNEMainWindow
        else:
            self._plotter_class = Plotter

        self._nrows, self._ncols = self.store['shape']
        self._azimuth = self._elevation = None

    def _build(self):
        if self.plotter is None:
            if not self.notebook:
                out = _init_mne_qtapp(
                    enable_icon=hasattr(self._plotter_class, 'set_icon'),
                    splash=self.splash)
                # replace it with the Qt object
                if self.splash:
                    self.splash = out[1]
                    app = out[0]
                else:
                    app = out
                self.store['app'] = app
            plotter = self._plotter_class(**self.store)
            plotter.background_color = self.background_color
            self._plotter = plotter
        if self.plotter.iren is not None:
            self.plotter.iren.initialize()
        _process_events(self.plotter)
        _process_events(self.plotter)
        return self.plotter

    def _is_active(self):
        if self.plotter is None:
            return False
        return hasattr(self.plotter, 'ren_win')


class _Projection(object):
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


@copy_base_doc_to_subclass_doc
class _PyVistaRenderer(_AbstractRenderer):
    """Class managing rendering scene.

    Attributes
    ----------
    plotter: Plotter
        Main PyVista access point.
    name: str
        Name of the window.
    """

    def __init__(self, fig=None, size=(600, 600), bgcolor='black',
                 name="PyVista Scene", show=False, shape=(1, 1),
                 notebook=None, smooth_shading=True, splash=False,
                 multi_samples=None):
        from .._3d import _get_3d_option
        _require_version('pyvista', 'use 3D rendering', '0.32')
        multi_samples = _get_3d_option('multi_samples')
        # multi_samples > 1 is broken on macOS + Intel Iris + volume rendering
        if sys.platform == 'darwin':
            multi_samples = 1
        figure = PyVistaFigure()
        figure._init(
            show=show, title=name, size=size, shape=shape,
            background_color=bgcolor, notebook=notebook,
            smooth_shading=smooth_shading, splash=splash,
            multi_samples=multi_samples)
        self.font_family = "arial"
        self.tube_n_sides = 20
        self.antialias = _get_3d_option('antialias')
        self.depth_peeling = _get_3d_option('depth_peeling')
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
            self.figure.store['off_screen'] = True

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            # pyvista theme may enable depth peeling by default so
            # we disable it initially to better control the value afterwards
            with _disabled_depth_peeling():
                self.plotter = self.figure._build()
            self._hide_axes()
            self._enable_antialias()
            self._enable_depth_peeling()

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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.plotter.subplot(x, y)

    def scene(self):
        return self.figure

    def update_lighting(self):
        # Inspired from Mayavi's version of Raymond Maple 3-lights illumination
        for renderer in self._all_renderers:
            lights = list(renderer.GetLights())
            headlight = lights.pop(0)
            headlight.SetSwitch(False)
            # below and centered, left and above, right and above
            az_el_in = ((0, -45, 0.7), (-60, 30, 0.7), (60, 30, 0.7))
            for li, light in enumerate(lights):
                if li < len(az_el_in):
                    light.SetSwitch(True)
                    light.SetPosition(_to_pos(*az_el_in[li][:2]))
                    light.SetIntensity(az_el_in[li][2])
                else:
                    light.SetSwitch(False)
                    light.SetPosition(_to_pos(0.0, 0.0))
                    light.SetIntensity(0.0)
                light.SetColor(1.0, 1.0, 1.0)

    def set_interaction(self, interaction):
        if not hasattr(self.plotter, "iren") or self.plotter.iren is None:
            return
        if interaction == "rubber_band_2d":
            for renderer in self._all_renderers:
                renderer.enable_parallel_projection()
            if hasattr(self.plotter, 'enable_rubber_band_2d_style'):
                self.plotter.enable_rubber_band_2d_style()
            else:
                from vtkmodules.vtkInteractionStyle import\
                    vtkInteractorStyleRubberBand2D
                style = vtkInteractorStyleRubberBand2D()
                self.plotter.interactor.SetInteractorStyle(style)
        else:
            for renderer in self._all_renderers:
                renderer.disable_parallel_projection()
            kwargs = dict()
            if interaction == 'terrain':
                kwargs['mouse_wheel_zooms'] = True
            getattr(self.plotter, f'enable_{interaction}_style')(**kwargs)

    def legend(self, labels, border=False, size=0.1, face='triangle',
               loc='upper left'):
        return self.plotter.add_legend(
            labels, size=(size, size), face=face, loc=loc)

    def polydata(self, mesh, color=None, opacity=1.0, normals=None,
                 backface_culling=False, scalars=None, colormap=None,
                 vmin=None, vmax=None, interpolate_before_map=True,
                 representation='surface', line_width=1.,
                 polygon_offset=None, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            rgba = False
            if color is not None and len(color) == mesh.n_points:
                if color.shape[1] == 3:
                    scalars = np.c_[color, np.ones(mesh.n_points)]
                else:
                    scalars = color
                scalars = (scalars * 255).astype('ubyte')
                color = None
                rgba = True
            if isinstance(colormap, np.ndarray):
                if colormap.dtype == np.uint8:
                    colormap = colormap.astype(np.float64) / 255.
                from matplotlib.colors import ListedColormap
                colormap = ListedColormap(colormap)
            if normals is not None:
                _point_data(mesh)["Normals"] = normals
                mesh.GetPointData().SetActiveNormals("Normals")
            else:
                _compute_normals(mesh)
            if 'rgba' in kwargs:
                rgba = kwargs["rgba"]
                kwargs.pop('rgba')
            smooth_shading = self.smooth_shading
            if representation == 'wireframe':
                smooth_shading = False  # never use smooth shading for wf
            actor = _add_mesh(
                plotter=self.plotter,
                mesh=mesh, color=color, scalars=scalars, edge_color=color,
                rgba=rgba, opacity=opacity, cmap=colormap,
                backface_culling=backface_culling,
                rng=[vmin, vmax], show_scalar_bar=False,
                smooth_shading=smooth_shading,
                interpolate_before_map=interpolate_before_map,
                style=representation, line_width=line_width, **kwargs,
            )

            if polygon_offset is not None:
                mapper = actor.GetMapper()
                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(
                    polygon_offset, polygon_offset)

            return actor, mesh

    def mesh(self, x, y, z, triangles, color, opacity=1.0, shading=False,
             backface_culling=False, scalars=None, colormap=None,
             vmin=None, vmax=None, interpolate_before_map=True,
             representation='surface', line_width=1., normals=None,
             polygon_offset=None, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
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
            polygon_offset=polygon_offset,
            **kwargs,
        )

    def contour(self, surface, scalars, contours, width=1.0, opacity=1.0,
                vmin=None, vmax=None, colormap=None,
                normalized_colormap=False, kind='line', color=None):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            if colormap is not None:
                colormap = _get_colormap_from_array(colormap,
                                                    normalized_colormap)
            vertices = np.array(surface['rr'])
            triangles = np.array(surface['tris'])
            n_triangles = len(triangles)
            triangles = np.c_[np.full(n_triangles, 3), triangles]
            mesh = PolyData(vertices, triangles)
            _point_data(mesh)['scalars'] = scalars
            contour = mesh.contour(isosurfaces=contours)
            line_width = width
            if kind == 'tube':
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

    def surface(self, surface, color=None, opacity=1.0,
                vmin=None, vmax=None, colormap=None,
                normalized_colormap=False, scalars=None,
                backface_culling=False, polygon_offset=None):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            normals = surface.get('nn', None)
            vertices = np.array(surface['rr'])
            triangles = np.array(surface['tris'])
            triangles = np.c_[np.full(len(triangles), 3), triangles]
            mesh = PolyData(vertices, triangles)
        colormap = _get_colormap_from_array(colormap, normalized_colormap)
        if scalars is not None:
            _point_data(mesh)['scalars'] = scalars
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
            polygon_offset=polygon_offset,
        )

    def sphere(self, center, color, scale, opacity=1.0,
               resolution=8, backface_culling=False,
               radius=None):
        from vtkmodules.vtkFiltersSources import vtkSphereSource
        factor = 1.0 if radius is not None else scale
        center = np.array(center, dtype=float)
        if len(center) == 0:
            return None, None
        _check_option('center.ndim', center.ndim, (1, 2))
        _check_option('center.shape[-1]', center.shape[-1], (3,))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            sphere = vtkSphereSource()
            sphere.SetThetaResolution(resolution)
            sphere.SetPhiResolution(resolution)
            if radius is not None:
                sphere.SetRadius(radius)
            sphere.Update()
            geom = sphere.GetOutput()
            mesh = PolyData(center)
            glyph = mesh.glyph(orient=False, scale=False,
                               factor=factor, geom=geom)
            actor = _add_mesh(
                self.plotter,
                mesh=glyph, color=color, opacity=opacity,
                backface_culling=backface_culling,
                smooth_shading=self.smooth_shading
            )
            return actor, glyph

    def tube(self, origin, destination, radius=0.001, color='white',
             scalars=None, vmin=None, vmax=None, colormap='RdBu',
             normalized_colormap=False, reverse_lut=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            cmap = _get_colormap_from_array(colormap, normalized_colormap)
            for (pointa, pointb) in zip(origin, destination):
                line = Line(pointa, pointb)
                if scalars is not None:
                    _point_data(line)['scalars'] = scalars[0, :]
                    scalars = 'scalars'
                    color = None
                else:
                    scalars = None
                tube = line.tube(radius, n_sides=self.tube_n_sides)
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
                )
        return actor, tube

    def quiver3d(self, x, y, z, u, v, w, color, scale, mode, resolution=8,
                 glyph_height=None, glyph_center=None, glyph_resolution=None,
                 opacity=1.0, scale_mode='none', scalars=None, colormap=None,
                 backface_culling=False, line_width=2., name=None,
                 glyph_width=None, glyph_depth=None, glyph_radius=0.15,
                 solid_transform=None, *, clim=None):
        _check_option('mode', mode, ALLOWED_QUIVER_MODES)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            factor = scale
            vectors = np.c_[u, v, w]
            points = np.vstack(np.c_[x, y, z])
            n_points = len(points)
            cell_type = np.full(n_points, VTK_VERTEX)
            cells = np.c_[np.full(n_points, 1), range(n_points)]
            args = (cells, cell_type, points)
            if not VTK9:
                args = (np.arange(n_points) * 3,) + args
            grid = UnstructuredGrid(*args)
            if scalars is None:
                scalars = np.ones((n_points,))
            _point_data(grid)['scalars'] = np.array(scalars)
            _point_data(grid)['vec'] = vectors
            if mode == '2darrow':
                return _arrow_glyph(grid, factor), grid
            elif mode == 'arrow':
                alg = _glyph(
                    grid,
                    orient='vec',
                    scalars='scalars',
                    factor=factor
                )
                mesh = pyvista.wrap(alg.GetOutput())
            else:
                tr = None
                if mode == 'cone':
                    glyph = vtkConeSource()
                    glyph.SetCenter(0.5, 0, 0)
                    if glyph_radius is not None:
                        glyph.SetRadius(glyph_radius)
                elif mode == 'cylinder':
                    glyph = vtkCylinderSource()
                    if glyph_radius is not None:
                        glyph.SetRadius(glyph_radius)
                elif mode == 'oct':
                    glyph = vtkPlatonicSolidSource()
                    glyph.SetSolidTypeToOctahedron()
                else:
                    assert mode == 'sphere', mode  # guaranteed above
                    glyph = vtkSphereSource()
                if mode == 'cylinder':
                    if glyph_height is not None:
                        glyph.SetHeight(glyph_height)
                    if glyph_center is not None:
                        glyph.SetCenter(glyph_center)
                    if glyph_resolution is not None:
                        glyph.SetResolution(glyph_resolution)
                    tr = vtkTransform()
                    tr.RotateWXYZ(90, 0, 0, 1)
                elif mode == 'oct':
                    if solid_transform is not None:
                        assert solid_transform.shape == (4, 4)
                        tr = vtkTransform()
                        tr.SetMatrix(
                            solid_transform.astype(np.float64).ravel())
                if tr is not None:
                    # fix orientation
                    glyph.Update()
                    trp = vtkTransformPolyDataFilter()
                    trp.SetInputData(glyph.GetOutput())
                    trp.SetTransform(tr)
                    glyph = trp
                glyph.Update()
                geom = glyph.GetOutput()
                mesh = grid.glyph(orient='vec', scale=scale_mode == 'vector',
                                  factor=factor, geom=geom)
            actor = _add_mesh(
                self.plotter,
                mesh=mesh,
                color=color,
                opacity=opacity,
                scalars=None,
                colormap=colormap,
                show_scalar_bar=False,
                backface_culling=backface_culling,
                clim=clim,
            )
        return actor, mesh

    def text2d(self, x_window, y_window, text, size=14, color='white',
               justification=None):
        size = 14 if size is None else size
        position = (x_window, y_window)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            actor = self.plotter.add_text(text, position=position,
                                          font_size=size,
                                          color=color,
                                          viewport=True)
            if isinstance(justification, str):
                if justification == 'left':
                    actor.GetTextProperty().SetJustificationToLeft()
                elif justification == 'center':
                    actor.GetTextProperty().SetJustificationToCentered()
                elif justification == 'right':
                    actor.GetTextProperty().SetJustificationToRight()
                else:
                    raise ValueError('Expected values for `justification`'
                                     'are `left`, `center` or `right` but '
                                     'got {} instead.'.format(justification))
        _hide_testing_actor(actor)
        return actor

    def text3d(self, x, y, z, text, scale, color='white'):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            kwargs = dict(
                points=np.array([x, y, z]).astype(float),
                labels=[text],
                point_size=scale,
                text_color=color,
                font_family=self.font_family,
                name=text,
                shape_opacity=0,
            )
            if ('always_visible'
                    in signature(self.plotter.add_point_labels).parameters):
                kwargs['always_visible'] = True
            actor = self.plotter.add_point_labels(**kwargs)
        _hide_testing_actor(actor)
        return actor

    def scalarbar(self, source, color="white", title=None, n_labels=4,
                  bgcolor=None, **extra_kwargs):
        if isinstance(source, vtkMapper):
            mapper = source
        elif isinstance(source, vtkActor):
            mapper = source.GetMapper()
        else:
            mapper = None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            kwargs = dict(color=color, title=title, n_labels=n_labels,
                          use_opacity=False, n_colors=256, position_x=0.15,
                          position_y=0.05, width=0.7, shadow=False, bold=True,
                          label_font_size=22, font_family=self.font_family,
                          background_color=bgcolor, mapper=mapper)
            kwargs.update(extra_kwargs)
            actor = self.plotter.add_scalar_bar(**kwargs)
        _hide_testing_actor(actor)
        return actor

    def show(self):
        self.plotter.show()

    def close(self):
        _close_3d_figure(figure=self.figure)

    def get_camera(self):
        return _get_3d_view(self.figure)

    def set_camera(self, azimuth=None, elevation=None, distance=None,
                   focalpoint='auto', roll=None, reset_camera=True,
                   rigid=None, update=True):
        _set_3d_view(self.figure, azimuth=azimuth, elevation=elevation,
                     distance=distance, focalpoint=focalpoint, roll=roll,
                     reset_camera=reset_camera, rigid=rigid, update=update)

    def reset_camera(self):
        self.plotter.reset_camera()

    def screenshot(self, mode='rgb', filename=None):
        return _take_3d_screenshot(figure=self.figure, mode=mode,
                                   filename=filename)

    def project(self, xyz, ch_names):
        xy = _3d_to_2d(self.plotter, xyz)
        xy = dict(zip(ch_names, xy))
        # pts = self.fig.children[-1]
        pts = self.plotter.renderer.GetActors().GetLastItem()

        return _Projection(xy=xy, pts=pts, plotter=self.plotter)

    def _enable_depth_peeling(self):
        if not self.depth_peeling:
            return
        if not self.figure.store['off_screen']:
            for renderer in self._all_renderers:
                renderer.enable_depth_peeling()

    def _enable_antialias(self):
        """Enable it everywhere except Azure."""
        if not self.antialias:
            return
        # XXX for some reason doing this on Azure causes access violations:
        #     ##[error]Cmd.exe exited with code '-1073741819'
        # So for now don't use it there. Maybe has to do with setting these
        # before the window has actually been made "active"...?
        # For Mayavi we have an "on activated" event or so, we should look into
        # using this for Azure at some point, too.
        if self.figure._is_active():
            # macOS, Azure
            bad_system = (
                sys.platform == 'darwin' or
                os.getenv('AZURE_CI_WINDOWS', 'false').lower() == 'true')
            bad_system |= _is_mesa(self.plotter)
            if not bad_system:
                for renderer in self._all_renderers:
                    renderer.enable_anti_aliasing()
            for plotter in self._all_plotters:
                plotter.ren_win.LineSmoothingOn()

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

    def _update_picking_callback(self,
                                 on_mouse_move,
                                 on_button_press,
                                 on_button_release,
                                 on_pick):
        add_obs = self.plotter.iren.add_observer
        add_obs(vtkCommand.RenderEvent, on_mouse_move)
        add_obs(vtkCommand.LeftButtonPressEvent, on_button_press)
        add_obs(vtkCommand.EndInteractionEvent, on_button_release)
        self.plotter.picker = vtkCellPicker()
        self.plotter.picker.AddObserver(
            vtkCommand.EndPickEvent,
            on_pick
        )
        self.plotter.picker.SetVolumeOpacityIsovalue(0.)

    def _set_mesh_scalars(self, mesh, scalars, name):
        # Catch:  FutureWarning: Conversion of the second argument of
        # issubdtype from `complex` to `np.complexfloating` is deprecated.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            _point_data(mesh)[name] = scalars

    def _set_colormap_range(self, actor, ctable, scalar_bar, rng=None,
                            background_color=None):
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

    def _set_volume_range(self, volume, ctable, alpha, scalar_bar, rng):
        color_tf = vtkColorTransferFunction()
        opacity_tf = vtkPiecewiseFunction()
        for loc, color in zip(np.linspace(*rng, num=len(ctable)), ctable):
            color_tf.AddRGBPoint(loc, *(color[:-1] / 255.))
            opacity_tf.AddPoint(loc, color[-1] * alpha / 255.)
        color_tf.ClampingOn()
        opacity_tf.ClampingOn()
        prop = volume.GetProperty()
        prop.SetColor(color_tf)
        prop.SetScalarOpacity(opacity_tf)
        prop.ShadeOn()
        prop.SetInterpolationTypeToLinear()
        if scalar_bar is not None:
            lut = vtkLookupTable()
            lut.SetRange(*rng)
            lut.SetTable(numpy_to_vtk(ctable))
            scalar_bar.SetLookupTable(lut)

    def _sphere(self, center, color, radius):
        from vtkmodules.vtkFiltersSources import vtkSphereSource
        sphere = vtkSphereSource()
        sphere.SetThetaResolution(8)
        sphere.SetPhiResolution(8)
        sphere.SetRadius(radius)
        sphere.SetCenter(center)
        sphere.Update()
        mesh = pyvista.wrap(sphere.GetOutput())
        actor = _add_mesh(
            self.plotter,
            mesh=mesh,
            color=color
        )
        return actor, mesh

    def _volume(self, dimensions, origin, spacing, scalars,
                surface_alpha, resolution, blending, center):
        # Now we can actually construct the visualization
        grid = pyvista.UniformGrid()
        grid.dimensions = dimensions + 1  # inject data on the cells
        grid.origin = origin
        grid.spacing = spacing
        _cell_data(grid)['values'] = scalars

        # Add contour of enclosed volume (use GetOutput instead of
        # GetOutputPort below to avoid updating)
        if surface_alpha > 0 or resolution is not None:
            grid_alg = vtkCellDataToPointData()
            grid_alg.SetInputDataObject(grid)
            grid_alg.SetPassCellData(False)
            grid_alg.Update()
        else:
            grid_alg = None

        if surface_alpha > 0:
            grid_surface = vtkMarchingContourFilter()
            grid_surface.ComputeNormalsOn()
            grid_surface.ComputeScalarsOff()
            grid_surface.SetInputData(grid_alg.GetOutput())
            grid_surface.SetValue(0, 0.1)
            grid_surface.Update()
            grid_mesh = vtkPolyDataMapper()
            grid_mesh.SetInputData(grid_surface.GetOutput())
        else:
            grid_mesh = None

        mapper = vtkSmartVolumeMapper()
        if resolution is None:  # native
            mapper.SetScalarModeToUseCellData()
            mapper.SetInputDataObject(grid)
        else:
            upsampler = vtkImageReslice()
            upsampler.SetInterpolationModeToLinear()  # default anyway
            upsampler.SetOutputSpacing(*([resolution] * 3))
            upsampler.SetInputConnection(grid_alg.GetOutputPort())
            mapper.SetInputConnection(upsampler.GetOutputPort())
        # Additive, AverageIntensity, and Composite might also be reasonable
        remap = dict(composite='Composite', mip='MaximumIntensity')
        getattr(mapper, f'SetBlendModeTo{remap[blending]}')()
        volume_pos = vtkVolume()
        volume_pos.SetMapper(mapper)
        dist = grid.length / (np.mean(grid.dimensions) - 1)
        volume_pos.GetProperty().SetScalarOpacityUnitDistance(dist)
        if center is not None and blending == 'mip':
            # We need to create a minimum intensity projection for the neg half
            mapper_neg = vtkSmartVolumeMapper()
            if resolution is None:  # native
                mapper_neg.SetScalarModeToUseCellData()
                mapper_neg.SetInputDataObject(grid)
            else:
                mapper_neg.SetInputConnection(upsampler.GetOutputPort())
            mapper_neg.SetBlendModeToMinimumIntensity()
            volume_neg = vtkVolume()
            volume_neg.SetMapper(mapper_neg)
            volume_neg.GetProperty().SetScalarOpacityUnitDistance(dist)
        else:
            volume_neg = None
        return grid, grid_mesh, volume_pos, volume_neg

    def _silhouette(self, mesh, color=None, line_width=None, alpha=None,
                    decimate=None):
        mesh = mesh.decimate(decimate) if decimate is not None else mesh
        silhouette_filter = vtkPolyDataSilhouette()
        silhouette_filter.SetInputData(mesh)
        silhouette_filter.SetCamera(self.plotter.renderer.GetActiveCamera())
        silhouette_filter.SetEnableFeatureAngle(0)
        silhouette_mapper = vtkPolyDataMapper()
        silhouette_mapper.SetInputConnection(
            silhouette_filter.GetOutputPort())
        actor, prop = self.plotter.add_actor(
            silhouette_mapper, reset_camera=False, name=None,
            culling=False, pickable=False, render=False)
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
    if 'Normals' not in _point_data(mesh):
        mesh.compute_normals(
            cell_normals=False,
            consistent_normals=False,
            non_manifold_traversal=False,
            inplace=True,
        )


def _add_mesh(plotter, *args, **kwargs):
    """Patch PyVista add_mesh."""
    mesh = kwargs.get('mesh')
    if 'smooth_shading' in kwargs:
        smooth_shading = kwargs.pop('smooth_shading')
    else:
        smooth_shading = True
    # disable rendering pass for add_mesh, render()
    # is called in show()
    if 'render' not in kwargs:
        kwargs['render'] = False
    actor = plotter.add_mesh(*args, **kwargs)
    if smooth_shading and 'Normals' in _point_data(mesh):
        prop = actor.GetProperty()
        prop.SetInterpolationToPhong()
    _hide_testing_actor(actor)
    return actor


def _hide_testing_actor(actor):
    from . import renderer
    if renderer.MNE_3D_BACKEND_TESTING:
        actor.SetVisibility(False)


def _deg2rad(deg):
    return deg * np.pi / 180.


def _rad2deg(rad):
    return rad * 180. / np.pi


def _to_pos(azimuth, elevation):
    theta = azimuth * np.pi / 180.0
    phi = (90.0 - elevation) * np.pi / 180.0
    x = np.sin(theta) * np.sin(phi)
    y = np.cos(phi)
    z = np.cos(theta) * np.sin(phi)
    return x, y, z


def _mat_to_array(vtk_mat):
    e = [vtk_mat.GetElement(i, j) for i in range(4) for j in range(4)]
    arr = np.array(e, dtype=float)
    arr.shape = (4, 4)
    return arr


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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        close_all()
    _FIGURES.clear()


def _get_camera_direction(focalpoint, position):
    x, y, z = position - focalpoint
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def _get_3d_view(figure):
    position = np.array(figure.plotter.camera_position[0])
    focalpoint = np.array(figure.plotter.camera_position[1])
    _, theta, phi = _get_camera_direction(focalpoint, position)
    azimuth, elevation = _rad2deg(phi), _rad2deg(theta)
    return (figure.plotter.camera.GetRoll(),
            figure.plotter.camera.GetDistance(),
            azimuth, elevation, focalpoint)


def _set_3d_view(figure, azimuth=None, elevation=None, focalpoint='auto',
                 distance=None, roll=None, reset_camera=True, rigid=None,
                 update=True):
    rigid = np.eye(4) if rigid is None else rigid
    position = np.array(figure.plotter.camera_position[0])
    bounds = np.array(figure.plotter.renderer.ComputeVisiblePropBounds())
    if reset_camera:
        figure.plotter.reset_camera(render=False)

    # focalpoint: if 'auto', we use the center of mass of the visible
    # bounds, if None, we use the existing camera focal point otherwise
    # we use the values given by the user
    if isinstance(focalpoint, str):
        _check_option('focalpoint', focalpoint, ('auto',),
                      extra='when a string')
        focalpoint = (bounds[1::2] + bounds[::2]) * 0.5
    elif focalpoint is None:
        focalpoint = np.array(figure.plotter.camera_position[1])
    else:
        focalpoint = np.asarray(focalpoint)

    # work in the transformed space
    position = apply_trans(rigid, position)
    focalpoint = apply_trans(rigid, focalpoint)
    _, theta, phi = _get_camera_direction(focalpoint, position)

    if azimuth is not None:
        phi = _deg2rad(azimuth)
    if elevation is not None:
        theta = _deg2rad(elevation)

    # set the distance
    if distance is None:
        distance = max(bounds[1::2] - bounds[::2]) * 2.0

    # Now calculate the view_up vector of the camera.  If the view up is
    # close to the 'z' axis, the view plane normal is parallel to the
    # camera which is unacceptable, so we use a different view up.
    if elevation is None or 5. <= abs(elevation) <= 175.:
        view_up = [0, 0, 1]
    else:
        view_up = [0, 1, 0]

    position = [
        distance * np.cos(phi) * np.sin(theta),
        distance * np.sin(phi) * np.sin(theta),
        distance * np.cos(theta)]

    figure._azimuth = _rad2deg(phi)
    figure._elevation = _rad2deg(theta)

    # restore to the original frame
    rigid = np.linalg.inv(rigid)
    position = apply_trans(rigid, position)
    focalpoint = apply_trans(rigid, focalpoint)
    view_up = apply_trans(rigid, view_up, move=False)
    figure.plotter.camera_position = [
        position, focalpoint, view_up]
    # We need to add the requested roll to the roll dictated by the
    # transformed view_up
    if roll is not None:
        figure.plotter.camera.SetRoll(figure.plotter.camera.GetRoll() + roll)

    if update:
        figure.plotter.update()
        _process_events(figure.plotter)


def _set_3d_title(figure, title, size=16):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        figure.plotter.add_text(title, font_size=size, color='white',
                                name='title')
    figure.plotter.update()
    _process_events(figure.plotter)


def _check_3d_figure(figure):
    _validate_type(figure, PyVistaFigure, 'figure')


def _close_3d_figure(figure):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        # copy the plotter locally because figure.plotter is modified
        plotter = figure.plotter
        # close the window
        plotter.close()  # additional cleaning following signal_close
        _process_events(plotter)
        # free memory and deregister from the scraper
        plotter.deep_clean()  # remove internal references
        _ALL_PLOTTERS.pop(plotter._id_name, None)
        _process_events(plotter)


def _take_3d_screenshot(figure, mode='rgb', filename=None):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        _process_events(figure.plotter)
        return figure.plotter.screenshot(
            transparent_background=(mode == 'rgba'),
            filename=filename)


def _process_events(plotter):
    if hasattr(plotter, 'app'):
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings('ignore', 'constrained_layout')
            plotter.app.processEvents()


def _add_camera_callback(camera, callback):
    camera.AddObserver(vtkCommand.ModifiedEvent, callback)


def _arrow_glyph(grid, factor):
    glyph = vtkGlyphSource2D()
    glyph.SetGlyphTypeToArrow()
    glyph.FilledOff()
    glyph.Update()

    # fix position
    tr = vtkTransform()
    tr.Translate(0.5, 0., 0.)
    trp = vtkTransformPolyDataFilter()
    trp.SetInputConnection(glyph.GetOutputPort())
    trp.SetTransform(tr)
    trp.Update()

    alg = _glyph(
        grid,
        scale_mode='vector',
        scalars=False,
        orient='vec',
        factor=factor,
        geom=trp.GetOutputPort(),
    )
    mapper = vtkDataSetMapper()
    mapper.SetInputConnection(alg.GetOutputPort())
    return mapper


def _glyph(dataset, scale_mode='scalar', orient=True, scalars=True, factor=1.0,
           geom=None, tolerance=0.0, absolute=False, clamping=False, rng=None):
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
    if scale_mode == 'scalar':
        alg.SetScaleModeToScaleByScalar()
    elif scale_mode == 'vector':
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
    try:
        from pyvista import global_theme
    except Exception:  # workaround for older PyVista
        from pyvista import rcParams
        depth_peeling = rcParams['depth_peeling']
    else:
        depth_peeling = global_theme.depth_peeling
    depth_peeling_enabled = depth_peeling["enabled"]
    depth_peeling["enabled"] = False
    try:
        yield
    finally:
        depth_peeling["enabled"] = depth_peeling_enabled


def _is_mesa(plotter):
    # MESA (could use GPUInfo / _get_gpu_info here, but it takes
    # > 700 ms to make a new window + report capabilities!)
    # CircleCI's is: "Mesa 20.0.8 via llvmpipe (LLVM 10.0.0, 256 bits)"
    gpu_info = plotter.ren_win.ReportCapabilities()
    gpu_info = re.findall("OpenGL renderer string:(.+)\n", gpu_info)
    return ' mesa ' in ' '.join(gpu_info).lower().split()
