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
from distutils.version import LooseVersion
import os
import sys
import warnings

import numpy as np
import vtk

from .base_renderer import _BaseRenderer
from ._utils import (_get_colormap_from_array, _alpha_blend_background,
                     ALLOWED_QUIVER_MODES)
from ...fixes import _get_args
from ...utils import copy_base_doc_to_subclass_doc, _check_option
from ...externals.decorator import decorator


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pyvista
    from pyvista import Plotter, PolyData, Line, close_all, UnstructuredGrid
    try:
        from pyvistaqt import BackgroundPlotter  # noqa
    except ImportError:
        from pyvista import BackgroundPlotter
    from pyvista.utilities import try_callback
    from pyvista.plotting.plotting import _ALL_PLOTTERS
VTK9 = LooseVersion(getattr(vtk, 'VTK_VERSION', '9.0')) >= LooseVersion('9.0')


_FIGURES = dict()


class _Figure(object):
    def __init__(self, plotter=None,
                 plotter_class=None,
                 display=None,
                 show=False,
                 title='PyVista Scene',
                 size=(600, 600),
                 shape=(1, 1),
                 background_color='black',
                 smooth_shading=True,
                 off_screen=False,
                 notebook=False):
        self.plotter = plotter
        self.plotter_class = plotter_class
        self.display = display
        self.background_color = background_color
        self.smooth_shading = smooth_shading
        self.notebook = notebook

        self.store = dict()
        self.store['show'] = show
        self.store['title'] = title
        self.store['window_size'] = size
        self.store['shape'] = shape
        self.store['off_screen'] = off_screen
        self.store['border'] = False
        self.store['auto_update'] = False
        # multi_samples > 1 is broken on macOS + Intel Iris + volume rendering
        self.store['multi_samples'] = 1 if sys.platform == 'darwin' else 4

    def build(self):
        if self.plotter_class is None:
            self.plotter_class = BackgroundPlotter
        if self.notebook:
            self.plotter_class = Plotter

        if self.plotter_class is Plotter:
            self.store.pop('show', None)
            self.store.pop('title', None)
            self.store.pop('auto_update', None)

        if self.plotter is None:
            if self.plotter_class is BackgroundPlotter:
                from PyQt5.QtWidgets import QApplication
                app = QApplication.instance()
                if app is None:
                    app = QApplication(["MNE"])
                self.store['app'] = app
            plotter = self.plotter_class(**self.store)
            plotter.background_color = self.background_color
            self.plotter = plotter
            if self.plotter_class is BackgroundPlotter and \
                    hasattr(BackgroundPlotter, 'set_icon'):
                _init_resources()
                _process_events(plotter)
                plotter.set_icon(":/mne-icon.png")
        _process_events(self.plotter)
        _process_events(self.plotter)
        return self.plotter

    def is_active(self):
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

    def __init__(self, xy=None, pts=None):
        """Store input projection information into attributes."""
        self.xy = xy
        self.pts = pts

    def visible(self, state):
        """Modify visibility attribute of the sensors."""
        self.pts.SetVisibility(state)


def _enable_aa(figure, plotter):
    """Enable it everywhere except Azure."""
    # XXX for some reason doing this on Azure causes access violations:
    #     ##[error]Cmd.exe exited with code '-1073741819'
    # So for now don't use it there. Maybe has to do with setting these
    # before the window has actually been made "active"...?
    # For Mayavi we have an "on activated" event or so, we should look into
    # using this for Azure at some point, too.
    if os.getenv('AZURE_CI_WINDOWS', 'false').lower() == 'true':
        return
    if figure.is_active():
        if sys.platform != 'darwin':
            plotter.enable_anti_aliasing()
        plotter.ren_win.LineSmoothingOn()


@copy_base_doc_to_subclass_doc
class _Renderer(_BaseRenderer):
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
                 notebook=None, smooth_shading=True):
        from .renderer import MNE_3D_BACKEND_TESTING
        from .._3d import _get_3d_option
        figure = _Figure(show=show, title=name, size=size, shape=shape,
                         background_color=bgcolor, notebook=notebook,
                         smooth_shading=smooth_shading)
        self.font_family = "arial"
        self.tube_n_sides = 20
        self.shape = shape
        antialias = _get_3d_option('antialias')
        self.antialias = antialias and not MNE_3D_BACKEND_TESTING
        if isinstance(fig, int):
            saved_fig = _FIGURES.get(fig)
            # Restore only active plotter
            if saved_fig is not None and saved_fig.is_active():
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
            if MNE_3D_BACKEND_TESTING:
                self.tube_n_sides = 3
                # smooth_shading=True fails on MacOS CIs
                self.figure.smooth_shading = False
            with _disabled_depth_peeling():
                self.plotter = self.figure.build()
            self.plotter.hide_axes()
            if hasattr(self.plotter, "default_camera_tool_bar"):
                self.plotter.default_camera_tool_bar.close()
            if hasattr(self.plotter, "saved_cameras_tool_bar"):
                self.plotter.saved_cameras_tool_bar.close()
            if self.antialias:
                _enable_aa(self.figure, self.plotter)

        # FIX: https://github.com/pyvista/pyvistaqt/pull/68
        if LooseVersion(pyvista.__version__) >= '0.27.0':
            if not hasattr(self.plotter, "iren"):
                self.plotter.iren = None

        self.update_lighting()

    @contextmanager
    def ensure_minimum_sizes(self):
        sz = self.figure.store['window_size']
        # plotter:            pyvista.plotting.qt_plotting.BackgroundPlotter
        # plotter.interactor: vtk.qt.QVTKRenderWindowInteractor.QVTKRenderWindowInteractor -> QWidget  # noqa
        # plotter.app_window: pyvista.plotting.qt_plotting.MainWindow -> QMainWindow  # noqa
        # plotter.frame:      QFrame with QVBoxLayout with plotter.interactor as centralWidget  # noqa
        # plotter.ren_win:    vtkXOpenGLRenderWindow
        self.plotter.interactor.setMinimumSize(*sz)
        try:
            yield  # show
        finally:
            # 1. Process events
            _process_events(self.plotter)
            _process_events(self.plotter)
            # 2. Get the window size that accommodates the size
            sz = self.plotter.app_window.size()
            # 3. Call app_window.setBaseSize and resize (in pyvistaqt)
            self.plotter.window_size = (sz.width(), sz.height())
            # 4. Undo the min size setting and process events
            self.plotter.interactor.setMinimumSize(0, 0)
            _process_events(self.plotter)
            _process_events(self.plotter)
            # 5. Resize the window (again!) to the correct size
            #    (not sure why, but this is required on macOS at least)
            self.plotter.window_size = (sz.width(), sz.height())
            _process_events(self.plotter)
            _process_events(self.plotter)

    def subplot(self, x, y):
        x = np.max([0, np.min([x, self.shape[0] - 1])])
        y = np.max([0, np.min([y, self.shape[1] - 1])])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.plotter.subplot(x, y)
            if self.antialias:
                _enable_aa(self.figure, self.plotter)

    def scene(self):
        return self.figure

    def _orient_lights(self):
        lights = list(self.plotter.renderer.GetLights())
        lights.pop(0)  # unused headlight
        lights[0].SetPosition(_to_pos(45.0, -45.0))
        lights[1].SetPosition(_to_pos(-30.0, 60.0))
        lights[2].SetPosition(_to_pos(-30.0, -60.0))

    def update_lighting(self):
        # Inspired from Mayavi's version of Raymond Maple 3-lights illumination
        lights = list(self.plotter.renderer.GetLights())
        headlight = lights.pop(0)
        headlight.SetSwitch(False)
        for i in range(len(lights)):
            if i < 3:
                lights[i].SetSwitch(True)
                lights[i].SetIntensity(1.0)
                lights[i].SetColor(1.0, 1.0, 1.0)
            else:
                lights[i].SetSwitch(False)
                lights[i].SetPosition(_to_pos(0.0, 0.0))
                lights[i].SetIntensity(1.0)
                lights[i].SetColor(1.0, 1.0, 1.0)

        lights[0].SetPosition(_to_pos(45.0, 45.0))
        lights[1].SetPosition(_to_pos(-30.0, -60.0))
        lights[1].SetIntensity(0.6)
        lights[2].SetPosition(_to_pos(-30.0, 60.0))
        lights[2].SetIntensity(0.5)

    def set_interaction(self, interaction):
        if not hasattr(self.plotter, "iren") or self.plotter.iren is None:
            return
        if interaction == "rubber_band_2d":
            for renderer in self.plotter.renderers:
                renderer.enable_parallel_projection()
            if hasattr(self.plotter, 'enable_rubber_band_2d_style'):
                self.plotter.enable_rubber_band_2d_style()
            else:
                style = vtk.vtkInteractorStyleRubberBand2D()
                self.plotter.interactor.SetInteractorStyle(style)
        else:
            for renderer in self.plotter.renderers:
                renderer.disable_parallel_projection()
            getattr(self.plotter, f'enable_{interaction}_style')()

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
                mesh.point_arrays["Normals"] = normals
                mesh.GetPointData().SetActiveNormals("Normals")
            else:
                _compute_normals(mesh)
            if 'rgba' in kwargs:
                rgba = kwargs["rgba"]
                kwargs.pop('rgba')
            actor = _add_mesh(
                plotter=self.plotter,
                mesh=mesh, color=color, scalars=scalars,
                rgba=rgba, opacity=opacity, cmap=colormap,
                backface_culling=backface_culling,
                rng=[vmin, vmax], show_scalar_bar=False,
                smooth_shading=self.figure.smooth_shading,
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
            vertices = np.c_[x, y, z]
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
            mesh.point_arrays['scalars'] = scalars
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
                smooth_shading=self.figure.smooth_shading
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
            mesh.point_arrays['scalars'] = scalars
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
        factor = 1.0 if radius is not None else scale
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            sphere = vtk.vtkSphereSource()
            sphere.SetThetaResolution(resolution)
            sphere.SetPhiResolution(resolution)
            if radius is not None:
                sphere.SetRadius(radius)
            sphere.Update()
            geom = sphere.GetOutput()
            mesh = PolyData(np.array(center))
            glyph = mesh.glyph(orient=False, scale=False,
                               factor=factor, geom=geom)
            actor = _add_mesh(
                self.plotter,
                mesh=glyph, color=color, opacity=opacity,
                backface_culling=backface_culling,
                smooth_shading=self.figure.smooth_shading
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
                    line.point_arrays['scalars'] = scalars[0, :]
                    scalars = 'scalars'
                    color = None
                else:
                    scalars = None
                tube = line.tube(radius, n_sides=self.tube_n_sides)
                _add_mesh(
                    plotter=self.plotter,
                    mesh=tube,
                    scalars=scalars,
                    flip_scalars=reverse_lut,
                    rng=[vmin, vmax],
                    color=color,
                    show_scalar_bar=False,
                    cmap=cmap,
                    smooth_shading=self.figure.smooth_shading,
                )
        return tube

    def quiver3d(self, x, y, z, u, v, w, color, scale, mode, resolution=8,
                 glyph_height=None, glyph_center=None, glyph_resolution=None,
                 opacity=1.0, scale_mode='none', scalars=None,
                 backface_culling=False, line_width=2., name=None,
                 glyph_width=None, glyph_depth=None,
                 solid_transform=None):
        _check_option('mode', mode, ALLOWED_QUIVER_MODES)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            factor = scale
            vectors = np.c_[u, v, w]
            points = np.vstack(np.c_[x, y, z])
            n_points = len(points)
            cell_type = np.full(n_points, vtk.VTK_VERTEX)
            cells = np.c_[np.full(n_points, 1), range(n_points)]
            args = (cells, cell_type, points)
            if not VTK9:
                args = (np.arange(n_points) * 3,) + args
            grid = UnstructuredGrid(*args)
            grid.point_arrays['vec'] = vectors
            if scale_mode == 'scalar':
                grid.point_arrays['mag'] = np.array(scalars)
                scale = 'mag'
            else:
                scale = False
            if mode == '2darrow':
                return _arrow_glyph(grid, factor), grid
            elif mode == 'arrow':
                alg = _glyph(
                    grid,
                    orient='vec',
                    scalars=scale,
                    factor=factor
                )
                mesh = pyvista.wrap(alg.GetOutput())
            else:
                tr = None
                if mode == 'cone':
                    glyph = vtk.vtkConeSource()
                    glyph.SetCenter(0.5, 0, 0)
                    glyph.SetRadius(0.15)
                elif mode == 'cylinder':
                    glyph = vtk.vtkCylinderSource()
                    glyph.SetRadius(0.15)
                elif mode == 'oct':
                    glyph = vtk.vtkPlatonicSolidSource()
                    glyph.SetSolidTypeToOctahedron()
                else:
                    assert mode == 'sphere', mode  # guaranteed above
                    glyph = vtk.vtkSphereSource()
                if mode == 'cylinder':
                    if glyph_height is not None:
                        glyph.SetHeight(glyph_height)
                    if glyph_center is not None:
                        glyph.SetCenter(glyph_center)
                    if glyph_resolution is not None:
                        glyph.SetResolution(glyph_resolution)
                    tr = vtk.vtkTransform()
                    tr.RotateWXYZ(90, 0, 0, 1)
                elif mode == 'oct':
                    if solid_transform is not None:
                        assert solid_transform.shape == (4, 4)
                        tr = vtk.vtkTransform()
                        tr.SetMatrix(
                            solid_transform.astype(np.float64).ravel())
                if tr is not None:
                    # fix orientation
                    glyph.Update()
                    trp = vtk.vtkTransformPolyDataFilter()
                    trp.SetInputData(glyph.GetOutput())
                    trp.SetTransform(tr)
                    glyph = trp
                glyph.Update()
                geom = glyph.GetOutput()
                mesh = grid.glyph(orient='vec', scale=scale, factor=factor,
                                  geom=geom)
            _add_mesh(
                self.plotter,
                mesh=mesh,
                color=color,
                opacity=opacity,
                backface_culling=backface_culling
            )

    def text2d(self, x_window, y_window, text, size=14, color='white',
               justification=None):
        size = 14 if size is None else size
        position = (x_window, y_window)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            actor = self.plotter.add_text(text, position=position,
                                          font_size=size,
                                          font=self.font_family,
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
        return actor

    def text3d(self, x, y, z, text, scale, color='white'):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            kwargs = dict(
                points=[x, y, z],
                labels=[text],
                point_size=scale,
                text_color=color,
                font_family=self.font_family,
                name=text,
                shape_opacity=0,
            )
            if 'always_visible' in _get_args(self.plotter.add_point_labels):
                kwargs['always_visible'] = True
            self.plotter.add_point_labels(**kwargs)

    def scalarbar(self, source, color="white", title=None, n_labels=4,
                  bgcolor=None, **extra_kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            kwargs = dict(color=color, title=title, n_labels=n_labels,
                          use_opacity=False, n_colors=256, position_x=0.15,
                          position_y=0.05, width=0.7, shadow=False, bold=True,
                          label_font_size=22, font_family=self.font_family,
                          background_color=bgcolor)
            kwargs.update(extra_kwargs)
            self.plotter.add_scalar_bar(**kwargs)

    def show(self):
        self.figure.display = self.plotter.show()
        if hasattr(self.plotter, "app_window"):
            with self.ensure_minimum_sizes():
                self.plotter.app_window.show()
        return self.scene()

    def close(self):
        _close_3d_figure(figure=self.figure)

    def set_camera(self, azimuth=None, elevation=None, distance=None,
                   focalpoint=None, roll=None, reset_camera=True):
        _set_3d_view(self.figure, azimuth=azimuth, elevation=elevation,
                     distance=distance, focalpoint=focalpoint, roll=roll,
                     reset_camera=reset_camera)

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

        return _Projection(xy=xy, pts=pts)

    def enable_depth_peeling(self):
        if not self.figure.store['off_screen']:
            for renderer in self.plotter.renderers:
                renderer.enable_depth_peeling()

    def remove_mesh(self, mesh_data):
        actor, _ = mesh_data
        self.plotter.remove_actor(actor)


def _create_actor(mapper=None):
    """Create a vtkActor."""
    actor = vtk.vtkActor()
    if mapper is not None:
        actor.SetMapper(mapper)
    return actor


def _compute_normals(mesh):
    """Patch PyVista compute_normals."""
    if 'Normals' not in mesh.point_arrays:
        mesh.compute_normals(
            cell_normals=False,
            consistent_normals=False,
            non_manifold_traversal=False,
            inplace=True,
        )


def _add_mesh(plotter, *args, **kwargs):
    """Patch PyVista add_mesh."""
    from . import renderer
    _process_events(plotter)
    mesh = kwargs.get('mesh')
    if 'smooth_shading' in kwargs:
        smooth_shading = kwargs.pop('smooth_shading')
    else:
        smooth_shading = True
    # disable rendering pass for add_mesh, render()
    # is called in show()
    if 'render' not in kwargs and 'render' in _get_args(plotter.add_mesh):
        kwargs['render'] = False
    actor = plotter.add_mesh(*args, **kwargs)
    if smooth_shading and 'Normals' in mesh.point_arrays:
        prop = actor.GetProperty()
        prop.SetInterpolationToPhong()
    if renderer.MNE_3D_BACKEND_TESTING:
        actor.SetVisibility(False)
    return actor


def _deg2rad(deg):
    return deg * np.pi / 180.


def _rad2deg(rad):
    return rad * 180. / np.pi


def _to_pos(elevation, azimuth):
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
    size = plotter.window_size
    xyz = np.column_stack([xyz, np.ones(xyz.shape[0])])

    # Transform points into 'unnormalized' view coordinates
    comb_trans_mat = _get_world_to_view_matrix(plotter)
    view_coords = np.dot(comb_trans_mat, xyz.T).T

    # Divide through by the fourth element for normalized view coords
    norm_view_coords = view_coords / (view_coords[:, 3].reshape(-1, 1))

    # Transform from normalized view coordinates to display coordinates.
    view_to_disp_mat = _get_view_to_display_matrix(size)
    xy = np.dot(view_to_disp_mat, norm_view_coords.T).T

    # Pull the first two columns since they're meaningful for 2d plotting
    xy = xy[:, :2]
    return xy


def _get_world_to_view_matrix(plotter):
    cam = plotter.renderer.camera

    scene_size = plotter.window_size
    clip_range = cam.GetClippingRange()
    aspect_ratio = float(scene_size[0]) / scene_size[1]

    vtk_comb_trans_mat = cam.GetCompositeProjectionTransformMatrix(
        aspect_ratio, clip_range[0], clip_range[1])
    vtk_comb_trans_mat = _mat_to_array(vtk_comb_trans_mat)
    return vtk_comb_trans_mat


def _get_view_to_display_matrix(size):
    x, y = size
    view_to_disp_mat = np.array([[x / 2.0,       0.,   0.,   x / 2.0],
                                 [0.,      -y / 2.0,   0.,   y / 2.0],
                                 [0.,            0.,   1.,        0.],
                                 [0.,            0.,   0.,        1.]])
    return view_to_disp_mat


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
    return r, theta, phi, focalpoint


def _set_3d_view(figure, azimuth, elevation, focalpoint, distance, roll=None,
                 reset_camera=True):
    position = np.array(figure.plotter.camera_position[0])
    if reset_camera:
        figure.plotter.reset_camera()
    if focalpoint is None:
        focalpoint = np.array(figure.plotter.camera_position[1])
    r, theta, phi, fp = _get_camera_direction(focalpoint, position)

    if azimuth is not None:
        phi = _deg2rad(azimuth)
    if elevation is not None:
        theta = _deg2rad(elevation)

    # set the distance
    renderer = figure.plotter.renderer
    bounds = np.array(renderer.ComputeVisiblePropBounds())
    if distance is None:
        distance = max(bounds[1::2] - bounds[::2]) * 2.0

    if focalpoint is not None:
        focalpoint = np.asarray(focalpoint)
    else:
        focalpoint = (bounds[1::2] + bounds[::2]) * 0.5

    # Now calculate the view_up vector of the camera.  If the view up is
    # close to the 'z' axis, the view plane normal is parallel to the
    # camera which is unacceptable, so we use a different view up.
    if elevation is None or 5. <= abs(elevation) <= 175.:
        view_up = [0, 0, 1]
    else:
        view_up = [np.sin(phi), np.cos(phi), 0]

    position = [
        distance * np.cos(phi) * np.sin(theta),
        distance * np.sin(phi) * np.sin(theta),
        distance * np.cos(theta)]
    figure.plotter.camera_position = [
        position, focalpoint, view_up]
    if roll is not None:
        figure.plotter.camera.SetRoll(roll)

    figure.plotter.renderer._azimuth = azimuth
    figure.plotter.renderer._elevation = elevation
    figure.plotter.renderer._distance = distance
    figure.plotter.renderer._roll = roll
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
    if not isinstance(figure, _Figure):
        raise TypeError('figure must be an instance of _Figure.')


def _close_3d_figure(figure):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        # close the window
        figure.plotter.close()
        _process_events(figure.plotter)
        # free memory and deregister from the scraper
        figure.plotter.deep_clean()
        del _ALL_PLOTTERS[figure.plotter._id_name]
        _process_events(figure.plotter)


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


def _set_colormap_range(actor, ctable, scalar_bar, rng=None,
                        background_color=None):
    from vtk.util.numpy_support import numpy_to_vtk
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
        lut.SetTable(numpy_to_vtk(ctable, array_type=vtk.VTK_UNSIGNED_CHAR))
        lut.SetRange(*rng)


def _set_volume_range(volume, ctable, alpha, scalar_bar, rng):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    color_tf = vtk.vtkColorTransferFunction()
    opacity_tf = vtk.vtkPiecewiseFunction()
    for loc, color in zip(np.linspace(*rng, num=len(ctable)), ctable):
        color_tf.AddRGBPoint(loc, *(color[:-1] / 255.))
        opacity_tf.AddPoint(loc, color[-1] * alpha / 255.)
    color_tf.ClampingOn()
    opacity_tf.ClampingOn()
    volume.GetProperty().SetColor(color_tf)
    volume.GetProperty().SetScalarOpacity(opacity_tf)
    if scalar_bar is not None:
        lut = vtk.vtkLookupTable()
        lut.SetRange(*rng)
        lut.SetTable(numpy_to_vtk(ctable))
        scalar_bar.SetLookupTable(lut)


def _set_mesh_scalars(mesh, scalars, name):
    # Catch:  FutureWarning: Conversion of the second argument of
    # issubdtype from `complex` to `np.complexfloating` is deprecated.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        mesh.point_arrays[name] = scalars


def _update_slider_callback(slider, callback, event_type):
    _check_option('event_type', event_type, ['start', 'end', 'always'])

    def _the_callback(widget, event):
        value = widget.GetRepresentation().GetValue()
        if hasattr(callback, '__call__'):
            try_callback(callback, value)
        return

    if event_type == 'start':
        event = vtk.vtkCommand.StartInteractionEvent
    elif event_type == 'end':
        event = vtk.vtkCommand.EndInteractionEvent
    else:
        assert event_type == 'always', event_type
        event = vtk.vtkCommand.InteractionEvent

    slider.RemoveObserver(event)
    slider.AddObserver(event, _the_callback)


def _add_camera_callback(camera, callback):
    camera.AddObserver(vtk.vtkCommand.ModifiedEvent, callback)


def _update_picking_callback(plotter,
                             on_mouse_move,
                             on_button_press,
                             on_button_release,
                             on_pick):
    interactor = plotter.iren
    interactor.AddObserver(
        vtk.vtkCommand.RenderEvent,
        on_mouse_move
    )
    interactor.AddObserver(
        vtk.vtkCommand.LeftButtonPressEvent,
        on_button_press
    )
    interactor.AddObserver(
        vtk.vtkCommand.EndInteractionEvent,
        on_button_release
    )
    picker = vtk.vtkCellPicker()
    picker.AddObserver(
        vtk.vtkCommand.EndPickEvent,
        on_pick
    )
    picker.SetVolumeOpacityIsovalue(0.)
    plotter.picker = picker


def _remove_picking_callback(interactor, picker):
    interactor.RemoveObservers(vtk.vtkCommand.RenderEvent)
    interactor.RemoveObservers(vtk.vtkCommand.LeftButtonPressEvent)
    interactor.RemoveObservers(vtk.vtkCommand.EndInteractionEvent)
    picker.RemoveObservers(vtk.vtkCommand.EndPickEvent)


def _arrow_glyph(grid, factor):
    glyph = vtk.vtkGlyphSource2D()
    glyph.SetGlyphTypeToArrow()
    glyph.FilledOff()
    glyph.Update()

    # fix position
    tr = vtk.vtkTransform()
    tr.Translate(0.5, 0., 0.)
    trp = vtk.vtkTransformPolyDataFilter()
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
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(alg.GetOutputPort())
    return mapper


def _glyph(dataset, scale_mode='scalar', orient=True, scalars=True, factor=1.0,
           geom=None, tolerance=0.0, absolute=False, clamping=False, rng=None):
    if geom is None:
        arrow = vtk.vtkArrowSource()
        arrow.Update()
        geom = arrow.GetOutputPort()
    alg = vtk.vtkGlyph3D()
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


def _sphere(plotter, center, color, radius):
    sphere = vtk.vtkSphereSource()
    sphere.SetThetaResolution(8)
    sphere.SetPhiResolution(8)
    sphere.SetRadius(radius)
    sphere.SetCenter(center)
    sphere.Update()
    mesh = pyvista.wrap(sphere.GetOutput())
    actor = _add_mesh(
        plotter,
        mesh=mesh,
        color=color
    )
    return actor, mesh


def _volume(dimensions, origin, spacing, scalars,
            surface_alpha, resolution, blending, center):
    # Now we can actually construct the visualization
    grid = pyvista.UniformGrid()
    grid.dimensions = dimensions + 1  # inject data on the cells
    grid.origin = origin
    grid.spacing = spacing
    grid.cell_arrays['values'] = scalars

    # Add contour of enclosed volume (use GetOutput instead of
    # GetOutputPort below to avoid updating)
    grid_alg = vtk.vtkCellDataToPointData()
    grid_alg.SetInputDataObject(grid)
    grid_alg.SetPassCellData(False)
    grid_alg.Update()

    if surface_alpha > 0:
        grid_surface = vtk.vtkMarchingContourFilter()
        grid_surface.ComputeNormalsOn()
        grid_surface.ComputeScalarsOff()
        grid_surface.SetInputData(grid_alg.GetOutput())
        grid_surface.SetValue(0, 0.1)
        grid_surface.Update()
        grid_mesh = vtk.vtkPolyDataMapper()
        grid_mesh.SetInputData(grid_surface.GetOutput())
    else:
        grid_mesh = None

    mapper = vtk.vtkSmartVolumeMapper()
    if resolution is None:  # native
        mapper.SetScalarModeToUseCellData()
        mapper.SetInputDataObject(grid)
    else:
        upsampler = vtk.vtkImageReslice()
        upsampler.SetInterpolationModeToLinear()  # default anyway
        upsampler.SetOutputSpacing(*([resolution] * 3))
        upsampler.SetInputConnection(grid_alg.GetOutputPort())
        mapper.SetInputConnection(upsampler.GetOutputPort())
    # Additive, AverageIntensity, and Composite might also be reasonable
    remap = dict(composite='Composite', mip='MaximumIntensity')
    getattr(mapper, f'SetBlendModeTo{remap[blending]}')()
    volume_pos = vtk.vtkVolume()
    volume_pos.SetMapper(mapper)
    dist = grid.length / (np.mean(grid.dimensions) - 1)
    volume_pos.GetProperty().SetScalarOpacityUnitDistance(dist)
    if center is not None and blending == 'mip':
        # We need to create a minimum intensity projection for the neg half
        mapper_neg = vtk.vtkSmartVolumeMapper()
        if resolution is None:  # native
            mapper_neg.SetScalarModeToUseCellData()
            mapper_neg.SetInputDataObject(grid)
        else:
            mapper_neg.SetInputConnection(upsampler.GetOutputPort())
        mapper_neg.SetBlendModeToMinimumIntensity()
        volume_neg = vtk.vtkVolume()
        volume_neg.SetMapper(mapper_neg)
        volume_neg.GetProperty().SetScalarOpacityUnitDistance(dist)
    else:
        volume_neg = None
    return grid, grid_mesh, volume_pos, volume_neg


def _require_minimum_version(version_required):
    from distutils.version import LooseVersion
    version = LooseVersion(pyvista.__version__)
    if version < version_required:
        raise ImportError('pyvista>={} is required for this module but the '
                          'version found is {}'.format(version_required,
                                                       version))


@contextmanager
def _testing_context(interactive):
    from . import renderer
    orig_offscreen = pyvista.OFF_SCREEN
    orig_testing = renderer.MNE_3D_BACKEND_TESTING
    orig_interactive = renderer.MNE_3D_BACKEND_INTERACTIVE
    renderer.MNE_3D_BACKEND_TESTING = True
    if interactive:
        pyvista.OFF_SCREEN = False
        renderer.MNE_3D_BACKEND_INTERACTIVE = True
    else:
        pyvista.OFF_SCREEN = True
        renderer.MNE_3D_BACKEND_INTERACTIVE = False
    try:
        yield
    finally:
        pyvista.OFF_SCREEN = orig_offscreen
        renderer.MNE_3D_BACKEND_TESTING = orig_testing
        renderer.MNE_3D_BACKEND_INTERACTIVE = orig_interactive


@contextmanager
def _disabled_depth_peeling():
    from pyvista import rcParams
    depth_peeling_enabled = rcParams["depth_peeling"]["enabled"]
    rcParams["depth_peeling"]["enabled"] = False
    try:
        yield
    finally:
        rcParams["depth_peeling"]["enabled"] = depth_peeling_enabled


@contextmanager
def _disabled_interaction(renderer):
    plotter = renderer.plotter
    if not plotter.renderer.GetInteractive():
        yield
    else:
        plotter.disable()
        try:
            yield
        finally:
            plotter.enable()


@decorator
def run_once(fun, *args, **kwargs):
    """Run the function only once."""
    if not hasattr(fun, "_has_run"):
        fun._has_run = True
        return fun(*args, **kwargs)


@run_once
def _init_resources():
    from ...icons import resources
    resources.qInitResources()
