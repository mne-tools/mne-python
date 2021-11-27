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
import platform
import sys
import warnings

import numpy as np
import vtk

from ._abstract import _AbstractRenderer
from ._utils import (_get_colormap_from_array, _alpha_blend_background,
                     ALLOWED_QUIVER_MODES, _init_qt_resources)
from ...fixes import _get_args, _point_data, _cell_data
from ...transforms import apply_trans
from ...utils import copy_base_doc_to_subclass_doc, _check_option


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pyvista
    from pyvista import Plotter, PolyData, Line, close_all, UnstructuredGrid
    try:
        from pyvistaqt import BackgroundPlotter  # noqa
    except ImportError:
        from pyvista import BackgroundPlotter
    from pyvista.plotting.plotting import _ALL_PLOTTERS
VTK9 = LooseVersion(getattr(vtk, 'VTK_VERSION', '9.0')) >= LooseVersion('9.0')


_FIGURES = dict()


class _Figure(object):
    def __init__(self,
                 plotter=None,
                 show=False,
                 title='PyVista Scene',
                 size=(600, 600),
                 shape=(1, 1),
                 background_color='black',
                 smooth_shading=True,
                 off_screen=False,
                 notebook=False):
        self.plotter = plotter
        self.display = None
        self.background_color = background_color
        self.smooth_shading = smooth_shading
        self.notebook = notebook

        self.store = dict()
        self.store['window_size'] = size
        self.store['shape'] = shape
        self.store['off_screen'] = off_screen
        self.store['border'] = False
        # multi_samples > 1 is broken on macOS + Intel Iris + volume rendering
        self.store['multi_samples'] = 1 if sys.platform == 'darwin' else 4

        if not self.notebook:
            self.store['show'] = show
            self.store['title'] = title
            self.store['auto_update'] = False
            self.store['menu_bar'] = False
            self.store['toolbar'] = False
            self.store['update_app_icon'] = False

        self._nrows, self._ncols = self.store['shape']
        self._azimuth = self._elevation = None

    def build(self):
        if self.notebook:
            plotter_class = Plotter
        else:
            plotter_class = BackgroundPlotter

        if self.plotter is None:
            if not self.notebook:
                from PyQt5.QtWidgets import QApplication
                app = QApplication.instance()
                if app is None:
                    app = QApplication(["MNE"])
                self.store['app'] = app
            plotter = plotter_class(**self.store)
            plotter.background_color = self.background_color
            self.plotter = plotter
            if not self.notebook and hasattr(plotter_class, 'set_icon'):
                _init_qt_resources()
                _process_events(plotter)
                kind = 'bigsur-' if platform.mac_ver()[0] >= '10.16' else ''
                plotter.set_icon(f":/mne-{kind}icon.png")
        if self.plotter.iren is not None:
            self.plotter.iren.initialize()
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
                 notebook=None, smooth_shading=True):
        from .renderer import MNE_3D_BACKEND_TESTING
        from .._3d import _get_3d_option
        figure = _Figure(show=show, title=name, size=size, shape=shape,
                         background_color=bgcolor, notebook=notebook,
                         smooth_shading=smooth_shading)
        self.font_family = "arial"
        self.tube_n_sides = 20
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
            self._hide_axes()
            self._enable_aa()

        # FIX: https://github.com/pyvista/pyvistaqt/pull/68
        if LooseVersion(pyvista.__version__) >= '0.27.0':
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
                style = vtk.vtkInteractorStyleRubberBand2D()
                self.plotter.interactor.SetInteractorStyle(style)
        else:
            for renderer in self._all_renderers:
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
                _point_data(mesh)["Normals"] = normals
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
        factor = 1.0 if radius is not None else scale
        center = np.array(center, dtype=float)
        if len(center) == 0:
            return None, None
        _check_option('center.ndim', center.ndim, (1, 2))
        _check_option('center.shape[-1]', center.shape[-1], (3,))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            sphere = vtk.vtkSphereSource()
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
                    smooth_shading=self.figure.smooth_shading,
                )
        return actor, tube

    def quiver3d(self, x, y, z, u, v, w, color, scale, mode, resolution=8,
                 glyph_height=None, glyph_center=None, glyph_resolution=None,
                 opacity=1.0, scale_mode='none', scalars=None,
                 backface_culling=False, line_width=2., name=None,
                 glyph_width=None, glyph_depth=None, glyph_radius=0.15,
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
            _point_data(grid)['vec'] = vectors
            if scale_mode == 'scalar':
                _point_data(grid)['mag'] = np.array(scalars)
                scale = 'mag'
            elif scale_mode == 'vector':
                scale = True
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
                    if glyph_radius is not None:
                        glyph.SetRadius(glyph_radius)
                elif mode == 'cylinder':
                    glyph = vtk.vtkCylinderSource()
                    if glyph_radius is not None:
                        glyph.SetRadius(glyph_radius)
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
            actor = _add_mesh(
                self.plotter,
                mesh=mesh,
                color=color,
                opacity=opacity,
                backface_culling=backface_culling
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
            actor = self.plotter.add_point_labels(**kwargs)
        _hide_testing_actor(actor)
        return actor

    def scalarbar(self, source, color="white", title=None, n_labels=4,
                  bgcolor=None, **extra_kwargs):
        if isinstance(source, vtk.vtkMapper):
            mapper = source
        elif isinstance(source, vtk.vtkActor):
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

    def enable_depth_peeling(self):
        if not self.figure.store['off_screen']:
            for renderer in self._all_renderers:
                renderer.enable_depth_peeling()

    def _enable_aa(self):
        """Enable it everywhere except Azure."""
        if not self.antialias:
            return
        # XXX for some reason doing this on Azure causes access violations:
        #     ##[error]Cmd.exe exited with code '-1073741819'
        # So for now don't use it there. Maybe has to do with setting these
        # before the window has actually been made "active"...?
        # For Mayavi we have an "on activated" event or so, we should look into
        # using this for Azure at some point, too.
        if os.getenv('AZURE_CI_WINDOWS', 'false').lower() == 'true':
            return
        if self.figure.is_active():
            if sys.platform != 'darwin':
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
        actor = vtk.vtkActor()
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
        add_obs(vtk.vtkCommand.RenderEvent, on_mouse_move)
        add_obs(vtk.vtkCommand.LeftButtonPressEvent, on_button_press)
        add_obs(vtk.vtkCommand.EndInteractionEvent, on_button_release)
        self.plotter.picker = vtk.vtkCellPicker()
        self.plotter.picker.AddObserver(
            vtk.vtkCommand.EndPickEvent,
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
            lut.SetTable(numpy_to_vtk(ctable,
                                      array_type=vtk.VTK_UNSIGNED_CHAR))
            lut.SetRange(*rng)

    def _set_volume_range(self, volume, ctable, alpha, scalar_bar, rng):
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk
        color_tf = vtk.vtkColorTransferFunction()
        opacity_tf = vtk.vtkPiecewiseFunction()
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
            lut = vtk.vtkLookupTable()
            lut.SetRange(*rng)
            lut.SetTable(numpy_to_vtk(ctable))
            scalar_bar.SetLookupTable(lut)

    def _sphere(self, center, color, radius):
        sphere = vtk.vtkSphereSource()
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
            grid_alg = vtk.vtkCellDataToPointData()
            grid_alg.SetInputDataObject(grid)
            grid_alg.SetPassCellData(False)
            grid_alg.Update()
        else:
            grid_alg = None

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

    def _silhouette(self, mesh, color=None, line_width=None, alpha=None,
                    decimate=None):
        mesh = mesh.decimate(decimate) if decimate is not None else mesh
        silhouette_filter = vtk.vtkPolyDataSilhouette()
        silhouette_filter.SetInputData(mesh)
        silhouette_filter.SetCamera(self.plotter.renderer.GetActiveCamera())
        silhouette_filter.SetEnableFeatureAngle(0)
        silhouette_mapper = vtk.vtkPolyDataMapper()
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
    import vtk
    coordinate = vtk.vtkCoordinate()
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


def _set_3d_view(figure, azimuth=None, elevation=None, focalpoint='auto',
                 distance=None, roll=None, reset_camera=True, rigid=None,
                 update=True):
    rigid = np.eye(4) if rigid is None else rigid
    position = np.array(figure.plotter.camera_position[0])
    bounds = np.array(figure.plotter.renderer.ComputeVisiblePropBounds())
    if reset_camera:
        figure.plotter.reset_camera()

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
    if not isinstance(figure, _Figure):
        raise TypeError('figure must be an instance of _Figure.')


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
        del _ALL_PLOTTERS[plotter._id_name]
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
    camera.AddObserver(vtk.vtkCommand.ModifiedEvent, callback)


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


def _require_minimum_version(version_required):
    from distutils.version import LooseVersion
    version = LooseVersion(pyvista.__version__)
    if version < version_required:
        raise ImportError('pyvista>={} is required for this module but the '
                          'version found is {}'.format(version_required,
                                                       version))


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
