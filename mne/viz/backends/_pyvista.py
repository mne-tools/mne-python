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

import numpy as np
import warnings
import vtk
from .base_renderer import _BaseRenderer
from ._utils import _get_colormap_from_array
from ...utils import copy_base_doc_to_subclass_doc

_FIGURES = dict()


class _Figure(object):
    def __init__(self, plotter=None,
                 plotter_class=None,
                 display=None,
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
        self.store['title'] = title
        self.store['window_size'] = size
        self.store['shape'] = shape
        self.store['off_screen'] = off_screen
        self.store['border'] = False

    def build(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from pyvista import Plotter, BackgroundPlotter

        if self.plotter_class is None:
            self.plotter_class = BackgroundPlotter
        if self.notebook:
            self.plotter_class = Plotter

        if self.plotter_class == Plotter:
            self.store.pop('title', None)

        if self.plotter is None:
            plotter = self.plotter_class(**self.store)
            plotter.background_color = self.background_color
            self.plotter = plotter
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
                 name="PyVista Scene", show=False, shape=(1, 1)):
        from pyvista import OFF_SCREEN
        from mne.viz.backends.renderer import MNE_3D_BACKEND_TESTING
        figure = _Figure(title=name, size=size, shape=shape,
                         background_color=bgcolor, notebook=_check_notebook())
        self.font_family = "arial"
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
        if OFF_SCREEN or MNE_3D_BACKEND_TESTING:
            self.figure.store['off_screen'] = True

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            if MNE_3D_BACKEND_TESTING:
                from pyvista import Plotter
                self.figure.plotter_class = Plotter

            self.plotter = self.figure.build()
            self.plotter.hide_axes()

    def subplot(self, x, y):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.plotter.subplot(x, y)

    def scene(self):
        return self.figure

    def set_interactive(self):
        self.plotter.enable_terrain_style()

    def mesh(self, x, y, z, triangles, color, opacity=1.0, shading=False,
             backface_culling=False, scalars=None, colormap=None,
             vmin=None, vmax=None, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            from pyvista import PolyData
            smooth_shading = self.figure.smooth_shading
            vertices = np.c_[x, y, z]
            n_vertices = len(vertices)
            triangles = np.c_[np.full(len(triangles), 3), triangles]
            pd = PolyData(vertices, triangles)
            rgba = False
            if color is not None and len(color) == n_vertices:
                if color.shape[1] == 3:
                    scalars = np.c_[color, np.ones(n_vertices)]
                else:
                    scalars = color
                scalars = (scalars * 255).astype('ubyte')
                color = None
                # Disabling normal computation for smooth shading
                # is a temporary workaround of:
                # https://github.com/pyvista/pyvista-support/issues/15
                smooth_shading = False
                rgba = True
            if isinstance(colormap, np.ndarray):
                if colormap.dtype == np.uint8:
                    colormap = colormap.astype(np.float) / 255.
                from matplotlib.colors import ListedColormap
                colormap = ListedColormap(colormap)

            self.plotter.add_mesh(mesh=pd, color=color, scalars=scalars,
                                  rgba=rgba, opacity=opacity, cmap=colormap,
                                  backface_culling=backface_culling,
                                  rng=[vmin, vmax], show_scalar_bar=False,
                                  smooth_shading=smooth_shading)

    def contour(self, surface, scalars, contours, width=1.0, opacity=1.0,
                vmin=None, vmax=None, colormap=None,
                normalized_colormap=False, kind='line', color=None):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            from pyvista import PolyData
            if colormap is not None:
                colormap = _get_colormap_from_array(colormap,
                                                    normalized_colormap)
            vertices = np.array(surface['rr'])
            triangles = np.array(surface['tris'])
            n_triangles = len(triangles)
            triangles = np.c_[np.full(n_triangles, 3), triangles]
            pd = PolyData(vertices, triangles)
            pd.point_arrays['scalars'] = scalars
            mesh = pd.contour(isosurfaces=contours, rng=(vmin, vmax))
            line_width = width
            if kind == 'tube':
                mesh = mesh.tube(radius=width)
                line_width = 1.0
            self.plotter.add_mesh(mesh,
                                  show_scalar_bar=False,
                                  line_width=line_width,
                                  color=color,
                                  cmap=colormap,
                                  opacity=opacity,
                                  smooth_shading=self.figure.smooth_shading)

    def surface(self, surface, color=None, opacity=1.0,
                vmin=None, vmax=None, colormap=None,
                normalized_colormap=False, scalars=None,
                backface_culling=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            from pyvista import PolyData
            cmap = _get_colormap_from_array(colormap, normalized_colormap)
            vertices = np.array(surface['rr'])
            triangles = np.array(surface['tris'])
            n_triangles = len(triangles)
            triangles = np.c_[np.full(n_triangles, 3), triangles]
            pd = PolyData(vertices, triangles)
            if scalars is not None:
                pd.point_arrays['scalars'] = scalars
            self.plotter.add_mesh(mesh=pd, color=color,
                                  rng=[vmin, vmax],
                                  show_scalar_bar=False,
                                  opacity=opacity,
                                  cmap=cmap,
                                  backface_culling=backface_culling,
                                  smooth_shading=self.figure.smooth_shading)

    def sphere(self, center, color, scale, opacity=1.0,
               resolution=8, backface_culling=False,
               radius=None):
        factor = 1.0 if radius is not None else scale
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            from pyvista import PolyData
            sphere = vtk.vtkSphereSource()
            sphere.SetThetaResolution(resolution)
            sphere.SetPhiResolution(resolution)
            if radius is not None:
                sphere.SetRadius(radius)
            sphere.Update()
            geom = sphere.GetOutput()
            pd = PolyData(center)
            self.plotter.add_mesh(pd.glyph(orient=False, scale=False,
                                           factor=factor, geom=geom),
                                  color=color, opacity=opacity,
                                  backface_culling=backface_culling,
                                  smooth_shading=self.figure.smooth_shading)

    def tube(self, origin, destination, radius=0.001, color='white',
             scalars=None, vmin=None, vmax=None, colormap='RdBu',
             normalized_colormap=False, reverse_lut=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            from pyvista import Line
            cmap = _get_colormap_from_array(colormap, normalized_colormap)
            for (pointa, pointb) in zip(origin, destination):
                line = Line(pointa, pointb)
                if scalars is not None:
                    line.point_arrays['scalars'] = scalars[0, :]
                    scalars = 'scalars'
                    color = None
                else:
                    scalars = None
                tube = line.tube(radius)
                self.plotter.add_mesh(mesh=tube,
                                      scalars=scalars,
                                      flip_scalars=reverse_lut,
                                      rng=[vmin, vmax],
                                      color=color,
                                      show_scalar_bar=False,
                                      cmap=cmap,
                                      smooth_shading=self.
                                      figure.smooth_shading)
        return tube

    def quiver3d(self, x, y, z, u, v, w, color, scale, mode, resolution=8,
                 glyph_height=None, glyph_center=None, glyph_resolution=None,
                 opacity=1.0, scale_mode='none', scalars=None,
                 backface_culling=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            from pyvista import UnstructuredGrid
            factor = scale
            vectors = np.c_[u, v, w]
            points = np.vstack(np.c_[x, y, z])
            n_points = len(points)
            offset = np.arange(n_points) * 3
            cell_type = np.full(n_points, vtk.VTK_VERTEX)
            cells = np.c_[np.full(n_points, 1), range(n_points)]
            grid = UnstructuredGrid(offset, cells, cell_type, points)
            grid.point_arrays['vec'] = vectors
            if scale_mode == "scalar":
                grid.point_arrays['mag'] = np.array(scalars)
                scale = 'mag'
            else:
                scale = False
            if mode == "arrow":
                self.plotter.add_mesh(grid.glyph(orient='vec',
                                                 scale=scale,
                                                 factor=factor),
                                      color=color,
                                      opacity=opacity,
                                      backface_culling=backface_culling,
                                      smooth_shading=self.figure.
                                      smooth_shading)
            elif mode == "cone":
                cone = vtk.vtkConeSource()
                if glyph_height is not None:
                    cone.SetHeight(glyph_height)
                if glyph_center is not None:
                    cone.SetCenter(glyph_center)
                if glyph_resolution is not None:
                    cone.SetResolution(glyph_resolution)
                cone.Update()

                geom = cone.GetOutput()
                self.plotter.add_mesh(grid.glyph(orient='vec',
                                                 scale=scale,
                                                 factor=factor,
                                                 geom=geom),
                                      color=color,
                                      opacity=opacity,
                                      backface_culling=backface_culling,
                                      smooth_shading=self.figure.
                                      smooth_shading)

            elif mode == "cylinder":
                cylinder = vtk.vtkCylinderSource()
                cylinder.SetHeight(glyph_height)
                cylinder.SetRadius(0.15)
                cylinder.SetCenter(glyph_center)
                cylinder.SetResolution(glyph_resolution)
                cylinder.Update()

                # fix orientation
                tr = vtk.vtkTransform()
                tr.RotateWXYZ(90, 0, 0, 1)
                trp = vtk.vtkTransformPolyDataFilter()
                trp.SetInputData(cylinder.GetOutput())
                trp.SetTransform(tr)
                trp.Update()

                geom = trp.GetOutput()
                self.plotter.add_mesh(grid.glyph(orient='vec',
                                                 scale=scale,
                                                 factor=factor,
                                                 geom=geom),
                                      color=color,
                                      opacity=opacity,
                                      backface_culling=backface_culling,
                                      smooth_shading=self.figure.
                                      smooth_shading)

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

    def text3d(self, x, y, z, text, scale, color='white'):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.plotter.add_point_labels(points=[x, y, z],
                                          labels=[text],
                                          point_size=scale,
                                          text_color=color,
                                          font_family=self.font_family,
                                          name=text,
                                          shape_opacity=0)

    def scalarbar(self, source, title=None, n_labels=4, bgcolor=None):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.plotter.add_scalar_bar(title=title, n_labels=n_labels,
                                        use_opacity=False, n_colors=256,
                                        position_x=0.15, width=0.7,
                                        label_font_size=22,
                                        font_family=self.font_family,
                                        background_color=bgcolor)

    def show(self):
        self.figure.display = self.plotter.show()
        return self.scene()

    def close(self):
        _close_3d_figure(figure=self.figure)

    def set_camera(self, azimuth=None, elevation=None, distance=None,
                   focalpoint=None):
        _set_3d_view(self.figure, azimuth=azimuth, elevation=elevation,
                     distance=distance, focalpoint=focalpoint)

    def screenshot(self, mode='rgb', filename=None):
        return _take_3d_screenshot(figure=self.figure, mode=mode,
                                   filename=filename)

    def project(self, xyz, ch_names):
        xy = _3d_to_2d(self.plotter, xyz)
        xy = dict(zip(ch_names, xy))
        # pts = self.fig.children[-1]
        pts = self.plotter.renderer.GetActors().GetLastItem()

        return _Projection(xy=xy, pts=pts)


def _deg2rad(deg):
    return deg * np.pi / 180.


def _rad2deg(rad):
    return rad * 180. / np.pi


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
    view_to_disp_mat = np.array([[x / 2.0,       0.,   0.,   x / 2.0],  # noqa: E241,E501
                                 [0.,      -y / 2.0,   0.,   y / 2.0],  # noqa: E241,E501
                                 [0.,            0.,   1.,        0.],  # noqa: E241,E501
                                 [0.,            0.,   0.,        1.]])  # noqa: E241,E501
    return view_to_disp_mat


def _close_all():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from pyvista import close_all
        close_all()


def _check_notebook():
    if _run_from_ipython():
        try:
            return type(get_ipython()).__module__.startswith('ipykernel.')
        except NameError:
            return False


def _run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def _get_camera_direction(focalpoint, position):
    x, y, z = position - focalpoint
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi, focalpoint


def _set_3d_view(figure, azimuth, elevation, focalpoint, distance):
    position = np.array(figure.plotter.camera_position[0])
    focalpoint = np.array(figure.plotter.camera_position[1])
    r, theta, phi, fp = _get_camera_direction(focalpoint, position)

    if azimuth is not None:
        phi = _deg2rad(azimuth)
    if elevation is not None:
        theta = _deg2rad(elevation)

    renderer = figure.plotter.renderer
    bounds = np.array(renderer.ComputeVisiblePropBounds())
    if distance is not None:
        r = distance
    else:
        r = max(bounds[1::2] - bounds[::2]) * 2.0

    cen = (bounds[1::2] + bounds[::2]) * 0.5
    if focalpoint is not None:
        cen = np.asarray(focalpoint)

    position = [
        r * np.cos(phi) * np.sin(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(theta)]
    figure.plotter.camera_position = [
        position, cen, [0, 0, 1]]


def _set_3d_title(figure, title, size=40):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        figure.plotter.add_text(title, font_size=size, color='white')


def _check_3d_figure(figure):
    if not isinstance(figure, _Figure):
        raise TypeError('figure must be an instance of _Figure.')


def _close_3d_figure(figure):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        figure.plotter.close()


def _take_3d_screenshot(figure, mode='rgb', filename=None):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        return figure.plotter.screenshot(
            transparent_background=(mode == 'rgba'),
            filename=filename)


def _try_3d_backend():
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            import pyvista  # noqa: F401
    except Exception:
        pass
