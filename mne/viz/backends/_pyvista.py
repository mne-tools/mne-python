"""
Core visualization operations based on PyVista.

Actual implementation of _Renderer and _Projection classes.
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import vtk
import pyvista
import warnings
import numpy as np
from .base_renderer import _BaseRenderer
from ...utils import copy_base_doc_to_subclass_doc


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
    plotter: pyvista.Plotter
        Main PyVista access point.
    off_screen: bool
        State of the offscreen.
    name: str
        Name of the window.
    """

    def __init__(self, fig=None, size=(600, 600), bgcolor=(0., 0., 0.),
                 name="PyVista Scene", show=False):
        from mne.viz.backends.renderer import MNE_3D_BACKEND_TEST_DATA
        self.off_screen = False
        self.name = name
        if MNE_3D_BACKEND_TEST_DATA:
            self.off_screen = True
        if fig is None:
            self.plotter = pyvista.Plotter(
                window_size=size, off_screen=self.off_screen)
            self.plotter.background_color = bgcolor
            # this is a hack to avoid using a deleled ren_win
            self.plotter._window_size = size
        else:
            # import basic properties
            self.plotter = pyvista.Plotter(
                window_size=fig._window_size, off_screen=fig.off_screen)
            # import background
            self.plotter.background_color = fig.background_color
            # import actors
            for actor in fig.renderer.GetActors():
                self.plotter.renderer.AddActor(actor)
            # import camera
            self.plotter.camera_position = fig.camera_position
            self.plotter.reset_camera()

    def scene(self):
        return self.plotter

    def set_interactive(self):
        self.plotter.enable_terrain_style()

    def mesh(self, x, y, z, triangles, color, opacity=1.0, shading=False,
             backface_culling=False, **kwargs):
        vertices = np.c_[x, y, z]
        n_triangles = len(triangles)
        triangles = np.c_[np.full(n_triangles, 3), triangles]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            pd = pyvista.PolyData(vertices, triangles)
            self.plotter.add_mesh(mesh=pd, color=color, opacity=opacity,
                                  backface_culling=backface_culling)

    def contour(self, surface, scalars, contours, line_width=1.0, opacity=1.0,
                vmin=None, vmax=None, colormap=None):
        from matplotlib import cm
        from matplotlib.colors import ListedColormap
        if colormap is None:
            cmap = cm.get_cmap('coolwarm')
        else:
            cmap = ListedColormap(colormap / 255.0)
        vertices = np.array(surface['rr'])
        triangles = np.array(surface['tris'])
        n_triangles = len(triangles)
        triangles = np.c_[np.full(n_triangles, 3), triangles]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            pd = pyvista.PolyData(vertices, triangles)
            pd.point_arrays['scalars'] = scalars
            self.plotter.add_mesh(pd.contour(isosurfaces=contours,
                                             rng=(vmin, vmax)),
                                  show_scalar_bar=False,
                                  line_width=line_width,
                                  cmap=cmap,
                                  opacity=opacity)

    def surface(self, surface, color=None, opacity=1.0,
                vmin=None, vmax=None, colormap=None, scalars=None,
                backface_culling=False):
        from matplotlib import cm
        from matplotlib.colors import ListedColormap
        if colormap is None:
            cmap = cm.get_cmap('coolwarm')
        else:
            cmap = ListedColormap(colormap / 255.0)
        vertices = np.array(surface['rr'])
        triangles = np.array(surface['tris'])
        n_triangles = len(triangles)
        triangles = np.c_[np.full(n_triangles, 3), triangles]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            pd = pyvista.PolyData(vertices, triangles)
            if scalars is not None:
                pd.point_arrays['scalars'] = scalars
            self.plotter.add_mesh(mesh=pd, color=color,
                                  rng=[vmin, vmax],
                                  show_scalar_bar=False,
                                  opacity=opacity,
                                  cmap=cmap,
                                  backface_culling=backface_culling)

    def sphere(self, center, color, scale, opacity=1.0,
               resolution=8, backface_culling=False):
        sphere = vtk.vtkSphereSource()
        sphere.SetThetaResolution(resolution)
        sphere.SetPhiResolution(resolution)
        sphere.Update()
        geom = sphere.GetOutput()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            pd = pyvista.PolyData(center)
            self.plotter.add_mesh(pd.glyph(orient=False, scale=False,
                                           factor=scale, geom=geom),
                                  color=color, opacity=opacity,
                                  backface_culling=backface_culling)

    def quiver3d(self, x, y, z, u, v, w, color, scale, mode, resolution=8,
                 glyph_height=None, glyph_center=None, glyph_resolution=None,
                 opacity=1.0, scale_mode='none', scalars=None,
                 backface_culling=False):
        factor = scale
        vectors = np.c_[u, v, w]
        points = np.vstack(np.c_[x, y, z])
        n_points = len(points)
        offset = np.arange(n_points) * 3
        cell_type = np.full(n_points, vtk.VTK_VERTEX)
        cells = np.c_[np.full(n_points, 1), range(n_points)]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            grid = pyvista.UnstructuredGrid(offset, cells, cell_type, points)
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
                                      backface_culling=backface_culling)
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
                                      backface_culling=backface_culling)

            elif mode == "cylinder":
                cylinder = vtk.vtkCylinderSource()
                cylinder.SetHeight(glyph_height)
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
                                      backface_culling=backface_culling)

    def text(self, x, y, text, width, color=(1.0, 1.0, 1.0)):
        self.plotter.add_text(text, position=(x, y),
                              font_size=int(width * 100),
                              color=color)

    def show(self):
        self.plotter.show(title=self.name)

    def close(self):
        self.plotter.close()

    def set_camera(self, azimuth=0.0, elevation=0.0, distance=1.0,
                   focalpoint=(0, 0, 0)):
        phi = _deg2rad(azimuth)
        theta = _deg2rad(elevation)
        position = [
            distance * np.cos(phi) * np.sin(theta),
            distance * np.sin(phi) * np.sin(theta),
            distance * np.cos(theta)]
        self.plotter.camera_position = [
            position, focalpoint, [0, 0, 1]]
        self.plotter.reset_camera()

    def screenshot(self):
        return self.plotter.screenshot()

    def project(self, xyz, ch_names):
        xy = _3d_to_2d(self.plotter, xyz)
        xy = dict(zip(ch_names, xy))
        # pts = self.fig.children[-1]
        pts = self.plotter.renderer.GetActors().GetLastItem()

        return _Projection(xy=xy, pts=pts)


def _deg2rad(deg):
    from numpy import pi
    return deg * pi / 180.


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
