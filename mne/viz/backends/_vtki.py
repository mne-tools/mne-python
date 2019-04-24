"""
Core visualization operations based on VTKi.

Actual implementation of _Renderer and _Projection classes.
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import vtk
import vtki
import warnings
import numpy as np


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


class _Renderer(object):
    """Class managing rendering scene.

    Attributes
    ----------
    plotter: vtki.Plotter
        Main VTKI access point.
    off_screen: bool
        State of the offscreen.
    name: str
        Name of the window.
    """

    def __init__(self, fig=None, size=(600, 600), bgcolor=(0., 0., 0.),
                 name="VTKI Scene", show=False):
        """Set up the scene.

        Parameters
        ----------
        fig: instance of vtki.Plotter
            Scene handle.
        size : tuple
            The dimensions of the context window: (width, height).
        bgcolor: tuple
            The color definition of the background: (red, green, blue).
        name: str | None
            The name of the scene.
        """
        from mne.viz.backends.renderer import MNE_3D_BACKEND_TEST_DATA
        self.off_screen = False
        self.name = name
        if MNE_3D_BACKEND_TEST_DATA:
            self.off_screen = True
        if fig is None:
            self.plotter = vtki.Plotter(window_size=size,
                                        off_screen=self.off_screen)
            self.plotter.background_color = bgcolor
            # this is a hack to avoid using a deleled ren_win
            self.plotter._window_size = size
        else:
            # import basic properties
            self.plotter = vtki.Plotter(window_size=fig._window_size,
                                        off_screen=fig.off_screen)
            # import background
            self.plotter.background_color = fig.background_color
            # import actors
            for actor in fig.renderer.GetActors():
                self.plotter.renderer.AddActor(actor)
            # import camera
            self.plotter.camera_position = fig.camera_position
            self.plotter.reset_camera()

    def scene(self):
        """Return scene handle."""
        return self.plotter

    def set_interactive(self):
        """Enable interactive mode."""
        self.plotter.enable_terrain_style()

    def mesh(self, x, y, z, triangles, color, opacity=1.0, shading=False,
             backface_culling=False, **kwargs):
        """Add a mesh in the scene.

        Parameters
        ----------
        x: array, shape (n_vertices,)
           The array containing the X component of the vertices.
        y: array, shape (n_vertices,)
           The array containing the Y component of the vertices.
        z: array, shape (n_vertices,)
           The array containing the Z component of the vertices.
        triangles: array, shape (n_polygons, 3)
           The array containing the indices of the polygons.
        color: tuple
            The color of the mesh: (red, green, blue).
        opacity: float
            The opacity of the mesh.
        shading: bool
            If True, enable the mesh shading.
        backface_culling: bool
            If True, enable backface culling on the mesh.
        kwargs: args
            The arguments to pass to triangular_mesh
        """
        vertices = np.c_[x, y, z]
        n_triangles = len(triangles)
        triangles = np.c_[np.full(n_triangles, 3), triangles]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            pd = vtki.PolyData(vertices, triangles)
            self.plotter.add_mesh(mesh=pd, color=color, opacity=opacity,
                                  backface_culling=backface_culling)

    def contour(self, surface, scalars, contours, line_width=1.0, opacity=1.0,
                vmin=None, vmax=None, colormap=None):
        """Add a contour in the scene.

        Parameters
        ----------
        surface: surface object
            The mesh to use as support for contour.
        scalars: ndarray, shape (n_vertices,)
            The scalar valued associated to the vertices.
        contours: int | list
             Specifying a list of values will only give the requested contours.
        line_width: float
            The width of the lines.
        opacity: float
            The opacity of the contour.
        vmin: float | None
            vmin is used to scale the colormap.
            If None, the min of the data will be used
        vmax: float | None
            vmax is used to scale the colormap.
            If None, the max of the data will be used
        colormap:
            The colormap to use.
        """
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
            pd = vtki.PolyData(vertices, triangles)
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
        """Add a surface in the scene.

        Parameters
        ----------
        surface: surface object
            The information describing the surface.
        color: tuple
            The color of the surface: (red, green, blue).
        opacity: float
            The opacity of the surface.
        vmin: float | None
            vmin is used to scale the colormap.
            If None, the min of the data will be used
        vmax: float | None
            vmax is used to scale the colormap.
            If None, the max of the data will be used
        colormap:
            The colormap to use.
        scalars: ndarray, shape (n_vertices,)
            The scalar valued associated to the vertices.
        backface_culling: bool
            If True, enable backface culling on the surface.
        """
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
            pd = vtki.PolyData(vertices, triangles)
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
        """Add sphere in the scene.

        Parameters
        ----------
        center: ndarray, shape(n_center, 3)
            The list of centers to use for the sphere(s).
        color: tuple
            The color of the sphere(s): (red, green, blue).
        scale: float
            The scale of the sphere(s).
        opacity: float
            The opacity of the sphere(s).
        resolution: int
            The resolution of the sphere.
        backface_culling: bool
            If True, enable backface culling on the sphere(s).
        """
        sphere = vtk.vtkSphereSource()
        sphere.SetThetaResolution(resolution)
        sphere.SetPhiResolution(resolution)
        sphere.Update()
        geom = sphere.GetOutput()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            pd = vtki.PolyData(center)
            self.plotter.add_mesh(pd.glyph(orient=False, scale=False,
                                           factor=scale, geom=geom),
                                  color=color, opacity=opacity,
                                  backface_culling=backface_culling)

    def quiver3d(self, x, y, z, u, v, w, color, scale, mode, resolution=8,
                 glyph_height=None, glyph_center=None, glyph_resolution=None,
                 opacity=1.0, scale_mode='none', scalars=None,
                 backface_culling=False):
        """Add quiver3d in the scene.

        Parameters
        ----------
        x: array, shape (n_quivers,)
            The X component of the position of the quiver.
        y: array, shape (n_quivers,)
            The Y component of the position of the quiver.
        z: array, shape (n_quivers,)
            The Z component of the position of the quiver.
        u: array, shape (n_quivers,)
            The last X component of the quiver.
        v: array, shape (n_quivers,)
            The last Y component of the quiver.
        w: array, shape (n_quivers,)
            The last Z component of the quiver.
        color: tuple
            The color of the quiver: (red, green, blue).
        scale: float
            The scale of the quiver.
        mode: 'arrow', 'cone' or 'cylinder'
            The type of the quiver.
        resolution: int
            The resolution of the arrow.
        glyph_height: float
            The height of the glyph used with the quiver.
        glyph_center: tuple
            The center of the glyph used with the quiver: (x, y, z).
        glyph_resolution: float
            The resolution of the glyph used with the quiver.
        opacity: float
            The opacity of the quiver.
        scale_mode: 'vector', 'scalar' or 'none'
            The scaling mode for the glyph.
        scalars: array, shape (n_quivers,) | None
            The optional scalar data to use.
        backface_culling: bool
            If True, enable backface culling on the quiver.
        """
        factor = scale
        vectors = np.c_[u, v, w]
        points = np.vstack(np.c_[x, y, z])
        n_points = len(points)
        offset = np.arange(n_points) * 3
        cell_type = np.full(n_points, vtk.VTK_VERTEX)
        cells = np.c_[np.full(n_points, 1), range(n_points)]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            grid = vtki.UnstructuredGrid(offset, cells, cell_type, points)
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
        """Add test in the scene.

        Parameters
        ----------
        x: float
            The X component to use as position of the text.
        y: float
            The Y component to use as position of the text.
        text: str
            The content of the text.
        width: float
            The width of the text.
        color: tuple
            The color of the text.
        """
        self.plotter.add_text(text, position=(x, y),
                              font_size=int(width * 100),
                              color=color)

    def show(self):
        """Render the scene."""
        self.plotter.show(title=self.name)

    def close(self):
        """Close the scene."""
        self.plotter.close()

    def set_camera(self, azimuth=0.0, elevation=0.0, distance=1.0,
                   focalpoint=(0, 0, 0)):
        """Configure the camera of the scene.

        Parameters
        ----------
        azimuth: float
            The azimuthal angle of the camera.
        elevation: float
            The zenith angle of the camera.
        distance: float
            The distance to the focal point.
        focalpoint: tuple
            The focal point of the camera: (x, y, z).
        """
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
        """Take a screenshot of the scene."""
        return self.plotter.screenshot()

    def project(self, xyz, ch_names):
        """Convert 3d points to a 2d perspective.

        Parameters
        ----------
        xyz: array, shape(n_points, 3)
            The points to project.
        ch_names: array, shape(_n_points,)
            Names of the channels.
        """
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
