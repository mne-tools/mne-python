"""
Core visualization operations based on VTKi.

Actual implementation of _Renderer and _Projection classes.
"""

import vtk
import vtki
import warnings
import numpy as np


class _Projection(object):

    def __init__(self):
        return 0


class _Renderer(object):

    def __init__(self, fig=None, size=(600, 600), bgcolor=(0., 0., 0.),
                 name=None, show=False):
        from mne.viz.backends.renderer import MNE_3D_BACKEND_TEST_DATA
        self.off_screen = False
        if MNE_3D_BACKEND_TEST_DATA:
            self.off_screen = True
        if fig is None:
            self.plotter = vtki.Plotter(window_size=size,
                                        off_screen=self.off_screen)
        else:
            self.plotter = fig
        self.plotter.background_color = bgcolor

    def scene(self):
        return self.plotter

    def set_interactive(self):
        return 0

    def mesh(self, x, y, z, triangles, color, opacity=1.0, shading=False,
             backface_culling=False, **kwargs):
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
                                  opacity=opacity)

    def surface(self, surface, color=None, opacity=1.0,
                vmin=None, vmax=None, colormap=None, scalars=None,
                backface_culling=False):
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
                                  backface_culling=backface_culling)

    def sphere(self, center, color, scale, opacity=1.0,
               backface_culling=False):
        sphere = vtk.vtkSphereSource()
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
        self.plotter.add_text(text, position=(x, y),
                              font_size=int(width * 100),
                              color=color)

    def show(self):
        self.plotter.show()

    def set_camera(self, azimuth=None, elevation=None, distance=None,
                   focalpoint=None):
        return 0

    def screenshot(self):
        return self.plotter.screenshot()

    def project(self, xyz, ch_names):
        return 0
