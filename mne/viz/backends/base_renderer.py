"""Core visualization operations."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

from abc import ABCMeta, abstractclassmethod


class _BaseRenderer(metaclass=ABCMeta):
    @abstractclassmethod
    def __init__(self, fig=None, size=(600, 600), bgcolor=(0., 0., 0.),
                 name=None, show=False):
        """Set up the scene."""
        pass

    @abstractclassmethod
    def scene(self):
        """Return scene handle."""
        pass

    @abstractclassmethod
    def set_interactive(self):
        """Enable interactive mode."""
        pass

    @abstractclassmethod
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
        pass

    @abstractclassmethod
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
        pass

    @abstractclassmethod
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
        pass

    @abstractclassmethod
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
        pass

    @abstractclassmethod
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
        pass

    @abstractclassmethod
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
        pass

    @abstractclassmethod
    def show(self):
        """Render the scene."""
        pass

    @abstractclassmethod
    def close(self):
        """Close the scene."""
        pass

    @abstractclassmethod
    def set_camera(self, azimuth=None, elevation=None, distance=None,
                   focalpoint=None):
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
        pass

    @abstractclassmethod
    def screenshot(self):
        """Take a screenshot of the scene."""
        pass

    @abstractclassmethod
    def project(self, xyz, ch_names):
        """Convert 3d points to a 2d perspective.

        Parameters
        ----------
        xyz: array, shape(n_points, 3)
            The points to project.
        ch_names: array, shape(_n_points,)
            Names of the channels.
        """
        pass
