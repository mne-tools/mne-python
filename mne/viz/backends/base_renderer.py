"""Core visualization operations."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

from abc import ABCMeta, abstractclassmethod


class _BaseRenderer(metaclass=ABCMeta):
    @abstractclassmethod
    def __init__(self, fig=None, size=(600, 600), bgcolor=(0., 0., 0.),
                 name=None, show=False, shape=(1, 1)):
        """Set up the scene."""
        pass

    @abstractclassmethod
    def subplot(self, x, y):
        """Set the active subplot."""
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
        color: tuple | str
            The color of the mesh as a tuple (red, green, blue) of float
            values between 0 and 1 or a valid color name (i.e. 'white'
            or 'w').
        opacity: float
            The opacity of the mesh.
        shading: bool
            If True, enable the mesh shading.
        backface_culling: bool
            If True, enable backface culling on the mesh.
        kwargs: args
            The arguments to pass to triangular_mesh

        Returns
        -------
        surface:
            Handle of the mesh in the scene.
        """
        pass

    @abstractclassmethod
    def contour(self, surface, scalars, contours, width=1.0, opacity=1.0,
                vmin=None, vmax=None, colormap=None,
                normalized_colormap=False, kind='line', color=None):
        """Add a contour in the scene.

        Parameters
        ----------
        surface: surface object
            The mesh to use as support for contour.
        scalars: ndarray, shape (n_vertices,)
            The scalar valued associated to the vertices.
        contours: int | list
             Specifying a list of values will only give the requested contours.
        width: float
            The width of the lines or radius of the tubes.
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
        normalized_colormap: bool
            Specify if the values of the colormap are between 0 and 1.
        kind: 'line' | 'tube'
            The type of the primitives to use to display the contours.
        color:
            The color of the mesh as a tuple (red, green, blue) of float
            values between 0 and 1 or a valid color name (i.e. 'white'
            or 'w').
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
        color: tuple | str
            The color of the surface as a tuple (red, green, blue) of float
            values between 0 and 1 or a valid color name (i.e. 'white'
            or 'w').
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
               resolution=8, backface_culling=False,
               radius=None):
        """Add sphere in the scene.

        Parameters
        ----------
        center: ndarray, shape(n_center, 3)
            The list of centers to use for the sphere(s).
        color: tuple | str
            The color of the sphere as a tuple (red, green, blue) of float
            values between 0 and 1 or a valid color name (i.e. 'white'
            or 'w').
        scale: float
            The scaling applied to the spheres. The given value specifies
            the maximum size in drawing units.
        opacity: float
            The opacity of the sphere(s).
        resolution: int
            The resolution of the sphere created. This is the number
            of divisions along theta and phi.
        backface_culling: bool
            If True, enable backface culling on the sphere(s).
        radius: float | None
            Replace the glyph scaling by a fixed radius value for each
            sphere (not supported by mayavi).
        """
        pass

    @abstractclassmethod
    def tube(self, origin, destination, radius=0.001, color='white',
             scalars=None, vmin=None, vmax=None, colormap='RdBu',
             normalized_colormap=False, reverse_lut=False):
        """Add tube in the scene.

        Parameters
        ----------
        origin: array, shape(n_lines, 3)
            The coordinates of the first end of the tube(s).
        destination: array, shape(n_lines, 3)
            The coordinates of the other end of the tube(s).
        radius: float
            The radius of the tube(s).
        color: tuple | str
            The color of the tube as a tuple (red, green, blue) of float
            values between 0 and 1 or a valid color name (i.e. 'white'
            or 'w').
        scalars: array, shape (n_quivers,) | None
            The optional scalar data to use.
        vmin: float | None
            vmin is used to scale the colormap.
            If None, the min of the data will be used
        vmax: float | None
            vmax is used to scale the colormap.
            If None, the max of the data will be used
        colormap:
            The colormap to use.
        opacity: float
            The opacity of the tube(s).
        backface_culling: bool
            If True, enable backface culling on the tube(s).
        reverse_lut: bool
            If True, reverse the lookup table.

        Returns
        -------
        surface:
            Handle of the tube in the scene.
        """
        pass

    @abstractclassmethod
    def quiver3d(self, x, y, z, u, v, w, color, scale, mode, resolution=8,
                 glyph_height=None, glyph_center=None, glyph_resolution=None,
                 opacity=1.0, scale_mode='none', scalars=None,
                 backface_culling=False, colormap=None, vmin=None, vmax=None,
                 line_width=2., name=None):
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
        color: tuple | str
            The color of the quiver as a tuple (red, green, blue) of float
            values between 0 and 1 or a valid color name (i.e. 'white'
            or 'w').
        scale: float
            The scaling applied to the glyphs. The size of the glyph
            is by default calculated from the inter-glyph spacing.
            The given value specifies the maximum glyph size in drawing units.
        mode: 'arrow', 'cone' or 'cylinder'
            The type of the quiver.
        resolution: int
            The resolution of the glyph created. Depending on the type of
            glyph, it represents the number of divisions in its geometric
            representation.
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
        colormap:
            The colormap to use.
        vmin: float | None
            vmin is used to scale the colormap.
            If None, the min of the data will be used
        vmax: float | None
            vmax is used to scale the colormap.
            If None, the max of the data will be used
        line_width: float
            The width of the 2d arrows.
        """
        pass

    @abstractclassmethod
    def text2d(self, x_window, y_window, text, size=14, color='white'):
        """Add 2d text in the scene.

        Parameters
        ----------
        x: float
            The X component to use as position of the text in the
            window coordinates system (window_width, window_height).
        y: float
            The Y component to use as position of the text in the
            window coordinates system (window_width, window_height).
        text: str
            The content of the text.
        size: int
            The size of the font.
        color: tuple | str
            The color of the text as a tuple (red, green, blue) of float
            values between 0 and 1 or a valid color name (i.e. 'white'
            or 'w').
        """
        pass

    @abstractclassmethod
    def text3d(self, x, y, z, text, width, color='white'):
        """Add 2d text in the scene.

        Parameters
        ----------
        x: float
            The X component to use as position of the text.
        y: float
            The Y component to use as position of the text.
        z: float
            The Z component to use as position of the text.
        text: str
            The content of the text.
        width: float
            The width of the text.
        color: tuple | str
            The color of the text as a tuple (red, green, blue) of float
            values between 0 and 1 or a valid color name (i.e. 'white'
            or 'w').
        """
        pass

    @abstractclassmethod
    def scalarbar(self, source, title=None, n_labels=4):
        """Add a scalar bar in the scene.

        Parameters
        ----------
        source:
            The object of the scene used for the colormap.
        title: str | None
            The title of the scalar bar.
        n_labels: int | None
            The number of labels to display on the scalar bar.
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
    def reset_camera(self):
        """Reset the camera properties."""
        pass

    @abstractclassmethod
    def screenshot(self, mode='rgb', filename=None):
        """Take a screenshot of the scene.

        Parameters
        ----------
        mode: str
            Either 'rgb' or 'rgba' for values to return.
            Default is 'rgb'.
        filename: str | None
            If not None, save the figure to the disk.
        """
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
