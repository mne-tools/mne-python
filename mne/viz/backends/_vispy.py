"""
Core visualization operations based on VisPy.

Actual implementation of _Renderer and _Projection classes.
"""

import numpy as np
from vispy import app, scene
from vispy.visuals.filters import Alpha
from vispy.visuals.transforms import STTransform

default_sphere_radius = 0.5
default_mesh_shininess = 0.0

# TODO: _Projection


class _Renderer(object):
    """Class managing rendering scene.

    Attributes
    ----------
    canvas: Instance of vispy.scene.canvas
        The support allowing automatic drawing of the scene.
    view : Instance of vispy.scene.widgets.ViewBox
        The window through which we can view the scene.
    """

    def __init__(self, fig=None, size=(600, 600), bgcolor=(0., 0., 0.),
                 name="VisPy Scene", show=False):
        """Set up the scene.

        Parameters
        ----------
        fig: instance of vispy.scene.canvas
            The scene handle.
        size: tuple
            The dimensions of the context window: (width, height).
        bgcolor: tuple
            The color definition of the background: (red, green, blue).
        name: str | None
            The name of the scene.
        """
        if fig is None:
            self.canvas = scene.SceneCanvas(keys='interactive',
                                            size=size,
                                            title=name,
                                            show=show)
            self.canvas.bgcolor = bgcolor
        else:
            self.canvas = fig
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = \
            scene.cameras.TurntableCamera(interactive=True, fov=60,
                                          azimuth=180.0, elevation=0.0,
                                          distance=0.5,
                                          parent=self.view.scene)

    def scene(self):
        """Return scene handle."""
        return self.canvas

    def set_interactive(self):
        """Enable interactive mode."""
        self.view.camera.interactive = True

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
            Unused (kept for compatibility).
        """
        so = 'smooth' if shading else None
        vertices = np.column_stack((x, y, z)).astype(np.float32)
        mesh = scene.visuals.Mesh(vertices=vertices, faces=triangles,
                                  color=color, parent=self.view.scene,
                                  shading=so)
        mesh.shininess = default_mesh_shininess
        mesh.attach(Alpha(opacity))
        if backface_culling:
            mesh.set_gl_state(cull_face=True)

    def contour(self, surface, scalars, contours, line_width=1.0, opacity=1.0,
                vmin=None, vmax=None, colormap=None):
        """Add a contour in the scene.

        Parameters
        ----------
        surface: surface object
            The mesh to use as support for contour.
        scalars: ndarray, shape (n_vertices)
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
        vertices = surface['rr']
        tris = surface['tris']

        if isinstance(contours, int):
            cmin = min(scalars)
            cmax = max(scalars)
            levels = np.linspace(cmin, cmax, num=contours)
        else:
            levels = np.array(contours)

        iso = scene.visuals.Isoline(vertices=vertices, tris=tris,
                                    width=line_width,
                                    levels=levels, color_lev='winter',
                                    data=scalars, parent=self.view.scene)
        iso.attach(Alpha(opacity))
        return 0

    def surface(self, surface, color=(0.7, 0.7, 0.7), opacity=1.0,
                vmin=None, vmax=None, colormap=None,
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
        backface_culling: bool
            If True, enable backface culling on the surface.
        """
        # TODO: add colormap
        mesh = scene.visuals.Mesh(vertices=surface['rr'],
                                  faces=surface['tris'],
                                  color=color, parent=self.view.scene,
                                  shading='smooth')
        mesh.shininess = default_mesh_shininess
        mesh.attach(Alpha(opacity))
        if backface_culling:
            mesh.set_gl_state(cull_face=True)

    def sphere(self, center, color, scale, opacity=1.0,
               backface_culling=False):
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
        backface_culling: bool
            If True, enable backface culling on the sphere(s).
        """
        for c in center:
            sphere = scene.visuals.Sphere(radius=scale * default_sphere_radius,
                                          color=color, parent=self.view.scene)
            sphere.transform = STTransform(translate=c)
            sphere.attach(Alpha(opacity))
            if backface_culling:
                sphere.set_gl_state(cull_face=True)

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
        mode: 'arrow' or 'cylinder'
            The type of the quiver.
        resolution: float
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
        source = np.column_stack((x, y, z))
        destination = source + np.column_stack((u, v, w))
        arr_pos = np.array(list(zip(source, destination)))
        arr_pos.reshape(-1, 3)
        for i in range(len(source)):
            _create_quiver(mode=mode,
                           source=source[i, :],
                           destination=destination[i, :],
                           scale=scale,
                           scale_mode=scale_mode,
                           view=self.view,
                           color=color,
                           opacity=opacity,
                           backface_culling=backface_culling)
        return 0

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
        # normalize font size
        font_size = 100 * width
        # normalize default placement
        h = self.canvas.size[0]
        scene.visuals.Text(pos=(x, h - y), text=text, font_size=font_size,
                           anchor_x='left', anchor_y='top',
                           color=color, parent=self.view)

    def show(self):
        """Render the scene."""
        self.canvas.show()
        app.run()

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
        if azimuth is not None:
            self.view.camera.azimuth = azimuth
        if elevation is not None:
            self.view.camera.elevation = elevation
        if distance is not None:
            self.view.camera.distance = distance
        if focalpoint is not None:
            self.view.camera.center = focalpoint

    def screenshot(self):
        """Take a screenshot of the scene."""
        return self.canvas.render()


def _create_quiver(mode, source, destination, view, color,
                   scale, scale_mode='none',
                   resolution=8, opacity=1.0,
                   backface_culling=False):
    from vispy.geometry.generation import create_arrow, create_cylinder
    from vispy.visuals.transforms import MatrixTransform

    v1 = destination - source
    vn = np.linalg.norm(v1)
    v1 = v1 / vn

    v2 = np.array([0, 0, 1])

    cosangle = np.dot(v1, v2)
    axis = np.cross(v2, v1)

    length = vn
    if scale_mode == 'none':
        length = scale
    radius = length / 20.0

    meshdata = None
    if mode == 'arrow':
        cone_radius = radius * 3.0
        cone_length = length / 4.0
        meshdata = create_arrow(rows=resolution, cols=resolution,
                                length=length, radius=radius,
                                cone_radius=cone_radius,
                                cone_length=cone_length)
    elif mode == 'cylinder':
        meshdata = create_cylinder(rows=resolution, cols=resolution,
                                   length=length, radius=[radius, radius])

    if meshdata is not None:
        arr = scene.visuals.Mesh(meshdata=meshdata, color=color,
                                 shading='flat', parent=view.scene)
        arr.attach(Alpha(opacity))
        if backface_culling:
            arr.set_gl_state(cull_face=True)
        # apply transform
        mat = MatrixTransform()
        if cosangle != 1:
            mat.rotate(np.degrees(np.arccos(cosangle)), axis)
        mat.translate(source)
        arr.transform = mat
