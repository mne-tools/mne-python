"""
Core visualization operations based on VisPy.

Actual implementation of _Renderer and _Projection classes.
"""

import numpy as np
import warnings
from .base_renderer import _BaseRenderer
from ...utils import copy_base_doc_to_subclass_doc
with warnings.catch_warnings():  # catch the VisPy warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from vispy import scene
    from vispy.color import Colormap
    from vispy.visuals.filters import Alpha

default_sphere_radius = 0.5
default_mesh_shininess = 0.0


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
        if isinstance(self.pts, list):
            for p in self.pts:
                p.visible = state
        else:
            self.pts.visible = state


@copy_base_doc_to_subclass_doc
class _Renderer(_BaseRenderer):
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
        from .renderer import MNE_3D_BACKEND_TEST_DATA
        fig = MNE_3D_BACKEND_TEST_DATA
        if fig is None:
            self.canvas = scene.SceneCanvas(keys='interactive',
                                            size=size,
                                            title=name,
                                            show=show)
            self.canvas.bgcolor = bgcolor
            self.view = None
        else:
            self.canvas = fig
            if len(self.canvas.central_widget._widgets) > 0:
                # by default, the viewbox is the first widget
                self.view = self.canvas.central_widget._widgets[0]
            else:
                self.view = None

        if self.view is None:
            self.view = self.canvas.central_widget.add_view()
            self.view.camera = \
                scene.cameras.TurntableCamera(interactive=True, fov=50,
                                              azimuth=180.0, elevation=0.0,
                                              distance=0.5,
                                              parent=self.view.scene)

    def scene(self):
        return self.canvas

    def set_interactive(self):
        self.view.camera.interactive = True

    def mesh(self, x, y, z, triangles, color, opacity=1.0, shading=True,
             backface_culling=False, offset=None, **kwargs):
        so = 'smooth' if shading else None
        vertices = np.column_stack((x, y, z)).astype(np.float32)
        mesh = scene.visuals.Mesh(vertices=vertices, faces=triangles,
                                  color=color, parent=self.view.scene,
                                  shading=so)
        mesh.shininess = default_mesh_shininess
        mesh.attach(Alpha(opacity))
        if opacity < 1.0:
            mesh.set_gl_state('translucent', depth_test=False)
        if backface_culling:
            mesh.set_gl_state(cull_face=True)
        if offset is not None:
            mesh.set_gl_state(depth_test=True,
                              polygon_offset=(offset, offset),
                              polygon_offset_fill=True)

    def contour(self, surface, scalars, contours, line_width=1.0, opacity=1.0,
                vmin=None, vmax=None, colormap=None, offset=None):
        vertices = surface['rr']
        tris = surface['tris']

        if colormap is None:
            cm = "coolwarm"
        else:
            cm = Colormap(colormap / 255.0)

        if isinstance(contours, int):
            if vmin is None:
                vmin = min(scalars)
            if vmax is None:
                vmax = max(scalars)
            levels = np.linspace(vmin, vmax, num=contours)
        else:
            levels = np.array(contours)

        iso = scene.visuals.Isoline(vertices=vertices, tris=tris,
                                    width=line_width,
                                    levels=levels, color_lev=cm,
                                    data=scalars, parent=self.view.scene)
        iso.attach(Alpha(opacity))
        if offset is not None:
            iso.set_gl_state(depth_test=True,
                             polygon_offset=(offset, offset),
                             polygon_offset_fill=True)
        return 0

    def surface(self, surface, color=(0.7, 0.7, 0.7), opacity=1.0,
                vmin=None, vmax=None, colormap=None, scalars=None,
                backface_culling=False, offset=None):
        mesh = None
        vertices = np.array(surface['rr']).astype(np.float32)
        faces = np.array(surface['tris']).astype(np.uint32)
        if colormap is not None and scalars is not None:
            cm = Colormap(colormap / 255.0)
            if vmin is None:
                vmin = min(scalars)
            if vmax is None:
                vmax = max(scalars)
            nscalars = (scalars - vmin) / (vmax - vmin)
            vcolors = cm.map(nscalars)
            mesh = scene.visuals.Mesh(vertices=vertices,
                                      faces=faces,
                                      vertex_colors=vcolors,
                                      shading='flat',
                                      parent=self.view.scene)
        else:
            mesh = scene.visuals.Mesh(vertices=vertices,
                                      faces=faces,
                                      color=color, parent=self.view.scene,
                                      shading='flat')
        if mesh is not None:
            mesh.shininess = default_mesh_shininess
            mesh.attach(Alpha(opacity))
            if opacity < 1.0:
                mesh.set_gl_state('translucent', depth_test=False)
            if backface_culling:
                mesh.set_gl_state(cull_face=True)
        if offset is not None:
            mesh.set_gl_state(depth_test=True,
                              polygon_offset=(offset, offset),
                              polygon_offset_fill=True)

    def sphere(self, center, color, scale, opacity=1.0,
               resolution=8, backface_culling=False):
        from vispy.geometry import create_sphere
        from vispy.util.transforms import translate

        if center.ndim == 1:
            center = np.array([center])

        # fetch original data
        meshdata = create_sphere(radius=scale * default_sphere_radius,
                                 cols=resolution, rows=resolution)
        orig_vertices = meshdata.get_vertices()
        n_vertices = len(orig_vertices)
        orig_vertices = np.c_[orig_vertices, np.ones(n_vertices)].T
        orig_faces = meshdata.get_faces()
        n_faces = len(orig_faces)
        n_center = len(center)
        voffset = 0
        foffset = 0

        # accumulate mesh data
        acc_vertices = np.empty((n_center * n_vertices, 3), dtype=np.float32)
        acc_faces = np.empty((n_center * n_faces, 3), dtype=np.uint32)
        for c in center:
            # apply index shifting and accumulate faces
            current_faces = orig_faces + voffset
            acc_faces[foffset:foffset + n_faces, :] = current_faces
            foffset += n_faces

            # apply translation and accumulate vertices
            mat = translate(c).T
            current_vertices = mat.dot(orig_vertices)
            acc_vertices[voffset:voffset + n_vertices, :] = \
                current_vertices.T[:, 0:3]
            voffset += n_vertices

        sphere = scene.visuals.Mesh(vertices=acc_vertices,
                                    faces=acc_faces,
                                    color=color,
                                    parent=self.view.scene)
        sphere.attach(Alpha(opacity))
        if opacity < 1.0:
            sphere.set_gl_state('translucent', depth_test=False)
        if backface_culling:
            sphere.set_gl_state(cull_face=True)

    def quiver3d(self, x, y, z, u, v, w, color, scale, mode='arrow',
                 resolution=8, glyph_height=None, glyph_center=None,
                 glyph_resolution=None, opacity=1.0, scale_mode='none',
                 scalars=None, backface_culling=False):
        source = np.column_stack((x, y, z))
        destination = source + np.column_stack((u, v, w))
        arr_pos = np.array(list(zip(source, destination)))
        arr_pos.reshape(-1, 3)

        acc_vertices = list()
        acc_faces = list()
        foffset = 0
        voffset = 0
        for i in range(len(source)):
            scalar = scalars[i] if scalars is not None else None
            meshdata, mat = _create_quiver(mode=mode,
                                           source=source[i, :],
                                           destination=destination[i, :],
                                           scale=scale,
                                           scale_mode=scale_mode,
                                           scalar=scalar)
            if meshdata is not None:
                orig_vertices = meshdata.get_vertices()
                orig_faces = meshdata.get_faces()
                n_faces = len(orig_faces)
                n_vertices = len(orig_vertices)

                # accumulate faces
                current_faces = orig_faces + voffset
                acc_faces.append(current_faces)
                foffset += n_faces

                # accumulate vertices
                mat = mat.T
                current_vertices = np.c_[orig_vertices, np.ones(n_vertices)]
                current_vertices = mat.dot(current_vertices.T)
                current_vertices = current_vertices.T[:, 0:3]
                acc_vertices.append(current_vertices)
                voffset += n_vertices

        # concatenate
        faces = np.concatenate(acc_faces, axis=0)
        vertices = np.concatenate(acc_vertices, axis=0)

        quiver = scene.visuals.Mesh(vertices=vertices,
                                    faces=faces,
                                    color=color,
                                    shading='flat',
                                    parent=self.view.scene)
        quiver.attach(Alpha(opacity))
        if opacity < 1.0:
            quiver.set_gl_state('translucent', depth_test=False)
        if backface_culling:
            quiver.set_gl_state(cull_face=True)
        return 0

    def text(self, x, y, text, width, color=(1.0, 1.0, 1.0)):
        # normalize font size
        font_size = 100 * width
        # normalize default placement
        h = self.canvas.size[0]
        scene.visuals.Text(pos=(x, h - y), text=text, font_size=font_size,
                           anchor_x='left', anchor_y='top',
                           color=color, parent=self.view)

    def show(self):
        from .renderer import MNE_3D_BACKEND_TEST_DATA
        if MNE_3D_BACKEND_TEST_DATA is None:
            self.canvas.show()
            self.canvas.app.run()

    def close(self):
        self.canvas.close()

    def set_camera(self, azimuth=None, elevation=None, distance=None,
                   focalpoint=None):
        if azimuth is not None:
            self.view.camera.azimuth = 90 + azimuth
        if elevation is not None:
            self.view.camera.elevation = 90 - elevation
        if distance is not None:
            self.view.camera.fov = 50
            self.view.camera.distance = distance / 2.0
        if focalpoint is not None:
            self.view.camera.center = focalpoint

    def screenshot(self):
        return self.canvas.render()

    def title(self, text, height, color=(1.0, 1.0, 1.0)):
        # normalize font size
        font_size = 100 * height
        scene.visuals.Text(pos=(0, 0), text=text, font_size=font_size,
                           anchor_x='center', anchor_y='top',
                           color=color, parent=self.view)

    def project(self, xyz, ch_names):
        xy = _3d_to_2d(xyz, self.canvas, self.view.camera)
        xy = dict(zip(ch_names, xy))
        root = self.canvas.scene.children[0]

        # looking for the SubScene
        node = root
        class_name = root.__class__.__name__
        while class_name != 'SubScene':
            if node.children:
                node = node.children[0]
                class_name = node.__class__.__name__
            else:
                raise RuntimeError('No SubScene object have been found '
                                   'in the SceneGraph.')
        subscene = node

        # looking for sensors
        # NB: for now, implementation of spheres use multiples Meshes
        # so this is a list. Ideally, only one mesh is necessary.
        pts = [node for node in subscene.children
               if node.__class__.__name__ == 'Mesh']

        # Do not pick the first Mesh, it's not a sensor
        pts = pts[1:]

        return _Projection(xy=xy, pts=pts)


def _3d_to_2d(xyz, canvas, camera):
    xyz = np.column_stack([xyz, np.ones(xyz.shape[0])])

    # Transform points into 'unnormalized' view coordinates
    comb_trans_mat = _get_world_to_view_matrix(camera)
    view_coords = np.dot(comb_trans_mat, xyz.T).T

    # Divide through by the fourth element for normalized view coords
    norm_view_coords = view_coords / (view_coords[:, 3].reshape(-1, 1))

    # Transform from normalized view coordinates to display coordinates.
    view_to_disp_mat = _get_view_to_display_matrix(canvas)
    xy = np.dot(view_to_disp_mat, norm_view_coords.T).T

    # Pull the first two columns since they're meaningful for 2d plotting
    xy = xy[:, :2]
    return xy


def _get_world_to_view_matrix(camera):
    return camera.transform.matrix


def _get_view_to_display_matrix(canvas):
    x, y = canvas.size
    view_to_disp_mat = np.array([[x / 2.0, 0., 0., x / 2.0],
                                 [0., -y / 2.0, 0., y / 2.0],
                                 [0., 0., 1., 0.],
                                 [0., 0., 0., 1.]])

    return view_to_disp_mat


def _create_quiver(mode, source, destination, scale, scale_mode='none',
                   scalar=None, resolution=8):
    from vispy.geometry.generation import (create_arrow, create_cylinder,
                                           create_cone)
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
    elif scale_mode == 'scalar' and scalar is not None:
        length = scale * scalar

    radius = length / 20.0

    meshdata = None
    if mode == 'arrow':
        cone_radius = radius * 3.0
        cone_length = length / 4.0
        meshdata = create_arrow(rows=resolution, cols=resolution,
                                length=length, radius=radius,
                                cone_radius=cone_radius,
                                cone_length=cone_length)
    elif mode == 'cone':
        meshdata = create_cone(cols=resolution, length=length,
                               radius=radius)
    elif mode == 'cylinder':
        meshdata = create_cylinder(rows=resolution, cols=resolution,
                                   length=length, radius=[radius, radius])

    # apply transform
    mat = MatrixTransform()
    if cosangle != 1:
        mat.rotate(np.degrees(np.arccos(cosangle)), axis)
    mat.translate(source)
    return meshdata, mat.matrix


def _close_all():
    pass
