"""
Core visualization operations based on ipyvolume.

Actual implementation of _Renderer and _Projection classes.
"""

import ipyvolume as ipv
import numpy as np
import pythreejs as p3js


class _Renderer(object):
    """Ipyvolume-based rendering."""
    def __init__(self, fig=None, size=(600, 600), bgcolor=(0., 0., 0.),
                 name="Ipyvolume Scene", show=False):
        # why do I need to check for it?
        from mne.viz.backends.renderer import MNE_3D_BACKEND_TEST_DATA
        self.off_screen = False
        self.name = name

        if MNE_3D_BACKEND_TEST_DATA:
            self.off_screen = True

        if fig is None:
            fig_w, fig_h = size
            self.plotter = ipv.figure(width=fig_w, height=fig_h)
            fig.animation = 0
            ipv.style.box_off()
            ipv.style.axes_off()
            ipv.style.background_color(bgcolor)
        else:
            # not sure this is the exact behavior, needs a thorough look
            self.plotter = ipv.figure(key=fig)

    def scene(self):
        return self.plotter

    def set_interactive(self):
        # what is this for?
        #self.plotter.enable_terrain_style()
        pass

    def mesh(self, x, y, z, triangles, color, opacity=1.0, shading=False,
             backface_culling=False, **kwargs):
        # opacity should be intergrated into color, i.e. rgba model
        ipv.plot_trisurf(x, y, z, triangles=triangles, color=color)

    def contour(self, surface, scalars, contours, line_width=1.0, opacity=1.0,
                vmin=None, vmax=None, colormap=None):
        # stub
        # there is no quick way to plot a contour with ipyvolume
        # what is vmin, vmax for?
        # opacity should be intergrated into colors, no other way for ipv
        # what is contours, line_width for?
        from matplotlib import cm
        from matplotlib.colors import ListedColormap

        if colormap is None:
            cmap = cm.get_cmap('coolwarm')
        else:
            cmap = ListedColormap(colormap / 255.0)

        vertices = np.array(surface['rr'])
        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]
        triangles = np.array(surface['tris'])
        color = cmap(scalars)

        ipv.plot_trisurf(x, y, z, triangles=triangles, color=color)

    def surface(self, surface, color=None, opacity=1.0,
                vmin=None, vmax=None, colormap=None, scalars=None,
                backface_culling=False):
        # what is vmin, vmax for?
        # opacity should be intergrated into colors, no other way for ipv
        # should we use ipv.plot_surface?
        from matplotlib import cm
        from matplotlib.colors import ListedColormap

        if colormap is None:
            cmap = cm.get_cmap('coolwarm')
        else:
            cmap = ListedColormap(colormap / 255.0)

        vertices = np.array(surface['rr'])
        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]
        triangles = np.array(surface['tris'])
        color = cmap(scalars)

        ipv.plot_trisurf(x, y, z, triangles=triangles, color=color)

    def sphere(self, center, color, scale, opacity=1.0,
               backface_culling=False):
        pass
        # it seems like there is no way to add a random geometry to the scene
        # so we might try to tweak ipyvolume to be able to do that??
        # what is scale for?
        # intergrate opacity into color, check whether it is possible
        # geometry = p3js.SphereGeometry()
        # # https://stackoverflow.com/questions/12835361/three-js-move-custom-geometry-to-origin
        # geometry.translate(*center)
        # material = p3js.MeshBasicMaterial({'color': color})
        # sphere = p3js.Mesh(geometry, material)
        # scene.add(sphere)

    def quiver3d(self, x, y, z, u, v, w, color, scale, mode, resolution=8,
                 glyph_height=None, glyph_center=None, glyph_resolution=None,
                 opacity=1.0, scale_mode='none', scalars=None,
                 backface_culling=False):
        # possible geo/marker values
        # this.geos = {
        #     diamond: this.geo_diamond,
        #     box: this.geo_box,
        #     arrow: this.geo_arrow,
        #     sphere: this.geo_sphere,
        #     cat: this.geo_cat,
        #     square_2d: this.geo_square_2d,
        #     point_2d: this.geo_point_2d,
        #     circle_2d: this.geo_circle_2d,
        #     triangle_2d: this.geo_triangle_2d
        # }
        ipv.quiver(x, y, z, u, v, w, marker=mode, color=color)

    def text(self, x, y, text, width, color=(1.0, 1.0, 1.0)):
        pass
        # how to add text in ipyvolume/pythreejs?
        # self.plotter.add_text(text, position=(x, y),
        #                       font_size=int(width * 100),
        #                       color=color)

    def show(self):
        ipv.show()

    def close(self):
        # ???
        self.plotter.close()

    def set_camera(self, azimuth=0.0, elevation=0.0, distance=1.0,
                   focalpoint=(0, 0, 0)):
        # edit this 
        phi = _deg2rad(azimuth)
        theta = _deg2rad(elevation)
        position = [
            distance * np.cos(phi) * np.sin(theta),
            distance * np.sin(phi) * np.sin(theta),
            distance * np.cos(theta)]
        self.plotter.camera_position = [
            position, focalpoint, [0, 0, 1]]
        self.plotter.reset_camera()

