"""Mayavi/traits GUI elements"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

from mayavi.mlab import pipeline, text3d
from mayavi.modules.glyph import Glyph
from mayavi.modules.surface import Surface
from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.modules.text3d import Text3D
import numpy as np
from pyface.api import error
from scipy.spatial import Delaunay
from traits.api import HasTraits, HasPrivateTraits, on_trait_change, cached_property, Instance, Property, \
                       Array, Bool, Button, Color, Enum, Float, List, Range, Str, Tuple
from traitsui.api import View, Item, Group, HGroup, VGroup

from .transforms import apply_trans


headview_borders = VGroup(Item('headview', style='custom', show_label=False),
                          show_border=True, label='View')


class HeadViewController(HasTraits):
    """
    Set head views for Anterior-Left-Superior coordinate system

    Parameters
    ----------
    system : 'RAS' | 'ALS' | 'ARI'
        Coordinate system described as initials for directions associated with
        the x, y, and z axes. Relevant terms are: Anterior, Right, Left,
        Superior, Inferior.
    """
    system = Enum("RAS", "ALS", "ARI", desc="Coordinate system: directions of "
                  "the x, y, and z axis.")

    right = Button()
    front = Button()
    left = Button()
    top = Button()

    scale = Float(0.16)

    scene = Instance(MlabSceneModel)

    view = View(Group(HGroup('72', Item('top', show_label=False), '100',
                             Item('scale', label='Scale')),
                      HGroup('right', 'front', 'left', show_labels=False)))

    @on_trait_change('scene.activated')
    def _init_view(self):
        self.scene.parallel_projection = True
        self.sync_trait('scale', self.scene.camera, 'parallel_scale')
        # this alone seems not to be enough to sync the camera scale (see
        # ._on_view_scale_update() method below

    @on_trait_change('scale')
    def _on_view_scale_update(self):
        if self.scene is not None:
            self.scene.camera.parallel_scale = self.scale
            self.scene.render()

    @on_trait_change('top,left,right,front')
    def on_set_view(self, view, _):
        if self.scene is None:
            return

        system = self.system
        kwargs = None

        if system == 'ALS':
            if view == 'front':
                kwargs = dict(azimuth=0, elevation=90, roll= -90)
            elif view == 'left':
                kwargs = dict(azimuth=90, elevation=90, roll=180)
            elif view == 'right':
                kwargs = dict(azimuth= -90, elevation=90, roll=0)
            elif view == 'top':
                kwargs = dict(azimuth=0, elevation=0, roll= -90)
        elif system == 'RAS':
            if view == 'front':
                kwargs = dict(azimuth=90, elevation=90, roll=180)
            elif view == 'left':
                kwargs = dict(azimuth=180, elevation=90, roll=90)
            elif view == 'right':
                kwargs = dict(azimuth=0, elevation=90, roll=270)
            elif view == 'top':
                kwargs = dict(azimuth=90, elevation=0, roll=180)
        elif system == 'ARI':
            if view == 'front':
                kwargs = dict(azimuth=0, elevation=90, roll=90)
            elif view == 'left':
                kwargs = dict(azimuth= -90, elevation=90, roll=180)
            elif view == 'right':
                kwargs = dict(azimuth=90, elevation=90, roll=0)
            elif view == 'top':
                kwargs = dict(azimuth=0, elevation=180, roll=90)
        else:
            raise ValueError("Invalid system: %r" % system)

        if kwargs is None:
            raise ValueError("Invalid view: %r" % view)

        self.scene.mlab.view(distance=None, reset_roll=True,
                             figure=self.scene.mayavi_scene, **kwargs)



class Object(HasPrivateTraits):
    """Represents a 3d object in a mayavi scene"""
    points = Array(float, shape=(None, 3))
    trans = Array()
    name = Str

    scene = Instance(MlabSceneModel, ())
    src = Instance(VTKDataSource)

    color = Color()
    rgbcolor = Property(depends_on='color')
    point_scale = Float(10, label='Point Scale')
    opacity = Range(low=0., high=1., value=1.)
    visible = Bool(True)

    @cached_property
    def _get_rgbcolor(self):
        return tuple(v / 255. for v in self.color.Get())

    @on_trait_change('trans')
    def _update_points(self):
        """Update the location of the plotted points"""
        if not hasattr(self.src, 'data'):
            return

        trans = self.trans
        if np.any(trans):
            if trans.ndim == 0 or trans.shape == (3,) or trans.shape == (1, 3):
                pts = self.points * trans
            elif trans.shape == (3, 3):
                pts = np.dot(self.points, trans.T)
            elif trans.shape == (4, 4):
                pts = apply_trans(trans, self.points)
            else:
                err = ("trans must be a scalar, a length 3 sequence, or an "
                       "array of shape (1,3), (3, 3) or (4, 4). "
                       "Got %s" % str(trans))
                error(None, err, "Display Error")
                raise ValueError(err)
        else:
            pts = self.points

        self.src.data.points = pts
        self.scene.reset_zoom()




class PointObject(Object):
    """Represents a group of individual opints in a mayavi scene"""
    label = Bool(False, enabled_when='visible')
    text3d = List  # (None, [])

    glyph = Instance(Glyph)

    view = View(HGroup(Item('point_scale', label='Size'), 'color',
                      'visible', 'label'))

    @on_trait_change('label')
    def _show_labels(self, show):
        self.scene.disable_render = True
        while self.text3d:
            text = self.text3d.pop()
            text.remove()

        if show:
            fig = self.scene.mayavi_scene
            for i, pt in enumerate(np.array(self.src.data.points)):
                x, y, z = pt
                t = text3d(x, y, z, ' %i' % i, scale=.01, color=self.rgbcolor,
                           figure=fig)
                self.text3d.append(t)

        self.scene.disable_render = False

    @on_trait_change('visible')
    def _on_hide(self):
        if not self.visible:
            self.label = False

    @on_trait_change('scene.activated')
    def _plot_points(self):
        """Add the points to the mayavi pipeline"""
        _scale = self.scene.camera.parallel_scale

        if hasattr(self.glyph, 'remove'):
            self.glyph.remove()
        if hasattr(self.src, 'remove'):
            self.src.remove()

        fig = self.scene.mayavi_scene

        x, y, z = self.points.T
        scatter = pipeline.scalar_scatter(x, y, z)
        glyph = pipeline.glyph(scatter, color=self.rgbcolor, figure=fig,
                               scale_factor=self.point_scale, opacity=1.)
        self.src = scatter
        self.glyph = glyph

        self.sync_trait('point_scale', self.glyph.glyph.glyph, 'scale_factor')
        self.sync_trait('rgbcolor', self.glyph.actor.property, 'color', mutual=False)
        self.sync_trait('visible', self.glyph, 'visible')
        self.sync_trait('opacity', self.glyph.actor.property, 'opacity')
        self.on_trait_change(self._update_points, 'points')

        self.scene.camera.parallel_scale = _scale



class SurfaceObject(Object):
    """Represents a solid object in a mayavi scene

    Notes
    -----
    Doesn't automatically update plot because update requires both
    :attr:`points` and :attr:`tri`. Call :meth:`plot` after updateing both
    attributes.

    """
    rep = Enum("Surface", "Wireframe")
    tri = Array(int, shape=(None, 3))

    surf = Instance(Surface)

    view = View(HGroup('color', 'visible', 'opacity'))

    def clear(self):
        if hasattr(self.src, 'remove'):
            self.src.remove()
        if hasattr(self.surf, 'remove'):
            self.surf.remove()

    @on_trait_change('scene.activated')
    def plot(self):
        """Add the points to the mayavi pipeline"""
        _scale = self.scene.camera.parallel_scale
        self.clear()

        if not np.any(self.tri):
            return

        fig = self.scene.mayavi_scene

        x, y, z = self.points.T

        if self.rep == 'Wireframe':
            rep = 'wireframe'
        else:
            rep = 'surface'

        src = pipeline.triangular_mesh_source(x, y, z, self.tri, figure=fig)
        surf = pipeline.surface(src, figure=fig, color=self.rgbcolor, opacity=self.opacity,
                                representation=rep, line_width=1)

        self.src = src
        self.surf = surf

        self.sync_trait('visible', self.surf, 'visible')
        self.sync_trait('rgbcolor', self.surf.actor.property, 'color', mutual=False)
        self.sync_trait('opacity', self.surf.actor.property, 'opacity')

        self.scene.camera.parallel_scale = _scale
