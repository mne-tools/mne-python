"""Mayavi/traits GUI visualization elements"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os
import numpy as np

# allow import without traits
try:
    from mayavi.mlab import pipeline, text3d
    from mayavi.modules.glyph import Glyph
    from mayavi.modules.surface import Surface
    from mayavi.sources.vtk_data_source import VTKDataSource
    from mayavi.tools.mlab_scene_model import MlabSceneModel
    from pyface.api import error
    from traits.api import (HasTraits, HasPrivateTraits, on_trait_change,
                            cached_property, Instance, Property, Array, Bool,
                            Button, Color, Enum, Float, Int, List, Range, Str)
    from traitsui.api import View, Item, Group, HGroup, VGrid, VGroup
except:
    from ..utils import trait_wraith
    HasTraits = HasPrivateTraits = object
    cached_property = on_trait_change = MlabSceneModel = Array = Bool = \
        Button = Color = Enum = Float = Instance = Int = List = Property = \
        Range = Str = View = Item = Group = HGroup = VGrid = VGroup = \
        Glyph = Surface = VTKDataSource = trait_wraith

from ..transforms import apply_trans


headview_item = Item('headview', style='custom', show_label=False)
headview_borders = VGroup(Item('headview', style='custom', show_label=False),
                          show_border=True, label='View')
defaults = {'mri_fid_scale': 1e-2, 'hsp_fid_scale': 3e-2,
            'hsp_fid_opacity': 0.3, 'hsp_points_scale': 4e-3,
            'mri_color': (252, 227, 191), 'hsp_point_color': (255, 255, 255),
            'lpa_color': (255, 0, 0), 'nasion_color': (0, 255, 0),
            'rpa_color': (0, 0, 255)}


def _testing_mode():
    """Helper to determine if we're running tests"""
    return (os.getenv('_MNE_GUI_TESTING_MODE', '') == 'true')


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

    view = View(VGrid('0', 'top', '0', Item('scale', label='Scale',
                                            show_label=True),
                      'right', 'front', 'left', show_labels=False, columns=4))

    @on_trait_change('scene.activated')
    def _init_view(self):
        self.scene.parallel_projection = True

        # apparently scene,activated happens several times
        if self.scene.renderer:
            self.sync_trait('scale', self.scene.camera, 'parallel_scale')
            # and apparently this does not happen by default:
            self.on_trait_change(self.scene.render, 'scale')

    @on_trait_change('top,left,right,front')
    def on_set_view(self, view, _):
        if self.scene is None:
            return

        system = self.system
        kwargs = None

        if system == 'ALS':
            if view == 'front':
                kwargs = dict(azimuth=0, elevation=90, roll=-90)
            elif view == 'left':
                kwargs = dict(azimuth=90, elevation=90, roll=180)
            elif view == 'right':
                kwargs = dict(azimuth=-90, elevation=90, roll=0)
            elif view == 'top':
                kwargs = dict(azimuth=0, elevation=0, roll=-90)
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
                kwargs = dict(azimuth=-90, elevation=90, roll=180)
            elif view == 'right':
                kwargs = dict(azimuth=90, elevation=90, roll=0)
            elif view == 'top':
                kwargs = dict(azimuth=0, elevation=180, roll=90)
        else:
            raise ValueError("Invalid system: %r" % system)

        if kwargs is None:
            raise ValueError("Invalid view: %r" % view)

        if not _testing_mode():
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
        if hasattr(self.color, 'Get'):  # wx
            color = tuple(v / 255. for v in self.color.Get())
        else:
            color = self.color.getRgbF()[:3]
        return color

    @on_trait_change('trans,points')
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


class PointObject(Object):
    """Represents a group of individual points in a mayavi scene"""
    label = Bool(False, enabled_when='visible')
    text3d = List

    glyph = Instance(Glyph)
    resolution = Int(8)

    def __init__(self, view='points', *args, **kwargs):
        """
        Parameters
        ----------
        view : 'points' | 'cloud'
            Whether the view options should be tailored to individual points
            or a point cloud.
        """
        self._view = view
        super(PointObject, self).__init__(*args, **kwargs)

    def default_traits_view(self):
        color = Item('color', show_label=False)
        scale = Item('point_scale', label='Size')
        if self._view == 'points':
            visible = Item('visible', label='Show', show_label=True)
            view = View(HGroup(visible, color, scale, 'label'))
        elif self._view == 'cloud':
            visible = Item('visible', show_label=False)
            view = View(HGroup(visible, color, scale))
        else:
            raise ValueError("PointObject(view = %r)" % self._view)
        return view

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
#         _scale = self.scene.camera.parallel_scale

        if hasattr(self.glyph, 'remove'):
            self.glyph.remove()
        if hasattr(self.src, 'remove'):
            self.src.remove()

        if not _testing_mode():
            fig = self.scene.mayavi_scene
        else:
            fig = None

        x, y, z = self.points.T
        scatter = pipeline.scalar_scatter(x, y, z)
        glyph = pipeline.glyph(scatter, color=self.rgbcolor, figure=fig,
                               scale_factor=self.point_scale, opacity=1.,
                               resolution=self.resolution)
        self.src = scatter
        self.glyph = glyph

        self.sync_trait('point_scale', self.glyph.glyph.glyph, 'scale_factor')
        self.sync_trait('rgbcolor', self.glyph.actor.property, 'color',
                        mutual=False)
        self.sync_trait('visible', self.glyph)
        self.sync_trait('opacity', self.glyph.actor.property)
        self.on_trait_change(self._update_points, 'points')

#         self.scene.camera.parallel_scale = _scale

    def _resolution_changed(self, new):
        if not self.glyph:
            return

        self.glyph.glyph.glyph_source.glyph_source.phi_resolution = new
        self.glyph.glyph.glyph_source.glyph_source.theta_resolution = new


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

    view = View(HGroup(Item('visible', show_label=False),
                       Item('color', show_label=False), Item('opacity')))

    def clear(self):
        if hasattr(self.src, 'remove'):
            self.src.remove()
        if hasattr(self.surf, 'remove'):
            self.surf.remove()
        self.reset_traits(['src', 'surf'])

    @on_trait_change('scene.activated')
    def plot(self):
        """Add the points to the mayavi pipeline"""
        _scale = self.scene.camera.parallel_scale if not _testing_mode() else 1
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
        surf = pipeline.surface(src, figure=fig, color=self.rgbcolor,
                                opacity=self.opacity,
                                representation=rep, line_width=1)

        self.src = src
        self.surf = surf

        self.sync_trait('visible', self.surf, 'visible')
        self.sync_trait('rgbcolor', self.surf.actor.property, 'color',
                        mutual=False)
        self.sync_trait('opacity', self.surf.actor.property, 'opacity')

        if not _testing_mode():
            self.scene.camera.parallel_scale = _scale
