"""Mayavi/traits GUI for averaging two sets of KIT marker points"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import cPickle as pickle
import os

from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.mlab import text3d
from mayavi.modules.glyph import Glyph
from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi.tools.mlab_scene_model import MlabSceneModel
import numpy as np
from pyface.api import confirm, error, FileDialog, OK, YES
from traits.api import HasTraits, on_trait_change, Instance, Property, \
                       Array, Bool, Button, Color, Enum, File, Float, List, \
                       Str, Tuple
from traitsui.api import View, Item, Group, HGroup, VGroup, CheckListEditor
from traitsui.menu import NoButtons
from tvtk.pyface.scene_editor import SceneEditor

from .coreg import fit_matched_pts
from .transforms import apply_trans, rotation, translation
from ..fiff.kit.coreg import read_mrk



out_wildcard = ("Pickled marker coordinates (*.pickled)|*.pickled|"
                "Hpi (text) file (*.hpi)|*.hpi")
out_ext = ['.pickled', '.hpi']

use_editor = CheckListEditor(cols=1, values=[(i, str(i)) for i in xrange(5)])



class ALSHeadViewController(HasTraits):
    """Set head views for Anterior-Left-Superior coordinate system"""
    right = Button()
    front = Button()
    left = Button()
    top = Button()
    view_scale = Float(0.13)

    scene = Instance(MlabSceneModel)

    view = View(Group(HGroup('72', Item('top', show_label=False), '100',
                             Item('view_scale', label='Scale')),
                      HGroup('right', 'front', 'left', show_labels=False),
                      label='View', show_border=True))

    @on_trait_change('scene.activated')
    def _init_view(self):
        self.scene.camera.view_up = (0, 0, 1)
        self.sync_trait('view_scale', self.scene.camera, 'parallel_scale')
        self.view_scale = 0.16
        self.front = True

    @on_trait_change('view_scale')
    def _on_view_scale_update(self):
        self.scene.camera.parallel_scale = self.view_scale
        self.scene.render()

    @on_trait_change('top,left,right,front')
    def on_set_view(self, view='front', info=None):
        self.scene.parallel_projection = True
        kwargs = dict(azimuth=0, elevation=90, distance=None, roll= -90,
                      reset_roll=True, figure=self.scene.mayavi_scene)
        if view == 'left':
            kwargs.update(azimuth=90, elevation=90, roll=180)
        elif view == 'right':
            kwargs.update(azimuth= -90, elevation=90, roll=0)
        elif view == 'top':
            kwargs.update(azimuth=0, elevation=0, roll= -90)
        self.scene.mlab.view(**kwargs)



class MarkerPoints(HasTraits):
    """Represent 5 marker points"""
    points = Array(float, (5, 3))
    label = Bool(False)
    color = Color()
    rgbcolor = Property(Tuple(Float, Float, Float), depends_on='color')
    save_as = Button()

    file = File
    name = Property(Str, depends_on='file')
    dir = Property(Str, depends_on='file')

    scene = Instance(MlabSceneModel, ())
    glyph = Instance(Glyph)
    src = Instance(VTKDataSource)

    view = View(VGroup('points', HGroup('color', 'save_as')))

    def _get_dir(self):
        return os.path.dirname(self.file)

    def _get_name(self):
        return os.path.basename(self.file)

    def _get_rgbcolor(self):
        return tuple(v / 255. for v in self.color.Get())

    @on_trait_change('label')
    def show_labels(self, show):
        self.scene.disable_render = True
        if hasattr(self, '_text3d'):
            for text in self._text3d:
                text.remove()
            del self._text3d

        if show:
            self._text3d = []
            fig = self.scene.mayavi_scene
            for i, pt in enumerate(np.array(self.points)):
                x, y, z = pt
                t = text3d(x, y, z, ' %i' % i, scale=.01, color=self.rgbcolor,
                           figure=fig)
                self._text3d.append(t)

        self.scene.disable_render = False

    def plot_points(self, scale=1e-2, color=None):
        from mayavi.tools import pipeline
        fig = self.scene.mayavi_scene
        if color is not None:
            self.color = tuple(int(c * 255) for c in color)

        x, y, z = self.points.T
        src = pipeline.scalar_scatter(x, y, z)
        glyph = pipeline.glyph(src, color=self.rgbcolor, figure=fig, scale_factor=scale,
                               opacity=1.)
        self.glyph = glyph
        self.src = src

        self.sync_trait('rgbcolor', self.glyph.actor.property, 'color', mutual=False)
        self.sync_trait('points', self.src.data, 'points', mutual=False)

    def _save_as_fired(self):
        dlg = FileDialog(action="save as", wildcard=out_wildcard,
                         default_filename=self.name,
                         default_directory=self.dir)
        dlg.open()
        if dlg.return_code != OK:
            return

        ext = out_ext[dlg.wildcard_index]
        path = dlg.path
        if not path.endswith(ext):
            path = path + ext

        if os.path.exists(path):
            answer = confirm(None, "The file %r already exists. Should it be "
                             "replaced?", "Overwrite File?")
            if answer != YES:
                return

        mrk = np.array(self.points)
        if ext == '.pickled':
            with open(path, 'w') as fid:
                pickle.dump(mrk, fid)
        elif ext == '.hpi':
            np.savetxt(path, mrk, fmt='%.18e', delimiter='\t', newline='\n')
        else:
            error(None, "Not Implemented: %r" % ext)



class MarkerPointSource(MarkerPoints):
    """MarkerPoints subclass for source files"""
    file = File(filter=['*.sqd'], exists=True)
    use = List(range(5), desc="Which points to use for the interpolated "
               "marker.")

    view = View(VGroup(Item('name', style='readonly'),
                       'file',
                       HGroup(
                              Item('use', editor=use_editor, style='custom'),
                              'points',
                              ),
                       HGroup('label',
                              Item('color', show_label=False),
                              Item('save_as', show_label=False)),
                       show_border=True, label="Source Marker"))

    @on_trait_change('file')
    def load(self, fname):
        pts = read_mrk(fname)
        self.points = pts



class MarkerPointDest(MarkerPoints):
    """MarkerPoints subclass that serves for derived points"""
    src1 = Instance(MarkerPointSource)
    src2 = Instance(MarkerPointSource)

    name = Property(Str, depends_on='src1.name,src2.name')
    dir = Property(Str, depends_on='src1.dir,src2.dir')

    method = Enum('Transform', 'Average', desc="Transform: estimate a rotation"
                  "/translation from mrk1 to mrk2; Average: use the average "
                  "of the mrk1 and mrk2 coordinates for each point.")

    view = View(VGroup(Item('method', style='custom'),
                       HGroup('label', 'color',
                              Item('save_as', show_label=False)),
                       show_border=True, label="New Marker"))

    def _get_dir(self):
        return self.src1.dir

    def _get_name(self):
        i = 0
        n1 = self.src1.name
        n2 = self.src2.name
        while n1[i] == n2[i]:
            i += 1

        return n1[:i]

    @on_trait_change('method')
    def update(self):
        if self.method == 'Average':
            if len(np.union1d(self.src1.use, self.src2.use)) < 5:
                error("Need at least one source for each point.")
                return

            pts = (self.src1.points + self.src2.points) / 2
            for i in np.setdiff1d(self.src1.use, self.src2.use):
                pts[i] = self.src1.points[i]
            for i in np.setdiff1d(self.src2.use, self.src1.use):
                pts[i] = self.src2.points[i]

            self.points = pts
            return

        idx = np.intersect1d(self.src1.use, self.src2.use, assume_unique=True)
        if len(idx) < 3:
            error("Need at least three shared points for transformation.")
            return

        src_pts = self.src1.points[idx]
        tgt_pts = self.src2.points[idx]
        _, rot, tra = fit_matched_pts(src_pts, tgt_pts, params=True)

        if len(self.src1.use) == 5:
            trans = np.dot(translation(*(tra / 2)), rotation(*(rot / 2)))
            pts = apply_trans(trans, self.src1.points)
        elif len(self.src2.use) == 5:
            trans = np.dot(translation(*(-tra / 2)), rotation(*(-rot / 2)))
            pts = apply_trans(trans, self.src2.points)
        else:
            trans1 = np.dot(translation(*(tra / 2)), rotation(*(rot / 2)))
            pts = apply_trans(trans1, self.src1.points)
            trans2 = np.dot(translation(*(-tra / 2)), rotation(*(-rot / 2)))
            for i in np.setdiff1d(self.src2.use, self.src1.use):
                pts[i] = apply_trans(trans2, self.src2.points[i])
        self.points = pts



class ControlPanel(HasTraits):
    """Has two marker points sources and interpolates to a third one"""
    mrk1 = File
    mrk2 = File
    markers_1 = Instance(MarkerPointSource)
    markers_2 = Instance(MarkerPointSource)
    markers = Instance(MarkerPointDest)
    scene = Instance(MlabSceneModel, ())
    head_view = Instance(ALSHeadViewController)

    def _head_view_default(self):
        return ALSHeadViewController(scene=self.scene)

    def _markers_default(self):
        mrk = MarkerPointDest(scene=self.scene, src1=self.markers_1,
                              src2=self.markers_2)
        return mrk

    def _markers_1_default(self):
        if os.path.exists(self.mrk1):
            return MarkerPointSource(scene=self.scene, file=self.mrk1)
        else:
            return MarkerPointSource(scene=self.scene)

    def _markers_2_default(self):
        if os.path.exists(self.mrk2):
            return MarkerPointSource(scene=self.scene, file=self.mrk2)
        else:
            return MarkerPointSource(scene=self.scene)

    view = View(VGroup(Item('markers_1', springy=True, style='custom'),
                       Item('markers_2', style='custom'),
                       Item('markers', style='custom'),
                       Item('head_view', style='custom'),
                       show_labels=False,
                       ))

    @on_trait_change('scene.activated')
    def _init_plot(self):
        self.markers_1.plot_points(color=(.1, .9, 1))
        self.markers_2.plot_points(color=(1, .5, .1))
        self.markers.plot_points(color=(0, 0, 0))
        self.markers.update()

        self.markers_1.on_trait_change(self.markers.update, 'points')
        self.markers_1.on_trait_change(self.markers.update, 'use')
        self.markers_2.on_trait_change(self.markers.update, 'points')
        self.markers_2.on_trait_change(self.markers.update, 'use')



class MainWindow(HasTraits):
    """GUI for interpolating between two KIT marker files

    Parameters
    ----------
    mrk1, mrk2 : str
        Path to pre- and post measurement marker files (*.sqd) or empty string.
    """
    panel = Instance(ControlPanel)
    scene = Instance(MlabSceneModel, ())

    def _panel_default(self):
        return ControlPanel(scene=self.scene, mrk1=self._mrk1, mrk2=self._mrk2)

    view = View(HGroup(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                            dock='vertical'),
                       Item('panel', style="custom"),
                       show_labels=False,
                      ),
                resizable=True,
                height=0.75, width=0.75,
                buttons=NoButtons)

    def __init__(self, mrk1='', mrk2=''):
        self._mrk1 = mrk1
        self._mrk2 = mrk2
