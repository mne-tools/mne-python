"""Mayavi/traits GUI for averaging two sets of KIT marker points"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import cPickle as pickle
import os

from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.modules.glyph import Glyph
from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi.tools.mlab_scene_model import MlabSceneModel
import numpy as np
from pyface.api import confirm, error, FileDialog, OK, YES
from traits.api import HasTraits, HasPrivateTraits, on_trait_change, cached_property, \
                       Instance, Property, \
                       Array, Bool, Button, Color, Enum, File, Float, List, \
                       Str, Tuple
from traitsui.api import View, Item, Group, HGroup, VGroup, CheckListEditor
from traitsui.menu import NoButtons
from tvtk.pyface.scene_editor import SceneEditor

from .coreg import fit_matched_pts
from .transforms import apply_trans, rotation, translation
from .viewer import HeadViewController, headview_borders, PointObject
from ..fiff.kit.coreg import read_mrk



out_wildcard = ("Pickled KIT parameters (*.pickled)|*.pickled|"
                "Tab separated values file (*.txt)|*.txt")
out_ext = ['.pickled', '.txt']

use_editor = CheckListEditor(cols=1, values=[(i, str(i)) for i in xrange(5)])



class MarkerPoints(HasPrivateTraits):
    """Represent 5 marker points"""
    points = Array(float, (5, 3))

    can_save = Property(depends_on='points')
    save_as = Button()

    view = View(VGroup('points',
                       Item('save_as', enabled_when='can_save')))

    @cached_property
    def _get_can_save(self):
        return np.any(self.points)

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
                pickle.dump({'mrk': mrk}, fid)
        elif ext == '.txt':
            np.savetxt(path, mrk, fmt='%.18e', delimiter='\t', newline='\n')



class MarkerPointSource(MarkerPoints):
    """MarkerPoints subclass for source files"""
    file = File(filter=['Sqd marker file (*.sqd)|*.sqd',
                        'Text marker file (*.txt)|*.txt',
                        'Pickled markers (*.pickled)|*.pickled'], exists=True)
    name = Property(Str, depends_on='file')
    dir = Property(Str, depends_on='file')

    use = List(range(5), desc="Which points to use for the interpolated "
               "marker.")
    enabled = Property(Bool, depends_on=['points', 'use'])
    clear = Button(desc="Clear the current marker data")

    view = View(VGroup('file',
                       Item('name', show_label=False, style='readonly'),
                       HGroup(
                              Item('use', editor=use_editor, enabled_when="enabled", style='custom'),
                              'points',
                              ),
                       HGroup(Item('clear', enabled_when="can_save", show_label=False),
                              Item('save_as', enabled_when="can_save", show_label=False)),
                       ))

    @cached_property
    def _get_enabled(self):
        return np.any(self.points)

    @cached_property
    def _get_dir(self):
        if self.file:
            return os.path.dirname(self.file)

    @cached_property
    def _get_name(self):
        if self.file:
            return os.path.basename(self.file)

    @on_trait_change('file')
    def load(self, fname):
        if not fname:
            self.reset_traits(['points'])
            return

        try:
            pts = read_mrk(fname)
        except Exception as err:
            error(None, str(err), "Error Reading mrk")
            self.reset_traits(['points'])
        else:
            self.points = pts

    def _clear_fired(self):
        self.reset_traits(['file', 'points'])



class MarkerPointDest(MarkerPoints):
    """MarkerPoints subclass that serves for derived points"""
    src1 = Instance(MarkerPointSource)
    src2 = Instance(MarkerPointSource)

    name = Property(Str, depends_on='src1.name,src2.name')
    dir = Property(Str, depends_on='src1.dir,src2.dir')

    points = Property(Array(float, (5, 3)),
                      depends_on=['method', 'src1.points', 'src1.use',
                                  'src2.points', 'src2.use'])
    enabled = Property(Bool, depends_on=['points'])

    method = Enum('Transform', 'Average', desc="Transform: estimate a rotation"
                  "/translation from mrk1 to mrk2; Average: use the average "
                  "of the mrk1 and mrk2 coordinates for each point.")

    view = View(VGroup(Item('method', style='custom'),
                       Item('save_as', enabled_when='can_save', show_label=False)))

    @cached_property
    def _get_dir(self):
        return self.src1.dir

    @cached_property
    def _get_name(self):
        n1 = self.src1.name
        n2 = self.src2.name

        if not n1:
            if n2:
                return n2
            else:
                return ''
        elif not n2:
            return n1

        if n1 == n2:
            return n1

        i = 0
        l1 = len(n1) - 1
        l2 = len(n1) - 2
        while n1[i] == n2[i]:
            if i == l1:
                return n1
            elif i == l2:
                return n2

            i += 1

        return n1[:i]

    @cached_property
    def _get_enabled(self):
        return np.any(self.points)

    @cached_property
    def _get_points(self):
        # in case only one or no source is enabled
        if not (self.src1 and self.src1.enabled):
            if (self.src2 and self.src2.enabled):
                return self.src2.points
            else:
                return np.zeros((5, 3))
        elif not (self.src2 and self.src2.enabled):
            return self.src1.points

        # Average method
        if self.method == 'Average':
            if len(np.union1d(self.src1.use, self.src2.use)) < 5:
                error(None, "Need at least one source for each point.",
                      "Marker Average Error")
                return np.zeros((5, 3))

            pts = (self.src1.points + self.src2.points) / 2
            for i in np.setdiff1d(self.src1.use, self.src2.use):
                pts[i] = self.src1.points[i]
            for i in np.setdiff1d(self.src2.use, self.src1.use):
                pts[i] = self.src2.points[i]

            return pts

        # Transform method
        idx = np.intersect1d(self.src1.use, self.src2.use, assume_unique=True)
        if len(idx) < 3:
            error(None, "Need at least three shared points for trans"
                  "formation.", "Marker Interpolation Error")
            return np.zeros((5, 3))

        src_pts = self.src1.points[idx]
        tgt_pts = self.src2.points[idx]
        est = fit_matched_pts(src_pts, tgt_pts)
        rot = np.array(est[:3]) / 2
        tra = np.array(est[3:]) / 2

        if len(self.src1.use) == 5:
            trans = np.dot(translation(*tra), rotation(*rot))
            pts = apply_trans(trans, self.src1.points)
        elif len(self.src2.use) == 5:
            trans = np.dot(translation(* -tra), rotation(* -rot))
            pts = apply_trans(trans, self.src2.points)
        else:
            trans1 = np.dot(translation(*tra), rotation(*rot))
            pts = apply_trans(trans1, self.src1.points)
            trans2 = np.dot(translation(* -tra), rotation(* -rot))
            for i in np.setdiff1d(self.src2.use, self.src1.use):
                pts[i] = apply_trans(trans2, self.src2.points[i])

        return pts



class MarkerPanel(HasTraits):
    """Has two marker points sources and interpolates to a third one"""
    mrk1_file = File
    mrk2_file = File
    mrk1 = Instance(MarkerPointSource)
    mrk2 = Instance(MarkerPointSource)
    mrk3 = Instance(MarkerPointDest)

    # Visualization
    scene = Instance(MlabSceneModel, ())
    scale = Float(5e-3)
    mrk1_obj = Instance(PointObject)
    mrk2_obj = Instance(PointObject)
    mrk3_obj = Instance(PointObject)

    def _mrk1_default(self):
        if os.path.exists(self.mrk1_file):
            return MarkerPointSource(file=self.mrk1_file)
        else:
            return MarkerPointSource()

    def _mrk2_default(self):
        if os.path.exists(self.mrk2_file):
            return MarkerPointSource(file=self.mrk2_file)
        else:
            return MarkerPointSource()

    def _mrk3_default(self):
        mrk = MarkerPointDest(src1=self.mrk1, src2=self.mrk2)
        return mrk

    view = View(VGroup(VGroup(Item('mrk1', style='custom'),
                              Item('mrk1_obj', style='custom'),
                              show_labels=False,
                              label="Source Marker 1", show_border=True),
                       VGroup(Item('mrk2', style='custom'),
                              Item('mrk2_obj', style='custom'),
                              show_labels=False,
                              label="Source Marker 2", show_border=True),
                       VGroup(Item('mrk3', style='custom'),
                              Item('mrk3_obj', style='custom'),
                              show_labels=False,
                              label="New Marker", show_border=True),
                       ))

    @on_trait_change('scene.activated')
    def _init_plot(self):
        self.mrk1_obj = PointObject(scene=self.scene, color=(155, 55, 55),
                                    point_scale=self.scale)
        self.mrk1.sync_trait('points', self.mrk1_obj, 'points', mutual=False)
        self.mrk1.sync_trait('enabled', self.mrk1_obj, 'visible', mutual=False)

        self.mrk2_obj = PointObject(scene=self.scene, color=(55, 155, 55),
                                    point_scale=self.scale)
        self.mrk2.sync_trait('points', self.mrk2_obj, 'points', mutual=False)
        self.mrk2.sync_trait('enabled', self.mrk2_obj, 'visible', mutual=False)

        self.mrk3_obj = PointObject(scene=self.scene, color=(150, 200, 255),
                                    point_scale=self.scale)
        self.mrk3.sync_trait('points', self.mrk3_obj, 'points', mutual=False)
        self.mrk3.sync_trait('enabled', self.mrk3_obj, 'visible', mutual=False)



class MainWindow(HasTraits):
    """GUI for interpolating between two KIT marker files

    Parameters
    ----------
    mrk1, mrk2 : str
        Path to pre- and post measurement marker files (*.sqd) or empty string.
    """
    scene = Instance(MlabSceneModel, ())
    headview = Instance(HeadViewController)
    panel = Instance(MarkerPanel)

    def _headview_default(self):
        return HeadViewController(scene=self.scene, system='ALS')

    def _panel_default(self):
        return MarkerPanel(scene=self.scene, mrk1_file=self._mrk1, mrk2_file=self._mrk2)

    view = View(HGroup(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                            dock='vertical'),
                       VGroup(headview_borders,
                              Item('panel', style="custom"),
                              show_labels=False),
                       show_labels=False,
                      ),
                resizable=True,
                height=0.75, width=0.75,
                buttons=NoButtons)

    def __init__(self, mrk1='', mrk2=''):
        self._mrk1 = mrk1
        self._mrk2 = mrk2
        super(MainWindow, self).__init__()
