"""Mayavi/traits GUI for averaging two sets of KIT marker points."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os
import sys

import numpy as np

from mayavi.tools.mlab_scene_model import MlabSceneModel
from pyface.api import confirm, error, FileDialog, OK, YES
from traits.api import (HasTraits, HasPrivateTraits, on_trait_change,
                        cached_property, Instance, Property, Array, Bool,
                        Button, Enum, File, Float, List, Str, ArrayOrNone)
from traitsui.api import View, Item, HGroup, VGroup, CheckListEditor
from traitsui.menu import Action, CancelButton

from ..transforms import apply_trans, rotation, translation
from ..coreg import fit_matched_points
from ..io.kit import read_mrk
from ..io._digitization import _write_dig_points
from ._viewer import PointObject
from ._backend import _get_pyface_backend


if _get_pyface_backend() == 'wx':
    mrk_wildcard = [
        'Supported Files (*.sqd, *.mrk, *.txt, *.pickled)|*.sqd;*.mrk;*.txt;*.pickled',  # noqa:E501
        'Sqd marker file (*.sqd;*.mrk)|*.sqd;*.mrk',
        'Text marker file (*.txt)|*.txt',
        'Pickled markers (*.pickled)|*.pickled']
    mrk_out_wildcard = ["Tab separated values file (*.txt)|*.txt"]
else:
    if sys.platform in ('win32', 'linux2'):
        # on Windows and Ubuntu, multiple wildcards does not seem to work
        mrk_wildcard = ["*.sqd", "*.mrk", "*.txt", "*.pickled"]
    else:
        mrk_wildcard = ["*.sqd;*.mrk;*.txt;*.pickled"]
    mrk_out_wildcard = "*.txt"
out_ext = '.txt'


use_editor_v = CheckListEditor(cols=1, values=[(i, str(i)) for i in range(5)])
use_editor_h = CheckListEditor(cols=5, values=[(i, str(i)) for i in range(5)])

mrk_view_editable = View(
    VGroup('file',
           Item('name', show_label=False, style='readonly'),
           HGroup(
               Item('use', editor=use_editor_v, enabled_when="enabled",
                    style='custom'),
               'points',
           ),
           HGroup(Item('clear', enabled_when="can_save", show_label=False),
                  Item('save_as', enabled_when="can_save",
                       show_label=False)),
           ))

mrk_view_basic = View(
    VGroup('file',
           Item('name', show_label=False, style='readonly'),
           Item('use', editor=use_editor_h, enabled_when="enabled",
                style='custom'),
           HGroup(Item('clear', enabled_when="can_save", show_label=False),
                  Item('edit', show_label=False),
                  Item('switch_left_right', label="Switch Left/Right",
                       show_label=False),
                  Item('reorder', show_label=False),
                  Item('save_as', enabled_when="can_save",
                       show_label=False)),
           ))

mrk_view_edit = View(VGroup('points'))


class ReorderDialog(HasPrivateTraits):
    """Dialog for reordering marker points."""

    order = Str("0 1 2 3 4")
    index = Property(List, depends_on='order')
    is_ok = Property(Bool, depends_on='order')

    view = View(
        Item('order', label='New order (five space delimited numbers)'),
        buttons=[CancelButton, Action(name='OK', enabled_when='is_ok')])

    def _get_index(self):
        try:
            return [int(i) for i in self.order.split()]
        except ValueError:
            return []

    def _get_is_ok(self):
        return sorted(self.index) == [0, 1, 2, 3, 4]


class MarkerPoints(HasPrivateTraits):
    """Represent 5 marker points."""

    points = Array(float, (5, 3))

    can_save = Property(depends_on='points')
    save_as = Button()

    view = View(VGroup('points',
                       Item('save_as', enabled_when='can_save')))

    @cached_property
    def _get_can_save(self):
        return np.any(self.points)

    def _save_as_fired(self):
        dlg = FileDialog(action="save as", wildcard=mrk_out_wildcard,
                         default_filename=self.name,
                         default_directory=self.dir)
        dlg.open()
        if dlg.return_code != OK:
            return

        path, ext = os.path.splitext(dlg.path)
        if not path.endswith(out_ext) and len(ext) != 0:
            ValueError("The extension '%s' is not supported." % ext)
        path = path + out_ext

        if os.path.exists(path):
            answer = confirm(None, "The file %r already exists. Should it "
                             "be replaced?", "Overwrite File?")
            if answer != YES:
                return
        self.save(path)

    def save(self, path):
        """Save the marker points.

        Parameters
        ----------
        path : str
            Path to the file to write. The kind of file to write is determined
            based on the extension: '.txt' for tab separated text file,
            '.pickled' for pickled file.
        """
        _write_dig_points(path, self.points)


class MarkerPointSource(MarkerPoints):  # noqa: D401
    """MarkerPoints subclass for source files."""

    file = File(filter=mrk_wildcard, exists=True)
    name = Property(Str, depends_on='file')
    dir = Property(Str, depends_on='file')

    use = List(list(range(5)), desc="Which points to use for the interpolated "
               "marker.")
    enabled = Property(Bool, depends_on=['points', 'use'])
    clear = Button(desc="Clear the current marker data")
    edit = Button(desc="Edit the marker coordinates manually")
    switch_left_right = Button(
        desc="Switch left and right marker points; this is intended to "
             "correct for markers that were attached in the wrong order")
    reorder = Button(desc="Change the order of the marker points")

    view = mrk_view_basic

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
        self.reset_traits(['file', 'points', 'use'])

    def _edit_fired(self):
        self.edit_traits(view=mrk_view_edit)

    def _reorder_fired(self):
        dlg = ReorderDialog()
        ui = dlg.edit_traits(kind='modal')
        if not ui.result:  # user pressed cancel
            return
        self.points = self.points[dlg.index]

    def _switch_left_right_fired(self):
        self.points = self.points[[1, 0, 2, 4, 3]]


class MarkerPointDest(MarkerPoints):  # noqa: D401
    """MarkerPoints subclass that serves for derived points."""

    src1 = Instance(MarkerPointSource)
    src2 = Instance(MarkerPointSource)

    name = Property(Str, depends_on='src1.name,src2.name')
    dir = Property(Str, depends_on='src1.dir,src2.dir')

    points = Property(ArrayOrNone(float, (5, 3)),
                      depends_on=['method', 'src1.points', 'src1.use',
                                  'src2.points', 'src2.use'])
    enabled = Property(Bool, depends_on=['points'])

    method = Enum('Transform', 'Average', desc="Transform: estimate a rotation"
                  "/translation from mrk1 to mrk2; Average: use the average "
                  "of the mrk1 and mrk2 coordinates for each point.")

    view = View(VGroup(Item('method', style='custom'),
                       Item('save_as', enabled_when='can_save',
                            show_label=False)))

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

            pts = (self.src1.points + self.src2.points) / 2.
            for i in np.setdiff1d(self.src1.use, self.src2.use):
                pts[i] = self.src1.points[i]
            for i in np.setdiff1d(self.src2.use, self.src1.use):
                pts[i] = self.src2.points[i]

            return pts

        # Transform method
        idx = np.intersect1d(np.array(self.src1.use),
                             np.array(self.src2.use), assume_unique=True)
        if len(idx) < 3:
            error(None, "Need at least three shared points for trans"
                  "formation.", "Marker Interpolation Error")
            return np.zeros((5, 3))

        src_pts = self.src1.points[idx]
        tgt_pts = self.src2.points[idx]
        est = fit_matched_points(src_pts, tgt_pts, out='params')
        rot = np.array(est[:3]) / 2.
        tra = np.array(est[3:]) / 2.

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


class CombineMarkersModel(HasPrivateTraits):
    """Combine markers model."""

    mrk1_file = Instance(File)
    mrk2_file = Instance(File)
    mrk1 = Instance(MarkerPointSource)
    mrk2 = Instance(MarkerPointSource)
    mrk3 = Instance(MarkerPointDest)

    clear = Button(desc="Clear the current marker data")

    # stats
    distance = Property(Str, depends_on=['mrk1.points', 'mrk2.points'])

    def _clear_fired(self):
        self.mrk1.clear = True
        self.mrk2.clear = True
        self.mrk3.reset_traits(['method'])

    def _mrk1_default(self):
        return MarkerPointSource()

    def _mrk1_file_default(self):
        return self.mrk1.trait('file')

    def _mrk2_default(self):
        return MarkerPointSource()

    def _mrk2_file_default(self):
        return self.mrk2.trait('file')

    def _mrk3_default(self):
        return MarkerPointDest(src1=self.mrk1, src2=self.mrk2)

    @cached_property
    def _get_distance(self):
        if (self.mrk1 is None or self.mrk2 is None or
                (not np.any(self.mrk1.points)) or
                (not np.any(self.mrk2.points))):
            return ""

        ds = np.sqrt(np.sum((self.mrk1.points - self.mrk2.points) ** 2, 1))
        desc = '\t'.join('%.1f mm' % (d * 1000) for d in ds)
        return desc


class CombineMarkersPanel(HasTraits):  # noqa: D401
    """Has two marker points sources and interpolates to a third one."""

    model = Instance(CombineMarkersModel, ())

    # model references for UI
    mrk1 = Instance(MarkerPointSource)
    mrk2 = Instance(MarkerPointSource)
    mrk3 = Instance(MarkerPointDest)
    distance = Str

    # Visualization
    scene = Instance(MlabSceneModel)
    scale = Float(5e-3)
    mrk1_obj = Instance(PointObject)
    mrk2_obj = Instance(PointObject)
    mrk3_obj = Instance(PointObject)
    trans = Array()

    view = View(VGroup(VGroup(Item('mrk1', style='custom'),
                              Item('mrk1_obj', style='custom'),
                              show_labels=False,
                              label="Source Marker 1", show_border=True),
                       VGroup(Item('mrk2', style='custom'),
                              Item('mrk2_obj', style='custom'),
                              show_labels=False,
                              label="Source Marker 2", show_border=True),
                       VGroup(Item('distance', style='readonly'),
                              label='Stats', show_border=True),
                       VGroup(Item('mrk3', style='custom'),
                              Item('mrk3_obj', style='custom'),
                              show_labels=False,
                              label="New Marker", show_border=True),
                       ))

    def _mrk1_default(self):
        return self.model.mrk1

    def _mrk2_default(self):
        return self.model.mrk2

    def _mrk3_default(self):
        return self.model.mrk3

    def __init__(self, *args, **kwargs):  # noqa: D102
        super(CombineMarkersPanel, self).__init__(*args, **kwargs)

        self.model.sync_trait('distance', self, 'distance', mutual=False)

        self.mrk1_obj = PointObject(scene=self.scene,
                                    color=(0.608, 0.216, 0.216),
                                    point_scale=self.scale)
        self.model.mrk1.sync_trait(
            'enabled', self.mrk1_obj, 'visible', mutual=False)

        self.mrk2_obj = PointObject(scene=self.scene,
                                    color=(0.216, 0.608, 0.216),
                                    point_scale=self.scale)
        self.model.mrk2.sync_trait(
            'enabled', self.mrk2_obj, 'visible', mutual=False)

        self.mrk3_obj = PointObject(scene=self.scene,
                                    color=(0.588, 0.784, 1.),
                                    point_scale=self.scale)
        self.model.mrk3.sync_trait(
            'enabled', self.mrk3_obj, 'visible', mutual=False)

    @on_trait_change('model:mrk1:points,trans')
    def _update_mrk1(self):
        if self.mrk1_obj is not None:
            self.mrk1_obj.points = apply_trans(self.trans,
                                               self.model.mrk1.points)

    @on_trait_change('model:mrk2:points,trans')
    def _update_mrk2(self):
        if self.mrk2_obj is not None:
            self.mrk2_obj.points = apply_trans(self.trans,
                                               self.model.mrk2.points)

    @on_trait_change('model:mrk3:points,trans')
    def _update_mrk3(self):
        if self.mrk3_obj is not None:
            self.mrk3_obj.points = apply_trans(self.trans,
                                               self.model.mrk3.points)
