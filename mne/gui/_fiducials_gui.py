# -*- coding: utf-8 -*-
"""Mayavi/traits GUI for setting MRI fiducials."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os

from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.tools.mlab_scene_model import MlabSceneModel
import numpy as np
from pyface.api import confirm, error, FileDialog, OK, YES
from traits.api import (HasTraits, HasPrivateTraits, on_trait_change,
                        cached_property, DelegatesTo, Event, Instance,
                        Property, Array, Bool, Button, Enum)
from traitsui.api import HGroup, Item, VGroup, View, Handler, ArrayEditor
from traitsui.menu import NoButtons
from tvtk.pyface.scene_editor import SceneEditor

from ..coreg import (fid_fname, _find_fiducials_files, _find_head_bem,
                     get_mni_fiducials)
from ..defaults import DEFAULTS
from ..io import write_fiducials
from ..io.constants import FIFF
from ..surface import complete_surface_info, decimate_surface
from ..utils import get_subjects_dir, logger, warn
from ..viz.backends._pysurfer_mayavi import _toggle_mlab_render
from ._file_traits import (SurfaceSource, fid_wildcard, FiducialsSource,
                           MRISubjectSource, SubjectSelectorPanel,
                           Surf)
from ._viewer import (HeadViewController, PointObject, SurfaceObject,
                      headview_borders, _BUTTON_WIDTH,
                      _MRI_FIDUCIALS_WIDTH, _MM_WIDTH,
                      _RESET_LABEL, _RESET_WIDTH, _mm_fmt)

defaults = DEFAULTS['coreg']


class MRIHeadWithFiducialsModel(HasPrivateTraits):
    """Represent an MRI head shape (high and low res) with fiducials.

    Attributes
    ----------
    points : array (n_points, 3)
        MRI head surface points.
    tris : array (n_tris, 3)
        Triangles based on points.
    lpa : array (1, 3)
        Left peri-auricular point coordinates.
    nasion : array (1, 3)
        Nasion coordinates.
    rpa : array (1, 3)
        Right peri-auricular point coordinates.
    """

    subject_source = Instance(MRISubjectSource, ())
    bem_low_res = Instance(SurfaceSource, ())
    bem_high_res = Instance(SurfaceSource, ())
    fid = Instance(FiducialsSource, ())

    fid_file = DelegatesTo('fid', 'file')
    fid_fname = DelegatesTo('fid', 'fname')
    fid_points = DelegatesTo('fid', 'points')
    subjects_dir = DelegatesTo('subject_source')
    subject = DelegatesTo('subject_source')
    subject_has_bem = DelegatesTo('subject_source')
    lpa = Array(float, (1, 3))
    nasion = Array(float, (1, 3))
    rpa = Array(float, (1, 3))

    reset = Event(desc="Reset fiducials to the file.")

    # info
    can_save = Property(depends_on=['file', 'can_save_as'])
    can_save_as = Property(depends_on=['lpa', 'nasion', 'rpa'])
    can_reset = Property(depends_on=['file', 'fid.points', 'lpa', 'nasion',
                                     'rpa'])
    fid_ok = Property(depends_on=['lpa', 'nasion', 'rpa'], desc="All points "
                      "are set")
    default_fid_fname = Property(depends_on=['subjects_dir', 'subject'],
                                 desc="the default file name for the "
                                 "fiducials fif file")

    # switch for the GUI (has no effect in the model)
    lock_fiducials = Bool(False, desc="Used by GIU, has no effect in the "
                          "model.")

    @on_trait_change('fid_points')
    def reset_fiducials(self):  # noqa: D102
        if self.fid_points is not None:
            self.lpa = self.fid_points[0:1]
            self.nasion = self.fid_points[1:2]
            self.rpa = self.fid_points[2:3]

    def save(self, fname=None):
        """Save the current fiducials to a file.

        Parameters
        ----------
        fname : str
            Destination file path. If None, will use the current fid filename
            if available, or else use the default pattern.
        """
        if fname is None:
            fname = self.fid_file
        if not fname:
            fname = self.default_fid_fname

        dig = [{'kind': FIFF.FIFFV_POINT_CARDINAL,
                'ident': FIFF.FIFFV_POINT_LPA,
                'r': np.array(self.lpa[0])},
               {'kind': FIFF.FIFFV_POINT_CARDINAL,
                'ident': FIFF.FIFFV_POINT_NASION,
                'r': np.array(self.nasion[0])},
               {'kind': FIFF.FIFFV_POINT_CARDINAL,
                'ident': FIFF.FIFFV_POINT_RPA,
                'r': np.array(self.rpa[0])}]
        write_fiducials(fname, dig, FIFF.FIFFV_COORD_MRI)
        self.fid_file = fname

    @cached_property
    def _get_can_reset(self):
        if not self.fid_file:
            return False
        elif np.any(self.lpa != self.fid.points[0:1]):
            return True
        elif np.any(self.nasion != self.fid.points[1:2]):
            return True
        elif np.any(self.rpa != self.fid.points[2:3]):
            return True
        return False

    @cached_property
    def _get_can_save_as(self):
        can = not (np.all(self.nasion == self.lpa) or
                   np.all(self.nasion == self.rpa) or
                   np.all(self.lpa == self.rpa))
        return can

    @cached_property
    def _get_can_save(self):
        if not self.can_save_as:
            return False
        elif self.fid_file:
            return True
        elif self.subjects_dir and self.subject:
            return True
        else:
            return False

    @cached_property
    def _get_default_fid_fname(self):
        fname = fid_fname.format(subjects_dir=self.subjects_dir,
                                 subject=self.subject)
        return fname

    @cached_property
    def _get_fid_ok(self):
        return all(np.any(pt) for pt in (self.nasion, self.lpa, self.rpa))

    def _reset_fired(self):
        self.reset_fiducials()

    # if subject changed because of a change of subjects_dir this was not
    # triggered
    @on_trait_change('subjects_dir,subject')
    def _subject_changed(self):
        subject = self.subject
        subjects_dir = self.subjects_dir
        if not subjects_dir or not subject:
            return

        # find high-res head model (if possible)
        high_res_path = _find_head_bem(subject, subjects_dir, high_res=True)
        low_res_path = _find_head_bem(subject, subjects_dir, high_res=False)
        if high_res_path is None and low_res_path is None:
            msg = 'No standard head model was found for subject %s' % subject
            error(None, msg, "No head surfaces found")
            raise RuntimeError(msg)
        if high_res_path is not None:
            self.bem_high_res.file = high_res_path
        else:
            self.bem_high_res.file = low_res_path
        if low_res_path is None:
            # This should be very rare!
            warn('No low-resolution head found, decimating high resolution '
                 'mesh (%d vertices): %s' % (len(self.bem_high_res.surf.rr),
                                             high_res_path,))
            # Create one from the high res one, which we know we have
            rr, tris = decimate_surface(self.bem_high_res.surf.rr,
                                        self.bem_high_res.surf.tris,
                                        n_triangles=5120)
            surf = complete_surface_info(dict(rr=rr, tris=tris),
                                         copy=False, verbose=False)
            # directly set the attributes of bem_low_res
            self.bem_low_res.surf = Surf(tris=surf['tris'], rr=surf['rr'],
                                         nn=surf['nn'])
        else:
            self.bem_low_res.file = low_res_path

        # Set MNI points
        try:
            fids = get_mni_fiducials(subject, subjects_dir)
        except Exception:  # some problem, leave at origin
            self.fid.mni_points = None
        else:
            self.fid.mni_points = np.array([f['r'] for f in fids], float)

        # find fiducials file
        fid_files = _find_fiducials_files(subject, subjects_dir)
        if len(fid_files) == 0:
            self.fid.reset_traits(['file'])
            self.lock_fiducials = False
        else:
            self.fid_file = fid_files[0].format(subjects_dir=subjects_dir,
                                                subject=subject)
            self.lock_fiducials = True

        # does not seem to happen by itself ... so hard code it:
        self.reset_fiducials()


class SetHandler(Handler):
    """Handler to change style when setting MRI fiducials."""

    def object_set_changed(self, info):  # noqa: D102
        return self.object_locked_changed(info)

    def object_locked_changed(self, info):  # noqa: D102
        if info.object.locked:
            ss = ''
        else:
            ss = 'border-style: solid; border-color: red; border-width: 2px;'
        # This will only work for Qt, but hopefully that's most users!
        try:
            _color_children(info.ui.info.ui.control, ss)
        except AttributeError:  # safeguard for wxpython
            pass


def _color_children(obj, ss):
    """Qt helper."""
    for child in obj.children():
        if 'QRadioButton' in repr(child):
            child.setStyleSheet(ss if child.isChecked() else '')
        elif 'QLineEdit' in repr(child):
            child.setStyleSheet(ss)
        elif 'QWidget' in repr(child):  # on Linux it's nested
            _color_children(child, ss)


_SET_TOOLTIP = ('Click on the MRI image to set the position, '
                'or enter values below')


class FiducialsPanel(HasPrivateTraits):
    """Set fiducials on an MRI surface."""

    model = Instance(MRIHeadWithFiducialsModel)

    fid_file = DelegatesTo('model')
    fid_fname = DelegatesTo('model')
    lpa = DelegatesTo('model')
    nasion = DelegatesTo('model')
    rpa = DelegatesTo('model')
    can_save = DelegatesTo('model')
    can_save_as = DelegatesTo('model')
    can_reset = DelegatesTo('model')
    fid_ok = DelegatesTo('model')
    locked = DelegatesTo('model', 'lock_fiducials')

    set = Enum('LPA', 'Nasion', 'RPA')
    current_pos_mm = Array(float, (1, 3))

    save_as = Button(label='Save as...')
    save = Button(label='Save')
    reset_fid = Button(label=_RESET_LABEL)

    headview = Instance(HeadViewController)
    hsp_obj = Instance(SurfaceObject)

    picker = Instance(object)

    # the layout of the dialog created
    view = View(VGroup(
        HGroup(Item('fid_file', width=_MRI_FIDUCIALS_WIDTH,
                    tooltip='MRI fiducials file'), show_labels=False),
        HGroup(Item('set', width=_MRI_FIDUCIALS_WIDTH,
                    format_func=lambda x: x, style='custom',
                    tooltip=_SET_TOOLTIP), show_labels=False),
        HGroup(Item('current_pos_mm',
                    editor=ArrayEditor(width=_MM_WIDTH, format_func=_mm_fmt),
                    tooltip='MRI fiducial position (mm)'), show_labels=False),
        HGroup(Item('save', enabled_when='can_save',
                    tooltip="If a filename is currently specified, save to "
                    "that file, otherwise save to the default file name",
                    width=_BUTTON_WIDTH),
               Item('save_as', enabled_when='can_save_as',
                    width=_BUTTON_WIDTH),
               Item('reset_fid', enabled_when='can_reset', width=_RESET_WIDTH,
                    tooltip='Reset to file values (if available)'),
               show_labels=False),
        enabled_when="locked==False", show_labels=False), handler=SetHandler())

    def __init__(self, *args, **kwargs):  # noqa: D102
        super(FiducialsPanel, self).__init__(*args, **kwargs)

    @on_trait_change('current_pos_mm')
    def _update_pos(self):
        attr = self.set.lower()
        if not np.allclose(getattr(self, attr), self.current_pos_mm * 1e-3):
            setattr(self, attr, self.current_pos_mm * 1e-3)

    @on_trait_change('model:lpa')
    def _update_lpa(self, name):
        if self.set == 'LPA':
            self.current_pos_mm = self.lpa * 1000

    @on_trait_change('model:nasion')
    def _update_nasion(self, name):
        if self.set.lower() == 'Nasion':
            self.current_pos_mm = self.nasion * 1000

    @on_trait_change('model:rpa')
    def _update_rpa(self, name):
        if self.set.lower() == 'RPA':
            self.current_pos_mm = self.rpa * 1000

    def _reset_fid_fired(self):
        self.model.reset = True

    def _save_fired(self):
        self.model.save()

    def _save_as_fired(self):
        if self.fid_file:
            default_path = self.fid_file
        else:
            default_path = self.model.default_fid_fname

        dlg = FileDialog(action="save as", wildcard=fid_wildcard,
                         default_path=default_path)
        dlg.open()
        if dlg.return_code != OK:
            return

        path = dlg.path
        if not path.endswith('.fif'):
            path = path + '.fif'
            if os.path.exists(path):
                answer = confirm(None, "The file %r already exists. Should it "
                                 "be replaced?", "Overwrite File?")
                if answer != YES:
                    return

        self.model.save(path)

    def _on_pick(self, picker):
        if self.locked:
            return

        self.picker = picker
        n_pos = len(picker.picked_positions)

        if n_pos == 0:
            logger.debug("GUI: picked empty location")
            return

        if picker.actor is self.hsp_obj.surf.actor.actor:
            idxs = []
            idx = None
            pt = [picker.pick_position]
        elif self.hsp_obj.surf.actor.actor in picker.actors:
            idxs = [i for i in range(n_pos) if picker.actors[i] is
                    self.hsp_obj.surf.actor.actor]
            idx = idxs[-1]
            pt = [picker.picked_positions[idx]]
        else:
            logger.debug("GUI: picked object other than MRI")

        def round_(x):
            return round(x, 3)

        poss = [map(round_, pos) for pos in picker.picked_positions]
        pos = map(round_, picker.pick_position)
        msg = ["Pick Event: %i picked_positions:" % n_pos]

        line = str(pos)
        if idx is None:
            line += " <-pick_position"
        msg.append(line)

        for i, pos in enumerate(poss):
            line = str(pos)
            if i == idx:
                line += " <- MRI mesh"
            elif i in idxs:
                line += " (<- also MRI mesh)"
            msg.append(line)
        logger.debug('\n'.join(msg))

        if self.set == 'Nasion':
            self.nasion = pt
        elif self.set == 'LPA':
            self.lpa = pt
        elif self.set == 'RPA':
            self.rpa = pt
        else:
            raise ValueError("set = %r" % self.set)

    @on_trait_change('set')
    def _on_set_change(self, obj, name, old, new):
        if new == 'Nasion':
            self.current_pos_mm = self.nasion * 1000
            self.headview.front = True
        elif new == 'LPA':
            self.current_pos_mm = self.lpa * 1000
            self.headview.left = True
        elif new == 'RPA':
            self.current_pos_mm = self.rpa * 1000
            self.headview.right = True


# FiducialsPanel view that allows manipulating all coordinates numerically
view2 = View(VGroup(Item('fid_file', label='Fiducials File'),
                    Item('fid_fname', show_label=False, style='readonly'),
                    Item('set', style='custom'), 'lpa', 'nasion', 'rpa',
                    HGroup(Item('save', enabled_when='can_save'),
                           Item('save_as', enabled_when='can_save_as'),
                           Item('reset_fid', enabled_when='can_reset'),
                           show_labels=False),
                    enabled_when="locked==False"))


class FiducialsFrame(HasTraits):
    """GUI for interpolating between two KIT marker files.

    Parameters
    ----------
    subject : None | str
        Set the subject which is initially selected.
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable.
    """

    model = Instance(MRIHeadWithFiducialsModel, ())

    scene = Instance(MlabSceneModel, ())
    headview = Instance(HeadViewController)

    spanel = Instance(SubjectSelectorPanel)
    panel = Instance(FiducialsPanel)

    mri_obj = Instance(SurfaceObject)
    point_scale = float(defaults['mri_fid_scale'])
    lpa_obj = Instance(PointObject)
    nasion_obj = Instance(PointObject)
    rpa_obj = Instance(PointObject)

    def _headview_default(self):
        return HeadViewController(scene=self.scene, system='RAS')

    def _panel_default(self):
        panel = FiducialsPanel(model=self.model, headview=self.headview)
        panel.trait_view('view', view2)
        return panel

    def _spanel_default(self):
        return SubjectSelectorPanel(model=self.model.subject_source)

    view = View(HGroup(Item('scene',
                            editor=SceneEditor(scene_class=MayaviScene),
                            dock='vertical'),
                       VGroup(headview_borders,
                              VGroup(Item('spanel', style='custom'),
                                     label="Subject", show_border=True,
                                     show_labels=False),
                              VGroup(Item('panel', style="custom"),
                                     label="Fiducials", show_border=True,
                                     show_labels=False),
                              show_labels=False),
                       show_labels=False),
                resizable=True,
                buttons=NoButtons)

    def __init__(self, subject=None, subjects_dir=None,
                 **kwargs):  # noqa: D102
        super(FiducialsFrame, self).__init__(**kwargs)

        subjects_dir = get_subjects_dir(subjects_dir)
        if subjects_dir is not None:
            self.spanel.subjects_dir = subjects_dir

        if subject is not None:
            if subject in self.spanel.subjects:
                self.spanel.subject = subject

    @on_trait_change('scene.activated')
    def _init_plot(self):
        _toggle_mlab_render(self, False)

        lpa_color = defaults['lpa_color']
        nasion_color = defaults['nasion_color']
        rpa_color = defaults['rpa_color']

        # bem
        color = defaults['mri_color']
        self.mri_obj = SurfaceObject(points=self.model.points, color=color,
                                     tri=self.model.tris, scene=self.scene)
        self.model.on_trait_change(self._on_mri_src_change, 'tris')
        self.panel.hsp_obj = self.mri_obj

        # fiducials
        self.lpa_obj = PointObject(scene=self.scene, color=lpa_color,
                                   has_norm=True,
                                   point_scale=self.point_scale)
        self.panel.sync_trait('lpa', self.lpa_obj, 'points', mutual=False)
        self.sync_trait('point_scale', self.lpa_obj, mutual=False)

        self.nasion_obj = PointObject(scene=self.scene, color=nasion_color,
                                      has_norm=True,
                                      point_scale=self.point_scale)
        self.panel.sync_trait('nasion', self.nasion_obj, 'points',
                              mutual=False)
        self.sync_trait('point_scale', self.nasion_obj, mutual=False)

        self.rpa_obj = PointObject(scene=self.scene, color=rpa_color,
                                   has_norm=True,
                                   point_scale=self.point_scale)
        self.panel.sync_trait('rpa', self.rpa_obj, 'points', mutual=False)
        self.sync_trait('point_scale', self.rpa_obj, mutual=False)

        self.headview.left = True
        _toggle_mlab_render(self, True)

        # picker
        self.scene.mayavi_scene.on_mouse_pick(self.panel._on_pick, type='cell')

    def _on_mri_src_change(self):
        if (not np.any(self.model.points)) or (not np.any(self.model.tris)):
            self.mri_obj.clear()
            return

        self.mri_obj.points = self.model.points
        self.mri_obj.tri = self.model.tris
        self.mri_obj.plot()
