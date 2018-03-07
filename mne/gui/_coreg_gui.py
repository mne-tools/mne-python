# -*- coding: utf-8 -*-
"""Traits-based GUI for head-MRI coregistration.

Hierarchy
---------
This is the hierarchy of classes for control. Brackets like [1] denote
properties that are set to be equivalent.

::

  CoregFrame: GUI for head-MRI coregistration.
  |-- CoregModel (model): Traits object for estimating the head mri transform.
  |   |-- MRIHeadWithFiducialsModel (mri) [1]: Represent an MRI head shape (high and low res) with fiducials.
  |   |   |-- SurfaceSource (bem_high_res): High-res MRI head
  |   |   |-- SurfaceSource (bem_low_res): Low-res MRI head
  |   |   +-- MRISubjectSource (subject_source) [2]: Find subjects in SUBJECTS_DIR and select one.
  |   |-- FiducialsSource (fid): Expose points of a given fiducials fif file.
  |   +-- DigSource (hsp): Expose measurement information from a inst file.
  |-- MlabSceneModel (scene) [3]: mayavi.core.ui.mayavi_scene
  |-- HeadViewController (headview) [4]: Set head views for the given coordinate system.
  |   +-- MlabSceneModel (scene) [3*]: ``HeadViewController(scene=CoregFrame.scene)``
  |-- SubjectSelectorPanel (subject_panel): Subject selector panel
  |   +-- MRISubjectSource (model) [2*]: ``SubjectSelectorPanel(model=self.model.mri.subject_source)``
  |-- SurfaceObject (mri_obj) [5]: Represent a solid object in a mayavi scene.
  |-- FiducialsPanel (fid_panel): Set fiducials on an MRI surface.
  |   |-- MRIHeadWithFiducialsModel (model) [1*]: ``FiducialsPanel(model=CoregFrame.model.mri, headview=CoregFrame.headview)``
  |   |-- HeadViewController (headview) [4*]: ``FiducialsPanel(model=CoregFrame.model.mri, headview=CoregFrame.headview)``
  |   +-- SurfaceObject (hsp_obj) [5*]: ``CoregFrame.fid_panel.hsp_obj = CoregFrame.mri_obj``
  |-- CoregPanel (coreg_panel): Coregistration panel for Head<->MRI with scaling.
  +-- PointObject ({hsp, eeg, lpa, nasion, rpa, hsp_lpa, hsp_nasion, hsp_rpa} + _obj): Represent a group of individual points in a mayavi scene.

In the MRI viewing frame, MRI points and transformed via scaling, then by
mri_head_t to the Neuromag head coordinate frame. Digitized points (in head
coordinate frame) are never transformed.
"""  # noqa: E501

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os
from ..externals.six.moves import queue
import re
from threading import Thread
import traceback
import warnings

import numpy as np

from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.tools.mlab_scene_model import MlabSceneModel
from pyface.api import (error, confirm, OK, YES, NO, CANCEL, information,
                        FileDialog, GUI)
from traits.api import (Bool, Button, cached_property, DelegatesTo, Directory,
                        Enum, Float, HasTraits, HasPrivateTraits, Instance,
                        Int, on_trait_change, Property, Str, List)
from traitsui.api import (View, Item, Group, HGroup, VGroup, VGrid, EnumEditor,
                          Handler, Label, TextEditor, Spring, InstanceEditor)
from traitsui.menu import Action, UndoButton, CancelButton, NoButtons
from tvtk.pyface.scene_editor import SceneEditor

from ..bem import make_bem_solution, write_bem_solution
from ..coreg import bem_fname, trans_fname
from ..defaults import DEFAULTS
from ..surface import _compute_nearest
from ..transforms import (write_trans, read_trans, apply_trans, rotation,
                          rotation_angles, Transform, _ensure_trans)
from ..coreg import fit_matched_points, scale_mri, _find_fiducials_files
from ..viz._3d import _toggle_mlab_render
from ..utils import logger, set_config
from ._fiducials_gui import MRIHeadWithFiducialsModel, FiducialsPanel
from ._file_traits import trans_wildcard, DigSource, SubjectSelectorPanel
from ._viewer import (HeadViewController, PointObject, SurfaceObject,
                      _RAD_WIDTH, _M_WIDTH, _MM_WIDTH, _BUTTON_WIDTH,
                      _SHOW_BORDER, _TEXT_WIDTH, _COREG_WIDTH,
                      _INC_BUTTON_WIDTH, _SCALE_WIDTH, _WEIGHT_WIDTH,
                      _M_STEP_WIDTH, _RAD_STEP_WIDTH,
                      laggy_float_editor_mm, laggy_float_editor_m)

defaults = DEFAULTS['coreg']

laggy_float_editor_w = TextEditor(auto_set=False, enter_set=True,
                                  evaluate=float,
                                  format_func=lambda x: '%0.2f' % x)

laggy_float_editor_scale = TextEditor(auto_set=False, enter_set=True,
                                      evaluate=float,
                                      format_func=lambda x: '%0.3f' % x)
laggy_float_editor_rad = TextEditor(auto_set=False, enter_set=True,
                                    evaluate=float,
                                    format_func=lambda x: '%0.4f' % x)


class CoregModel(HasPrivateTraits):
    """Traits object for estimating the head mri transform.

    Notes
    -----
    Transform from head to mri space is modelled with the following steps:

    * move the head shape to its nasion position
    * rotate the head shape with user defined rotation around its nasion
    * move the head shape by user defined translation
    * move the head shape origin to the mri nasion

    If MRI scaling is enabled,

    * the MRI is scaled relative to its origin center (prior to any
      transformation of the digitizer head)

    Don't sync transforms to anything to prevent them from being recomputed
    upon every parameter change.
    """

    # data sources
    mri = Instance(MRIHeadWithFiducialsModel, ())
    hsp = Instance(DigSource, ())

    # parameters
    guess_mri_subject = Bool(True)  # change MRI subject when dig file changes
    grow_hair = Float(label="ΔHair [mm]", desc="Move the back of the MRI "
                      "head outwards to compensate for hair on the digitizer "
                      "head shape")
    n_scale_params = Enum(0, 1, 3, desc="Scale the MRI to better fit the "
                          "subject's head shape (a new MRI subject will be "
                          "created with a name specified upon saving)")
    scale_x = Float(1, label="X")
    scale_y = Float(1, label="Y")
    scale_z = Float(1, label="Z")
    trans_x = Float(0, label="ΔX")
    trans_y = Float(0, label="ΔY")
    trans_z = Float(0, label="ΔZ")
    rot_x = Float(0, label=u"∠X")
    rot_y = Float(0, label=u"∠Y")
    rot_z = Float(0, label=u"∠Z")
    parameters = List()
    lpa_weight = Float(1.)
    nasion_weight = Float(1.)
    rpa_weight = Float(1.)
    hsp_weight = Float(1.)
    eeg_weight = Float(0.)
    hpi_weight = Float(0.)
    coord_frame = Str('mri')

    # options during scaling
    scale_labels = Bool(True, desc="whether to scale *.label files")
    copy_annot = Bool(True, desc="whether to copy *.annot files for scaled "
                      "subject")
    prepare_bem_model = Bool(True, desc="whether to run mne_prepare_bem_model "
                             "after scaling the MRI")

    # secondary to parameters
    scale = Property(depends_on=['n_scale_params', 'parameters[]'])
    has_nasion_data = Property(
        Bool,
        desc="Nasion data is present.",
        depends_on=['mri:nasion', 'hsp:nasion'])
    has_lpa_data = Property(
        Bool,
        desc="LPA data is present.",
        depends_on=['mri:lpa', 'hsp:lpa'])
    has_rpa_data = Property(
        Bool,
        desc="RPA data is present.",
        depends_on=['mri:rpa', 'hsp:rpa'])
    has_hsp_data = Property(
        Bool,
        depends_on=['mri:points', 'hsp:points'])
    has_eeg_data = Property(
        Bool,
        depends_on=['mri:points', 'hsp:eeg_points'])
    has_hpi_data = Property(
        Bool,
        depends_on=['mri:points', 'hsp:hpi_points'])

    # target transforms
    mri_head_t = Property(
        desc="Transformation of the scaled MRI to the head coordinate frame.",
        depends_on=['parameters[]'])
    head_mri_t = Property(depends_on=['mri_head_t'])
    mri_trans = Property(depends_on=['mri_head_t', 'parameters[]',
                                     'coord_frame'])
    hsp_trans = Property(depends_on=['head_mri_t', 'coord_frame'])

    # info
    subject_has_bem = DelegatesTo('mri')
    lock_fiducials = DelegatesTo('mri')
    can_prepare_bem_model = Property(
        Bool,
        depends_on=['n_scale_params', 'subject_has_bem'])
    can_save = Property(Bool, depends_on=['mri_head_t'])
    raw_subject = Property(
        desc="Subject guess based on the raw file name.",
        depends_on=['hsp:inst_fname'])

    # MRI geometry transformed to viewing coordinate system
    processed_high_res_mri_points = Property(
        depends_on=['mri:bem_high_res:points', 'grow_hair'])
    processed_low_res_mri_points = Property(
        depends_on=['mri:bem_low_res:points', 'grow_hair'])
    transformed_high_res_mri_points = Property(
        depends_on=['processed_high_res_mri_points', 'mri_trans'])
    transformed_low_res_mri_points = Property(
        depends_on=['processed_low_res_mri_points', 'mri_trans'])
    nearest_transformed_low_res_mri_idx_hsp = Property(
        depends_on=['transformed_low_res_mri_points',
                    'transformed_hsp_points'])
    nearest_transformed_low_res_mri_idx_orig_hsp = Property(
        depends_on=['transformed_low_res_mri_points',
                    'transformed_orig_hsp_points'])
    nearest_transformed_low_res_mri_idx_eeg = Property(
        depends_on=['transformed_low_res_mri_points',
                    'transformed_hsp_eeg_points'])
    nearest_transformed_low_res_mri_idx_hpi = Property(
        depends_on=['transformed_low_res_mri_points',
                    'transformed_hsp_hpi'])
    transformed_mri_lpa = Property(
        depends_on=['mri:lpa', 'mri_trans'])
    transformed_mri_nasion = Property(
        depends_on=['mri:nasion', 'mri_trans'])
    transformed_mri_rpa = Property(
        depends_on=['mri:rpa', 'mri_trans'])
    # HSP geometry transformed to viewing coordinate system
    transformed_hsp_points = Property(
        depends_on=['hsp:points', 'hsp_trans'])
    transformed_orig_hsp_points = Property(
        depends_on=['hsp:_hsp_points', 'hsp_trans'])
    transformed_hsp_lpa = Property(
        depends_on=['hsp:lpa', 'hsp_trans'])
    transformed_hsp_nasion = Property(
        depends_on=['hsp:nasion', 'hsp_trans'])
    transformed_hsp_rpa = Property(
        depends_on=['hsp:rpa', 'hsp_trans'])
    transformed_hsp_eeg_points = Property(
        depends_on=['hsp:eeg_points', 'hsp_trans'])
    transformed_hsp_hpi = Property(
        depends_on=['hsp:hpi', 'hsp_trans'])

    # fit properties
    lpa_distance = Property(
        depends_on=['transformed_mri_lpa', 'transformed_hsp_lpa'])
    nasion_distance = Property(
        depends_on=['transformed_mri_nasion', 'transformed_hsp_nasion'])
    rpa_distance = Property(
        depends_on=['transformed_mri_rpa', 'transformed_hsp_rpa'])
    point_distance = Property(  # use low res points
        depends_on=['nearest_transformed_low_res_mri_idx_hsp',
                    'nearest_transformed_low_res_mri_idx_eeg',
                    'nearest_transformed_low_res_mri_idx_hpi',
                    'transformed_low_res_mri_points',
                    'transformed_hsp_points',
                    'hsp_weight',
                    'eeg_weight',
                    'hpi_weight'])
    orig_hsp_point_distance = Property(  # use low res points
        depends_on=['nearest_transformed_low_res_mri_idx_orig_hsp',
                    'transformed_low_res_mri_points',
                    'transformed_orig_hsp_points',
                    'hpi_weight'])

    # fit property info strings
    fid_eval_str = Property(
        depends_on=['lpa_distance', 'nasion_distance', 'rpa_distance'])
    points_eval_str = Property(
        depends_on=['point_distance'])

    @cached_property
    def _get_can_prepare_bem_model(self):
        return self.subject_has_bem and self.n_scale_params > 0

    @cached_property
    def _get_can_save(self):
        return np.any(self.mri_head_t != np.eye(4))

    @cached_property
    def _get_has_lpa_data(self):
        return (np.any(self.mri.lpa) and np.any(self.hsp.lpa))

    @cached_property
    def _get_has_nasion_data(self):
        return (np.any(self.mri.nasion) and np.any(self.hsp.nasion))

    @cached_property
    def _get_has_rpa_data(self):
        return (np.any(self.mri.rpa) and np.any(self.hsp.rpa))

    @cached_property
    def _get_has_hsp_data(self):
        return (np.any(self.mri.bem_low_res.points) and
                np.any(self.hsp.points))

    @cached_property
    def _get_has_eeg_data(self):
        return (np.any(self.mri.bem_low_res.points) and
                np.any(self.hsp.eeg_points))

    @cached_property
    def _get_has_hpi_data(self):
        return (np.any(self.mri.bem_low_res.points) and
                np.any(self.hsp.hpi_points))

    @cached_property
    def _get_scale(self):
        if self.n_scale_params == 0:
            return np.array(1)
        return np.array([self.scale_x, self.scale_y, self.scale_z])

    @cached_property
    def _get_mri_head_t(self):
        # rotate and translate hsp
        trans = rotation(self.rot_x, self.rot_y, self.rot_z)
        trans[:3, 3] = [self.trans_x, self.trans_y, self.trans_z]
        return trans

    @cached_property
    def _get_head_mri_t(self):
        trans = rotation(self.rot_x, self.rot_y, self.rot_z).T
        trans[:3, 3] = -np.dot(trans[:3, :3],
                               [self.trans_x, self.trans_y, self.trans_z])
        # should be the same as np.linalg.inv(self.mri_head_t)
        return trans

    @cached_property
    def _get_processed_high_res_mri_points(self):
        if self.grow_hair:
            if len(self.mri.bem_high_res.norms):
                scaled_hair_dist = self.grow_hair / (self.scale * 1000)
                points = self.mri.bem_high_res.points.copy()
                hair = points[:, 2] > points[:, 1]
                points[hair] += (self.mri.bem_high_res.norms[hair] *
                                 scaled_hair_dist)
                return points
            else:
                error(None, "Norms missing from bem, can't grow hair")
                self.grow_hair = 0
        else:
            return self.mri.bem_high_res.points

    @cached_property
    def _get_processed_low_res_mri_points(self):
        if self.grow_hair:
            if len(self.mri.bem_low_res.norms):
                scaled_hair_dist = self.grow_hair / (self.scale * 1000)
                points = self.mri.bem_low_res.points.copy()
                hair = points[:, 2] > points[:, 1]
                points[hair] += (self.mri.bem_low_res.norms[hair] *
                                 scaled_hair_dist)
                return points
            else:
                error(None, "Norms missing from bem, can't grow hair")
                self.grow_hair = 0
        else:
            return self.mri.bem_low_res.points

    @cached_property
    def _get_mri_trans(self):
        mri_scaling = np.ones(4)
        mri_scaling[:3] = self.scale
        if self.coord_frame.lower() == 'head':
            t = self.mri_head_t
        else:
            t = np.eye(4)
        return t * mri_scaling

    @cached_property
    def _get_hsp_trans(self):
        if self.coord_frame.lower() == 'head':
            t = np.eye(4)
        else:
            t = self.head_mri_t
        return t

    @cached_property
    def _get_nearest_transformed_low_res_mri_idx_hsp(self):
        return _compute_nearest(self.transformed_low_res_mri_points,
                                self.transformed_hsp_points)

    @cached_property
    def _get_nearest_transformed_low_res_mri_idx_orig_hsp(self):
        # This is redundant to some extent with the one above due to
        # overlapping points, but it's fast and the refactoring to
        # remove redundancy would be a pain.
        return _compute_nearest(self.transformed_low_res_mri_points,
                                self.transformed_orig_hsp_points)

    @cached_property
    def _get_nearest_transformed_low_res_mri_idx_eeg(self):
        return _compute_nearest(self.transformed_low_res_mri_points,
                                self.transformed_hsp_eeg_points)

    @cached_property
    def _get_nearest_transformed_low_res_mri_idx_hpi(self):
        return _compute_nearest(self.transformed_low_res_mri_points,
                                self.transformed_hsp_hpi)

    # MRI view-transformed data
    @cached_property
    def _get_transformed_low_res_mri_points(self):
        points = apply_trans(self.mri_trans,
                             self.processed_low_res_mri_points)
        return points

    @cached_property
    def _get_transformed_high_res_mri_points(self):
        points = apply_trans(self.mri_trans,
                             self.processed_high_res_mri_points)
        return points

    @cached_property
    def _get_transformed_mri_lpa(self):
        return apply_trans(self.mri_trans, self.mri.lpa)

    @cached_property
    def _get_transformed_mri_nasion(self):
        return apply_trans(self.mri_trans, self.mri.nasion)

    @cached_property
    def _get_transformed_mri_rpa(self):
        return apply_trans(self.mri_trans, self.mri.rpa)

    # HSP view-transformed data
    @cached_property
    def _get_transformed_hsp_points(self):
        return apply_trans(self.hsp_trans, self.hsp.points)

    @cached_property
    def _get_transformed_orig_hsp_points(self):
        return apply_trans(self.hsp_trans, self.hsp._hsp_points)

    @cached_property
    def _get_transformed_hsp_lpa(self):
        return apply_trans(self.hsp_trans, self.hsp.lpa)

    @cached_property
    def _get_transformed_hsp_nasion(self):
        return apply_trans(self.hsp_trans, self.hsp.nasion)

    @cached_property
    def _get_transformed_hsp_rpa(self):
        return apply_trans(self.hsp_trans, self.hsp.rpa)

    @cached_property
    def _get_transformed_hsp_eeg_points(self):
        return apply_trans(self.hsp_trans, self.hsp.eeg_points)

    @cached_property
    def _get_transformed_hsp_hpi(self):
        return apply_trans(self.hsp_trans, self.hsp.hpi_points)

    # Distances, etc.
    @cached_property
    def _get_lpa_distance(self):
        d = np.ravel(self.transformed_mri_lpa - self.transformed_hsp_lpa)
        return np.sqrt(np.dot(d, d))

    @cached_property
    def _get_nasion_distance(self):
        d = np.ravel(self.transformed_mri_nasion - self.transformed_hsp_nasion)
        return np.sqrt(np.dot(d, d))

    @cached_property
    def _get_rpa_distance(self):
        d = np.ravel(self.transformed_mri_rpa - self.transformed_hsp_rpa)
        return np.sqrt(np.dot(d, d))

    @cached_property
    def _get_point_distance(self):
        mri_points = list()
        hsp_points = list()
        if self.hsp_weight > 0 and self.has_hsp_data:
            mri_points.append(self.transformed_low_res_mri_points[
                self.nearest_transformed_low_res_mri_idx_hsp])
            hsp_points.append(self.transformed_hsp_points)
        if self.eeg_weight > 0 and self.has_eeg_data:
            mri_points.append(self.transformed_low_res_mri_points[
                self.nearest_transformed_low_res_mri_idx_eeg])
            hsp_points.append(self.transformed_hsp_eeg_points)
        if self.hpi_weight > 0 and self.has_hpi_data:
            mri_points.append(self.transformed_low_res_mri_points[
                self.nearest_transformed_low_res_mri_idx_hpi])
            hsp_points.append(self.transformed_hsp_hpi)
        if all(len(h) == 0 for h in hsp_points):
            return None
        mri_points = np.concatenate(mri_points)
        hsp_points = np.concatenate(hsp_points)
        return np.linalg.norm(mri_points - hsp_points, axis=-1)

    @cached_property
    def _get_orig_hsp_point_distance(self):
        mri_points = self.transformed_low_res_mri_points[
            self.nearest_transformed_low_res_mri_idx_orig_hsp]
        hsp_points = self.transformed_orig_hsp_points
        return np.linalg.norm(mri_points - hsp_points, axis=-1)

    @cached_property
    def _get_fid_eval_str(self):
        d = (self.lpa_distance * 1000, self.nasion_distance * 1000,
             self.rpa_distance * 1000)
        return u'%.1f, %.1f, %.1f mm' % d

    @cached_property
    def _get_points_eval_str(self):
        if self.point_distance is None:
            return ""
        dists = 1000 * self.point_distance
        av_dist = np.mean(dists)
        std_dist = np.std(dists)
        kinds = [kind for kind, check in
                 (('HSP', self.hsp_weight > 0 and self.has_hsp_data),
                  ('EEG', self.eeg_weight > 0 and self.has_eeg_data),
                  ('HPI', self.hpi_weight > 0 and self.has_hpi_data))
                 if check]
        return (u"%s %s: %.1f ± %.1f mm"
                % (len(dists), '+'.join(kinds), av_dist, std_dist))

    def _get_raw_subject(self):
        # subject name guessed based on the inst file name
        if '_' in self.hsp.inst_fname:
            subject, _ = self.hsp.inst_fname.split('_', 1)
            if subject:
                return subject

    @on_trait_change('raw_subject')
    def _on_raw_subject_change(self, subject):
        if self.guess_mri_subject:
            if subject in self.mri.subject_source.subjects:
                self.mri.subject = subject
            elif 'fsaverage' in self.mri.subject_source.subjects:
                self.mri.subject = 'fsaverage'

    def omit_hsp_points(self, distance):
        """Exclude head shape points that are far away from the MRI head.

        Parameters
        ----------
        distance : float
            Exclude all points that are further away from the MRI head than
            this distance. Previously excluded points are still excluded unless
            reset=True is specified. A value of distance <= 0 excludes nothing.
        reset : bool
            Reset the filter before calculating new omission (default is
            False).
        """
        distance = float(distance)
        if distance <= 0:
            return

        # find the new filter
        mask = self.orig_hsp_point_distance <= distance
        n_excluded = np.sum(~mask)
        logger.info("Coregistration: Excluding %i head shape points with "
                    "distance >= %.3f m.", n_excluded, distance)
        # set the filter
        with warnings.catch_warnings(record=True):  # comp to None in Traits
            self.hsp.points_filter = mask

    def fit_fiducials(self, n_scale_params):
        """Find rotation and translation to fit all 3 fiducials."""
        head_pts = np.vstack((self.hsp.lpa, self.hsp.nasion, self.hsp.rpa))
        mri_pts = np.vstack((self.mri.lpa, self.mri.nasion, self.mri.rpa))
        weights = [self.lpa_weight, self.nasion_weight, self.rpa_weight]
        if n_scale_params == 0:
            mri_pts *= self.scale  # not done inside fit_matched_points
            x0 = (self.rot_x, self.rot_y, self.rot_z,
                  self.trans_x, self.trans_y, self.trans_z)
            est = fit_matched_points(mri_pts, head_pts, x0=x0, out='params',
                                     weights=weights)
            self.parameters[:6] = est
        else:
            x0 = (self.rot_x, self.rot_y, self.rot_z,
                  self.trans_x, self.trans_y, self.trans_z,
                  self.scale_x,)
            est = fit_matched_points(mri_pts, head_pts, scale=1, x0=x0,
                                     out='params', weights=weights)
            assert n_scale_params == 1  # guaranteed from GUI
            self.parameters[:] = np.concatenate([est, [est[-1]] * 2])

    def fit_icp(self, n_scale_params):
        """Find MRI scaling, translation, and rotation to match HSP."""
        head_pts = list()
        mri_pts = list()
        weights = list()
        if self.has_hsp_data and self.hsp_weight > 0:  # should be true
            head_pts.append(self.hsp.points)
            mri_pts.append(self.processed_low_res_mri_points[
                self.nearest_transformed_low_res_mri_idx_hsp])
            weights.append(np.full(len(head_pts[-1]), self.hsp_weight))
        for key in ('lpa', 'nasion', 'rpa'):
            if getattr(self, 'has_%s_data' % key):
                head_pts.append(getattr(self.hsp, key))
                mri_pts.append(getattr(self.mri, key))
                weights.append(np.full(len(mri_pts[-1]),
                                       getattr(self, '%s_weight' % key)))
        if self.has_eeg_data and self.eeg_weight > 0:
            head_pts.append(self.hsp.eeg_points)
            mri_pts.append(self.processed_low_res_mri_points[
                self.nearest_transformed_low_res_mri_idx_eeg])
            weights.append(np.full(len(mri_pts[-1]), self.eeg_weight))
        if self.has_hpi_data and self.hpi_weight > 0:
            head_pts.append(self.hsp.hpi_points)
            mri_pts.append(self.processed_low_res_mri_points[
                self.nearest_transformed_low_res_mri_idx_hpi])
            weights.append(np.full(len(mri_pts[-1]), self.hpi_weight))
        head_pts = np.concatenate(head_pts)
        mri_pts = np.concatenate(mri_pts)
        weights = np.concatenate(weights)

        if n_scale_params == 0:
            mri_pts *= self.scale  # not done inside fit_matched_points
            x0 = (self.rot_x, self.rot_y, self.rot_z,
                  self.trans_x, self.trans_y, self.trans_z)
            est = fit_matched_points(mri_pts, head_pts, x0=x0, out='params',
                                     weights=weights)
            self.parameters[:6] = est
        elif n_scale_params == 1:
            x0 = (self.rot_x, self.rot_y, self.rot_z,
                  self.trans_x, self.trans_y, self.trans_z,
                  self.scale_x)
            est = fit_matched_points(mri_pts, head_pts, scale=1, x0=x0,
                                     out='params', weights=weights)
            est = np.concatenate([est, [est[-1]] * 2])
            self.parameters[:] = est
        else:
            assert n_scale_params == 3
            x0 = (self.rot_x, self.rot_y, self.rot_z,
                  self.trans_x, self.trans_y, self.trans_z,
                  1. / self.scale_x, 1. / self.scale_y, 1. / self.scale_z)
            est = fit_matched_points(mri_pts, head_pts, scale=3, x0=x0,
                                     out='params', weights=weights)
            self.parameters[:] = est

    def get_scaling_job(self, subject_to, skip_fiducials):
        """Find all arguments needed for the scaling worker."""
        subjects_dir = self.mri.subjects_dir
        subject_from = self.mri.subject
        bem_names = []
        if self.can_prepare_bem_model and self.prepare_bem_model:
            pattern = bem_fname.format(subjects_dir=subjects_dir,
                                       subject=subject_from, name='(.+-bem)')
            bem_dir, pattern = os.path.split(pattern)
            for filename in os.listdir(bem_dir):
                match = re.match(pattern, filename)
                if match:
                    bem_names.append(match.group(1))

        return (subjects_dir, subject_from, subject_to, self.scale,
                skip_fiducials, self.scale_labels, self.copy_annot, bem_names)

    def load_trans(self, fname):
        """Load the head-mri transform from a fif file.

        Parameters
        ----------
        fname : str
            File path.
        """
        self.set_trans(_ensure_trans(read_trans(fname, return_all=True),
                                     'mri', 'head')['trans'])

    def reset(self):
        """Reset all the parameters affecting the coregistration."""
        self.reset_traits(('grow_hair', 'n_scaling_params', 'scale_x',
                           'scale_y', 'scale_z', 'rot_x', 'rot_y', 'rot_z',
                           'trans_x', 'trans_y', 'trans_z'))

    def set_trans(self, mri_head_t):
        """Set rotation and translation params from a transformation matrix.

        Parameters
        ----------
        mri_head_t : array, shape (4, 4)
            Transformation matrix from MRI to head space.
        """
        rot_x, rot_y, rot_z = rotation_angles(mri_head_t)
        x, y, z = mri_head_t[:3, 3]
        self.parameters[:6] = [rot_x, rot_y, rot_z, x, y, z]

    def save_trans(self, fname):
        """Save the head-mri transform as a fif file.

        Parameters
        ----------
        fname : str
            Target file path.
        """
        if not self.can_save:
            raise RuntimeError("Not enough information for saving transform")
        write_trans(fname, Transform('mri', 'head', self.mri_head_t))

    def _parameters_items_changed(self):
        # Update rot_x, rot_y, rot_z parameters if necessary
        for ii, key in enumerate(('rot_x', 'rot_y', 'rot_z',
                                  'trans_x', 'trans_y', 'trans_z',
                                  'scale_x', 'scale_y', 'scale_z')):
            if self.parameters[ii] != getattr(self, key):  # prevent circular
                setattr(self, key, self.parameters[ii])

    def _rot_x_changed(self):
        self.parameters[0] = self.rot_x

    def _rot_y_changed(self):
        self.parameters[1] = self.rot_y

    def _rot_z_changed(self):
        self.parameters[2] = self.rot_z

    def _trans_x_changed(self):
        self.parameters[3] = self.trans_x

    def _trans_y_changed(self):
        self.parameters[4] = self.trans_y

    def _trans_z_changed(self):
        self.parameters[5] = self.trans_z

    def _scale_x_changed(self):
        if self.n_scale_params == 1:
            self.parameters[6:9] = [self.scale_x] * 3
        else:
            self.parameters[6] = self.scale_x

    def _scale_y_changed(self):
        self.parameters[7] = self.scale_y

    def _scale_z_changed(self):
        self.parameters[8] = self.scale_z


class CoregFrameHandler(Handler):
    """Check for unfinished processes before closing its window."""

    def object_title_changed(self, info):
        """Set the title when it gets changed."""
        info.ui.title = info.object.title

    def close(self, info, is_ok):
        """Handle the close event."""
        if info.object.queue.unfinished_tasks:
            information(None, "Can not close the window while saving is still "
                        "in progress. Please wait until all MRIs are "
                        "processed.", "Saving Still in Progress")
            return False
        else:
            # store configuration, but don't prevent from closing on error
            try:
                info.object.save_config()
            except Exception as exc:
                warnings.warn("Error saving GUI configuration:\n%s" % (exc,))
            return True


def _make_view_coreg_panel(scrollable=False):
    """Generate View for CoregPanel."""
    view = View(VGroup(HGroup(Item('grow_hair',
                                   editor=laggy_float_editor_mm,
                                   width=_MM_WIDTH), Spring()),
                       HGroup(Item('n_scale_params', label='Scaling mode',
                                   editor=EnumEditor(
                                       values={0: '1:None',
                                               1: '2:Uniform',
                                               3: '3:3-axis'})), Spring()),
                       VGrid(Item('scale_x', editor=laggy_float_editor_scale,
                                  show_label=True, tooltip="Scale along "
                                  "right-left axis",
                                  enabled_when='n_scale_params > 0',
                                  width=_SCALE_WIDTH),
                             Item('scale_x_dec',
                                  enabled_when='n_scale_params > 0',
                                  width=_INC_BUTTON_WIDTH),
                             Item('scale_x_inc',
                                  enabled_when='n_scale_params > 0',
                                  width=_INC_BUTTON_WIDTH),
                             Item('scale_step', tooltip="Scaling step",
                                  enabled_when='n_scale_params > 0',
                                  width=_SCALE_WIDTH),
                             Spring(),

                             Item('scale_y', editor=laggy_float_editor_scale,
                                  show_label=True,
                                  enabled_when='n_scale_params > 1',
                                  tooltip="Scale along anterior-posterior "
                                  "axis", width=_SCALE_WIDTH),
                             Item('scale_y_dec',
                                  enabled_when='n_scale_params > 1',
                                  width=_INC_BUTTON_WIDTH),
                             Item('scale_y_inc',
                                  enabled_when='n_scale_params > 1',
                                  width=_INC_BUTTON_WIDTH),
                             Label('(Step)', width=_SCALE_WIDTH),
                             Spring(),

                             Item('scale_z', editor=laggy_float_editor_scale,
                                  enabled_when='n_scale_params > 1',
                                  show_label=True, tooltip="Scale along "
                                  "anterior-posterior axis",
                                  width=_SCALE_WIDTH),
                             Item('scale_z_dec',
                                  enabled_when='n_scale_params > 1',
                                  width=_INC_BUTTON_WIDTH),
                             Item('scale_z_inc',
                                  enabled_when='n_scale_params > 1',
                                  width=_INC_BUTTON_WIDTH),
                             '0',
                             Spring(),

                             label='Scaling parameters', show_labels=False,
                             columns=5),
                       VGrid(Item('fits_icp',
                                  enabled_when='n_scale_params',
                                  tooltip="Rotate, translate, and scale the "
                                  "MRI to minimize the distance from each "
                                  "digitizer point to the closest MRI point "
                                  "(one ICP iteration)", width=_BUTTON_WIDTH),
                             Item('fits_fid',
                                  enabled_when='n_scale_params == 1',
                                  tooltip="Rotate, translate, and scale the "
                                  "MRI to minimize the distance of the three "
                                  "fiducials.", width=_BUTTON_WIDTH),
                             Item('reset_scale',
                                  enabled_when='n_scale_params',
                                  tooltip="Reset scaling parameters",
                                  width=_BUTTON_WIDTH, height=-1),
                             '0',
                             show_labels=False, columns=2),
                       VGrid(Item('trans_x', editor=laggy_float_editor_m,
                                  show_label=True, tooltip="Move along "
                                  "right-left axis", width=_M_WIDTH),
                             Item('trans_x_dec', width=_INC_BUTTON_WIDTH),
                             Item('trans_x_inc', width=_INC_BUTTON_WIDTH),
                             Item('trans_step', tooltip="Movement step",
                                  width=_M_STEP_WIDTH),
                             Spring(),

                             Item('trans_y', editor=laggy_float_editor_m,
                                  show_label=True, tooltip="Move along "
                                  "anterior-posterior axis",
                                  width=_M_WIDTH),
                             Item('trans_y_dec', width=_INC_BUTTON_WIDTH),
                             Item('trans_y_inc', width=_INC_BUTTON_WIDTH),
                             Label('(Step)', width=_M_WIDTH),
                             Spring(),

                             Item('trans_z', editor=laggy_float_editor_m,
                                  show_label=True, tooltip="Move along "
                                  "anterior-posterior axis",
                                  width=_M_WIDTH),
                             Item('trans_z_dec', width=_INC_BUTTON_WIDTH),
                             Item('trans_z_inc', width=_INC_BUTTON_WIDTH),
                             Label(' '),
                             Spring(),

                             Item('rot_x', editor=laggy_float_editor_rad,
                                  show_label=True, tooltip="Rotate along "
                                  "right-left axis", width=_RAD_WIDTH),
                             Item('rot_x_dec', width=_INC_BUTTON_WIDTH),
                             Item('rot_x_inc', width=_INC_BUTTON_WIDTH),
                             Item('rot_step', tooltip="Rotation step",
                                  width=_RAD_STEP_WIDTH),
                             Spring(),

                             Item('rot_y', editor=laggy_float_editor_rad,
                                  show_label=True, tooltip="Rotate along "
                                  "anterior-posterior axis",
                                  width=_RAD_WIDTH),
                             Item('rot_y_dec', width=_INC_BUTTON_WIDTH),
                             Item('rot_y_inc', width=_INC_BUTTON_WIDTH),
                             Label('(Step)', width=_RAD_WIDTH),
                             Spring(),

                             Item('rot_z', editor=laggy_float_editor_rad,
                                  show_label=True, tooltip="Rotate along "
                                  "anterior-posterior axis",
                                  width=_RAD_WIDTH),
                             Item('rot_z_dec', width=_INC_BUTTON_WIDTH),
                             Item('rot_z_inc', width=_INC_BUTTON_WIDTH),
                             Label(' '),
                             Spring(),

                             show_labels=False, show_border=_SHOW_BORDER,
                             label=u'Translation (Δ) & rotation (∠)',
                             columns=5),
                       # buttons
                       VGroup(Item('fit_icp',
                                   enabled_when='has_hsp_data and '
                                   'hsp_weight > 0',
                                   tooltip="Rotate and translate the "
                                   "MRI to minimize the distance from each "
                                   "digitizer point to the closest MRI point "
                                   "(one ICP iteration)", width=_BUTTON_WIDTH),
                              Item('fit_fid',
                                   enabled_when='has_lpa_data and'
                                   'has_nasion_data and has_rpa_data',
                                   tooltip="Rotate and translate the "
                                   "MRI to minimize the distance of the three "
                                   "fiducials.", width=_BUTTON_WIDTH),
                              Item('reset_tr',
                                   tooltip="Reset translation and rotation.",
                                   width=_BUTTON_WIDTH),
                              Item('load_trans', width=_BUTTON_WIDTH),
                              show_labels=False, columns=2),
                       # Fitting weights
                       VGrid(Item('lpa_weight', editor=laggy_float_editor_w,
                                  tooltip="Relative weight for LPA",
                                  enabled_when='has_lpa_data',
                                  width=_WEIGHT_WIDTH),
                             Item('nasion_weight', editor=laggy_float_editor_w,
                                  tooltip="Relative weight for nasion",
                                  enabled_when='has_nasion_data',
                                  width=_WEIGHT_WIDTH),
                             Item('rpa_weight', editor=laggy_float_editor_w,
                                  tooltip="Relative weight for RPA",
                                  enabled_when='has_rpa_data',
                                  width=_WEIGHT_WIDTH),
                             Spring(),
                             show_labels=False, show_border=_SHOW_BORDER,
                             label='LPA/Nasion/RPA weights', columns=4),
                       Item('fid_eval_str', style='readonly',
                            tooltip='Fiducial differences', width=_TEXT_WIDTH),

                       VGrid(Item('hsp_weight', editor=laggy_float_editor_w,
                                  tooltip="Relative weight for head shape "
                                  "points", enabled_when='has_hsp_data and '
                                  'hsp_weight > 0', width=_WEIGHT_WIDTH,),
                             Item('eeg_weight', editor=laggy_float_editor_w,
                                  tooltip="Relative weight for EEG points",
                                  enabled_when='has_eeg_data',
                                  width=_WEIGHT_WIDTH),
                             Item('hpi_weight', editor=laggy_float_editor_w,
                                  tooltip="Relative weight for HPI points",
                                  enabled_when='has_hpi_data',
                                  width=_WEIGHT_WIDTH),
                             Spring(),
                             show_labels=False, show_border=_SHOW_BORDER,
                             label='HSP/EEG/HPI weights', columns=4),
                       Item('points_eval_str', style='readonly',
                            tooltip='Point error (μ ± σ)', width=_TEXT_WIDTH),
                       VGrid(
                           Item('scale_labels',
                                label="Scale *.label files",
                                enabled_when='n_scale_params > 0'),
                           Spring(),
                           Item('copy_annot',
                                label="Copy annotation files",
                                enabled_when='n_scale_params > 0'),
                           Spring(),
                           Item('prepare_bem_model',
                                label="Prepare BEM",
                                enabled_when='can_prepare_bem_model'),
                           Spring(),
                           show_left=False,
                           label='Subject-saving options', columns=2,
                           show_border=_SHOW_BORDER),
                       '_',
                       HGroup(Item('save', enabled_when='can_save',
                                   tooltip="Save the trans file and (if "
                                   "scaling is enabled) the scaled MRI"),
                              Item('reset_params', tooltip="Reset all "
                                   "coregistration parameters"),
                              show_labels=False),
                       Item('queue_feedback', style='readonly'),
                       Item('queue_current', style='readonly'),
                       Item('queue_len_str', style='readonly'),
                       Spring(),
                       show_labels=False),
                kind='panel', buttons=[UndoButton], scrollable=scrollable)
    return view


class CoregPanel(HasPrivateTraits):
    """Coregistration panel for Head<->MRI with scaling."""

    model = Instance(CoregModel)

    # parameters
    reset_params = Button(label='Reset')
    grow_hair = DelegatesTo('model')
    n_scale_params = DelegatesTo('model')
    parameters = DelegatesTo('model')
    scale_step = Float(0.01)
    scale_x = DelegatesTo('model')
    scale_x_dec = Button('-')
    scale_x_inc = Button('+')
    scale_y = DelegatesTo('model')
    scale_y_dec = Button('-')
    scale_y_inc = Button('+')
    scale_z = DelegatesTo('model')
    scale_z_dec = Button('-')
    scale_z_inc = Button('+')
    rot_step = Float(0.01)
    rot_x = DelegatesTo('model')
    rot_x_dec = Button('-')
    rot_x_inc = Button('+')
    rot_y = DelegatesTo('model')
    rot_y_dec = Button('-')
    rot_y_inc = Button('+')
    rot_z = DelegatesTo('model')
    rot_z_dec = Button('-')
    rot_z_inc = Button('+')
    trans_step = Float(0.001)
    trans_x = DelegatesTo('model')
    trans_x_dec = Button('-')
    trans_x_inc = Button('+')
    trans_y = DelegatesTo('model')
    trans_y_dec = Button('-')
    trans_y_inc = Button('+')
    trans_z = DelegatesTo('model')
    trans_z_dec = Button('-')
    trans_z_inc = Button('+')
    lpa_weight = DelegatesTo('model')
    nasion_weight = DelegatesTo('model')
    rpa_weight = DelegatesTo('model')
    hsp_weight = DelegatesTo('model')
    eeg_weight = DelegatesTo('model')
    hpi_weight = DelegatesTo('model')

    # fitting
    has_lpa_data = DelegatesTo('model')
    has_nasion_data = DelegatesTo('model')
    has_rpa_data = DelegatesTo('model')
    has_hsp_data = DelegatesTo('model')
    has_eeg_data = DelegatesTo('model')
    has_hpi_data = DelegatesTo('model')
    # fitting with scaling
    fits_icp = Button(label='Fit (ICP)')
    fits_fid = Button(label='Fit fiducials')
    reset_scale = Button(label='Reset')
    # fitting without scaling
    fit_icp = Button(label='Fit (ICP)')
    fit_fid = Button(label='Fit fiducials')
    reset_tr = Button(label='Reset')

    # fit info
    fid_eval_str = DelegatesTo('model')
    points_eval_str = DelegatesTo('model')

    # saving
    can_prepare_bem_model = DelegatesTo('model')
    can_save = DelegatesTo('model')
    scale_labels = DelegatesTo('model')
    copy_annot = DelegatesTo('model')
    prepare_bem_model = DelegatesTo('model')
    save = Button(label="Save as...")
    load_trans = Button(label='Load...')
    queue = Instance(queue.Queue, ())
    queue_feedback = Str('')
    queue_current = Str('')
    queue_len = Int(0)
    queue_len_str = Property(Str, depends_on=['queue_len'])

    view = _make_view_coreg_panel()

    def __init__(self, *args, **kwargs):  # noqa: D102
        super(CoregPanel, self).__init__(*args, **kwargs)
        self.model.parameters = [0., 0., 0., 0., 0., 0., 1., 1., 1.]

        # Setup scaling worker
        def worker():
            while True:
                (subjects_dir, subject_from, subject_to, scale, skip_fiducials,
                 include_labels, include_annot, bem_names) = self.queue.get()
                self.queue_len -= 1

                # Scale MRI files
                self.queue_current = 'Scaling %s...' % subject_to
                try:
                    scale_mri(subject_from, subject_to, scale, True,
                              subjects_dir, skip_fiducials, include_labels,
                              include_annot)
                except Exception:
                    logger.error('Error scaling %s:\n' % subject_to +
                                 traceback.format_exc())
                    self.queue_feedback = ('Error scaling %s (see Terminal)' %
                                           subject_to)
                    bem_names = ()  # skip bem solutions
                else:
                    self.queue_feedback = 'Done scaling %s.' % subject_to

                # Precompute BEM solutions
                for bem_name in bem_names:
                    self.queue_current = ('Computing %s solution...' %
                                          bem_name)
                    try:
                        bem_file = bem_fname.format(subjects_dir=subjects_dir,
                                                    subject=subject_to,
                                                    name=bem_name)
                        bemsol = make_bem_solution(bem_file)
                        write_bem_solution(bem_file[:-4] + '-sol.fif', bemsol)
                    except Exception:
                        logger.error('Error computing %s solution:\n' %
                                     bem_name + traceback.format_exc())
                        self.queue_feedback = ('Error computing %s solution '
                                               '(see Terminal)' % bem_name)
                    else:
                        self.queue_feedback = ('Done computing %s solution.' %
                                               bem_name)

                # Finalize
                self.queue_current = ''
                self.queue.task_done()

        t = Thread(target=worker)
        t.daemon = True
        t.start()

    @cached_property
    def _get_queue_len_str(self):
        if self.queue_len:
            return "Queue length: %i" % self.queue_len
        else:
            return ''

    @cached_property
    def _get_rotation(self):
        rot = np.array([self.rot_x, self.rot_y, self.rot_z])
        return rot

    @cached_property
    def _get_translation(self):
        trans = np.array([self.trans_x, self.trans_y, self.trans_z])
        return trans

    def _n_scale_params_fired(self):
        if self.n_scale_params == 0:
            use = [1] * 3
        elif self.n_scale_params == 1:
            use = [np.mean([self.scale_x, self.scale_y, self.scale_z])] * 3
        else:
            use = self.parameters[6:9]
        self.parameters[6:9] = use

    def _fit_fid_fired(self):
        GUI.set_busy()
        self.model.fit_fiducials(0)
        GUI.set_busy(False)

    def _fit_icp_fired(self):
        GUI.set_busy()
        self.model.fit_icp(0)
        GUI.set_busy(False)

    def _fits_fid_fired(self):
        GUI.set_busy()
        self.model.fit_fiducials(self.n_scale_params)
        GUI.set_busy(False)

    def _fits_icp_fired(self):
        GUI.set_busy()
        self.model.fit_icp(self.n_scale_params)
        GUI.set_busy(False)

    def _reset_scale_fired(self):
        self.reset_traits(('scale_x', 'scale_y', 'scale_z'))

    def _reset_tr_fired(self):
        self.reset_traits(('trans_x', 'trans_y', 'trans_z',
                           'rot_x', 'rot_y', 'rot_z'))

    def _reset_params_fired(self):
        self.model.reset()

    def _rot_x_dec_fired(self):
        self.rot_x -= self.rot_step

    def _rot_x_inc_fired(self):
        self.rot_x += self.rot_step

    def _rot_y_dec_fired(self):
        self.rot_y -= self.rot_step

    def _rot_y_inc_fired(self):
        self.rot_y += self.rot_step

    def _rot_z_dec_fired(self):
        self.rot_z -= self.rot_step

    def _rot_z_inc_fired(self):
        self.rot_z += self.rot_step

    def _load_trans_fired(self):
        # find trans file destination
        raw_dir = os.path.dirname(self.model.hsp.file)
        subject = self.model.mri.subject
        trans_file = trans_fname.format(raw_dir=raw_dir, subject=subject)
        dlg = FileDialog(action="open", wildcard=trans_wildcard,
                         default_path=trans_file)
        if dlg.open() != OK:
            return
        trans_file = dlg.path
        try:
            self.model.load_trans(trans_file)
        except Exception as e:
            error(None, "Error loading trans file %s: %s (See terminal "
                  "for details)" % (trans_file, e), "Error Loading Trans File")
            raise

    def _save_fired(self):
        subjects_dir = self.model.mri.subjects_dir
        subject_from = self.model.mri.subject

        # check that fiducials are saved
        skip_fiducials = False
        if self.n_scale_params and not _find_fiducials_files(subject_from,
                                                             subjects_dir):
            msg = ("No fiducials file has been found for {src}. If fiducials "
                   "are not saved, they will not be available in the scaled "
                   "MRI. Should the current fiducials be saved now? "
                   "Select Yes to save the fiducials at "
                   "{src}/bem/{src}-fiducials.fif. "
                   "Select No to proceed scaling the MRI without fiducials.".
                   format(src=subject_from))
            title = "Save Fiducials for %s?" % subject_from
            rc = confirm(None, msg, title, cancel=True, default=CANCEL)
            if rc == CANCEL:
                return
            elif rc == YES:
                self.model.mri.save(self.model.mri.default_fid_fname)
            elif rc == NO:
                skip_fiducials = True
            else:
                raise RuntimeError("rc=%s" % repr(rc))

        # find target subject
        if self.n_scale_params:
            subject_to = self.model.raw_subject or subject_from
            mridlg = NewMriDialog(subjects_dir=subjects_dir,
                                  subject_from=subject_from,
                                  subject_to=subject_to)
            ui = mridlg.edit_traits(kind='modal')
            if not ui.result:  # i.e., user pressed cancel
                return
            subject_to = mridlg.subject_to
        else:
            subject_to = subject_from

        # find trans file destination
        raw_dir = os.path.dirname(self.model.hsp.file)
        trans_file = trans_fname.format(raw_dir=raw_dir, subject=subject_to)
        dlg = FileDialog(action="save as", wildcard=trans_wildcard,
                         default_path=trans_file)
        dlg.open()
        if dlg.return_code != OK:
            return
        trans_file = dlg.path
        if not trans_file.endswith('.fif'):
            trans_file += '.fif'
            if os.path.exists(trans_file):
                answer = confirm(None, "The file %r already exists. Should it "
                                 "be replaced?", "Overwrite File?")
                if answer != YES:
                    return

        # save the trans file
        try:
            self.model.save_trans(trans_file)
        except Exception as e:
            error(None, "Error saving -trans.fif file: %s (See terminal for "
                  "details)" % (e,), "Error Saving Trans File")
            raise

        # save the scaled MRI
        if self.n_scale_params:
            job = self.model.get_scaling_job(subject_to, skip_fiducials)
            self.queue.put(job)
            self.queue_len += 1

    def _scale_x_dec_fired(self):
        self.scale_x -= self.scale_step

    def _scale_x_inc_fired(self):
        self.scale_x += self.scale_step

    def _scale_y_dec_fired(self):
        self.scale_y -= self.scale_step

    def _scale_y_inc_fired(self):
        self.scale_y += self.scale_step

    def _scale_z_dec_fired(self):
        self.scale_z -= self.scale_step

    def _scale_z_inc_fired(self):
        self.scale_z += self.scale_step

    def _trans_x_dec_fired(self):
        self.trans_x -= self.trans_step

    def _trans_x_inc_fired(self):
        self.trans_x += self.trans_step

    def _trans_y_dec_fired(self):
        self.trans_y -= self.trans_step

    def _trans_y_inc_fired(self):
        self.trans_y += self.trans_step

    def _trans_z_dec_fired(self):
        self.trans_z -= self.trans_step

    def _trans_z_inc_fired(self):
        self.trans_z += self.trans_step


class NewMriDialog(HasPrivateTraits):
    """New MRI dialog."""

    # Dialog to determine target subject name for a scaled MRI
    subjects_dir = Directory
    subject_to = Str
    subject_from = Str
    subject_to_dir = Property(depends_on=['subjects_dir', 'subject_to'])
    subject_to_exists = Property(Bool, depends_on='subject_to_dir')

    feedback = Str(' ' * 100)
    can_overwrite = Bool
    overwrite = Bool
    can_save = Bool

    view = View(Item('subject_to', label='New MRI Subject Name', tooltip="A "
                     "new folder with this name will be created in the "
                     "current subjects_dir for the scaled MRI files"),
                Item('feedback', show_label=False, style='readonly'),
                Item('overwrite', enabled_when='can_overwrite', tooltip="If a "
                     "subject with the chosen name exists, delete the old "
                     "subject"),
                buttons=[CancelButton,
                         Action(name='OK', enabled_when='can_save')])

    def _can_overwrite_changed(self, new):
        if not new:
            self.overwrite = False

    @cached_property
    def _get_subject_to_dir(self):
        return os.path.join(self.subjects_dir, self.subject_to)

    @cached_property
    def _get_subject_to_exists(self):
        if not self.subject_to:
            return False
        elif os.path.exists(self.subject_to_dir):
            return True
        else:
            return False

    @on_trait_change('subject_to_dir,overwrite')
    def update_dialog(self):
        if not self.subject_from:
            # weird trait state that occurs even when subject_from is set
            return
        elif not self.subject_to:
            self.feedback = "No subject specified..."
            self.can_save = False
            self.can_overwrite = False
        elif self.subject_to == self.subject_from:
            self.feedback = "Must be different from MRI source subject..."
            self.can_save = False
            self.can_overwrite = False
        elif self.subject_to_exists:
            if self.overwrite:
                self.feedback = "%s will be overwritten." % self.subject_to
                self.can_save = True
                self.can_overwrite = True
            else:
                self.feedback = "Subject already exists..."
                self.can_save = False
                self.can_overwrite = True
        else:
            self.feedback = "Name ok."
            self.can_save = True
            self.can_overwrite = False


def _make_view(tabbed=False, split=False, scene_width=500, scene_height=400,
               scrollable=True):
    """Create a view for the CoregFrame.

    Parameters
    ----------
    tabbed : bool
        Combine the data source panel and the coregistration panel into a
        single panel with tabs.
    split : bool
        Split the main panels with a movable splitter (good for QT4 but
        unnecessary for wx backend).
    scene_width : int
        Specify a minimum width for the 3d scene (in pixels).
    scrollable : bool
        Make the coregistration panel vertically scrollable (default True).

    Returns
    -------
    view : traits View
        View object for the CoregFrame.
    """
    # XXX This setting of scene_width and scene_height is a limiting factor
    scene = Item('scene', show_label=False,
                 width=scene_width, height=scene_height,
                 editor=SceneEditor(scene_class=MayaviScene),
                 dock='vertical')

    data_panel = VGroup(
        VGroup(Item('subject_panel', style='custom'), label="MRI Subject",
               show_border=_SHOW_BORDER, show_labels=False),
        VGroup(Item('lock_fiducials', style='custom',
                    editor=EnumEditor(cols=2, values={False: '2:Edit',
                                                      True: '1:Lock'}),
                    enabled_when='fid_ok'),
               HGroup(Item('hsp_always_visible',
                           label='Show head shape points', show_label=True,
                           enabled_when='not lock_fiducials', width=-1),
                      show_left=False),
               Item('fid_panel', style='custom'),
               label="MRI Fiducials",  show_border=_SHOW_BORDER,
               show_labels=False),
        VGroup(Item('raw_src', style="custom", width=_TEXT_WIDTH),
               HGroup('guess_mri_subject',
                      Label('Guess subject from filename'), show_labels=False),
               HGroup(Item('distance', show_label=False, width=_MM_WIDTH,
                           editor=laggy_float_editor_mm),
                      'omit_points',
                      'reset_omit_points',
                      Spring(), show_labels=False),
               Item('omitted_info', style='readonly', width=_TEXT_WIDTH),
               label='Digitization source',
               show_border=_SHOW_BORDER, show_labels=False),
        VGroup(HGroup(Item('headview', style='custom'), Spring(),
                      show_labels=False),
               Item('view_options', width=_TEXT_WIDTH),
               label='View', show_border=_SHOW_BORDER, show_labels=False),
        Spring(),
        show_labels=False, label="Data Source", show_border=True,
        enabled_when='True')

    # Setting `scrollable=True` for a Group does not seem to have any effect
    # (macOS), in order to be effective the parameter has to be set for a View
    # object; hence we use a special InstanceEditor to set the parameter
    # programmatically:
    coreg_panel = VGroup(
        Item('coreg_panel', style='custom',
             width=_COREG_WIDTH if scrollable else 1,
             editor=InstanceEditor(view=_make_view_coreg_panel(scrollable))),
        label="Coregistration", show_border=not scrollable, show_labels=False,
        enabled_when="fid_panel.locked")

    main_layout = 'split' if split else 'normal'

    if tabbed:
        main = HGroup(scene,
                      Group(data_panel, coreg_panel, show_labels=False,
                            layout='tabbed'),
                      layout=main_layout)
    else:
        main = HGroup(data_panel, scene, coreg_panel, show_labels=False,
                      layout=main_layout)

    # Here we set the width and height to impossibly small numbers to force the
    # window to be as tight as possible
    view = View(main, resizable=True, handler=CoregFrameHandler(),
                buttons=NoButtons, width=scene_width + 2 * _COREG_WIDTH,
                height=scene_height)
    return view


class ViewOptionsPanel(HasTraits):
    """View options panel."""

    mri_obj = Instance(SurfaceObject)
    hsp_obj = Instance(PointObject)
    eeg_obj = Instance(PointObject)
    hpi_obj = Instance(PointObject)
    hsp_cf_obj = Instance(PointObject)
    mri_cf_obj = Instance(PointObject)
    coord_frame = Enum('mri', 'head', label='Display coordinate frame')
    head_high_res = Bool(True, label='Show high-resolution head')
    view = View(
        VGroup(
            Item('mri_obj', style='custom', label="MRI"),
            Item('hsp_obj', style='custom', label="Head shape"),
            Item('eeg_obj', style='custom', label='EEG'),
            Item('hpi_obj', style='custom', label='HPI'),
            VGrid(Item('coord_frame', style='custom',
                       editor=EnumEditor(values={'mri': '1:MRI',
                                                 'head': '2:Head'}, cols=2,
                                         format_func=lambda x: x)),
                  Spring(),
                  Item('head_high_res'),
                  Spring(), columns=2, show_labels=True),
            Item('hsp_cf_obj', style='custom', label='Head axes'),
            Item('mri_cf_obj', style='custom', label='MRI axes'),
        ), title="Display options")


class CoregFrame(HasTraits):
    """GUI for head-MRI coregistration."""

    model = Instance(CoregModel)

    scene = Instance(MlabSceneModel, ())
    headview = Instance(HeadViewController)
    head_high_res = Bool(True)

    subject_panel = Instance(SubjectSelectorPanel)
    fid_panel = Instance(FiducialsPanel)
    coreg_panel = Instance(CoregPanel)
    view_options_panel = Instance(ViewOptionsPanel)

    raw_src = DelegatesTo('model', 'hsp')
    guess_mri_subject = DelegatesTo('model')
    project_to_surface = DelegatesTo('eeg_obj')
    orient_to_surface = DelegatesTo('hsp_obj')
    scale_by_distance = DelegatesTo('hsp_obj')
    mark_inside = DelegatesTo('hsp_obj')

    # Omit Points
    distance = Float(5., desc="maximal distance for head shape points from "
                     "the surface (mm)")
    omit_points = Button(label='Omit [mm]', desc="to omit head shape points "
                         "for the purpose of the automatic coregistration "
                         "procedure.")
    reset_omit_points = Button(label='Reset', desc="to reset the "
                               "omission of head shape points to include all.")
    omitted_info = Property(Str, depends_on=['model:hsp:n_omitted'])

    fid_ok = DelegatesTo('model', 'mri.fid_ok')
    lock_fiducials = DelegatesTo('model')
    hsp_always_visible = Bool(False, label="Always Show Head Shape")
    title = Str('MNE Coreg')

    # visualization (MRI)
    mri_obj = Instance(SurfaceObject)
    mri_lpa_obj = Instance(PointObject)
    mri_nasion_obj = Instance(PointObject)
    mri_rpa_obj = Instance(PointObject)
    # visualization (Digitization)
    hsp_obj = Instance(PointObject)
    eeg_obj = Instance(PointObject)
    hpi_obj = Instance(PointObject)
    hsp_lpa_obj = Instance(PointObject)
    hsp_nasion_obj = Instance(PointObject)
    hsp_rpa_obj = Instance(PointObject)
    hsp_visible = Property(depends_on=['hsp_always_visible', 'lock_fiducials'])
    # Coordinate frame axes
    hsp_cf_obj = Instance(PointObject)
    mri_cf_obj = Instance(PointObject)

    view_options = Button(label="Display options...")

    picker = Instance(object)

    # Processing
    queue = DelegatesTo('coreg_panel')

    view = _make_view()

    def _model_default(self):
        return CoregModel(
            scale_labels=self._config.get(
                'MNE_COREG_SCALE_LABELS', 'true') == 'true',
            copy_annot=self._config.get(
                'MNE_COREG_COPY_ANNOT', 'true') == 'true',
            prepare_bem_model=self._config.get(
                'MNE_COREG_PREPARE_BEM', 'true') == 'true')

    def _subject_panel_default(self):
        return SubjectSelectorPanel(model=self.model.mri.subject_source)

    def _fid_panel_default(self):
        return FiducialsPanel(model=self.model.mri, headview=self.headview)

    def _coreg_panel_default(self):
        return CoregPanel(model=self.model)

    def _headview_default(self):
        return HeadViewController(
            scene=self.scene, system='RAS',
            scale=self._initial_kwargs['scale'],
            interaction=self._initial_kwargs['interaction'])

    def __init__(self, raw=None, subject=None, subjects_dir=None,
                 guess_mri_subject=True, head_opacity=1.,
                 head_high_res=True, trans=None, config=None,
                 project_eeg=False, orient_to_surface=False,
                 scale_by_distance=False, mark_inside=False,
                 interaction='trackball', scale=0.16):  # noqa: D102
        self._config = config or {}
        super(CoregFrame, self).__init__(guess_mri_subject=guess_mri_subject,
                                         head_high_res=head_high_res)
        self._initial_kwargs = dict(project_eeg=project_eeg,
                                    orient_to_surface=orient_to_surface,
                                    scale_by_distance=scale_by_distance,
                                    mark_inside=mark_inside,
                                    head_opacity=head_opacity,
                                    interaction=interaction,
                                    scale=scale)
        if not 0 <= head_opacity <= 1:
            raise ValueError(
                "head_opacity needs to be a floating point number between 0 "
                "and 1, got %r" % (head_opacity,))

        if (subjects_dir is not None) and os.path.isdir(subjects_dir):
            self.model.mri.subjects_dir = subjects_dir

        if raw is not None:
            self.model.hsp.file = raw

        if subject is not None:
            if subject not in self.model.mri.subject_source.subjects:
                msg = "%s is not a valid subject. " % subject
                # no subjects -> ['']
                if any(self.model.mri.subject_source.subjects):
                    ss = ', '.join(self.model.mri.subject_source.subjects)
                    msg += ("The following subjects have been found: %s "
                            "(subjects_dir=%s). " %
                            (ss, self.model.mri.subjects_dir))
                else:
                    msg += ("No subjects were found in subjects_dir=%s. " %
                            self.model.mri.subjects_dir)
                msg += ("Make sure all MRI subjects have head shape files "
                        "(run $ mne make_scalp_surfaces).")
                raise ValueError(msg)
            self.model.mri.subject = subject
        if trans is not None:
            try:
                self.model.load_trans(trans)
            except Exception as e:
                error(None, "Error loading trans file %s: %s (See terminal "
                      "for details)" % (trans, e), "Error Loading Trans File")

    @on_trait_change('subject_panel:subject')
    def _set_title(self):
        self.title = '%s - MNE Coreg' % self.model.mri.subject

    @on_trait_change('scene:activated')
    def _init_plot(self):
        _toggle_mlab_render(self, False)

        lpa_color = defaults['lpa_color']
        nasion_color = defaults['nasion_color']
        rpa_color = defaults['rpa_color']

        # MRI scalp
        color = defaults['head_color']
        self.mri_obj = SurfaceObject(
            points=np.empty((0, 3)), color=color, tri=np.empty((0, 3)),
            scene=self.scene, name="MRI Scalp", block_behind=True,
            # opacity=self._initial_kwargs['head_opacity'],
            # setting opacity here causes points to be
            # [[0, 0, 0]] -- why??
        )
        self.mri_obj.opacity = self._initial_kwargs['head_opacity']
        self.fid_panel.hsp_obj = self.mri_obj
        # Do not do sync_trait here, instead use notifiers elsewhere

        # MRI Fiducials
        point_scale = defaults['mri_fid_scale']
        self.mri_lpa_obj = PointObject(scene=self.scene, color=lpa_color,
                                       point_scale=point_scale, name='LPA')
        self.model.sync_trait('transformed_mri_lpa',
                              self.mri_lpa_obj, 'points', mutual=False)
        self.mri_nasion_obj = PointObject(scene=self.scene, color=nasion_color,
                                          point_scale=point_scale,
                                          name='Nasion')
        self.model.sync_trait('transformed_mri_nasion',
                              self.mri_nasion_obj, 'points', mutual=False)
        self.mri_rpa_obj = PointObject(scene=self.scene, color=rpa_color,
                                       point_scale=point_scale, name='RPA')
        self.model.sync_trait('transformed_mri_rpa',
                              self.mri_rpa_obj, 'points', mutual=False)

        # Digitizer Head Shape
        kwargs = dict(
            view='cloud', scene=self.scene, resolution=20,
            orient_to_surface=self._initial_kwargs['orient_to_surface'],
            scale_by_distance=self._initial_kwargs['scale_by_distance'],
            mark_inside=self._initial_kwargs['mark_inside'])
        self.hsp_obj = PointObject(
            color=defaults['extra_color'], name='Extra',
            point_scale=defaults['extra_scale'], **kwargs)
        self.model.sync_trait('transformed_hsp_points',
                              self.hsp_obj, 'points', mutual=False)

        # Digitizer EEG
        self.eeg_obj = PointObject(
            color=defaults['eeg_color'], point_scale=defaults['eeg_scale'],
            name='EEG', projectable=True,
            project_to_surface=self._initial_kwargs['project_eeg'], **kwargs)
        self.model.sync_trait('transformed_hsp_eeg_points',
                              self.eeg_obj, 'points', mutual=False)

        # Digitizer HPI
        self.hpi_obj = PointObject(
            color=defaults['hpi_color'], name='HPI',
            point_scale=defaults['hpi_scale'], **kwargs)
        self.model.sync_trait('transformed_hsp_hpi',
                              self.hpi_obj, 'points', mutual=False)
        for p in (self.hsp_obj, self.eeg_obj, self.hpi_obj):
            self.model.mri.bem_low_res.sync_trait('tris', p, 'project_to_tris',
                                                  mutual=False)
            self.model.sync_trait('transformed_low_res_mri_points',
                                  p, 'project_to_points', mutual=False)
            p.inside_color = self.mri_obj.color
            self.mri_obj.sync_trait('color', p, 'inside_color',
                                    mutual=False)

        # Digitizer Fiducials
        point_scale = defaults['dig_fid_scale']
        opacity = defaults['dig_fid_opacity']
        self.hsp_lpa_obj = PointObject(
            scene=self.scene, color=lpa_color, opacity=opacity,
            point_scale=point_scale, name='HSP-LPA')
        self.model.sync_trait('transformed_hsp_lpa',
                              self.hsp_lpa_obj, 'points', mutual=False)
        self.hsp_nasion_obj = PointObject(
            scene=self.scene, color=nasion_color, opacity=opacity,
            point_scale=point_scale, name='HSP-Nasion')
        self.model.sync_trait('transformed_hsp_nasion',
                              self.hsp_nasion_obj, 'points', mutual=False)
        self.hsp_rpa_obj = PointObject(
            scene=self.scene, color=rpa_color, opacity=opacity,
            point_scale=point_scale, name='HSP-RPA')
        self.model.sync_trait('transformed_hsp_rpa',
                              self.hsp_rpa_obj, 'points', mutual=False)

        # All points share these
        for p in (self.hsp_obj, self.eeg_obj, self.hpi_obj,
                  self.hsp_lpa_obj, self.hsp_nasion_obj, self.hsp_rpa_obj):
            self.sync_trait('hsp_visible', p, 'visible', mutual=False)

        on_pick = self.scene.mayavi_scene.on_mouse_pick
        self.picker = on_pick(self.fid_panel._on_pick, type='cell')

        # Coordinate frame axes
        self.mri_cf_obj = PointObject(
            scene=self.scene, color=self.mri_obj.color,
            opacity=self.mri_obj.opacity, label_scale=5e-3,
            point_scale=0.02, name='MRI', view='arrow')
        self.mri_obj.sync_trait('color', self.mri_cf_obj, mutual=False)
        self._update_mri_axes()
        self.hsp_cf_obj = PointObject(
            scene=self.scene, color=self.hsp_obj.color,
            opacity=self.mri_obj.opacity, label_scale=5e-3,
            point_scale=0.02, name='Head', view='arrow')
        self.hsp_cf_obj.sync_trait('color', self.hsp_cf_obj, mutual=False)
        self._update_hsp_axes()

        self.headview.left = True
        self._on_mri_src_change()
        _toggle_mlab_render(self, True)
        self.scene.render()
        self.scene.camera.focal_point = (0., 0., 0.)
        self.view_options_panel = ViewOptionsPanel(
            mri_obj=self.mri_obj, hsp_obj=self.hsp_obj,
            eeg_obj=self.eeg_obj, hpi_obj=self.hpi_obj,
            hsp_cf_obj=self.hsp_cf_obj, mri_cf_obj=self.mri_cf_obj,
            head_high_res=self.head_high_res)
        self.view_options_panel.sync_trait('coord_frame', self.model,
                                           mutual=False)
        self.view_options_panel.sync_trait('head_high_res', self,
                                           mutual=False)

    @cached_property
    def _get_hsp_visible(self):
        return self.hsp_always_visible or self.lock_fiducials

    @cached_property
    def _get_omitted_info(self):
        if self.model.hsp.n_omitted == 0:
            return "No points omitted"
        elif self.model.hsp.n_omitted == 1:
            return "1 point omitted"
        else:
            return "%i points omitted" % self.model.hsp.n_omitted

    def _omit_points_fired(self):
        distance = self.distance / 1000.
        self.model.omit_hsp_points(distance)

    def _reset_omit_points_fired(self):
        self.model.omit_hsp_points(np.inf)

    @on_trait_change('model:mri_trans')
    def _update_mri_axes(self):
        if self.mri_cf_obj is None:
            return
        nn = apply_trans(self.model.mri_trans, np.eye(3), move=False)
        pts = apply_trans(self.model.mri_trans, np.zeros((3, 3)))
        self.mri_cf_obj.nn = nn
        self.mri_cf_obj.points = pts

    @on_trait_change('model:hsp_trans')
    def _update_hsp_axes(self):
        if self.hsp_cf_obj is None:
            return
        nn = apply_trans(self.model.hsp_trans, np.eye(3), move=False)
        pts = apply_trans(self.model.hsp_trans, np.zeros((3, 3)))
        self.hsp_cf_obj.nn = nn
        self.hsp_cf_obj.points = pts

    @on_trait_change('model:transformed_high_res_mri_points')
    def _update_mri_obj_points(self):
        if self.mri_obj is None:
            return
        self.mri_obj.points = getattr(
            self.model, 'transformed_%s_res_mri_points'
            % ('high' if self.head_high_res else 'low',))

    @on_trait_change('model:mri:bem_high_res.tris,head_high_res')
    def _on_mri_src_change(self):
        if self.mri_obj is None:
            return
        if not (np.any(self.model.mri.bem_low_res.points) and
                np.any(self.model.mri.bem_low_res.tris)):
            self.mri_obj.clear()
            return

        if self.head_high_res:
            bem = self.model.mri.bem_high_res
        else:
            bem = self.model.mri.bem_low_res
        self.mri_obj.tri = bem.tris
        self._update_mri_obj_points()
        self.mri_obj.plot()

    # automatically lock fiducials if a good fiducials file is loaded
    @on_trait_change('model:mri:fid_file')
    def _on_fid_file_loaded(self):
        if self.model.mri.fid_file:
            self.fid_panel.locked = True
        else:
            self.fid_panel.locked = False

    def _view_options_fired(self):
        self.view_options_panel.edit_traits()

    def save_config(self, home_dir=None):
        """Write configuration values."""
        def s_c(key, value, lower=True):
            value = str(value)
            if lower:
                value = value.lower()
            set_config(key, str(value).lower(), home_dir=home_dir,
                       set_env=False)

        s_c('MNE_COREG_GUESS_MRI_SUBJECT', self.model.guess_mri_subject)
        s_c('MNE_COREG_HEAD_HIGH_RES', self.head_high_res)
        s_c('MNE_COREG_HEAD_OPACITY', self.mri_obj.opacity)
        # This works on Qt variants. We cannot rely on
        # scene.renderer.size or scene.render_window.size because these
        # are in physical pixel units rather than logical (so HiDPI breaks
        # things). The problem remains that setting the size of the scene
        # sets the *minimum* as well as actual size of the scene, so
        # the window cannot be shrunk anymore :(. So we are stuck not
        # saving our size until we figure out how to have Traits not set
        # the minimum. (Not setting the scene size at all makes the
        # scene, left, and right panels grow at the same rate, which
        # is not helpful. The left and right should be fixed.)
        # try:
        #     w, h = self.scene.control.size()
        # except Exception:
        #     pass
        # else:
        #     s_c('MNE_COREG_SCENE_WIDTH', w)
        #     s_c('MNE_COREG_SCENE_HEIGHT', h)
        s_c('MNE_COREG_SCENE_SCALE', self.headview.scale)
        s_c('MNE_COREG_SCALE_LABELS', self.model.scale_labels)
        s_c('MNE_COREG_COPY_ANNOT', self.model.copy_annot)
        s_c('MNE_COREG_PREPARE_BEM', self.model.prepare_bem_model)
        if self.model.mri.subjects_dir:
            s_c('MNE_COREG_SUBJECTS_DIR', self.model.mri.subjects_dir, False)
        s_c('MNE_COREG_PROJECT_EEG', self.project_to_surface)
        s_c('MNE_COREG_ORIENT_TO_SURFACE', self.orient_to_surface)
        s_c('MNE_COREG_SCALE_BY_DISTANCE', self.scale_by_distance)
        s_c('MNE_COREG_MARK_INSIDE', self.mark_inside)
        s_c('MNE_COREG_INTERACTION', self.headview.interaction)
