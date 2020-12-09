# -*- coding: utf-8 -*-
u"""Traits-based GUI for head-MRI coregistration.

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
  |-- DataPanel (data_panel)
  |   |-- HeadViewController (headview) [4]: Set head views for the given coordinate system.
  |   |   +-- MlabSceneModel (scene) [3*]: ``HeadViewController(scene=CoregFrame.scene)``
  |   |-- SubjectSelectorPanel (subject_panel): Subject selector panel
  |   |   +-- MRISubjectSource (model) [2*]: ``SubjectSelectorPanel(model=self.model.mri.subject_source)``
  |   +-- FiducialsPanel (fid_panel): Set fiducials on an MRI surface.
  |       |-- MRIHeadWithFiducialsModel (model) [1*]: ``FiducialsPanel(model=CoregFrame.model.mri, headview=CoregFrame.headview)``
  |       |-- HeadViewController (headview) [4*]: ``FiducialsPanel(model=CoregFrame.model.mri, headview=CoregFrame.headview)``
  |       +-- SurfaceObject (hsp_obj) [5*]: ``CoregFrame.fid_panel.hsp_obj = CoregFrame.mri_obj``
  |-- CoregPanel (coreg_panel): Coregistration panel for Head<->MRI with scaling.
  |   +-- FittingOptionsPanel (fitting_options_panel): panel for fitting options.
  |-- SurfaceObject (mri_obj) [5]: Represent a solid object in a mayavi scene.
  +-- PointObject ({hsp, eeg, lpa, nasion, rpa, hsp_lpa, hsp_nasion, hsp_rpa} + _obj): Represent a group of individual points in a mayavi scene.

In the MRI viewing frame, MRI points are transformed via scaling, then by
mri_head_t to the Neuromag head coordinate frame. Digitized points (in head
coordinate frame) are never transformed.

Units
-----
User-facing GUI values are in readable units:

- ``scale_*`` are in %
- ``trans_*`` are in mm
- ``rot_*`` are in °

Internal computation quantities ``parameters`` are in units of (for X/Y/Z):

- ``parameters[:3]`` are in radians
- ``parameters[3:6]`` are in m
- ``paramteres[6:9]`` are in scale proportion

Conversions are handled via `np.deg2rad`, `np.rad2deg`, and appropriate
multiplications / divisions.
"""  # noqa: E501

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os
import queue
import re
import time
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
                          Handler, Label, Spring, InstanceEditor, StatusItem,
                          UIInfo)
from traitsui.menu import Action, UndoButton, CancelButton, NoButtons
from tvtk.pyface.scene_editor import SceneEditor

from ..bem import make_bem_solution, write_bem_solution
from ..coreg import bem_fname, trans_fname
from ..defaults import DEFAULTS
from ..surface import _DistanceQuery, _CheckInside
from ..transforms import (write_trans, read_trans, apply_trans, rotation,
                          rotation_angles, Transform, _ensure_trans,
                          rot_to_quat, _angle_between_quats)
from ..coreg import fit_matched_points, scale_mri, _find_fiducials_files
from ..viz.backends._pysurfer_mayavi import _toggle_mlab_render
from ..viz._3d import _get_3d_option
from ..utils import logger, set_config, _pl
from ._fiducials_gui import MRIHeadWithFiducialsModel, FiducialsPanel
from ._file_traits import trans_wildcard, DigSource, SubjectSelectorPanel
from ._viewer import (HeadViewController, PointObject, SurfaceObject,
                      _DEG_WIDTH, _MM_WIDTH, _BUTTON_WIDTH,
                      _SHOW_BORDER, _COREG_WIDTH, _SCALE_STEP_WIDTH,
                      _INC_BUTTON_WIDTH, _SCALE_WIDTH, _WEIGHT_WIDTH,
                      _MM_STEP_WIDTH, _DEG_STEP_WIDTH, _REDUCED_TEXT_WIDTH,
                      _RESET_LABEL, _RESET_WIDTH,
                      laggy_float_editor_scale, laggy_float_editor_deg,
                      laggy_float_editor_mm, laggy_float_editor_weight)

try:
    from traitsui.api import RGBColor
except ImportError:
    from traits.api import RGBColor

defaults = DEFAULTS['coreg']


class busy(object):
    """Set the GUI state to busy."""

    def __enter__(self):  # noqa: D105
        GUI.set_busy(True)

    def __exit__(self, type, value, traceback):  # noqa: D105
        GUI.set_busy(False)


def _pass(x):
    """Format text without changing it."""
    return x


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
    grow_hair = Float(label=u"ΔHair", desc="Move the back of the MRI "
                      "head outwards to compensate for hair on the digitizer "
                      "head shape (mm)")
    n_scale_params = Enum(0, 1, 3, desc="Scale the MRI to better fit the "
                          "subject's head shape (a new MRI subject will be "
                          "created with a name specified upon saving)")
    scale_x = Float(100, label="X")
    scale_y = Float(100, label="Y")
    scale_z = Float(100, label="Z")
    trans_x = Float(0, label=u"ΔX")
    trans_y = Float(0, label=u"ΔY")
    trans_z = Float(0, label=u"ΔZ")
    rot_x = Float(0, label=u"∠X")
    rot_y = Float(0, label=u"∠Y")
    rot_z = Float(0, label=u"∠Z")
    parameters = List()
    last_parameters = List()
    lpa_weight = Float(1.)
    nasion_weight = Float(10.)
    rpa_weight = Float(1.)
    hsp_weight = Float(1.)
    eeg_weight = Float(1.)
    hpi_weight = Float(1.)
    iteration = Int(-1)
    icp_iterations = Int(20)
    icp_start_time = Float(0.0)
    icp_angle = Float(0.2)
    icp_distance = Float(0.2)
    icp_scale = Float(0.2)
    icp_fid_match = Enum('nearest', 'matched')
    fit_icp_running = Bool(False)
    fits_icp_running = Bool(False)
    coord_frame = Enum('mri', 'head', desc='Display coordinate frame')
    status_text = Str()

    # options during scaling
    scale_labels = Bool(True, desc="whether to scale *.label files")
    copy_annot = Bool(True, desc="whether to copy *.annot files for scaled "
                      "subject")
    prepare_bem_model = Bool(True, desc="whether to run make_bem_solution "
                             "after scaling the MRI")

    # secondary to parameters
    has_nasion_data = Property(
        Bool, depends_on=['mri:nasion', 'hsp:nasion'])
    has_lpa_data = Property(
        Bool, depends_on=['mri:lpa', 'hsp:lpa'])
    has_rpa_data = Property(
        Bool, depends_on=['mri:rpa', 'hsp:rpa'])
    has_fid_data = Property(  # conjunction
        Bool, depends_on=['has_nasion_data', 'has_lpa_data', 'has_rpa_data'])
    has_mri_data = Property(
        Bool, depends_on=['transformed_high_res_mri_points'])
    has_hsp_data = Property(
        Bool, depends_on=['has_mri_data', 'hsp:points'])
    has_eeg_data = Property(
        Bool, depends_on=['has_mri_data', 'hsp:eeg_points'])
    has_hpi_data = Property(
        Bool, depends_on=['has_mri_data', 'hsp:hpi_points'])
    n_icp_points = Property(
        Int, depends_on=['has_nasion_data', 'nasion_weight',
                         'has_lpa_data', 'lpa_weight',
                         'has_rpa_data', 'rpa_weight',
                         'hsp:points', 'hsp_weight',
                         'hsp:eeg_points', 'eeg_weight',
                         'hsp:hpi_points', 'hpi_weight'])
    changes = Property(depends_on=['parameters', 'old_parameters'])

    # target transforms
    mri_head_t = Property(
        desc="Transformation of the scaled MRI to the head coordinate frame.",
        depends_on=['parameters[]'])
    head_mri_t = Property(depends_on=['mri_head_t'])
    mri_trans_noscale = Property(depends_on=['mri_head_t', 'coord_frame'])
    mri_trans = Property(depends_on=['mri_trans_noscale', 'parameters[]'])
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

    # Always computed in the MRI coordinate frame for speed
    # (building the nearest-neighbor tree is slow!)
    # though it will always need to be rebuilt in (non-uniform) scaling mode
    nearest_calc = Instance(_DistanceQuery)

    # MRI geometry transformed to viewing coordinate system
    processed_high_res_mri_points = Property(
        depends_on=['mri:bem_high_res:surf', 'grow_hair'])
    processed_low_res_mri_points = Property(
        depends_on=['mri:bem_low_res:surf', 'grow_hair'])
    transformed_high_res_mri_points = Property(
        depends_on=['processed_high_res_mri_points', 'mri_trans'])
    transformed_low_res_mri_points = Property(
        depends_on=['processed_low_res_mri_points', 'mri_trans'])
    nearest_transformed_high_res_mri_idx_lpa = Property(
        depends_on=['nearest_calc', 'hsp:lpa', 'head_mri_t'])
    nearest_transformed_high_res_mri_idx_nasion = Property(
        depends_on=['nearest_calc', 'hsp:nasion', 'head_mri_t'])
    nearest_transformed_high_res_mri_idx_rpa = Property(
        depends_on=['nearest_calc', 'hsp:rpa', 'head_mri_t'])
    nearest_transformed_high_res_mri_idx_hsp = Property(
        depends_on=['nearest_calc', 'hsp:points', 'head_mri_t'])
    nearest_transformed_high_res_mri_idx_orig_hsp = Property(
        depends_on=['nearest_calc', 'hsp:points', 'head_mri_t'])
    nearest_transformed_high_res_mri_idx_eeg = Property(
        depends_on=['nearest_calc', 'hsp:eeg_points', 'head_mri_t'])
    nearest_transformed_high_res_mri_idx_hpi = Property(
        depends_on=['nearest_calc', 'hsp:hpi_points', 'head_mri_t'])
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
        depends_on=['hsp:hpi_points', 'hsp_trans'])

    # fit properties
    lpa_distance = Property(
        depends_on=['transformed_mri_lpa', 'transformed_hsp_lpa'])
    nasion_distance = Property(
        depends_on=['transformed_mri_nasion', 'transformed_hsp_nasion'])
    rpa_distance = Property(
        depends_on=['transformed_mri_rpa', 'transformed_hsp_rpa'])
    point_distance = Property(  # use low res points
        depends_on=['nearest_transformed_high_res_mri_idx_hsp',
                    'nearest_transformed_high_res_mri_idx_eeg',
                    'nearest_transformed_high_res_mri_idx_hpi',
                    'hsp_weight',
                    'eeg_weight',
                    'hpi_weight'])
    orig_hsp_point_distance = Property(  # use low res points
        depends_on=['nearest_transformed_high_res_mri_idx_orig_hsp',
                    'hpi_weight'])

    # fit property info strings
    fid_eval_str = Property(
        depends_on=['lpa_distance', 'nasion_distance', 'rpa_distance'])
    points_eval_str = Property(
        depends_on=['point_distance'])

    def _parameters_default(self):
        return list(_DEFAULT_PARAMETERS)

    def _last_parameters_default(self):
        return list(_DEFAULT_PARAMETERS)

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
    def _get_has_fid_data(self):
        return self.has_nasion_data and self.has_lpa_data and self.has_rpa_data

    @cached_property
    def _get_has_mri_data(self):
        return len(self.transformed_high_res_mri_points) > 0

    @cached_property
    def _get_has_hsp_data(self):
        return (self.has_mri_data and
                len(self.nearest_transformed_high_res_mri_idx_hsp) > 0)

    @cached_property
    def _get_has_eeg_data(self):
        return (self.has_mri_data and
                len(self.nearest_transformed_high_res_mri_idx_eeg) > 0)

    @cached_property
    def _get_has_hpi_data(self):
        return (self.has_mri_data and
                len(self.nearest_transformed_high_res_mri_idx_hpi) > 0)

    @cached_property
    def _get_n_icp_points(self):
        """Get parameters for an ICP iteration."""
        n = (self.hsp_weight > 0) * len(self.hsp.points)
        for key in ('lpa', 'nasion', 'rpa'):
            if getattr(self, 'has_%s_data' % key):
                n += 1
        n += (self.eeg_weight > 0) * len(self.hsp.eeg_points)
        n += (self.hpi_weight > 0) * len(self.hsp.hpi_points)
        return n

    @cached_property
    def _get_changes(self):
        new = np.array(self.parameters, float)
        old = np.array(self.last_parameters, float)
        move = np.linalg.norm(old[3:6] - new[3:6]) * 1e3
        angle = np.rad2deg(_angle_between_quats(
            rot_to_quat(rotation(*new[:3])[:3, :3]),
            rot_to_quat(rotation(*old[:3])[:3, :3])))
        percs = 100 * (new[6:] - old[6:]) / old[6:]
        return move, angle, percs

    @cached_property
    def _get_mri_head_t(self):
        # rotate and translate hsp
        trans = rotation(*self.parameters[:3])
        trans[:3, 3] = np.array(self.parameters[3:6])
        return trans

    @cached_property
    def _get_head_mri_t(self):
        trans = rotation(*self.parameters[:3]).T
        trans[:3, 3] = -np.dot(trans[:3, :3], self.parameters[3:6])
        # should be the same as np.linalg.inv(self.mri_head_t)
        return trans

    @cached_property
    def _get_processed_high_res_mri_points(self):
        return self._get_processed_mri_points('high')

    @cached_property
    def _get_processed_low_res_mri_points(self):
        return self._get_processed_mri_points('low')

    def _get_processed_mri_points(self, res):
        bem = self.mri.bem_low_res if res == 'low' else self.mri.bem_high_res
        if self.grow_hair:
            if len(bem.surf.nn):
                scaled_hair_dist = (1e-3 * self.grow_hair /
                                    np.array(self.parameters[6:9]))
                points = bem.surf.rr.copy()
                hair = points[:, 2] > points[:, 1]
                points[hair] += bem.surf.nn[hair] * scaled_hair_dist
                return points
            else:
                error(None, "Norms missing from bem, can't grow hair")
                self.grow_hair = 0
        else:
            return bem.surf.rr

    @cached_property
    def _get_mri_trans(self):
        t = self.mri_trans_noscale.copy()
        t[:, :3] *= self.parameters[6:9]
        return t

    @cached_property
    def _get_mri_trans_noscale(self):
        if self.coord_frame == 'head':
            t = self.mri_head_t
        else:
            t = np.eye(4)
        return t

    @cached_property
    def _get_hsp_trans(self):
        if self.coord_frame == 'head':
            t = np.eye(4)
        else:
            t = self.head_mri_t
        return t

    @cached_property
    def _get_nearest_transformed_high_res_mri_idx_lpa(self):
        return self.nearest_calc.query(
            apply_trans(self.head_mri_t, self.hsp.lpa))[1]

    @cached_property
    def _get_nearest_transformed_high_res_mri_idx_nasion(self):
        return self.nearest_calc.query(
            apply_trans(self.head_mri_t, self.hsp.nasion))[1]

    @cached_property
    def _get_nearest_transformed_high_res_mri_idx_rpa(self):
        return self.nearest_calc.query(
            apply_trans(self.head_mri_t, self.hsp.rpa))[1]

    @cached_property
    def _get_nearest_transformed_high_res_mri_idx_hsp(self):
        return self.nearest_calc.query(
            apply_trans(self.head_mri_t, self.hsp.points))[1]

    @cached_property
    def _get_nearest_transformed_high_res_mri_idx_orig_hsp(self):
        # This is redundant to some extent with the one above due to
        # overlapping points, but it's fast and the refactoring to
        # remove redundancy would be a pain.
        return self.nearest_calc.query(
            apply_trans(self.head_mri_t, self.hsp._hsp_points))[1]

    @cached_property
    def _get_nearest_transformed_high_res_mri_idx_eeg(self):
        return self.nearest_calc.query(
            apply_trans(self.head_mri_t, self.hsp.eeg_points))[1]

    @cached_property
    def _get_nearest_transformed_high_res_mri_idx_hpi(self):
        return self.nearest_calc.query(
            apply_trans(self.head_mri_t, self.hsp.hpi_points))[1]

    # MRI view-transformed data
    @cached_property
    def _get_transformed_low_res_mri_points(self):
        points = apply_trans(self.mri_trans,
                             self.processed_low_res_mri_points)
        return points

    def _nearest_calc_default(self):
        return _DistanceQuery(
            self.processed_high_res_mri_points * self.parameters[6:9])

    @on_trait_change('processed_high_res_mri_points')
    def _update_nearest_calc(self):
        self.nearest_calc = self._nearest_calc_default()

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
        return np.linalg.norm(d)

    @cached_property
    def _get_nasion_distance(self):
        d = np.ravel(self.transformed_mri_nasion - self.transformed_hsp_nasion)
        return np.linalg.norm(d)

    @cached_property
    def _get_rpa_distance(self):
        d = np.ravel(self.transformed_mri_rpa - self.transformed_hsp_rpa)
        return np.linalg.norm(d)

    @cached_property
    def _get_point_distance(self):
        mri_points = list()
        hsp_points = list()
        if self.hsp_weight > 0 and self.has_hsp_data:
            mri_points.append(self.transformed_high_res_mri_points[
                self.nearest_transformed_high_res_mri_idx_hsp])
            hsp_points.append(self.transformed_hsp_points)
            assert len(mri_points[-1]) == len(hsp_points[-1])
        if self.eeg_weight > 0 and self.has_eeg_data:
            mri_points.append(self.transformed_high_res_mri_points[
                self.nearest_transformed_high_res_mri_idx_eeg])
            hsp_points.append(self.transformed_hsp_eeg_points)
            assert len(mri_points[-1]) == len(hsp_points[-1])
        if self.hpi_weight > 0 and self.has_hpi_data:
            mri_points.append(self.transformed_high_res_mri_points[
                self.nearest_transformed_high_res_mri_idx_hpi])
            hsp_points.append(self.transformed_hsp_hpi)
            assert len(mri_points[-1]) == len(hsp_points[-1])
        if all(len(h) == 0 for h in hsp_points):
            return None
        mri_points = np.concatenate(mri_points)
        hsp_points = np.concatenate(hsp_points)
        return np.linalg.norm(mri_points - hsp_points, axis=-1)

    @cached_property
    def _get_orig_hsp_point_distance(self):
        mri_points = self.transformed_high_res_mri_points[
            self.nearest_transformed_high_res_mri_idx_orig_hsp]
        hsp_points = self.transformed_orig_hsp_points
        return np.linalg.norm(mri_points - hsp_points, axis=-1)

    @cached_property
    def _get_fid_eval_str(self):
        d = (self.lpa_distance * 1000, self.nasion_distance * 1000,
             self.rpa_distance * 1000)
        return u'Fiducials: %.1f, %.1f, %.1f mm' % d

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

    def fit_fiducials(self, n_scale_params=None):
        """Find rotation and translation to fit all 3 fiducials."""
        if n_scale_params is None:
            n_scale_params = self.n_scale_params
        head_pts = np.vstack((self.hsp.lpa, self.hsp.nasion, self.hsp.rpa))
        mri_pts = np.vstack((self.mri.lpa, self.mri.nasion, self.mri.rpa))
        weights = [self.lpa_weight, self.nasion_weight, self.rpa_weight]
        assert n_scale_params in (0, 1)  # guaranteed by GUI
        if n_scale_params == 0:
            mri_pts *= self.parameters[6:9]  # not done in fit_matched_points
        x0 = np.array(self.parameters[:6 + n_scale_params])
        est = fit_matched_points(mri_pts, head_pts, x0=x0, out='params',
                                 scale=n_scale_params, weights=weights)
        if n_scale_params == 0:
            self.parameters[:6] = est
        else:
            self.parameters[:] = np.concatenate([est, [est[-1]] * 2])

    def _setup_icp(self, n_scale_params):
        """Get parameters for an ICP iteration."""
        head_pts = list()
        mri_pts = list()
        weights = list()
        if self.has_hsp_data and self.hsp_weight > 0:  # should be true
            head_pts.append(self.hsp.points)
            mri_pts.append(self.processed_high_res_mri_points[
                self.nearest_transformed_high_res_mri_idx_hsp])
            weights.append(np.full(len(head_pts[-1]), self.hsp_weight))
        for key in ('lpa', 'nasion', 'rpa'):
            if getattr(self, 'has_%s_data' % key):
                head_pts.append(getattr(self.hsp, key))
                if self.icp_fid_match == 'matched':
                    mri_pts.append(getattr(self.mri, key))
                else:
                    assert self.icp_fid_match == 'nearest'
                    mri_pts.append(self.processed_high_res_mri_points[
                        getattr(self, 'nearest_transformed_high_res_mri_idx_%s'
                                % (key,))])
                weights.append(np.full(len(mri_pts[-1]),
                                       getattr(self, '%s_weight' % key)))
        if self.has_eeg_data and self.eeg_weight > 0:
            head_pts.append(self.hsp.eeg_points)
            mri_pts.append(self.processed_high_res_mri_points[
                self.nearest_transformed_high_res_mri_idx_eeg])
            weights.append(np.full(len(mri_pts[-1]), self.eeg_weight))
        if self.has_hpi_data and self.hpi_weight > 0:
            head_pts.append(self.hsp.hpi_points)
            mri_pts.append(self.processed_high_res_mri_points[
                self.nearest_transformed_high_res_mri_idx_hpi])
            weights.append(np.full(len(mri_pts[-1]), self.hpi_weight))
        head_pts = np.concatenate(head_pts)
        mri_pts = np.concatenate(mri_pts)
        weights = np.concatenate(weights)
        if n_scale_params == 0:
            mri_pts *= self.parameters[6:9]  # not done in fit_matched_points
        return head_pts, mri_pts, weights

    def fit_icp(self, n_scale_params=None):
        """Find MRI scaling, translation, and rotation to match HSP."""
        if n_scale_params is None:
            n_scale_params = self.n_scale_params

        # Initial guess (current state)
        assert n_scale_params in (0, 1, 3)
        est = self.parameters[:[6, 7, None, 9][n_scale_params]]

        # Do the fits, assigning and evaluating at each step
        attr = 'fit_icp_running' if n_scale_params == 0 else 'fits_icp_running'
        setattr(self, attr, True)
        GUI.process_events()  # update the cancel button
        self.icp_start_time = time.time()
        for self.iteration in range(self.icp_iterations):
            head_pts, mri_pts, weights = self._setup_icp(n_scale_params)
            est = fit_matched_points(mri_pts, head_pts, scale=n_scale_params,
                                     x0=est, out='params', weights=weights)
            if n_scale_params == 0:
                self.parameters[:6] = est
            elif n_scale_params == 1:
                self.parameters[:] = list(est) + [est[-1]] * 2
            else:
                self.parameters[:] = est
            angle, move, scale = self.changes
            if angle <= self.icp_angle and move <= self.icp_distance and \
                    all(scale <= self.icp_scale):
                self.status_text = self.status_text[:-1] + '; converged)'
                break
            if not getattr(self, attr):  # canceled by user
                self.status_text = self.status_text[:-1] + '; cancelled)'
                break
            GUI.process_events()  # this will update the head view
        else:
            self.status_text = self.status_text[:-1] + '; did not converge)'
        setattr(self, attr, False)
        self.iteration = -1

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

        return (subjects_dir, subject_from, subject_to, self.parameters[6:9],
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
        with busy():
            self.reset_traits(('grow_hair', 'n_scaling_params'))
            self.parameters[:] = _DEFAULT_PARAMETERS
            self.omit_hsp_points(np.inf)

    def set_trans(self, mri_head_t):
        """Set rotation and translation params from a transformation matrix.

        Parameters
        ----------
        mri_head_t : array, shape (4, 4)
            Transformation matrix from MRI to head space.
        """
        with busy():
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
        write_trans(fname, Transform('head', 'mri', self.head_mri_t))

    def _parameters_items_changed(self):
        # Update GUI as necessary
        n_scale = self.n_scale_params
        for ii, key in enumerate(('rot_x', 'rot_y', 'rot_z')):
            val = np.rad2deg(self.parameters[ii])
            if val != getattr(self, key):  # prevent circular
                setattr(self, key, val)
        for ii, key in enumerate(('trans_x', 'trans_y', 'trans_z')):
            val = self.parameters[ii + 3] * 1e3
            if val != getattr(self, key):  # prevent circular
                setattr(self, key, val)
        for ii, key in enumerate(('scale_x', 'scale_y', 'scale_z')):
            val = self.parameters[ii + 6] * 1e2
            if val != getattr(self, key):  # prevent circular
                setattr(self, key, val)
        # Only update our nearest-neighbor if necessary
        if self.parameters[6:9] != self.last_parameters[6:9]:
            self._update_nearest_calc()
        # Update the status text
        move, angle, percs = self.changes
        text = u'Change:  Δ=%0.1f mm  ∠=%0.2f°' % (move, angle)
        if n_scale:
            text += '  Scale ' if n_scale == 1 else '  Sx/y/z '
            text += '/'.join(['%+0.1f%%' % p for p in percs[:n_scale]])
        if self.iteration >= 0:
            text += u' (iteration %d/%d, %0.1f sec)' % (
                self.iteration + 1, self.icp_iterations,
                time.time() - self.icp_start_time)
        self.last_parameters[:] = self.parameters[:]
        self.status_text = text

    def _rot_x_changed(self):
        self.parameters[0] = np.deg2rad(self.rot_x)

    def _rot_y_changed(self):
        self.parameters[1] = np.deg2rad(self.rot_y)

    def _rot_z_changed(self):
        self.parameters[2] = np.deg2rad(self.rot_z)

    def _trans_x_changed(self):
        self.parameters[3] = self.trans_x * 1e-3

    def _trans_y_changed(self):
        self.parameters[4] = self.trans_y * 1e-3

    def _trans_z_changed(self):
        self.parameters[5] = self.trans_z * 1e-3

    def _scale_x_changed(self):
        if self.n_scale_params == 1:
            self.parameters[6:9] = [self.scale_x * 1e-2] * 3
        else:
            self.parameters[6] = self.scale_x * 1e-2

    def _scale_y_changed(self):
        self.parameters[7] = self.scale_y * 1e-2

    def _scale_z_changed(self):
        self.parameters[8] = self.scale_z * 1e-2


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
            try:  # works on Qt only for now
                size = (info.ui.control.width(), info.ui.control.height())
            except AttributeError:
                size = None
            # store configuration, but don't prevent from closing on error
            try:
                info.object.save_config(size=size)
            except Exception as exc:
                warnings.warn("Error saving GUI configuration:\n%s" % (exc,))
            return True


class CoregPanelHandler(Handler):
    """Open other windows with proper parenting."""

    info = Instance(UIInfo)

    def object_fitting_options_panel_changed(self, info):  # noqa: D102
        self.info = info

    def object_fitting_options_changed(self, info):  # noqa: D102
        self.info.object.fitting_options_panel.edit_traits(
            parent=self.info.ui.control)

    def object_load_trans_changed(self, info):  # noqa: D102
        # find trans file destination
        model = self.info.object.model
        raw_dir = os.path.dirname(model.hsp.file)
        subject = model.mri.subject
        trans_file = trans_fname.format(raw_dir=raw_dir, subject=subject)
        dlg = FileDialog(action="open", wildcard=trans_wildcard,
                         default_path=trans_file, parent=self.info.ui.control)
        if dlg.open() != OK:
            return
        trans_file = dlg.path
        try:
            model.load_trans(trans_file)
        except Exception as e:
            error(None, "Error loading trans file %s: %s (See terminal "
                  "for details)" % (trans_file, e), "Error Loading Trans File")
            raise

    def object_save_changed(self, info):  # noqa: D102
        obj = self.info.object
        subjects_dir = obj.model.mri.subjects_dir
        subject_from = obj.model.mri.subject

        # check that fiducials are saved
        skip_fiducials = False
        if obj.n_scale_params and not _find_fiducials_files(subject_from,
                                                            subjects_dir):
            msg = ("No fiducials file has been found for {src}. If fiducials "
                   "are not saved, they will not be available in the scaled "
                   "MRI. Should the current fiducials be saved now? "
                   "Select Yes to save the fiducials at "
                   "{src}/bem/{src}-fiducials.fif. "
                   "Select No to proceed scaling the MRI without fiducials.".
                   format(src=subject_from))
            title = "Save Fiducials for %s?" % subject_from
            rc = confirm(self.info.ui.control, msg, title, cancel=True,
                         default=CANCEL)
            if rc == CANCEL:
                return
            elif rc == YES:
                obj.model.mri.save(obj.model.mri.default_fid_fname)
            elif rc == NO:
                skip_fiducials = True
            else:
                raise RuntimeError("rc=%s" % repr(rc))

        # find target subject
        if obj.n_scale_params:
            subject_to = obj.model.raw_subject or subject_from
            mridlg = NewMriDialog(subjects_dir=subjects_dir,
                                  subject_from=subject_from,
                                  subject_to=subject_to)
            ui = mridlg.edit_traits(kind='modal',
                                    parent=self.info.ui.control)
            if not ui.result:  # i.e., user pressed cancel
                return
            subject_to = mridlg.subject_to
        else:
            subject_to = subject_from

        # find trans file destination
        raw_dir = os.path.dirname(obj.model.hsp.file)
        trans_file = trans_fname.format(raw_dir=raw_dir, subject=subject_to)
        dlg = FileDialog(action="save as", wildcard=trans_wildcard,
                         default_path=trans_file,
                         parent=self.info.ui.control)
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
            obj.model.save_trans(trans_file)
        except Exception as e:
            error(None, "Error saving -trans.fif file: %s (See terminal for "
                  "details)" % (e,), "Error Saving Trans File")
            raise

        # save the scaled MRI
        if obj.n_scale_params:
            job = obj.model.get_scaling_job(subject_to, skip_fiducials)
            obj.queue.put(job)
            obj.queue_len += 1


def _make_view_data_panel(scrollable=False):
    view = View(VGroup(
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
               Item('fid_panel', style='custom'), label="MRI Fiducials",
               show_border=_SHOW_BORDER, show_labels=False),
        VGroup(Item('raw_src', style="custom"),
               HGroup('guess_mri_subject',
                      Label('Guess subject from name'), show_labels=False),
               VGrid(Item('grow_hair', editor=laggy_float_editor_mm,
                          width=_MM_WIDTH),
                     Label(u'ΔHair', show_label=True, width=-1), '0',
                     Item('distance', show_label=False, width=_MM_WIDTH,
                          editor=laggy_float_editor_mm),
                     Item('omit_points', width=_BUTTON_WIDTH),
                     Item('reset_omit_points', width=_RESET_WIDTH),
                     columns=3, show_labels=False),
               Item('omitted_info', style='readonly',
                    width=_REDUCED_TEXT_WIDTH), label='Digitization source',
               show_border=_SHOW_BORDER, show_labels=False),
        VGroup(HGroup(Item('headview', style='custom'), Spring(),
                      show_labels=False),
               Item('view_options', width=_REDUCED_TEXT_WIDTH),
               label='View', show_border=_SHOW_BORDER, show_labels=False),
        Spring(),
        show_labels=False), kind='panel', buttons=[UndoButton],
        scrollable=scrollable, handler=DataPanelHandler())
    return view


def _make_view_coreg_panel(scrollable=False):
    """Generate View for CoregPanel."""
    view = View(VGroup(
        # Scaling
        HGroup(Item('n_scale_params', label='Scaling mode',
                    editor=EnumEditor(values={0: '1:None',
                                              1: '2:Uniform',
                                              3: '3:3-axis'})), Spring()),
        VGrid(Item('scale_x', editor=laggy_float_editor_scale,
                   show_label=True, tooltip="Scale along right-left axis (%)",
                   enabled_when='n_scale_params > 0', width=_SCALE_WIDTH),
              Item('scale_x_dec', enabled_when='n_scale_params > 0',
                   width=_INC_BUTTON_WIDTH),
              Item('scale_x_inc', enabled_when='n_scale_params > 0',
                   width=_INC_BUTTON_WIDTH),
              Item('scale_step', tooltip="Scaling step (%)",
                   enabled_when='n_scale_params > 0', width=_SCALE_STEP_WIDTH),
              Spring(),

              Item('scale_y', editor=laggy_float_editor_scale, show_label=True,
                   enabled_when='n_scale_params > 1',
                   tooltip="Scale along anterior-posterior axis (%)",
                   width=_SCALE_WIDTH),
              Item('scale_y_dec', enabled_when='n_scale_params > 1',
                   width=_INC_BUTTON_WIDTH),
              Item('scale_y_inc', enabled_when='n_scale_params > 1',
                   width=_INC_BUTTON_WIDTH),
              Label('(Step)', width=_SCALE_WIDTH),
              Spring(),

              Item('scale_z', editor=laggy_float_editor_scale, show_label=True,
                   enabled_when='n_scale_params > 1', width=_SCALE_WIDTH,
                   tooltip="Scale along anterior-posterior axis (%)"),
              Item('scale_z_dec', enabled_when='n_scale_params > 1',
                   width=_INC_BUTTON_WIDTH),
              Item('scale_z_inc', enabled_when='n_scale_params > 1',
                   width=_INC_BUTTON_WIDTH),
              '0',
              Spring(),

              label='Scaling parameters', show_labels=False, columns=5,
              show_border=_SHOW_BORDER),
        VGrid(Item('fits_icp', enabled_when='n_scale_params > 0 and '
                   'n_icp_points >= 10',
                   tooltip="Rotate, translate, and scale the MRI to minimize "
                   "the distance from each digitizer point to the closest MRI "
                   "point (one ICP iteration)", width=_BUTTON_WIDTH),
              Item('fits_fid', enabled_when='n_scale_params == 1 and '
                   'has_fid_data',
                   tooltip="Rotate, translate, and scale the MRI to minimize "
                   "the distance of the three fiducials.",
                   width=_BUTTON_WIDTH),
              Item('cancels_icp', enabled_when="fits_icp_running",
                   tooltip='Stop ICP fitting', width=_RESET_WIDTH),
              Item('reset_scale', enabled_when='n_scale_params',
                   tooltip="Reset scaling parameters", width=_RESET_WIDTH),
              show_labels=False, columns=4),
        # Translation and rotation
        VGrid(Item('trans_x', editor=laggy_float_editor_mm, show_label=True,
                   tooltip="Move along right-left axis", width=_MM_WIDTH),
              Item('trans_x_dec', width=_INC_BUTTON_WIDTH),
              Item('trans_x_inc', width=_INC_BUTTON_WIDTH),
              Item('trans_step', tooltip="Movement step (mm)",
                   width=_MM_STEP_WIDTH),
              Spring(),

              Item('trans_y', editor=laggy_float_editor_mm, show_label=True,
                   tooltip="Move along anterior-posterior axis",
                   width=_MM_WIDTH),
              Item('trans_y_dec', width=_INC_BUTTON_WIDTH),
              Item('trans_y_inc', width=_INC_BUTTON_WIDTH),
              Label('(Step)', width=_MM_WIDTH),
              Spring(),

              Item('trans_z', editor=laggy_float_editor_mm, show_label=True,
                   tooltip="Move along anterior-posterior axis",
                   width=_MM_WIDTH),
              Item('trans_z_dec', width=_INC_BUTTON_WIDTH),
              Item('trans_z_inc', width=_INC_BUTTON_WIDTH),
              '0',
              Spring(),

              Item('rot_x', editor=laggy_float_editor_deg, show_label=True,
                   tooltip="Tilt the digitization backward (-) or forward (+)",
                   width=_DEG_WIDTH),
              Item('rot_x_dec', width=_INC_BUTTON_WIDTH),
              Item('rot_x_inc', width=_INC_BUTTON_WIDTH),
              Item('rot_step', tooltip=u"Rotation step (°)",
                   width=_DEG_STEP_WIDTH),
              Spring(),

              Item('rot_y', editor=laggy_float_editor_deg, show_label=True,
                   tooltip="Tilt the digitization rightward (-) or "
                   "leftward (+)", width=_DEG_WIDTH),
              Item('rot_y_dec', width=_INC_BUTTON_WIDTH),
              Item('rot_y_inc', width=_INC_BUTTON_WIDTH),
              Label('(Step)', width=_DEG_WIDTH),
              Spring(),

              Item('rot_z', editor=laggy_float_editor_deg, show_label=True,
                   tooltip="Turn the digitization leftward (-) or "
                   "rightward (+)", width=_DEG_WIDTH),
              Item('rot_z_dec', width=_INC_BUTTON_WIDTH),
              Item('rot_z_inc', width=_INC_BUTTON_WIDTH),
              '0',
              Spring(),

              columns=5, show_labels=False, show_border=_SHOW_BORDER,
              label=u'Translation (Δ) and Rotation (∠)'),
        VGroup(Item('fit_icp', enabled_when='n_icp_points >= 10',
                    tooltip="Rotate and translate the MRI to minimize the "
                    "distance from each digitizer point to the closest MRI "
                    "point (one ICP iteration)", width=_BUTTON_WIDTH),
               Item('fit_fid', enabled_when="has_fid_data",
                    tooltip="Rotate and translate the MRI to minimize the "
                    "distance of the three fiducials.", width=_BUTTON_WIDTH),
               Item('cancel_icp', enabled_when="fit_icp_running",
                    tooltip='Stop ICP iterations', width=_RESET_WIDTH),
               Item('reset_tr', tooltip="Reset translation and rotation.",
                    width=_RESET_WIDTH),
               show_labels=False, columns=4),
        # Fitting weights
        Item('fid_eval_str', style='readonly', tooltip='Fiducial differences',
             width=_REDUCED_TEXT_WIDTH),
        Item('points_eval_str', style='readonly',
             tooltip='Point error (μ ± σ)', width=_REDUCED_TEXT_WIDTH),
        Item('fitting_options', width=_REDUCED_TEXT_WIDTH, show_label=False),
        VGrid(Item('scale_labels', label="Scale label files",
                   enabled_when='n_scale_params > 0'),
              Item('copy_annot', label="Copy annotation files",
                   enabled_when='n_scale_params > 0'),
              Item('prepare_bem_model', label="Prepare BEM",
                   enabled_when='can_prepare_bem_model'),
              show_left=False, label='Subject-saving options', columns=1,
              show_border=_SHOW_BORDER),
        VGrid(Item('save', enabled_when='can_save',
                   tooltip="Save the trans file and (if scaling is enabled) "
                   "the scaled MRI", width=_BUTTON_WIDTH),
              Item('load_trans', width=_BUTTON_WIDTH,
                   tooltip="Load Head<->MRI trans file"),
              Item('reset_params', tooltip="Reset all coregistration "
                   "parameters", width=_RESET_WIDTH),
              show_labels=False, columns=3),
        Spring(),
        show_labels=False), kind='panel', buttons=[UndoButton],
        scrollable=scrollable, handler=CoregPanelHandler())
    return view


class FittingOptionsPanel(HasTraits):
    """View options panel."""

    model = Instance(CoregModel)
    lpa_weight = DelegatesTo('model')
    nasion_weight = DelegatesTo('model')
    rpa_weight = DelegatesTo('model')
    hsp_weight = DelegatesTo('model')
    eeg_weight = DelegatesTo('model')
    hpi_weight = DelegatesTo('model')
    has_lpa_data = DelegatesTo('model')
    has_nasion_data = DelegatesTo('model')
    has_rpa_data = DelegatesTo('model')
    has_hsp_data = DelegatesTo('model')
    has_eeg_data = DelegatesTo('model')
    has_hpi_data = DelegatesTo('model')
    icp_iterations = DelegatesTo('model')
    icp_start_time = DelegatesTo('model')
    icp_angle = DelegatesTo('model')
    icp_distance = DelegatesTo('model')
    icp_scale = DelegatesTo('model')
    icp_fid_match = DelegatesTo('model')
    n_scale_params = DelegatesTo('model')

    view = View(VGroup(
        VGrid(HGroup(Item('icp_iterations', label='Iterations',
                          width=_MM_WIDTH, tooltip='Maximum ICP iterations to '
                          'perform (per click)'),
                     Spring(), show_labels=True), label='ICP iterations (max)',
              show_border=_SHOW_BORDER),
        VGrid(Item('icp_angle', label=u'Angle (°)', width=_MM_WIDTH,
                   tooltip='Angle convergence threshold'),
              Item('icp_distance', label='Distance (mm)', width=_MM_WIDTH,
                   tooltip='Distance convergence threshold'),
              Item('icp_scale', label='Scale (%)',
                   tooltip='Scaling convergence threshold', width=_MM_WIDTH,
                   enabled_when='n_scale_params > 0'),
              show_labels=True, label='ICP convergence limits', columns=3,
              show_border=_SHOW_BORDER),
        VGrid(Item('icp_fid_match', width=-1, show_label=False,
                   editor=EnumEditor(values=dict(
                       nearest='1:Closest to surface',
                       matched='2:MRI fiducials'), cols=2,
                       format_func=lambda x: x),
                   tooltip='Match digitization fiducials to MRI fiducials or '
                   'the closest surface point', style='custom'),
              label='Fiducial point matching', show_border=_SHOW_BORDER),
        VGrid(
            VGrid(Item('lpa_weight', editor=laggy_float_editor_weight,
                       tooltip="Relative weight for LPA", width=_WEIGHT_WIDTH,
                       enabled_when='has_lpa_data', label='LPA'),
                  Item('nasion_weight', editor=laggy_float_editor_weight,
                       tooltip="Relative weight for nasion", label='Nasion',
                       width=_WEIGHT_WIDTH, enabled_when='has_nasion_data'),
                  Item('rpa_weight', editor=laggy_float_editor_weight,
                       tooltip="Relative weight for RPA", width=_WEIGHT_WIDTH,
                       enabled_when='has_rpa_data', label='RPA'),
                  columns=3, show_labels=True, show_border=_SHOW_BORDER,
                  label='Fiducials'),
            VGrid(Item('hsp_weight', editor=laggy_float_editor_weight,
                       tooltip="Relative weight for head shape points",
                       enabled_when='has_hsp_data',
                       label='HSP', width=_WEIGHT_WIDTH,),
                  Item('eeg_weight', editor=laggy_float_editor_weight,
                       tooltip="Relative weight for EEG points", label='EEG',
                       enabled_when='has_eeg_data', width=_WEIGHT_WIDTH),
                  Item('hpi_weight', editor=laggy_float_editor_weight,
                       tooltip="Relative weight for HPI points", label='HPI',
                       enabled_when='has_hpi_data', width=_WEIGHT_WIDTH),
                  columns=3, show_labels=True, show_border=_SHOW_BORDER,
                  label='Other points (closest-point matched)'),
            show_labels=False, label='Point weights', columns=2,
            show_border=_SHOW_BORDER),
    ), title="Fitting options")


_DEFAULT_PARAMETERS = (0., 0., 0., 0., 0., 0., 1., 1., 1.)


class CoregPanel(HasPrivateTraits):
    """Coregistration panel for Head<->MRI with scaling."""

    model = Instance(CoregModel)

    # parameters
    reset_params = Button(label=_RESET_LABEL)
    n_scale_params = DelegatesTo('model')
    parameters = DelegatesTo('model')
    scale_step = Float(1.)
    scale_x = DelegatesTo('model')
    scale_x_dec = Button('-')
    scale_x_inc = Button('+')
    scale_y = DelegatesTo('model')
    scale_y_dec = Button('-')
    scale_y_inc = Button('+')
    scale_z = DelegatesTo('model')
    scale_z_dec = Button('-')
    scale_z_inc = Button('+')
    rot_step = Float(1.)
    rot_x = DelegatesTo('model')
    rot_x_dec = Button('-')
    rot_x_inc = Button('+')
    rot_y = DelegatesTo('model')
    rot_y_dec = Button('-')
    rot_y_inc = Button('+')
    rot_z = DelegatesTo('model')
    rot_z_dec = Button('-')
    rot_z_inc = Button('+')
    trans_step = Float(1.)
    trans_x = DelegatesTo('model')
    trans_x_dec = Button('-')
    trans_x_inc = Button('+')
    trans_y = DelegatesTo('model')
    trans_y_dec = Button('-')
    trans_y_inc = Button('+')
    trans_z = DelegatesTo('model')
    trans_z_dec = Button('-')
    trans_z_inc = Button('+')

    # fitting
    has_lpa_data = DelegatesTo('model')
    has_nasion_data = DelegatesTo('model')
    has_rpa_data = DelegatesTo('model')
    has_fid_data = DelegatesTo('model')
    has_hsp_data = DelegatesTo('model')
    has_eeg_data = DelegatesTo('model')
    has_hpi_data = DelegatesTo('model')
    n_icp_points = DelegatesTo('model')
    # fitting with scaling
    fits_icp = Button(label='Fit (ICP)')
    fits_fid = Button(label='Fit Fid.')
    cancels_icp = Button(u'■')
    reset_scale = Button(label=_RESET_LABEL)
    fits_icp_running = DelegatesTo('model')
    # fitting without scaling
    fit_icp = Button(label='Fit (ICP)')
    fit_fid = Button(label='Fit Fid.')
    cancel_icp = Button(label=u'■')
    reset_tr = Button(label=_RESET_LABEL)
    fit_icp_running = DelegatesTo('model')

    # fit info
    fid_eval_str = DelegatesTo('model')
    points_eval_str = DelegatesTo('model')

    # saving
    can_prepare_bem_model = DelegatesTo('model')
    can_save = DelegatesTo('model')
    scale_labels = DelegatesTo('model')
    copy_annot = DelegatesTo('model')
    prepare_bem_model = DelegatesTo('model')
    save = Button(label="Save...")
    load_trans = Button(label='Load...')
    queue = Instance(queue.Queue, ())
    queue_feedback = Str('')
    queue_current = Str('')
    queue_len = Int(0)
    queue_status_text = Property(
        Str, depends_on=['queue_feedback', 'queue_current', 'queue_len'])

    fitting_options_panel = Instance(FittingOptionsPanel)
    fitting_options = Button('Fitting options...')

    def _fitting_options_panel_default(self):
        return FittingOptionsPanel(model=self.model)

    view = _make_view_coreg_panel()

    def __init__(self, *args, **kwargs):  # noqa: D102
        super(CoregPanel, self).__init__(*args, **kwargs)

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
                    self.queue_feedback = 'Done scaling %s' % subject_to

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
                        self.queue_feedback = ('Done computing %s solution' %
                                               bem_name)

                # Finalize
                self.queue_current = ''
                self.queue.task_done()

        t = Thread(target=worker)
        t.daemon = True
        t.start()

    @cached_property
    def _get_queue_status_text(self):
        items = []
        if self.queue_current:
            items.append(self.queue_current)
        if self.queue_feedback:
            items.append(self.queue_feedback)
        if self.queue_len:
            items.append("%i queued" % self.queue_len)
        return '    |    '.join(items)

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
            use = [np.mean([self.scale_x, self.scale_y, self.scale_z]) /
                   100.] * 3
        else:
            use = self.parameters[6:9]
        self.parameters[6:9] = use

    def _fit_fid_fired(self):
        with busy():
            self.model.fit_fiducials(0)

    def _fit_icp_fired(self):
        with busy():
            self.model.fit_icp(0)

    def _fits_fid_fired(self):
        with busy():
            self.model.fit_fiducials()

    def _fits_icp_fired(self):
        with busy():
            self.model.fit_icp()

    def _cancel_icp_fired(self):
        self.fit_icp_running = False

    def _cancels_icp_fired(self):
        self.fits_icp_running = False

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


def _make_view(tabbed=False, split=False, width=800, height=600,
               scrollable=True):
    """Create a view for the CoregFrame."""
    # Set the width to 0.99 to "push out" as much as possible, use
    # scene_width in the View below
    scene = Item('scene', show_label=False, width=0.99,
                 editor=SceneEditor(scene_class=MayaviScene))

    data_panel = VGroup(
        Item('data_panel', style='custom',
             width=_COREG_WIDTH if scrollable else 1,
             editor=InstanceEditor(view=_make_view_data_panel(scrollable))),
        label='Data', show_border=not scrollable, show_labels=False)

    # Setting `scrollable=True` for a Group does not seem to have any effect
    # (macOS), in order to be effective the parameter has to be set for a View
    # object; hence we use a special InstanceEditor to set the parameter
    # programmatically:
    coreg_panel = VGroup(
        Item('coreg_panel', style='custom',
             width=_COREG_WIDTH if scrollable else 1,
             editor=InstanceEditor(view=_make_view_coreg_panel(scrollable))),
        label="Coregistration", show_border=not scrollable, show_labels=False,
        enabled_when="data_panel.fid_panel.locked")

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
                buttons=NoButtons, width=width, height=height,
                statusbar=[StatusItem('status_text', width=0.55),
                           StatusItem('queue_status_text', width=0.45)])
    return view


class ViewOptionsPanel(HasTraits):
    """View options panel."""

    mri_obj = Instance(SurfaceObject)
    hsp_obj = Instance(PointObject)
    eeg_obj = Instance(PointObject)
    hpi_obj = Instance(PointObject)
    hsp_cf_obj = Instance(PointObject)
    mri_cf_obj = Instance(PointObject)
    bgcolor = RGBColor()
    coord_frame = Enum('mri', 'head', label='Display coordinate frame')
    head_high_res = Bool(True, label='Show high-resolution head')
    advanced_rendering = Bool(True, label='Use advanced OpenGL',
                              desc='Enable advanced OpenGL methods that do '
                              'not work with all renderers (e.g., depth '
                              'peeling)')

    view = View(
        VGroup(
            Item('mri_obj', style='custom', label="MRI"),
            Item('hsp_obj', style='custom', label="Head shape"),
            Item('eeg_obj', style='custom', label='EEG'),
            Item('hpi_obj', style='custom', label='HPI'),
            VGrid(Item('coord_frame', style='custom',
                       editor=EnumEditor(values={'mri': '1:MRI',
                                                 'head': '2:Head'}, cols=2,
                                         format_func=_pass)),
                  Item('head_high_res'), Spring(),
                  Item('advanced_rendering'),
                  Spring(), Spring(), columns=3, show_labels=True),
            Item('hsp_cf_obj', style='custom', label='Head axes'),
            Item('mri_cf_obj', style='custom', label='MRI axes'),
            HGroup(Item('bgcolor', label='Background'), Spring()),
        ), title="Display options")


class DataPanelHandler(Handler):
    """Open other windows with proper parenting."""

    info = Instance(UIInfo)

    def object_view_options_panel_changed(self, info):  # noqa: D102
        self.info = info

    def object_view_options_changed(self, info):  # noqa: D102
        self.info.object.view_options_panel.edit_traits(
            parent=self.info.ui.control)


class DataPanel(HasTraits):
    """Data loading panel."""

    # Set by CoregPanel
    model = Instance(CoregModel)
    scene = Instance(MlabSceneModel, ())
    lock_fiducials = DelegatesTo('model')
    guess_mri_subject = DelegatesTo('model')
    raw_src = DelegatesTo('model', 'hsp')
    # Set internally
    subject_panel = Instance(SubjectSelectorPanel)
    fid_panel = Instance(FiducialsPanel)
    headview = Instance(HeadViewController)
    view_options_panel = Instance(ViewOptionsPanel)
    hsp_always_visible = Bool(False, label="Always Show Head Shape")
    view_options = Button(label="Display options...")

    # Omit Points
    distance = Float(10., desc="maximal distance for head shape points from "
                     "the surface (mm)")
    omit_points = Button(label='Omit', desc="to omit head shape points "
                         "for the purpose of the automatic coregistration "
                         "procedure (mm).")
    grow_hair = DelegatesTo('model')
    reset_omit_points = Button(label=_RESET_LABEL, desc="to reset the "
                               "omission of head shape points to include all.")
    omitted_info = Str('No points omitted')

    def _subject_panel_default(self):
        return SubjectSelectorPanel(model=self.model.mri.subject_source)

    def _fid_panel_default(self):
        return FiducialsPanel(model=self.model.mri, headview=self.headview)

    def _headview_default(self):
        return HeadViewController(system='RAS', scene=self.scene)

    def _omit_points_fired(self):
        distance = self.distance / 1000.
        self.model.omit_hsp_points(distance)
        n_omitted = self.model.hsp.n_omitted
        self.omitted_info = (
            "%s pt%s omitted (%0.1f mm)"
            % (n_omitted if n_omitted > 0 else 'No', _pl(n_omitted),
               self.distance))

    @on_trait_change('model:hsp:file')
    def _file_change(self):
        self._reset_omit_points_fired()

    def _reset_omit_points_fired(self):
        self.model.omit_hsp_points(np.inf)
        self.omitted_info = 'No points omitted (reset)'


class CoregFrame(HasTraits):
    """GUI for head-MRI coregistration."""

    model = Instance(CoregModel)

    scene = Instance(MlabSceneModel, ())
    head_high_res = Bool(True)
    advanced_rendering = Bool(True)

    data_panel = Instance(DataPanel)
    coreg_panel = Instance(CoregPanel)  # right panel

    project_to_surface = DelegatesTo('eeg_obj')
    orient_to_surface = DelegatesTo('hsp_obj')
    scale_by_distance = DelegatesTo('hsp_obj')
    mark_inside = DelegatesTo('hsp_obj')
    status_text = DelegatesTo('model')
    queue_status_text = DelegatesTo('coreg_panel')

    fid_ok = DelegatesTo('model', 'mri.fid_ok')
    lock_fiducials = DelegatesTo('model')
    title = Str('MNE Coreg')

    # visualization (MRI)
    mri_obj = Instance(SurfaceObject)
    mri_lpa_obj = Instance(PointObject)
    mri_nasion_obj = Instance(PointObject)
    mri_rpa_obj = Instance(PointObject)
    bgcolor = RGBColor((0.5, 0.5, 0.5))
    # visualization (Digitization)
    hsp_obj = Instance(PointObject)
    eeg_obj = Instance(PointObject)
    hpi_obj = Instance(PointObject)
    hsp_lpa_obj = Instance(PointObject)
    hsp_nasion_obj = Instance(PointObject)
    hsp_rpa_obj = Instance(PointObject)
    hsp_visible = Property(depends_on=['data_panel:hsp_always_visible',
                                       'lock_fiducials'])
    # Coordinate frame axes
    hsp_cf_obj = Instance(PointObject)
    mri_cf_obj = Instance(PointObject)

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

    def _data_panel_default(self):
        return DataPanel(model=self.model, scene=self.scene)

    def _coreg_panel_default(self):
        return CoregPanel(model=self.model)

    def __init__(self, raw=None, subject=None, subjects_dir=None,
                 guess_mri_subject=True, head_opacity=1.,
                 head_high_res=True, trans=None, config=None,
                 project_eeg=False, orient_to_surface=False,
                 scale_by_distance=False, mark_inside=False,
                 interaction='trackball', scale=0.16,
                 advanced_rendering=True):  # noqa: D102
        self._config = config or {}
        super(CoregFrame, self).__init__(guess_mri_subject=guess_mri_subject,
                                         head_high_res=head_high_res,
                                         advanced_rendering=advanced_rendering)
        self._initial_kwargs = dict(project_eeg=project_eeg,
                                    orient_to_surface=orient_to_surface,
                                    scale_by_distance=scale_by_distance,
                                    mark_inside=mark_inside,
                                    head_opacity=head_opacity,
                                    interaction=interaction,
                                    scale=scale)
        self._locked_opacity = self._initial_kwargs['head_opacity']
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
        self._on_advanced_rendering_change()

        lpa_color = defaults['lpa_color']
        nasion_color = defaults['nasion_color']
        rpa_color = defaults['rpa_color']

        # MRI scalp
        #
        # Due to MESA rendering / z-order bugs, this should be added and
        # rendered first (see gh-5375).
        color = defaults['head_color']
        self.mri_obj = SurfaceObject(
            points=np.empty((0, 3)), color=color, tris=np.empty((0, 3)),
            scene=self.scene, name="MRI Scalp", block_behind=True,
            # opacity=self._initial_kwargs['head_opacity'],
            # setting opacity here causes points to be
            # [[0, 0, 0]] -- why??
        )
        self.mri_obj.opacity = self._initial_kwargs['head_opacity']
        self.data_panel.fid_panel.hsp_obj = self.mri_obj
        self._update_mri_obj()
        self.mri_obj.plot()
        # Do not do sync_trait here, instead use notifiers elsewhere

        # MRI Fiducials
        point_scale = defaults['mri_fid_scale']
        self.mri_lpa_obj = PointObject(scene=self.scene, color=lpa_color,
                                       has_norm=True, point_scale=point_scale,
                                       name='LPA', view='oct')
        self.model.sync_trait('transformed_mri_lpa',
                              self.mri_lpa_obj, 'points', mutual=False)
        self.mri_nasion_obj = PointObject(scene=self.scene, color=nasion_color,
                                          has_norm=True,
                                          point_scale=point_scale,
                                          name='Nasion', view='oct')
        self.model.sync_trait('transformed_mri_nasion',
                              self.mri_nasion_obj, 'points', mutual=False)
        self.mri_rpa_obj = PointObject(scene=self.scene, color=rpa_color,
                                       has_norm=True, point_scale=point_scale,
                                       name='RPA', view='oct')
        self.model.sync_trait('transformed_mri_rpa',
                              self.mri_rpa_obj, 'points', mutual=False)

        # Digitizer Head Shape
        kwargs = dict(
            view='cloud', scene=self.scene, resolution=20,
            orient_to_surface=self._initial_kwargs['orient_to_surface'],
            scale_by_distance=self._initial_kwargs['scale_by_distance'],
            mark_inside=self._initial_kwargs['mark_inside'])
        self.hsp_obj = PointObject(
            color=defaults['extra_color'], name='Extra', has_norm=True,
            point_scale=defaults['extra_scale'], **kwargs)
        self.model.sync_trait('transformed_hsp_points',
                              self.hsp_obj, 'points', mutual=False)

        # Digitizer EEG
        self.eeg_obj = PointObject(
            color=defaults['eeg_color'], point_scale=defaults['eeg_scale'],
            name='EEG', projectable=True, has_norm=True,
            project_to_surface=self._initial_kwargs['project_eeg'], **kwargs)
        self.model.sync_trait('transformed_hsp_eeg_points',
                              self.eeg_obj, 'points', mutual=False)

        # Digitizer HPI
        self.hpi_obj = PointObject(
            color=defaults['hpi_color'], name='HPI', has_norm=True,
            point_scale=defaults['hpi_scale'], **kwargs)
        self.model.sync_trait('transformed_hsp_hpi',
                              self.hpi_obj, 'points', mutual=False)
        for p in (self.hsp_obj, self.eeg_obj, self.hpi_obj):
            p.inside_color = self.mri_obj.color
            self.mri_obj.sync_trait('color', p, 'inside_color',
                                    mutual=False)

        # Digitizer Fiducials
        point_scale = defaults['dig_fid_scale']
        opacity = defaults['dig_fid_opacity']
        self.hsp_lpa_obj = PointObject(
            scene=self.scene, color=lpa_color, opacity=opacity,
            has_norm=True, point_scale=point_scale, name='HSP-LPA')
        self.model.sync_trait('transformed_hsp_lpa',
                              self.hsp_lpa_obj, 'points', mutual=False)
        self.hsp_nasion_obj = PointObject(
            scene=self.scene, color=nasion_color, opacity=opacity,
            has_norm=True, point_scale=point_scale, name='HSP-Nasion')
        self.model.sync_trait('transformed_hsp_nasion',
                              self.hsp_nasion_obj, 'points', mutual=False)
        self.hsp_rpa_obj = PointObject(
            scene=self.scene, color=rpa_color, opacity=opacity,
            has_norm=True, point_scale=point_scale, name='HSP-RPA')
        self.model.sync_trait('transformed_hsp_rpa',
                              self.hsp_rpa_obj, 'points', mutual=False)

        # All points share these
        for p in (self.hsp_obj, self.eeg_obj, self.hpi_obj,
                  self.hsp_lpa_obj, self.hsp_nasion_obj, self.hsp_rpa_obj):
            self.sync_trait('hsp_visible', p, 'visible', mutual=False)
            self.model.sync_trait('mri_trans_noscale', p, 'project_to_trans',
                                  mutual=False)

        on_pick = self.scene.mayavi_scene.on_mouse_pick
        self.picker = on_pick(self.data_panel.fid_panel._on_pick, type='cell')

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

        self.sync_trait('bgcolor', self.scene, 'background')

        self._update_projection_surf()

        _toggle_mlab_render(self, True)
        self.scene.render()
        self.scene.camera.focal_point = (0., 0., 0.)
        self.data_panel.view_options_panel = ViewOptionsPanel(
            mri_obj=self.mri_obj, hsp_obj=self.hsp_obj,
            eeg_obj=self.eeg_obj, hpi_obj=self.hpi_obj,
            hsp_cf_obj=self.hsp_cf_obj, mri_cf_obj=self.mri_cf_obj,
            head_high_res=self.head_high_res,
            bgcolor=self.bgcolor, advanced_rendering=self.advanced_rendering)
        self.data_panel.headview.scale = self._initial_kwargs['scale']
        self.data_panel.headview.interaction = \
            self._initial_kwargs['interaction']
        self.data_panel.headview.left = True
        self.data_panel.view_options_panel.sync_trait(
            'coord_frame', self.model)
        self.data_panel.view_options_panel.sync_trait('head_high_res', self)
        self.data_panel.view_options_panel.sync_trait('advanced_rendering',
                                                      self)
        self.data_panel.view_options_panel.sync_trait('bgcolor', self)

    @on_trait_change('advanced_rendering')
    def _on_advanced_rendering_change(self):
        renderer = getattr(self.scene, 'renderer', None)
        if renderer is None:
            return
        if self.advanced_rendering:
            renderer.use_depth_peeling = 1
            renderer.occlusion_ratio = 0.1
            renderer.maximum_number_of_peels = 100
            renderer.vtk_window.multi_samples = 0
            renderer.vtk_window.alpha_bit_planes = 1
        else:
            renderer.use_depth_peeling = 0
            renderer.vtk_window.multi_samples = 8
            renderer.vtk_window.alpha_bit_planes = 0
            if hasattr(renderer, 'use_fxaa'):
                self.scene.renderer.use_fxaa = _get_3d_option('antialias')
        self.scene.render()

    @on_trait_change('lock_fiducials')
    def _on_lock_change(self):
        if not self.lock_fiducials:
            if self.mri_obj is None:
                self._initial_kwargs['head_opacity'] = 1.
            else:
                self._locked_opacity = self.mri_obj.opacity
                self.mri_obj.opacity = 1.
        else:
            if self.mri_obj is not None:
                self.mri_obj.opacity = self._locked_opacity

    @cached_property
    def _get_hsp_visible(self):
        return self.data_panel.hsp_always_visible or self.lock_fiducials

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

    @on_trait_change('nearest_calc')
    def _update_projection_surf(self):
        if len(self.model.processed_low_res_mri_points) <= 1:
            return
        rr = (self.model.processed_low_res_mri_points *
              self.model.parameters[6:9])
        surf = dict(rr=rr, tris=self.model.mri.bem_low_res.surf.tris,
                    nn=self.model.mri.bem_low_res.surf.nn)
        check_inside = _CheckInside(surf)
        nearest = _DistanceQuery(rr)
        for p in (self.eeg_obj, self.hsp_obj, self.hpi_obj):
            if p is not None:
                p.check_inside = check_inside
                p.nearest = nearest

    @on_trait_change('model:mri:bem_low_res:surf,head_high_res,'
                     'model:transformed_high_res_mri_points')
    def _update_mri_obj(self):
        if self.mri_obj is None:
            return
        self.mri_obj.tris = getattr(
            self.model.mri, 'bem_%s_res'
            % ('high' if self.head_high_res else 'low',)).surf.tris
        self.mri_obj.points = getattr(
            self.model, 'transformed_%s_res_mri_points'
            % ('high' if self.head_high_res else 'low',))

    # automatically lock fiducials if a good fiducials file is loaded
    @on_trait_change('model:mri:fid_file')
    def _on_fid_file_loaded(self):
        self.data_panel.fid_panel.locked = bool(self.model.mri.fid_file)

    def save_config(self, home_dir=None, size=None):
        """Write configuration values."""
        def s_c(key, value, lower=True):
            value = str(value)
            if lower:
                value = value.lower()
            set_config(key, str(value).lower(), home_dir=home_dir,
                       set_env=False)

        s_c('MNE_COREG_GUESS_MRI_SUBJECT', self.model.guess_mri_subject)
        s_c('MNE_COREG_HEAD_HIGH_RES', self.head_high_res)
        s_c('MNE_COREG_ADVANCED_RENDERING', self.advanced_rendering)
        if self.lock_fiducials:
            opacity = self.mri_obj.opacity
        else:
            opacity = self._locked_opacity
        s_c('MNE_COREG_HEAD_OPACITY', opacity)
        if size is not None:
            s_c('MNE_COREG_WINDOW_WIDTH', size[0])
            s_c('MNE_COREG_WINDOW_HEIGHT', size[1])
        s_c('MNE_COREG_SCENE_SCALE', self.data_panel.headview.scale)
        s_c('MNE_COREG_SCALE_LABELS', self.model.scale_labels)
        s_c('MNE_COREG_COPY_ANNOT', self.model.copy_annot)
        s_c('MNE_COREG_PREPARE_BEM', self.model.prepare_bem_model)
        if self.model.mri.subjects_dir:
            s_c('MNE_COREG_SUBJECTS_DIR', self.model.mri.subjects_dir, False)
        s_c('MNE_COREG_PROJECT_EEG', self.project_to_surface)
        s_c('MNE_COREG_ORIENT_TO_SURFACE', self.orient_to_surface)
        s_c('MNE_COREG_SCALE_BY_DISTANCE', self.scale_by_distance)
        s_c('MNE_COREG_MARK_INSIDE', self.mark_inside)
        s_c('MNE_COREG_INTERACTION', self.data_panel.headview.interaction)
