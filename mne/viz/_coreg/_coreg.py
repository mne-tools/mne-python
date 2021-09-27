import os
import os.path as op
import numpy as np
from functools import partial
from ...defaults import DEFAULTS
from ...io import read_info, read_fiducials
from ...coreg import Coregistration, _is_mri_subject
from ...viz._3d import (_plot_head_surface, _plot_head_fiducials,
                        _plot_head_shape_points)
from ...transforms import (read_trans, write_trans, _ensure_trans,
                           rotation_angles, _get_transforms_to_coord_frame)
from ...utils import get_subjects_dir
from traitlets import observe, HasTraits, Unicode, Bool


class CoregistrationUI(HasTraits):
    _subject = Unicode()
    _subjects_dir = Unicode()
    _lock_fids = Bool()
    _fiducials_file = Unicode()
    _current_fiducial = Unicode()
    _info_file = Unicode()
    _head_shape_point = Bool()
    _head_resolution = Bool()
    _head_transparency = Bool()
    _scale_mode = Unicode()
    _icp_fid_match = Unicode()

    def __init__(self, info, subject=None, subjects_dir=None, fids='auto'):
        from ..backends.renderer import _get_renderer
        self._widgets = dict()
        self._verbose = True
        self._omit_hsp_distance = 0.0
        self._surface = "head-dense"
        self._opacity = 1.0
        self._default_fiducials = ("LPA", "Nasion", "RPA")
        self._default_icp_fid_matches = ('nearest', 'matched')
        self._default_icp_n_iterations = 20
        self._default_weights = {
            "lpa": 1.0,
            "nasion": 10.0,
            "rpa": 1.0,
            "hsp": 1.0,
            "eeg": 1.0,
            "hpi": 1.0,
        }

        self._actors = dict()
        self._renderer = _get_renderer()
        self._renderer._window_close_connect(self._clean)
        self._coreg = Coregistration(info, subject, subjects_dir, fids)
        self._fids = fids
        self._info = info
        self._subjects_dir = get_subjects_dir(subjects_dir=subjects_dir,
                                              raise_error=True)
        self._subject = subject if subject is not None else self._subjects[0]
        self._head_shape_point = True
        self._head_resolution = True
        self._icp_n_iterations = self._default_icp_n_iterations
        self._icp_fid_match = self._default_icp_fid_matches[0]
        for fid in self._default_weights.keys():
            setattr(self, f"_{fid}_weight", self._default_weights[fid])

        self._reset_fitting_parameters()
        self._configure_dock()
        self._renderer.show()

    def _set_subjects_dir(self, subjects_dir):
        self._subjects_dir = subjects_dir

    def _set_subject(self, subject):
        self._subject = subject

    def _set_lock_fids(self, state):
        self._lock_fids = bool(state)

    def _set_fiducials_file(self, fname):
        self._fiducials_file = fname

    def _set_current_fiducial(self, fid):
        self._current_fiducial = fid.lower()

    def _set_info_file(self, fname):
        self._info_file = fname

    def _set_omit_hsp_distance(self, distance):
        self._omit_hsp_distance = distance / 1000.0

    def _set_head_shape_points(self, state):
        self._head_shape_point = bool(state)

    def _set_head_resolution(self, state):
        self._head_resolution = bool(state)

    def _set_head_transparency(self, state):
        self._head_transparency = bool(state)

    def _set_scale_mode(self, mode):
        self._scale_mode = mode

    def _set_parameter(self, x, mode_name, coord):
        params = dict(
            rotation=self._coreg._rotation,
            translation=self._coreg._translation,
            scale=self._coreg._scale,
        )
        idx = ["X", "Y", "Z"].index(coord)
        if mode_name == "rotation":
            params[mode_name][idx] = np.deg2rad(x)
        else:
            params[mode_name][idx] = x / 1000.0
        self._coreg._update_params(
            rot=params["rotation"],
            tra=params["translation"],
            sca=params["scale"],
        )
        self._update(update_parameters=False)

    def _set_icp_n_iterations(self, n_iterations):
        self._icp_n_iterations = n_iterations

    def _set_icp_fid_match(self, method):
        self._icp_fid_match = method

    def _set_point_weight(self, weight, point):
        setattr(self, f"_{point}_weight", weight)

    @observe("_subjects_dir")
    def _subjects_dir_changed(self, change=None):
        # XXX: add coreg.set_subjects_dir
        self._coreg._subjects_dir = self._subjects_dir
        subjects = self._get_subjects()
        self._subject = subjects[0]
        self._reset()

    @observe("_subject")
    def _subject_changed(self, changed=None):
        # XXX: add coreg.set_subject()
        self._coreg._subject = self._subject
        self._reset()
        self._add_head_surface()
        self._add_head_fiducials()

    @observe("_lock_fids")
    def _lock_fids_changed(self, change=None):
        if "lock_fids" in self._widgets:
            self._widgets["lock_fids"].set_value(self._lock_fids)
        if "fid_file" in self._widgets:
            self._widgets["fid_file"].set_enabled(not self._lock_fids)
        if "fids" in self._widgets:
            self._widgets["fids"].set_enabled(not self._lock_fids)
        for coord in ("X", "Y", "Z"):
            name = f"fid_{coord}"
            if name in self._widgets:
                self._widgets[name].set_enabled(not self._lock_fids)

    @observe("_fiducials_file")
    def _fiducials_file_changed(self, change=None):
        fids, _ = read_fiducials(self._fiducials_file)
        self._coreg._setup_fiducials(fids)
        self._reset()
        self._add_head_fiducials()

    @observe("_current_fiducial")
    def _current_fiducial_changed(self, change=None):
        fid = self._current_fiducial
        val = getattr(self._coreg, f"_{fid}")[0] * 1000.0
        coords = ["X", "Y", "Z"]
        for coord in coords:
            name = f"fid_{coord}"
            idx = coords.index(coord)
            if name in self._widgets:
                self._widgets[name].set_value(val[idx])

    @observe("_info_file")
    def _info_file_changed(self, change=None):
        self._info = read_info(self._info_file)
        # XXX: add coreg.set_info()
        self._coreg._info = self._info
        self._reset()

    @observe("_head_shape_point")
    def _head_shape_point_changed(self, change=None):
        self._add_head_shape_points()

    @observe("_head_resolution")
    def _head_resolution_changed(self, change=None):
        self._surface = "head-dense" if self._head_resolution else "head"
        self._add_head_surface()

    @observe("_head_transparency")
    def _head_transparency_changed(self, change=None):
        self._opacity = 0.4 if self._head_transparency else 1.0
        self._actors["head"].GetProperty().SetOpacity(self._opacity)
        self._renderer._update()

    @observe("_scale_mode")
    def _scale_mode_changed(self, change=None):
        mode = None if self._scale_mode == "None" else self._scale_mode
        self._coreg.set_scale_mode(mode)

    @observe("_icp_fid_match")
    def _icp_fid_match_changed(self, change=None):
        self._coreg.set_fid_match(self._icp_fid_match)

    def _reset_fitting_parameters(self):
        if "icp_n_iterations" in self._widgets:
            self._widgets["icp_n_iterations"].set_value(
                self._default_icp_n_iterations)
        if "icp_fid_match" in self._widgets:
            self._widgets["icp_fid_match"].set_value(
                self._default_icp_fid_matches[0])
        for fid in self._default_weights.keys():
            widget_name = f"{fid}_weight"
            if widget_name in self._widgets:
                self._widgets[widget_name].set_value(
                    self._default_weights[fid])

    def _reset_fiducials(self):
        self._set_current_fiducial(self._default_fiducials[0])

    def _omit_hsp(self):
        self._coreg.omit_head_shape_points(self._omit_hsp_distance)

    def _reset(self):
        self._reset_fitting_parameters()
        self._reset_fiducials()
        self._coreg.reset()

    def _add_head_fiducials(self):
        if "head_fids" in self._actors:
            self._renderer.plotter.remove_actor(self._actors["head_fids"])
        coord_frame = 'mri'
        defaults = DEFAULTS['coreg']
        fid_colors = tuple(
            defaults[f'{key}_color'] for key in ('lpa', 'nasion', 'rpa'))
        to_cf_t = _get_transforms_to_coord_frame(
            self._info, self._coreg.trans, coord_frame=coord_frame)
        head_fids_actors = _plot_head_fiducials(
            self._renderer, self._info, to_cf_t, fid_colors)
        self._actors["head_fids"] = head_fids_actors
        self._renderer._update()

    def _add_head_shape_points(self):
        if "head_shape_points" in self._actors:
            self._renderer.plotter.remove_actor(
                self._actors["head_shape_points"])
        if self._head_shape_point:
            coord_frame = 'mri'
            to_cf_t = _get_transforms_to_coord_frame(
                self._info, self._coreg.trans, coord_frame=coord_frame)
            head_shape_points = _plot_head_shape_points(
                self._renderer, self._info, to_cf_t)
        else:
            head_shape_points = None
        self._actors["head_shape_points"] = head_shape_points
        self._renderer._update()

    def _add_head_surface(self):
        if "head" in self._actors:
            self._renderer.plotter.remove_actor(self._actors["head"])
        bem = None
        coord_frame = 'mri'
        to_cf_t = _get_transforms_to_coord_frame(
            self._info, self._coreg.trans, coord_frame=coord_frame)
        try:
            head_actor, _ = _plot_head_surface(
                self._renderer, self._surface, self._subject,
                self._subjects_dir, bem, coord_frame, to_cf_t,
                alpha=self._opacity)
        except IOError:
            head_actor, _ = _plot_head_surface(
                self._renderer, "head", self._subject, self._subjects_dir,
                bem, coord_frame, to_cf_t, alpha=self._opacity)
        self._actors["head"] = head_actor
        self._renderer._update()

    def _fit_fiducials(self):
        self._coreg.fit_fiducials(
            lpa_weight=self._lpa_weight,
            nasion_weight=self._nasion_weight,
            rpa_weight=self._rpa_weight,
            verbose=self._verbose,
        )
        # XXX: better way to update viz (traits maybe)?
        self._head_shape_point_changed()

    def _fit_icp(self):
        self._coreg.fit_icp(
            n_iterations=self._icp_n_iterations,
            lpa_weight=self._lpa_weight,
            nasion_weight=self._nasion_weight,
            rpa_weight=self._rpa_weight,
            verbose=self._verbose,
        )
        # XXX: better way to update viz (traits maybe)?
        self._head_shape_point_changed()

    def _save_trans(self, fname):
        write_trans(fname, self._coreg.trans)

    def _load_trans(self, fname):
        mri_head_t = _ensure_trans(read_trans(fname, return_all=True),
                                   'mri', 'head')['trans']
        rot_x, rot_y, rot_z = rotation_angles(mri_head_t)
        x, y, z = mri_head_t[:3, 3]
        self._coreg._update_params(
            rot=[rot_x, rot_y, rot_z],
            tra=[x, y, z],
        )

    def _get_subjects(self):
        # XXX: would be nice to move this function to util
        sdir = self._subjects_dir
        is_dir = sdir and op.isdir(sdir)
        if is_dir:
            dir_content = os.listdir(sdir)
            subjects = [s for s in dir_content if _is_mri_subject(s, sdir)]
            if len(subjects) == 0:
                subjects.append('')
        else:
            subjects = ['']
        return sorted(subjects)

    def _configure_dock(self):
        def noop(x):
            del x

        self._renderer._dock_initialize(name="Input", area="left")
        layout = self._renderer._dock_add_group_box("MRI Subject")
        self._widgets["subjects_dir"] = self._renderer._dock_add_file_button(
            name="subjects_dir",
            desc="Load",
            func=self._set_subjects_dir,
            value=self._subjects_dir,
            placeholder="Subjects Directory",
            directory=True,
            layout=layout,
        )
        self._widgets["subject"] = self._renderer._dock_add_combo_box(
            name="Subject",
            value=self._subject,
            rng=self._get_subjects(),
            callback=self._set_subject,
            compact=True,
            layout=layout
        )

        layout = self._renderer._dock_add_group_box("MRI Fiducials")
        self._widgets["lock_fids"] = self._renderer._dock_add_check_box(
            name="Lock fiducials",
            value=False,
            callback=self._set_lock_fids,
            layout=layout
        )
        self._widgets["fid_file"] = self._renderer._dock_add_file_button(
            name="fid_file",
            desc="Load",
            func=self._set_fiducials_file,
            placeholder="Path to fiducials",
            layout=layout,
        )
        self._widgets["fids"] = self._renderer._dock_add_radio_buttons(
            value=self._default_fiducials[0],
            rng=self._default_fiducials,
            callback=self._set_current_fiducial,
            vertical=False,
            layout=layout,
        )
        hlayout = self._renderer._dock_add_layout()
        for coord in ("X", "Y", "Z"):
            name = f"fid_{coord}"
            self._widgets[name] = self._renderer._dock_add_spin_box(
                name=coord,
                value=0.,
                rng=[-100., 100.],
                callback=noop,
                compact=True,
                double=True,
                decimals=1,
                layout=hlayout
            )
        self._set_current_fiducial(self._default_fiducials[0])  # init
        self._renderer._layout_add_widget(layout, hlayout)
        self._set_lock_fids(True)  # init

        layout = self._renderer._dock_add_group_box("Digitization Source")
        self._widgets["info_file"] = self._renderer._dock_add_file_button(
            name="info_file",
            desc="Load",
            func=self._set_info_file,
            placeholder="Path to info",
            layout=layout,
        )
        self._widgets["grow_hair"] = self._renderer._dock_add_spin_box(
            name="Grow Hair",
            value=0.0,
            rng=[0.0, 10.0],
            callback=noop,
            layout=layout,
        )
        hlayout = self._renderer._dock_add_layout(vertical=False)
        self._widgets["omit_distance"] = self._renderer._dock_add_spin_box(
            name="Omit Distance",
            value=10.,
            rng=[0.0, 100.0],
            callback=self._set_omit_hsp_distance,
            decimals=1,
            layout=hlayout,
        )
        self._widgets["omit"] = self._renderer._dock_add_button(
            name="Omit",
            callback=self._omit_hsp,
            layout=hlayout,
        )
        self._renderer._layout_add_widget(layout, hlayout)

        layout = self._renderer._dock_add_group_box("View")
        self._widgets["show_hsp"] = self._renderer._dock_add_check_box(
            name="Show Head Shape Points",
            value=True,
            callback=self._set_head_shape_points,
            layout=layout
        )
        self._widgets["high_res_head"] = self._renderer._dock_add_check_box(
            name="Show High Resolution Head",
            value=True,
            callback=self._set_head_resolution,
            layout=layout
        )
        self._widgets["make_transparent"] = self._renderer._dock_add_check_box(
            name="Make skin surface transparent",
            value=False,
            callback=self._set_head_transparency,
            layout=layout
        )
        self._renderer._dock_add_stretch()

        self._renderer._dock_initialize(name="Parameters", area="right")
        self._widgets["scaling_mode"] = self._renderer._dock_add_combo_box(
            name="Scaling Mode",
            value="None",
            rng=["None", "uniform", "3-axis"],
            callback=self._set_scale_mode,
            compact=True,
        )
        hlayout = self._renderer._dock_add_group_box(
            name="Scaling Parameters",
        )
        for coord in ("X", "Y", "Z"):
            name = f"s{coord}"
            self._widgets[name] = self._renderer._dock_add_spin_box(
                name=name,
                value=0.,
                rng=[-100., 100.],
                callback=noop,
                compact=True,
                double=True,
                decimals=1,
                layout=hlayout
            )

        for mode, mode_name in (("t", "Translation"), ("r", "Rotation")):
            hlayout = self._renderer._dock_add_group_box(
                f"{mode_name} ({mode})")
            for coord in ("X", "Y", "Z"):
                name = f"{mode}{coord}"
                self._widgets[name] = self._renderer._dock_add_spin_box(
                    name=name,
                    value=0.,
                    rng=[-1000., 1000.],
                    callback=partial(
                        self._set_parameter,
                        mode_name=mode_name.lower(),
                        coord=coord,
                    ),
                    compact=True,
                    double=True,
                    decimals=1,
                    layout=hlayout
                )

        layout = self._renderer._dock_add_group_box("Fitting")
        hlayout = self._renderer._dock_add_layout(vertical=False)
        self._renderer._dock_add_button(
            name="Fit Fiducials",
            callback=self._fit_fiducials,
            layout=hlayout,
        )
        self._renderer._dock_add_button(
            name="Fit ICP",
            callback=self._fit_icp,
            layout=hlayout,
        )
        self._renderer._layout_add_widget(layout, hlayout)
        self._widgets["icp_n_iterations"] = self._renderer._dock_add_spin_box(
            name="Number Of ICP Iterations",
            value=self._default_icp_n_iterations,
            rng=[1, 100],
            callback=self._set_icp_n_iterations,
            compact=True,
            double=False,
            layout=layout,
        )
        self._widgets["icp_fid_match"] = self._renderer._dock_add_combo_box(
            name="Fiducial point matching",
            value=self._default_icp_fid_matches[0],
            rng=self._default_icp_fid_matches,
            callback=self._set_icp_fid_match,
            compact=True,
            layout=layout
        )
        layout = self._renderer._dock_add_group_box(
            name="Weights",
            layout=layout,
        )
        hlayout = self._renderer._dock_add_layout(vertical=False)
        for fid in self._default_fiducials:
            fid_lower = fid.lower()
            name = f"{fid_lower}_weight"
            self._widgets[name] = self._renderer._dock_add_spin_box(
                name=fid,
                value=getattr(self, f"_{fid_lower}_weight"),
                rng=[1., 100.],
                callback=partial(self._set_point_weight, point=fid_lower),
                compact=True,
                double=True,
                layout=hlayout
            )
        self._renderer._layout_add_widget(layout, hlayout)
        hlayout = self._renderer._dock_add_layout(vertical=False)
        for point in ("HSP", "EEG", "HPI"):
            point_lower = point.lower()
            name = f"{point_lower}_weight"
            self._widgets[name] = self._renderer._dock_add_spin_box(
                name=point,
                value=getattr(self, f"_{point_lower}_weight"),
                rng=[1., 100.],
                callback=partial(self._set_point_weight, point=point_lower),
                compact=True,
                double=True,
                layout=hlayout
            )
        self._renderer._layout_add_widget(layout, hlayout)
        self._renderer._dock_add_button(
            name="Reset Fitting Options",
            callback=self._reset_fitting_parameters,
            layout=layout,
        )
        layout = self._renderer._dock_layout
        hlayout = self._renderer._dock_add_layout(vertical=False)
        self._renderer._dock_add_button(
            name="Reset",
            callback=self._reset,
            layout=hlayout,
        )
        self._widgets["save_trans"] = self._renderer._dock_add_file_button(
            name="save_trans",
            desc="Save...",
            func=self._save_trans,
            input_text_widget=False,
            layout=hlayout,
        )
        self._widgets["load_trans"] = self._renderer._dock_add_file_button(
            name="load_trans",
            desc="Load...",
            func=self._load_trans,
            input_text_widget=False,
            layout=hlayout,
        )
        self._renderer._layout_add_widget(layout, hlayout)
        self._renderer._dock_add_stretch()

    def _clean(self):
        self._renderer = None
        self._actors.clear()
