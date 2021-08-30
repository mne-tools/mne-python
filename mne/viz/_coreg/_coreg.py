import os
import os.path as op
from ...io import read_info, read_fiducials
from ...coreg import Coregistration, _is_mri_subject
from ...viz import plot_alignment
from ...transforms import (read_trans, write_trans, _ensure_trans,
                           rotation_angles)
from ...utils import get_subjects_dir
from traitlets import observe, HasTraits, Unicode, Bool


class CoregistrationUI(HasTraits):
    _subject = Unicode()
    _subjects_dir = Unicode()
    _lock_fids = Bool()
    _fiducials_file = Unicode()
    _current_fiducial = Unicode()
    _info_file = Unicode()
    _head_resolution = Bool()
    _head_transparency = Bool()
    _scale_mode = Unicode()

    def __init__(self, info, subject=None, subjects_dir=None, fids='auto'):
        from ..backends.renderer import _get_renderer
        self._widgets = dict()
        self._verbose = False
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
        self._reset_fitting_parameters()

        self._renderer = _get_renderer()
        self._renderer._window_close_connect(self._clean)
        self._renderer.show()
        self._coreg = Coregistration(info, subject, subjects_dir, fids)
        self._fids = fids
        self._info = info
        self._subjects_dir = get_subjects_dir(subjects_dir=subjects_dir,
                                              raise_error=True)
        self._subject = subject if subject is not None else self._subjects[0]

        self._configure_dock()
        self._update(clear=False)

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

    def _set_head_resolution(self, state):
        self._head_resolution = bool(state)

    def _set_head_transparency(self, state):
        self._head_transparency = bool(state)

    def _set_scale_mode(self, mode):
        self._scale_mode = mode

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

    @observe("_head_resolution")
    def _head_resolution_changed(self, change=None):
        self._surface = "head-dense" if self._head_resolution else "head"
        self._update()

    @observe("_head_transparency")
    def _head_transparency_changed(self, change=None):
        self._opacity = 0.4 if self._head_transparency else 1.0
        self._update()

    @observe("_scale_mode")
    def _scale_mode_changed(self, change=None):
        mode = None if self._scale_mode == "None" else self._scale_mode
        self._coreg.set_scale_mode(mode)

    def _reset_fitting_parameters(self):
        self._icp_n_iterations = self._default_icp_n_iterations
        if "icp_n_iterations" in self._widgets:
            self._widgets["icp_n_iterations"].set_value(
                self._icp_n_iterations)

        self._icp_fid_match = self._default_icp_fid_matches[0]
        if "icp_fid_match" in self._widgets:
            self._widgets["icp_fid_match"].set_value(
                self._icp_fid_match)

        for fid in self._default_weights.keys():
            widget_name = f"{fid}_weight"
            if widget_name in self._widgets:
                self._widgets[widget_name].set_value(
                    self._default_weights[fid])
            else:
                setattr(self, f"_{fid}_weight", self._default_weights[fid])

    def _reset_fiducials(self):
        self._set_current_fiducial(self._default_fiducials[0])

    def _omit_hsp(self):
        self._coreg.omit_head_shape_points(self._omit_hsp_distance)

    def _reset(self):
        self._reset_fitting_parameters()
        self._reset_fiducials()
        self._coreg.reset()
        self._update()

    def _update(self, clear=True, update_parameters=True):
        if clear:
            self._renderer.figure.plotter.clear()
        surfaces = dict()
        surfaces[self._surface] = self._opacity
        kwargs = dict(info=self._info, trans=self._coreg.trans,
                      subject=self._subject,
                      subjects_dir=self._subjects_dir,
                      surfaces=surfaces,
                      dig=True, eeg=[], meg=False,
                      coord_frame='meg', fig=self._renderer.figure,
                      show=False, verbose=self._verbose)
        try:
            plot_alignment(**kwargs)
        except IOError:
            kwargs.update(surfaces="head")
            plot_alignment(**kwargs)
        self._renderer.reset_camera()
        if update_parameters:
            coords = ["X", "Y", "Z"]
            for tr in ("translation", "rotation", "scale"):
                for coord in coords:
                    widget_name = tr[0] + coord
                    if widget_name in self._widgets:
                        idx = coords.index(coord)
                        val = getattr(self._coreg, f"_{tr}")
                        val_idx = val[idx]
                        if tr in ("translation", "scale"):
                            val_idx *= 1000.0
                        self._widgets[widget_name].set_value(val_idx)

    def _set_icp_fid_match(self, method):
        self._coreg.set_fid_match(method)
        self._update()

    def _set_icp_n_iterations(self, n_iterations):
        self._icp_n_iterations = n_iterations

    def _set_point_weight(self, point, weight):
        setattr(f"_{point}_weight", weight)

    def _fit_fiducials(self):
        self._coreg.fit_fiducials(
            lpa_weight=self._lpa_weight,
            nasion_weight=self._nasion_weight,
            rpa_weight=self._rpa_weight,
            verbose=self._verbose,
        )
        self._update()

    def _fit_icp(self):
        self._coreg.fit_icp(
            n_iterations=self._icp_n_iterations,
            lpa_weight=self._lpa_weight,
            nasion_weight=self._nasion_weight,
            rpa_weight=self._rpa_weight,
            verbose=self._verbose,
        )
        self._update()

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
        self._update()

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
            value=False,
            callback=noop,
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
                    rng=[-100., 100.],
                    callback=noop,
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
            name = f"{fid}_weight"
            self._widgets[name] = self._renderer._dock_add_spin_box(
                name=fid,
                value=getattr(self, f"_{fid_lower}_weight"),
                rng=[1., 100.],
                callback=lambda x: self._set_point_weight(fid_lower, x),
                compact=True,
                double=True,
                layout=hlayout
            )
        self._renderer._layout_add_widget(layout, hlayout)
        hlayout = self._renderer._dock_add_layout(vertical=False)
        for point in ("HSP", "EEG", "HPI"):
            point_lower = point.lower()
            name = f"{point}_weight"
            self._widgets[name] = self._renderer._dock_add_spin_box(
                name=point,
                value=getattr(self, f"_{point_lower}_weight"),
                rng=[1., 100.],
                callback=lambda x: self._set_point_weight(fid_lower, x),
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
