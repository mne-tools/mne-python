import os
import os.path as op
import numpy as np
from functools import partial
from ...defaults import DEFAULTS
from ...io import read_info, read_fiducials
from ...coreg import Coregistration, _is_mri_subject
from ...viz._3d import (_plot_head_surface, _plot_head_fiducials,
                        _plot_head_shape_points, _plot_mri_fiducials)
from ...transforms import (read_trans, write_trans, _ensure_trans,
                           rotation_angles, _get_transforms_to_coord_frame)
from ...utils import get_subjects_dir
from traitlets import observe, HasTraits, Unicode, Bool, Float


class CoregistrationUI(HasTraits):
    _subject = Unicode()
    _subjects_dir = Unicode()
    _lock_fids = Bool()
    _fiducials_file = Unicode()
    _current_fiducial = Unicode()
    _info_file = Unicode()
    _coreg_modified = Bool()
    _head_shape_point = Bool()
    _head_resolution = Bool()
    _head_transparency = Bool()
    _grow_hair = Float()
    _scale_mode = Unicode()
    _icp_fid_match = Unicode()

    def __init__(self, info, subject=None, subjects_dir=None, fids='auto'):
        from ..backends.renderer import _get_renderer
        self._widgets = dict()
        self._fids_to_pick = list()
        self._verbose = True
        self._coord_frame = "mri"
        self._mouse_no_mvt = -1
        self._omit_hsp_distance = 0.0
        self._surface = "head-dense"
        self._opacity = 1.0
        self._fid_colors = tuple(
            DEFAULTS['coreg'][f'{key}_color'] for key in
            ('lpa', 'nasion', 'rpa'))
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
        self._surfaces = dict()
        self._renderer = _get_renderer(bgcolor="grey")
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
        self._configure_picking()
        self._renderer.show()

    def _configure_picking(self):
        self._renderer._update_picking_callback(
            self._on_mouse_move,
            self._on_button_press,
            self._on_button_release,
            self._on_pick
        )
        self._actors["msg"] = self._renderer.text2d(0, 0, "")

    def _on_mouse_move(self, vtk_picker, event):
        if self._mouse_no_mvt:
            self._mouse_no_mvt -= 1

    def _on_button_press(self, vtk_picker, event):
        self._mouse_no_mvt = 2

    def _on_button_release(self, vtk_picker, event):
        if self._mouse_no_mvt > 0:
            x, y = vtk_picker.GetEventPosition()
            # XXX: plotter/renderer should not be exposed if possible
            plotter = self._renderer.figure.plotter
            picked_renderer = self._renderer.figure.plotter.renderer
            # trigger the pick
            plotter.picker.Pick(x, y, 0, picked_renderer)
        self._mouse_no_mvt = 0

    def _on_pick(self, vtk_picker, event):
        if self._lock_fids:
            return
        if len(self._fids_to_pick) == 0:
            return
        # XXX: taken from Brain, can be refactored
        cell_id = vtk_picker.GetCellId()
        mesh = vtk_picker.GetDataSet()
        if mesh is None or cell_id == -1 or not self._mouse_no_mvt:
            return
        pos = np.array(vtk_picker.GetPickPosition())
        vtk_cell = mesh.GetCell(cell_id)
        cell = [vtk_cell.GetPointId(point_id) for point_id
                in range(vtk_cell.GetNumberOfPoints())]
        vertices = mesh.points[cell]
        idx = np.argmin(abs(vertices - pos), axis=0)
        vertex_id = cell[idx[0]]
        # XXX: for debug only
        fid = self._fids_to_pick.pop()
        print(fid, vertex_id)
        if len(self._fids_to_pick) == 0:
            self._actors["msg"].SetInput("")
        else:
            next_fid = self._fids_to_pick[-1].upper()
            self._actors["msg"].SetInput(f"Picking {next_fid}...")
        self._renderer._update()

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
        self._emit_coreg_modified()

    def _set_head_resolution(self, state):
        self._head_resolution = bool(state)

    def _set_head_transparency(self, state):
        self._head_transparency = bool(state)

    def _set_grow_hair(self, value):
        self._grow_hair = value

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

    @observe("_lock_fids")
    def _lock_fids_changed(self, change=None):
        if "show_hsp" in self._widgets:
            if self._lock_fids:
                self._widgets["show_hsp"].set_enabled(True)
            else:
                self._widgets["show_hsp"].set_value(False)
                self._widgets["show_hsp"].set_enabled(False)
                self._fids_to_pick = ["lpa", "nasion", "rpa"]
                next_fid = self._fids_to_pick[-1].upper()
                self._actors["msg"].SetInput(f"Picking {next_fid}...")
                self._renderer._update()
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

    @observe("_coreg_modified")
    def _update(self, change=None):
        self._add_head_shape_points()
        self._add_head_fiducials()
        self._add_mri_fiducials()

    @observe("_head_shape_point")
    def _head_shape_point_changed(self, change=None):
        self._add_head_shape_points()

    @observe("_head_resolution")
    def _head_resolution_changed(self, change=None):
        self._surface = "head-dense" if self._head_resolution else "head"
        self._add_head_surface()
        self._grow_hair_changed()

    @observe("_head_transparency")
    def _head_transparency_changed(self, change=None):
        self._opacity = 0.4 if self._head_transparency else 1.0
        self._actors["head"].GetProperty().SetOpacity(self._opacity)
        self._renderer._update()

    @observe("_grow_hair")
    def _grow_hair_changed(self, change=None):
        self._coreg.set_grow_hair(self._grow_hair)
        if "head" in self._surfaces:
            res = "high" if self._head_resolution else "low"
            self._surfaces["head"].points = \
                self._coreg._get_processed_mri_points(res)
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
        self._emit_coreg_modified()

    def _emit_coreg_modified(self):
        self._coreg_modified = not self._coreg_modified

    def _update_actor(self, actor_name, actor):
        self._renderer.plotter.remove_actor(self._actors.get(actor_name))
        self._actors[actor_name] = actor

    def _add_mri_fiducials(self):
        to_cf_t = _get_transforms_to_coord_frame(
            self._info, self._coreg.trans, coord_frame=self._coord_frame)
        mri_fids_actors = _plot_mri_fiducials(
            self._renderer, self._coreg._fid_points, self._subjects_dir,
            self._subject, to_cf_t, self._fid_colors)
        self._update_actor("mri_fiducials", mri_fids_actors)

    def _add_head_fiducials(self):
        to_cf_t = _get_transforms_to_coord_frame(
            self._info, self._coreg.trans, coord_frame=self._coord_frame)
        head_fids_actors = _plot_head_fiducials(
            self._renderer, self._info, to_cf_t, self._fid_colors)
        self._update_actor("head_fiducials", head_fids_actors)

    def _add_head_shape_points(self):
        if self._head_shape_point:
            to_cf_t = _get_transforms_to_coord_frame(
                self._info, self._coreg.trans, coord_frame=self._coord_frame)
            hsp_actors = _plot_head_shape_points(
                self._renderer, self._info, to_cf_t)
        else:
            hsp_actors = None
        self._update_actor("head_shape_points", hsp_actors)

    def _add_head_surface(self):
        bem = None
        to_cf_t = _get_transforms_to_coord_frame(
            self._info, self._coreg.trans, coord_frame=self._coord_frame)
        try:
            head_actor, head_surf = _plot_head_surface(
                self._renderer, self._surface, self._subject,
                self._subjects_dir, bem, self._coord_frame, to_cf_t,
                alpha=self._opacity)
        except IOError:
            head_actor, head_surf = _plot_head_surface(
                self._renderer, "head", self._subject, self._subjects_dir,
                bem, self._coord_frame, to_cf_t, alpha=self._opacity)
        self._update_actor("head", head_actor)
        self._surfaces["head"] = head_surf

    def _fit_fiducials(self):
        self._coreg.fit_fiducials(
            lpa_weight=self._lpa_weight,
            nasion_weight=self._nasion_weight,
            rpa_weight=self._rpa_weight,
            verbose=self._verbose,
        )
        self._emit_coreg_modified()

    def _fit_icp(self):
        self._coreg.fit_icp(
            n_iterations=self._icp_n_iterations,
            lpa_weight=self._lpa_weight,
            nasion_weight=self._nasion_weight,
            rpa_weight=self._rpa_weight,
            verbose=self._verbose,
        )
        self._emit_coreg_modified()

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
        self._emit_coreg_modified()

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
            callback=self._set_grow_hair,
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
        self._surfaces.clear()
