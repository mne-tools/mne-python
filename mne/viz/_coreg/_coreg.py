import os
import os.path as op
from ...coreg import Coregistration, _is_mri_subject
from ...viz import plot_alignment
from ...utils import get_subjects_dir


class CoregistrationUI(object):
    def __init__(self, info, subject=None, subjects_dir=None, fids='auto'):
        from ..backends.renderer import _get_renderer
        self._widgets = dict()
        self._verbose = False
        self._first_time = True
        self._omit_hsp_distance = 0.0
        self._opacity = 1.0
        self._default_icp_n_iterations = 20
        self._default_weights = {
            "lpa": 1.0,
            "nasion": 10.0,
            "rpa": 1.0,
        }
        self._reset_fitting_parameters()

        self._fids = fids
        self._subjects_dir = get_subjects_dir(subjects_dir=subjects_dir,
                                              raise_error=True)
        self._subjects = self._get_subjects()
        self._subject = subject if subject is not None else self._subjects[0]
        self._info = info

        self._coreg = Coregistration(info, subject, subjects_dir, fids)
        self._renderer = _get_renderer()
        self._renderer._window_close_connect(self._clean)
        self._configure_dock()
        self._renderer.show()

        self._update()

    def _reset_fitting_parameters(self):
        self._icp_n_iterations = self._default_icp_n_iterations
        if "icp_n_iterations" in self._widgets:
            self._widgets["icp_n_iterations"].set_value(
                self._default_icp_n_iterations)

        for dig in ("lpa", "nasion", "rpa"):
            widget_name = f"{dig}_weight"
            if widget_name in self._widgets:
                self._widgets[widget_name].set_value(
                    self._default_weights[dig])
            else:
                setattr(self, f"_{dig}_weight", self._default_weights[dig])

    def _set_omit_hsp_distance(self, distance):
        self._omit_hsp_distance = distance

    def _omit_hsp(self):
        self._coreg.omit_head_shape_points(self._omit_hsp_distance)

    def _reset(self):
        self._reset_fitting_parameters()
        self._coreg.reset()
        self._update()

    def _update(self):
        if self._first_time:
            self._first_time = False
        else:
            self._renderer.figure.plotter.clear()
        plot_alignment(self._info, trans=self._coreg.trans,
                       subject=self._subject,
                       subjects_dir=self._subjects_dir,
                       surfaces=dict(head=self._opacity),
                       dig=True, eeg=[], meg=False,
                       coord_frame='meg', fig=self._renderer.figure,
                       show=False, verbose=self._verbose)
        self._renderer.reset_camera()
        coords = ["X", "Y", "Z"]
        for tr in ("translation", "rotation", "scale"):
            for coord in coords:
                widget_name = tr[0] + coord
                if widget_name in self._widgets:
                    idx = coords.index(coord)
                    val = getattr(self._coreg, f"_{tr}")
                    self._widgets[widget_name].set_value(val[idx])

    def _toggle_transparent(self, state):
        self._opacity = 0.4 if state else 1.0
        self._update()

    def _switch_subjects_dir(self, subjects_dir):
        self._subjects_dir = subjects_dir
        self._subjects = self._get_subjects()
        self._subject = self._subjects[0]
        # XXX: add coreg.set_subjects_dir
        self._coreg._subjects_dir = self._subjects_dir
        self._coreg._subject = self._subject
        self._reset()

    def _switch_subject(self, subject):
        self._subject = subject
        # XXX: add coreg.set_subject()
        self._coreg._subject = subject
        self._reset()

    def _set_icp_n_iterations(self, n_iterations):
        self._icp_n_iterations = n_iterations

    def _set_lpa_weight(self, value):
        self._lpa_weight = value

    def _set_nasion_weight(self, value):
        self._nasion_weight = value

    def _set_rpa_weight(self, value):
        self._rpa_weight = value

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
            func=self._switch_subjects_dir,
            value=self._subjects_dir,
            placeholder="Subjects Directory",
            directory=True,
            layout=layout,
        )
        self._widgets["subject"] = self._renderer._dock_add_combo_box(
            name="Subject",
            value=self._subject,
            rng=self._subjects,
            callback=self._switch_subject,
            compact=True,
            layout=layout
        )

        layout = self._renderer._dock_add_group_box("MRI Fiducials")
        digs_states = ["Lock", "Edit"]
        self._renderer._dock_add_radio_buttons(
            value=digs_states[0],
            rng=digs_states,
            callback=noop,
            vertical=False,
            layout=layout,
        )
        self._widgets["fid_file"] = self._renderer._dock_add_file_button(
            name="fid_file",
            desc="Load",
            func=noop,
            placeholder="Path to fiducials",
            layout=layout,
        )
        digs = ["LPA", "Nasion", "RPA"]
        self._renderer._dock_add_radio_buttons(
            value=digs[0],
            rng=digs,
            callback=noop,
            vertical=False,
            layout=layout,
        )
        hlayout = self._renderer._dock_add_layout()
        for coord in ("X", "Y", "Z"):
            name = f"dig_{coord}"
            self._widgets[name] = self._renderer._dock_add_spin_box(
                name=coord,
                value=0.,
                rng=[0., 1.],
                callback=noop,
                compact=True,
                double=True,
                layout=hlayout
            )
        self._renderer._layout_add_widget(layout, hlayout)

        layout = self._renderer._dock_add_group_box("Digitization Source")
        self._widgets["info_file"] = self._renderer._dock_add_file_button(
            name="info_file",
            desc="Load",
            func=noop,
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
            value=0.0,
            rng=[0.0, 100.0],
            callback=self._set_omit_hsp_distance,
            decimals=4,
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
        self._widgets["make_transparent"] = self._renderer._dock_add_check_box(
            name="Make skin surface transparent",
            value=False,
            callback=self._toggle_transparent,
            layout=layout
        )
        self._renderer._dock_add_stretch()

        self._renderer._dock_initialize(name="Parameters", area="right")
        self._widgets["scaling_mode"] = self._renderer._dock_add_combo_box(
            name="Scaling Mode",
            value="0",
            rng=["0", "1", "3"],
            callback=noop,
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
                decimals=4,
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
                    decimals=4,
                    layout=hlayout
                )
        layout = self._renderer._dock_layout
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
        layout = self._renderer._dock_add_group_box("Fitting Options")
        self._widgets["icp_n_iterations"] = self._renderer._dock_add_spin_box(
            name="Number Of ICP Iterations",
            value=self._default_icp_n_iterations,
            rng=[1, 100],
            callback=self._set_icp_n_iterations,
            compact=True,
            double=False,
            layout=layout,
        )
        for dig in digs:
            dig = dig.lower()
            name = f"{dig}_weight"
            self._widgets[name] = self._renderer._dock_add_spin_box(
                name=name,
                value=getattr(self, f"_{dig}_weight"),
                rng=[1., 100.],
                # XXX: does not work with lambda+setattr?
                callback=getattr(self, f"_set_{dig}_weight"),
                compact=True,
                double=True,
                layout=layout
            )
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
        self._renderer._dock_add_button(
            name="Save...",
            callback=noop,
            layout=hlayout,
        )
        self._renderer._dock_add_button(
            name="Load...",
            callback=noop,
            layout=hlayout,
        )
        self._renderer._layout_add_widget(layout, hlayout)
        self._renderer._dock_add_stretch()

    def _clean(self):
        self._renderer = None
