from ...coreg import Coregistration
from ...viz import plot_alignment


class CoregistrationUI(object):
    def __init__(self, info, subject, subjects_dir, fids='auto'):
        from ..backends.renderer import _get_renderer
        self._info = info
        self._subject = subject
        self._subjects_dir = subjects_dir
        self._fids = fids

        self._widgets = dict()
        self._coreg = Coregistration(info, subject, subjects_dir, fids)
        self._renderer = _get_renderer()
        self._renderer._window_close_connect(self._clean)
        self._configure_dock()
        self._renderer.show()

        self._plot()

    def _plot(self):
        plot_alignment(self._info, trans=self._coreg.trans,
                       subject=self._subject,
                       subjects_dir=self._subjects_dir,
                       surfaces=dict(head=0.4),
                       dig=True, eeg=[], meg=False,
                       coord_frame='meg', fig=self._renderer.figure)

    def _configure_dock(self):
        def noop(x):
            del x

        self._renderer._dock_initialize(name="Parameters", area="left")
        layout = self._renderer._dock_add_group_box("MRI Subject")
        self._widgets["subjects_dir"] = self._renderer._dock_add_file_button(
            name="subjects_dir",
            desc="Load",
            func=noop,
            placeholder="Subjects Directory",
            directory=True,
            layout=layout,
        )
        self._widgets["subject"] = self._renderer._dock_add_combo_box(
            name="Subject",
            value="sample",
            rng=["sample"],
            callback=noop,
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
            rng=[0.0, 10.0],
            callback=noop,
            layout=hlayout,
        )
        self._widgets["omit"] = self._renderer._dock_add_button(
            name="Omit",
            callback=noop,
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
            callback=noop,
            layout=layout
        )
        self._renderer._dock_add_stretch()

        self._renderer._dock_initialize(
            name="Fitting Parameters", area="right")
        layout = self._renderer._dock_add_group_box("Scaling")
        self._widgets["scaling_mode"] = self._renderer._dock_add_combo_box(
            name="Scaling Mode",
            value="0",
            rng=["0", "1", "3"],
            callback=noop,
            compact=True,
            layout=layout,
        )
        hlayout = self._renderer._dock_add_group_box(
            name="Scaling Parameters",
            layout=layout
        )
        for coord in ("X", "Y", "Z"):
            name = f"s{coord}"
            self._widgets[name] = self._renderer._dock_add_spin_box(
                name=name,
                value=0.,
                rng=[0., 1.],
                callback=noop,
                compact=True,
                double=True,
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
                    rng=[0., 1.],
                    callback=noop,
                    compact=True,
                    double=True,
                    layout=hlayout
                )
        hlayout = self._renderer._dock_add_layout(vertical=False)
        self._renderer._dock_add_button(
            name="Fit Fiducials",
            callback=noop,
            layout=hlayout,
        )
        self._renderer._dock_add_button(
            name="Fit ICP",
            callback=noop,
            layout=hlayout,
        )
        layout = self._renderer._dock_layout
        self._renderer._layout_add_widget(layout, hlayout)
        self._renderer._dock_add_stretch()

    def _clean(self):
        self._renderer = None
