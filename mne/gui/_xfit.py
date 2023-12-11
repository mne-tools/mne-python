from functools import partial

import numpy as np
import pyvista

from .. import pick_types
from ..beamformer import apply_lcmv, make_lcmv
from ..bem import (
    ConductorModel,
    _ensure_bem_surfaces,
    fit_sphere_to_headshape,
    make_sphere_model,
)
from ..cov import make_ad_hoc_cov
from ..dipole import fit_dipole
from ..forward import convert_forward_solution, make_field_map, make_forward_dipole
from ..minimum_norm import apply_inverse, make_inverse_operator
from ..surface import _normal_orth
from ..transforms import (
    _get_trans,
    _get_transforms_to_coord_frame,
    transform_surface_to,
)
from ..utils import _check_option, fill_doc, verbose
from ..viz import EvokedField, create_3d_figure
from ..viz._3d import _plot_head_surface, _plot_sensors_3d
from ..viz.ui_events import subscribe
from ..viz.utils import _get_color_list


@fill_doc
@verbose
def dipolefit(
    evoked,
    cov=None,
    bem=None,
    initial_time=None,
    trans=None,
    show_density=True,
    subject=None,
    subjects_dir=None,
    n_jobs=None,
    verbose=None,
):
    """GUI for interactive dipole fitting, inspired by MEGIN's XFit program.

    Parameters
    ----------
    evoked : instance of Evoked
        Evoked data to show fieldmap of and fit dipoles to.
    cov : instance of Covariance | None
        Noise covariance matrix. If None, an ad-hoc covariance matrix is used.
    bem : instance of ConductorModel | None
        Boundary element model. If None, a spherical model is used.
    initial_time : float | None
        Initial time point to show. If None, the time point of the maximum
        field strength is used.
    trans : instance of Transform | None
        The transformation from head coordinates to MRI coordinates. If None,
        the identity matrix is used.
    show_density : bool
        Whether to show the density of the fieldmap.
    subject : str | None
        The subject name. If None, no MRI data is shown.
    %(subjects_dir)s
    %(n_jobs)s
    %(verbose)s
    """
    return DipoleFitUI(
        evoked=evoked,
        cov=cov,
        bem=bem,
        initial_time=initial_time,
        trans=trans,
        show_density=show_density,
        subject=subject,
        subjects_dir=subjects_dir,
        n_jobs=n_jobs,
        verbose=verbose,
    )


@fill_doc
class DipoleFitUI:
    """GUI for interactive dipole fitting, inspired by MEGIN's XFit program.

    Parameters
    ----------
    evoked : instance of Evoked
        Evoked data to show fieldmap of and fit dipoles to.
    cov : instance of Covariance | None
        Noise covariance matrix. If None, an ad-hoc covariance matrix is used.
    cov_data : instance of Covariance | None
        Data covariance matrix. If None, LCMV method will be unavailable.
    bem : instance of ConductorModel | None
        Boundary element model. If None, a spherical model is used.
    initial_time : float | None
        Initial time point to show. If None, the time point of the maximum
        field strength is used.
    trans : instance of Transform | None
        The transformation from head coordinates to MRI coordinates. If None,
        the identity matrix is used.
    show_density : bool
        Whether to show the density of the fieldmap.
    subject : str | None
        The subject name. If None, no MRI data is shown.
    %(subjects_dir)s
    %(n_jobs)s
    %(verbose)s
    """

    def __init__(
        self,
        evoked,
        cov=None,
        cov_data=None,
        bem=None,
        initial_time=None,
        trans=None,
        show_density=True,
        subject=None,
        subjects_dir=None,
        ch_type=None,
        n_jobs=None,
        verbose=None,
    ):
        if cov is None:
            cov = make_ad_hoc_cov(evoked.info)
        if bem is None:
            bem = make_sphere_model("auto", "auto", evoked.info)
        bem = _ensure_bem_surfaces(bem, extra_allow=(ConductorModel, None))
        field_map = make_field_map(
            evoked,
            ch_type=ch_type,
            trans=trans,
            origin=bem["r0"] if bem["is_sphere"] else "auto",
            subject=subject,
            subjects_dir=subjects_dir,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        if initial_time is None:
            data = evoked.copy().pick(field_map[0]["ch_names"]).data
            initial_time = evoked.times[np.argmax(np.mean(data**2, axis=0))]

        # Get transforms to convert all the various meshes to head space
        head_mri_t = _get_trans(trans, "head", "mri")[0]
        to_cf_t = _get_transforms_to_coord_frame(
            evoked.info, head_mri_t, coord_frame="head"
        )

        # Transform the fieldmap surfaces to head space if needed
        if trans is not None:
            for fm in field_map:
                fm["surf"] = transform_surface_to(
                    fm["surf"], "head", [to_cf_t["mri"], to_cf_t["head"]], copy=False
                )

        self._actors = dict()
        self._arrows = list()
        self._bem = bem
        self._ch_type = ch_type
        self._cov = cov
        self._cov_data = cov_data
        self._current_time = initial_time
        self._dipoles = list()
        self._evoked = evoked
        self._field_map = field_map
        self._fig_sensors = None
        self._multi_dipole_method = "MNE"
        self._n_jobs = n_jobs
        self._show_density = show_density
        self._subjects_dir = subjects_dir
        self._subject = subject
        self._time_line = None
        self._to_cf_t = to_cf_t
        self._trans = trans
        self._verbose = verbose

        # Configure the GUI
        self._renderer = self._configure_main_display()
        self._configure_dock()

    def _configure_main_display(self):
        """Configure main 3D display of the GUI."""
        fig = create_3d_figure((1500, 1020), bgcolor="white", show=True)
        fig = EvokedField(
            self._evoked,
            self._field_map,
            time=self._current_time,
            interpolation="linear",
            alpha=0,
            show_density=self._show_density,
            foreground="black",
            background="white",
            fig=fig,
        )
        fig.separate_canvas = False  # needed to plot the timeline later
        fig.set_contour_line_width(2)
        fig._renderer.set_camera(
            focalpoint=fit_sphere_to_headshape(self._evoked.info)[1]
        )

        for surf_map in fig._surf_maps:
            if surf_map["map_kind"] == "meg":
                helmet_mesh = surf_map["mesh"]
                helmet_mesh._polydata.compute_normals()  # needed later
                self._actors["helmet"] = helmet_mesh._actor
            elif surf_map["map_kind"] == "eeg":
                head_mesh = surf_map["mesh"]
                head_mesh._polydata.compute_normals()  # needed later
                self._actors["head"] = head_mesh._actor

        show_meg = (self._ch_type is None or self._ch_type == "meg") and any(
            [m["kind"] == "meg" for m in self._field_map]
        )
        show_eeg = (self._ch_type is None or self._ch_type == "eeg") and any(
            [m["kind"] == "eeg" for m in self._field_map]
        )
        meg_picks = pick_types(self._evoked.info, meg=show_meg)
        eeg_picks = pick_types(self._evoked.info, meg=False, eeg=show_eeg)
        picks = np.concatenate((meg_picks, eeg_picks))
        self._ch_names = [self._evoked.ch_names[i] for i in picks]

        print(f"{show_meg=} {show_eeg=}")

        for m in self._field_map:
            if m["kind"] == "eeg":
                head_surf = m["surf"]
                break
        else:
            self._actors["head"], _, head_surf = _plot_head_surface(
                renderer=fig._renderer,
                head="head",
                subject=self._subject,
                subjects_dir=self._subjects_dir,
                bem=self._bem,
                coord_frame="head",
                to_cf_t=self._to_cf_t,
                alpha=0.2,
            )

        sensors = _plot_sensors_3d(
            renderer=fig._renderer,
            info=self._evoked.info,
            to_cf_t=self._to_cf_t,
            picks=picks,
            meg=show_meg,
            eeg=["original"] if show_eeg else False,
            fnirs=False,
            warn_meg=False,
            head_surf=head_surf,
            units="m",
            sensor_alpha=dict(meg=0.1, eeg=1.0),
            orient_glyphs=False,
            scale_by_distance=False,
            project_points=False,
            surf=None,
            check_inside=None,
            nearest=None,
            # sensor_colors=dict(meg="white", eeg="white"),
            sensor_colors=dict(
                meg=["white" for _ in meg_picks],
                eeg=["white" for _ in eeg_picks],
            ),
        )
        self._actors["sensors"] = list()
        for s in sensors.values():
            self._actors["sensors"].extend(s)

        subscribe(fig, "time_change", self._on_time_change)
        self._fig = fig
        return fig._renderer

    def _configure_dock(self):
        """Configure the left and right dock areas of the GUI."""
        r = self._renderer

        # Toggle buttons for various meshes
        layout = r._dock_add_group_box("Meshes")
        for actor_name in self._actors.keys():
            r._dock_add_check_box(
                name=actor_name,
                value=True,
                callback=partial(self.toggle_mesh, name=actor_name),
                layout=layout,
            )

        # Add view buttons
        layout = r._dock_add_group_box("Views")
        hlayout = None
        views = zip(
            [7, 8, 9, 4, 5, 6, 1, 2, 3],  # numpad order
            ["🢆", "🢃", "🢇", "🢂", "⊙", "🢀", "🢅", "🢁", "🢄"],
        )
        for i, (view, label) in enumerate(views):
            if i % 3 == 0:  # show in groups of 3
                hlayout = r._dock_add_layout(vertical=False)
                r._layout_add_widget(layout, hlayout)
            r._dock_add_button(
                label,
                callback=partial(self._set_view, view=view),
                layout=hlayout,
                style="pushbutton",
            )
            r.plotter.add_key_event(str(view), partial(self._set_view, view=view))

        # Right dock
        r._dock_initialize(name="Dipole fitting", area="right")
        r._dock_add_button("Sensor data", self._on_sensor_data)
        r._dock_add_button("Fit dipole", self._on_fit_dipole)
        methods = ["MNE", "Single-dipole"]
        if self._cov_data is not None:
            methods.append("LCMV")
        r._dock_add_combo_box(
            "Dipole model",
            value="MNE",
            rng=methods,
            callback=self._on_select_method,
        )
        self._dipole_box = r._dock_add_group_box(name="Dipoles")
        r._dock_add_stretch()

    def toggle_mesh(self, name, show=None):
        """Toggle a mesh on or off.

        Parameters
        ----------
        name : str
            Name of the mesh to toggle.
        show : bool | None
            Whether to show the mesh. If None, the visibility of the mesh is toggled.
        """
        _check_option("name", name, self._actors.keys())
        actors = self._actors[name]
        # self._actors[name] is sometimes a list and sometimes not. Make it
        # always be a list to simplify the code.
        if not isinstance(actors, list):
            actors = [actors]
        if show is None:
            show = not actors[0].GetVisibility()
        for act in actors:
            act.SetVisibility(show)
        self._renderer._update()

    def _set_view(self, view):
        kwargs = dict()
        if view == 1:
            kwargs = dict(azimuth=-135, roll=45, elevation=60, distance="auto")
        elif view == 2:
            kwargs = dict(azimuth=270, roll=180, elevation=90, distance="auto")
        elif view == 3:
            kwargs = dict(azimuth=-45, roll=-45, elevation=60, distance="auto")
        elif view == 4:
            kwargs = dict(azimuth=180, roll=90, elevation=90, distance="auto")
        elif view == 5:
            kwargs = dict(azimuth=0, roll=0, elevation=0, distance="auto")
        elif view == 6:
            kwargs = dict(azimuth=0, roll=-90, elevation=90, distance="auto")
        elif view == 7:
            kwargs = dict(azimuth=135, roll=90, elevation=60, distance="auto")
        elif view == 8:
            kwargs = dict(azimuth=90, roll=0, elevation=90, distance="auto")
        elif view == 9:
            kwargs = dict(azimuth=45, roll=-90, elevation=60, distance="auto")
        self._renderer.set_camera(**kwargs)

    def _on_time_change(self, event):
        new_time = np.clip(event.time, self._evoked.times[0], self._evoked.times[-1])
        self._current_time = new_time
        if self._time_line is not None:
            self._time_line.set_xdata([new_time])
            self._renderer._mplcanvas.update_plot()
        self._update_arrows()

    def _on_sensor_data(self):
        """Show sensor data and allow sensor selection."""
        if self._fig_sensors is not None:
            return
        fig = self._evoked.plot_topo(select=True)
        fig.canvas.mpl_connect("close_event", self._on_sensor_data_close)
        subscribe(fig, "channels_select", self._on_channels_select)
        self._fig_sensors = fig

    def _on_sensor_data_close(self, event):
        """Handle closing of the sensor selection window."""
        self._fig_sensors = None
        if "sensors" in self._actors:
            for act in self._actors["sensors"]:
                act.prop.SetColor(1, 1, 1)
            self._renderer._update()

    def _on_channels_select(self, event):
        """Color selected sensor meshes."""
        selected_channels = set(event.ch_names)
        if "sensors" in self._actors:
            for act, ch_name in zip(self._actors["sensors"], self._ch_names):
                if ch_name in selected_channels:
                    act.prop.SetColor(0, 1, 0)
                else:
                    act.prop.SetColor(1, 1, 1)
        self._renderer._update()

    def _on_fit_dipole(self):
        """Fit a single dipole."""
        evoked_picked = self._evoked.copy()
        cov_picked = self._cov
        if self._fig_sensors is not None:
            picks = self._fig_sensors.lasso.selection
            if len(picks) > 0:
                evoked_picked = evoked_picked.copy().pick(picks)
                evoked_picked.info.normalize_proj()
                cov_picked = cov_picked.copy().pick_channels(picks, ordered=False)
                cov_picked["projs"] = evoked_picked.info["projs"]
        evoked_picked.crop(self._current_time, self._current_time)

        dip = fit_dipole(
            evoked_picked,
            cov_picked,
            self._bem,
            trans=self._trans,
            min_dist=0,
            verbose=False,
        )[0]

        # Coordinates needed to draw the big arrow on the helmet.
        helmet_coords, helmet_pos = self._get_helmet_coords(dip)

        # Collect all relevant information on the dipole in a dict
        colors = _get_color_list()
        dip_num = len(self._dipoles)
        dip_name = f"dip{dip_num}"
        dip_color = colors[dip_num % len(colors)]
        if helmet_coords is not None:
            arrow_mesh = pyvista.PolyData(*_arrow_mesh())
        else:
            arrow_mesh = None
        dipole_dict = dict(
            active=True,
            arrow_actor=None,
            arrow_mesh=arrow_mesh,
            color=dip_color,
            dip=dip,
            fix_ori=True,
            fix_position=True,
            helmet_coords=helmet_coords,
            helmet_pos=helmet_pos,
            name=dip_name,
            num=dip_num,
        )
        self._dipoles.append(dipole_dict)

        # Add a row to the dipole list
        r = self._renderer
        hlayout = r._dock_add_layout(vertical=False)
        r._dock_add_check_box(
            name=dip_name,
            value=True,
            callback=partial(self._on_dipole_toggle, dip_name=dip_name),
            layout=hlayout,
        )
        r._dock_add_check_box(
            name="Fix pos",
            value=True,
            callback=partial(self._on_dipole_toggle_fix_position, dip_name=dip_name),
            layout=hlayout,
        )
        r._dock_add_check_box(
            name="Fix ori",
            value=True,
            callback=partial(self._on_dipole_toggle_fix_orientation, dip_name=dip_name),
            layout=hlayout,
        )
        r._layout_add_widget(self._dipole_box, hlayout)

        # Compute dipole timecourse, update arrow size
        self._fit_timecourses()

        # Show the dipole and arrow in the 3D view
        self._renderer.plotter.add_arrows(
            dip.pos[0], dip.ori[0], color=dip_color, mag=0.05
        )
        if arrow_mesh is not None:
            dipole_dict["arrow_actor"] = self._renderer.plotter.add_mesh(
                arrow_mesh, color=dip_color
            )

    def _get_helmet_coords(self, dip):
        """Compute the coordinate system used for drawing the big arrows on the helmet.

        In this coordinate system, Z is normal to the helmet surface, and XY
        are tangential to the helmet surface.
        """
        if "helmet" not in self._actors:
            return None, None

        # Get the closest vertex (=point) of the helmet mesh
        dip_pos = dip.pos[0]
        helmet = self._actors["helmet"].GetMapper().GetInput()
        distances = ((helmet.points - dip_pos) * helmet.point_normals).sum(axis=1)
        closest_point = np.argmin(distances)

        # Compute the position of the projected dipole on the helmet
        norm = helmet.point_normals[closest_point]
        helmet_pos = dip_pos + (distances[closest_point] + 0.003) * norm

        # Create a coordinate system where X and Y are tangential to the helmet
        helmet_coords = _normal_orth(norm)

        return helmet_coords, helmet_pos

    def _fit_timecourses(self):
        """Compute dipole timecourses using a multi-dipole model."""
        active_dips = [d for d in self._dipoles if d["active"]]
        if len(active_dips) == 0:
            return

        fwd, _ = make_forward_dipole(
            [d["dip"] for d in active_dips],
            self._bem,
            self._evoked.info,
            trans=self._trans,
        )
        fwd = convert_forward_solution(fwd, surf_ori=False)

        if self._multi_dipole_method == "MNE":
            inv = make_inverse_operator(
                self._evoked.info,
                fwd,
                self._cov,
                fixed=False,
                loose=1.0,
                depth=0,
            )
            stc = apply_inverse(
                self._evoked, inv, method="MNE", lambda2=0, pick_ori="vector"
            )
            timecourses = np.linalg.norm(stc.data, axis=1)
            orientations = stc.data / timecourses[:, np.newaxis, :]
        elif self._multi_dipole_method == "LCMV":
            lcmv = make_lcmv(
                self._evoked.info, fwd, self._cov_data, reg=0.05, noise_cov=self._cov
            )
            stc = apply_lcmv(self._evoked, lcmv)
            timecourses = stc.data
        elif self._multi_dipole_method == "Single-dipole":
            timecourses = list()
            for dip in active_dips:
                dip_timecourse = fit_dipole(
                    self._evoked,
                    self._cov,
                    self._bem,
                    pos=dip["dip"].pos[0],
                    ori=dip["dip"].ori[0],
                    trans=self._trans,
                    verbose=False,
                )[0].data[0]
                timecourses.append(dip_timecourse)

        # Update matplotlib canvas at the bottom of the window
        canvas = self._setup_mplcanvas()
        ymin, ymax = 0, 0
        for d, ori, timecourse in zip(active_dips, orientations, timecourses):
            d["ori"] = ori
            d["timecourse"] = timecourse
            if "line_artist" in d:
                d["line_artist"].set_ydata(timecourse)
            else:
                d["line_artist"] = canvas.plot(
                    self._evoked.times,
                    timecourse,
                    label=d["name"],
                    color=d["color"],
                )
            ymin = min(ymin, 1.1 * timecourse.min())
            ymax = max(ymax, 1.1 * timecourse.max())
        canvas.axes.set_ylim(ymin, ymax)
        canvas.update_plot()
        self._update_arrows()

    def _update_arrows(self):
        """Update the arrows to have the correct size and orientation."""
        active_dips = [d for d in self._dipoles if d["active"]]
        if len(active_dips) == 0:
            return
        orientations = [d["ori"] for d in active_dips]
        timecourses = [d["timecourse"] for d in active_dips]
        arrow_scaling = 0.05 / np.max(np.abs(timecourses))
        for d, ori, timecourse in zip(active_dips, orientations, timecourses):
            helmet_coords = d["helmet_coords"]
            if helmet_coords is None:
                continue
            dip_ori = [
                np.interp(self._current_time, self._evoked.times, o) for o in ori
            ]
            dip_moment = np.interp(self._current_time, self._evoked.times, timecourse)
            arrow_size = dip_moment * arrow_scaling
            arrow_mesh = d["arrow_mesh"]

            # Project the orientation of the dipole tangential to the helmet
            dip_ori_tan = helmet_coords[:2] @ dip_ori @ helmet_coords[:2]

            # Rotate the coordinate system such that Y lies along the dipole
            # orientation, now we have our desired coordinate system for the
            # arrows.
            arrow_coords = np.array(
                [np.cross(dip_ori_tan, helmet_coords[2]), dip_ori_tan, helmet_coords[2]]
            )
            arrow_coords /= np.linalg.norm(arrow_coords, axis=1, keepdims=True)

            # Update the arrow mesh to point in the right directions
            arrow_mesh.points = (_arrow_mesh()[0] * arrow_size) @ arrow_coords
            arrow_mesh.points += d["helmet_pos"]
        self._renderer._update()

    def _on_select_method(self, method):
        """Select the method to use for multi-dipole timecourse fitting."""
        self._multi_dipole_method = method
        self._fit_timecourses()

    def _on_dipole_toggle(self, active, dip_name):
        """Toggle a dipole on or off."""
        for d in self._dipoles:
            if d["name"] == dip_name:
                dipole = d
                break
        else:
            raise ValueError(f"Unknown dipole {dip_name}")
        active = bool(active)
        dipole["active"] = active
        dipole["line_artist"].set_visible(active)
        dipole["arrow_actor"].visibility = active
        self._fit_timecourses()
        self._renderer._update()
        self._renderer._mplcanvas.update_plot()

    def _on_dipole_toggle_fix_position(self, fix, dip_name):
        """Fix dipole position when fitting timecourse."""
        for d in self._dipoles:
            if d["name"] == dip_name:
                dipole = d
                break
        else:
            raise ValueError(f"Unknown dipole {dip_name}")
        dipole["fix_position"] = bool(fix)
        self._fit_timecourses()

    def _on_dipole_toggle_fix_orientation(self, fix, dip_name):
        """Fix dipole orientation when fitting timecourse."""
        for d in self._dipoles:
            if d["name"] == dip_name:
                dipole = d
                break
        else:
            raise ValueError(f"Unknown dipole {dip_name}")
        dipole["fix_ori"] = bool(fix)
        self._fit_timecourses()

    def _setup_mplcanvas(self):
        """Configure the matplotlib canvas at the bottom of the window."""
        if self._renderer._mplcanvas is None:
            self._renderer._mplcanvas = self._renderer._window_get_mplcanvas(
                self._fig, 0.3, False, False
            )
            self._renderer._window_adjust_mplcanvas_layout()
        if self._time_line is None:
            self._time_line = self._renderer._mplcanvas.plot_time_line(
                self._current_time,
                label="time",
                color="black",
            )
        return self._renderer._mplcanvas


def _arrow_mesh():
    """Obtain a PyVista mesh of an arrow."""
    vertices = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.3, 0.7, 0.0],
            [0.1, 0.7, 0.0],
            [0.1, -1.0, 0.0],
            [-0.1, -1.0, 0.0],
            [-0.1, 0.7, 0.0],
            [-0.3, 0.7, 0.0],
        ]
    )
    faces = np.array([[7, 0, 1, 2, 3, 4, 5, 6]])
    return vertices, faces
