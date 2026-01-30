# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import pyvista

from .._fiff.pick import pick_types
from ..bem import (
    ConductorModel,
    _ensure_bem_surfaces,
    make_sphere_model,
)
from ..cov import _ensure_cov, make_ad_hoc_cov
from ..dipole import Dipole, fit_dipole
from ..evoked import Evoked
from ..forward import convert_forward_solution, make_field_map
from ..forward._make_forward import _ForwardModeler
from ..minimum_norm import apply_inverse, make_inverse_operator
from ..source_estimate import (
    SourceEstimate,
    _BaseSurfaceSourceEstimate,
    read_source_estimate,
)
from ..source_space import setup_volume_source_space
from ..surface import _normal_orth
from ..transforms import _get_trans, _get_transforms_to_coord_frame, apply_trans
from ..utils import _check_option, _validate_type, fill_doc, logger, verbose
from ..viz import EvokedField, create_3d_figure
from ..viz._3d import _plot_head_surface, _plot_sensors_3d
from ..viz.backends._utils import _qt_app_exec
from ..viz.ui_events import link, subscribe
from ..viz.utils import _get_color_list


@fill_doc
class DipoleFitUI:
    """GUI for interactive dipole fitting, inspired by MEGIN's XFit program.

    Parameters
    ----------
    evoked : instance of Evoked | path-like
        Evoked data to show fieldmap of and fit dipoles to.
    %(baseline_evoked)s
    cov : instance of Covariance | "baseline" | None
        Noise covariance matrix. If ``None``, an ad-hoc covariance matrix is used with
        default values for the diagonal elements (see Notes). If ``"baseline"``, the
        diagonal elements is estimated from the baseline period of the evoked data.
    bem : instance of ConductorModel | None
        Boundary element model to use in forward calculations. If ``None``, a spherical
        model is used.
    initial_time : float | None
        Initial time point to show. If ``None``, the time point of the maximum field
        strength is used.
    trans : instance of Transform | None
        The transformation from head coordinates to MRI coordinates. If ``None``,
        the identity matrix is used and everything will be done in head coordinates.
    stc : instance of SourceEstimate | None
        An optional distributed source estimate to show alongside the fieldmap. The time
        samples need to match those of the evoked data.
    subject : str | None
        The subject name. If ``None``, no MRI data is shown.
    %(subjects_dir)s
    %(rank)s
    show_density : bool
        Whether to show the density of the fieldmap.
    ch_type : "meg" | "eeg" | None
        Type of channels to use for the dipole fitting. By default (``None``) both MEG
        and EEG channels will be used.
    %(n_jobs)s
    show : bool
        Show the GUI if True.
    block : bool
        Whether to halt program execution until the figure is closed.
    %(verbose)s

    Notes
    -----
    When using ``cov=None`` the default noise values are 5 fT/cm, 20 fT, and 0.2 µV for
    gradiometers, magnetometers, and EEG channels respectively.
    """

    def __init__(
        self,
        evoked=None,
        *,
        baseline=None,
        cov=None,
        bem=None,
        initial_time=None,
        trans=None,
        stc=None,
        subject=None,
        subjects_dir=None,
        rank="info",
        show_density=True,
        ch_type=None,
        n_jobs=None,
        show=True,
        block=False,
        verbose=None,
    ):
        _validate_type(evoked, Evoked, "evoked")
        evoked.apply_baseline(baseline)

        if cov is None:
            logger.info("Using ad-hoc noise covariance.")
            cov = make_ad_hoc_cov(evoked.info)
        elif cov == "baseline":
            logger.info(
                f"Estimating noise covariance from baseline ({evoked.baseline[0]:.3f} "
                f"to {evoked.baseline[1]:.3f} seconds)."
            )
            std = dict()
            for typ in set(evoked.get_channel_types(only_data_chs=True)):
                baseline = evoked.copy().pick(typ).crop(*evoked.baseline)
                std[typ] = baseline.data.std(axis=1).mean()
            cov = make_ad_hoc_cov(evoked.info, std)
        else:
            cov = _ensure_cov(cov)

        if bem is None:
            bem = make_sphere_model("auto", "auto", evoked.info)
        bem = _ensure_bem_surfaces(bem, extra_allow=(ConductorModel, None))

        if ch_type is not None:
            evoked = evoked.copy().pick(ch_type)

        field_map = make_field_map(
            evoked,
            trans=trans,
            origin=bem["r0"] if bem["is_sphere"] else "auto",
            subject=subject,
            subjects_dir=subjects_dir,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        if initial_time is None:
            # Set initial time to moment of maximum field power.
            data = evoked.copy().pick(field_map[0]["ch_names"]).data
            initial_time = evoked.times[np.argmax(np.mean(data**2, axis=0))]

        if stc is not None:
            _validate_type(stc, ("path-like", _BaseSurfaceSourceEstimate), "stc")
            if not isinstance(stc, _BaseSurfaceSourceEstimate):
                stc = read_source_estimate(stc)

            if len(stc.times) != len(evoked.times) or not np.allclose(
                stc.times, evoked.times
            ):
                raise ValueError(
                    "The time samples of the source estimate do not match those of the "
                    "evoked data."
                )
            if trans is None:
                raise ValueError(
                    "`trans` cannot be `None` when showing the fieldlines in "
                    "combination with a source estimate."
                )

        # Get transforms to convert all the various meshes to MRI space.
        head_mri_t = _get_trans(trans, "head", "mri")[0]
        to_cf_t = _get_transforms_to_coord_frame(
            evoked.info, head_mri_t, coord_frame="mri"
        )

        self.fwd = _ForwardModeler(
            info=evoked.info,
            trans=trans,
            bem=bem,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        # Initialize all the private attributes.
        self._actors = dict()
        self._bem = bem
        self._ch_type = ch_type
        self._cov = cov
        self._current_time = initial_time
        self._dipoles = dict()
        self._evoked = evoked
        self._field_map = field_map
        self._fig_sensors = None
        self._multi_dipole_method = "Multi dipole (MNE)"
        self._show_density = show_density
        self._stc = stc
        self._subjects_dir = subjects_dir
        self._subject = subject
        self._time_line = None
        self._head_mri_t = head_mri_t
        self._to_cf_t = to_cf_t
        self._rank = rank
        self._verbose = verbose
        self._n_jobs = n_jobs

        # Configure the GUI.
        self._renderer = self._configure_main_display(show)
        self._configure_dock()

        # must be done last
        if show:
            self._renderer.show()
        if block and self._renderer._kind != "notebook":
            _qt_app_exec(self._renderer.figure.store["app"])

    @property
    def dipoles(self):
        """A list of all the fitted dipoles that are enabled in the GUI."""
        return [d["dip"] for d in self._dipoles.values() if d["active"]]

    def _configure_main_display(self, show=True):
        """Configure main 3D display of the GUI."""
        fig = create_3d_figure((1500, 1020), bgcolor="white", show=show)

        self._fig_stc = None
        if self._stc is not None:
            kwargs = dict(
                subject=self._subject,
                subjects_dir=self._subjects_dir,
                hemi="both",
                time_viewer=False,
                initial_time=self._current_time,
                brain_kwargs=dict(units="m"),
                figure=fig,
            )
            if isinstance(self._stc, SourceEstimate):
                kwargs["surface"] = "white"
            fig = self._stc.plot(**kwargs)  # overwrite "fig" to be the STC plot
            self._fig_stc = fig
            self._actors["brain"] = fig._actors["data"]

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
        if self._stc is not None:
            link(self._fig_stc, fig)

        for surf_map in fig._surf_maps:
            if surf_map["map_kind"] == "meg":
                helmet_mesh = surf_map["mesh"]
                helmet_mesh._polydata.compute_normals()  # needed later
                helmet_mesh._actor.prop.culling = "back"
                self._actors["helmet"] = helmet_mesh._actor
                # For MEG fieldlines, we want to occlude the ones not facing us,
                # otherwise it's hard to interpret them. Since the "contours" object
                # does not support backface culling, we create an opaque mesh to put in
                # front of the contour lines with frontface culling.
                occl_surf = deepcopy(surf_map["surf"])
                occl_surf["rr"] -= 1e-3 * occl_surf["nn"]
                occl_act, _ = fig._renderer.surface(occl_surf, color="white")
                occl_act.prop.culling = "front"
                occl_act.prop.lighting = False
                self._actors["occlusion_surf"] = occl_act
            elif surf_map["map_kind"] == "eeg":
                head_mesh = surf_map["mesh"]
                head_mesh._polydata.compute_normals()  # needed later
                head_mesh._actor.prop.culling = "back"
                self._actors["head"] = head_mesh._actor

        show_meg = (self._ch_type is None or self._ch_type == "meg") and any(
            [m["kind"] == "meg" for m in self._field_map]
        )
        show_eeg = (self._ch_type is None or self._ch_type == "eeg") and any(
            [m["kind"] == "eeg" for m in self._field_map]
        )
        meg_picks = pick_types(self._evoked.info, meg=show_meg, ref_meg=False)
        eeg_picks = pick_types(self._evoked.info, meg=False, eeg=show_eeg)
        picks = np.concatenate((meg_picks, eeg_picks))
        self._ch_names = [self._evoked.ch_names[i] for i in picks]

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
                coord_frame="mri",
                to_cf_t=self._to_cf_t,
                alpha=0.2,
            )
            self._actors["head"].prop.culling = "back"

        sensors = _plot_sensors_3d(
            renderer=fig._renderer,
            info=self._evoked.info,
            to_cf_t=self._to_cf_t,
            picks=picks,
            meg=["sensors"] if show_meg else False,
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
            sensor_colors=dict(
                meg=["gray" for _ in meg_picks],
                eeg=["white" for _ in eeg_picks],
            ),
        )
        self._actors["sensors"] = list()
        for s in sensors.values():
            self._actors["sensors"].extend(s)

        # Adjust camera
        fig._renderer.set_camera(
            azimuth=180, elevation=90, roll=90, distance=0.55, focalpoint=[0, 0, 0.03]
        )

        subscribe(fig, "time_change", self._on_time_change)
        self._fig = fig
        return fig._renderer

    def _configure_dock(self):
        """Configure the left and right dock areas of the GUI."""
        r = self._renderer

        # Toggle buttons for various meshes
        layout = r._dock_add_group_box("Meshes")
        for actor_name in self._actors.keys():
            if actor_name == "occlusion_surf":
                continue
            r._dock_add_check_box(
                name=actor_name,
                value=True,
                callback=partial(self.toggle_mesh, name=actor_name),
                layout=layout,
            )

        # Right dock
        r._dock_initialize(name="Dipole fitting", area="right")
        r._dock_add_button("Sensor data", self._on_sensor_data)
        r._dock_add_button("Fit dipole", self._on_fit_dipole)
        methods = ["Multi dipole (MNE)", "Single dipole"]
        r._dock_add_combo_box(
            "Dipole model",
            value="Multi dipole (MNE)",
            rng=methods,
            callback=self._on_select_method,
        )
        self._dipole_box = r._dock_add_group_box(name="Dipoles")
        self._save_button = r._dock_add_file_button(
            name="save_dipoles",
            desc="Save dipoles",
            save=True,
            func=self.save,
            tooltip="Save the dipoles to disk",
            filter_="Dipole files (*.dip  *.bdip)",
            initial_directory=".",
        )
        self._save_button.set_enabled(False)
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
        print("sensor window closed.")

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
        cov_picked = self._cov.copy()
        if self._fig_sensors is not None:
            picks = self._fig_sensors.lasso.selection
            if len(picks) > 0:
                evoked_picked = evoked_picked.pick(picks)
                evoked_picked.info.normalize_proj()
                cov_picked = cov_picked.pick_channels(picks, ordered=False)
                cov_picked["projs"] = evoked_picked.info["projs"]
        evoked_picked.crop(self._current_time, self._current_time)

        dip = fit_dipole(
            evoked_picked,
            cov_picked,
            self._bem,
            trans=self._head_mri_t,
            rank=self._rank,
            n_jobs=self._n_jobs,
            verbose=False,
        )[0]

        self.add_dipole(dip)

    def add_dipole(self, dipole, name=None):
        """Add a dipole (or multiple dipoles) to the GUI.

        Parameters
        ----------
        dipole : Dipole
            The dipole to add. If the ``Dipole`` object defines multiple dipoles, they
            will all be added.
        name : str | list of str | None
            The name of the dipole. When the ``Dipole`` object defines multiple dipoles,
            this should be a list containing the name for each dipole. When ``None``,
            the ``.name`` attribute of the ``Dipole`` object itself will be used.
        """
        _validate_type(name, (str, list, None), "name")
        if isinstance(name, str):
            names = [name]
        elif name is None:
            # Try to obtain names from `dipole.name`. When multiple dipoles are saved,
            # the names are concatenated with `;` marks.
            if dipole.name is None:
                names = [None] * len(dipole)
            elif len(dipole.name.split(";")) == len(dipole):
                names = dipole.name.split(";")
            else:
                names = [dipole.name] * len(dipole)
        else:
            names = name
        if len(names) != len(dipole):
            raise ValueError(
                f"Number of names ({len(names)}) does not match the number of dipoles "
                f"({len(dipole)})."
            )

        # Ensure orientations are unit vectors. Due to rounding issues this is sometimes
        # not the case.
        dipole._ori /= np.linalg.norm(dipole._ori, axis=1, keepdims=True)

        new_dipoles = list()
        for dip, name in zip(dipole, names):
            # Coordinates needed to draw the big arrow on the helmet.
            helmet_coords, helmet_pos = self._get_helmet_coords(dip)

            # Collect all relevant information on the dipole in a dict.
            colors = _get_color_list()
            if len(self._dipoles) == 0:
                dip_num = 0
            else:
                dip_num = max(self._dipoles.keys()) + 1
            if name is None:
                dip.name = f"dip{dip_num}"
            else:
                dip.name = name
            dip_color = colors[dip_num % len(colors)]
            if helmet_coords is not None:
                arrow_mesh = pyvista.PolyData(*_arrow_mesh())
            else:
                arrow_mesh = None
            dipole_dict = dict(
                active=True,
                brain_arrow_actor=None,
                helmet_arrow_actor=None,
                arrow_mesh=arrow_mesh,
                color=dip_color,
                dip=dip,
                fix_ori=True,
                fix_position=True,
                helmet_coords=helmet_coords,
                helmet_pos=helmet_pos,
                num=dip_num,
                # fit_time=self._current_time,
            )
            self._dipoles[dip_num] = dipole_dict

            # Add a row to the dipole list
            r = self._renderer
            hlayout = r._dock_add_layout(vertical=False)
            widgets = []
            widgets.append(
                r._dock_add_check_box(
                    name="",
                    value=True,
                    callback=partial(self._on_dipole_toggle, dip_num=dip_num),
                    layout=hlayout,
                )
            )
            widgets.append(
                r._dock_add_text(
                    name=dip.name,
                    value=dip.name,
                    placeholder="name",
                    callback=partial(self._on_dipole_set_name, dip_num=dip_num),
                    layout=hlayout,
                )
            )
            widgets.append(
                r._dock_add_check_box(
                    name="Fix ori",
                    value=True,
                    callback=partial(
                        self._on_dipole_toggle_fix_orientation, dip_num=dip_num
                    ),
                    layout=hlayout,
                )
            )
            widgets.append(
                r._dock_add_button(
                    name="",
                    icon="clear",
                    callback=partial(self._on_dipole_delete, dip_num=dip_num),
                    layout=hlayout,
                )
            )
            dipole_dict["widgets"] = widgets
            r._layout_add_widget(self._dipole_box, hlayout)
            new_dipoles.append(dipole_dict)

        # Show the dipoles and arrows in the 3D view. Only do this after
        # `_fit_timecourses` so that they have the correct size straight away.
        self._fit_timecourses()
        for dipole_dict in new_dipoles:
            dip = dipole_dict["dip"]
            dipole_dict["brain_arrow_actor"] = self._renderer.plotter.add_arrows(
                apply_trans(self._head_mri_t, dip.pos[0]),
                apply_trans(self._head_mri_t, dip.ori[0]),
                color=dipole_dict["color"],
                mag=0.05,
            )
            if dipole_dict["arrow_mesh"] is not None:
                dipole_dict["helmet_arrow_actor"] = self._renderer.plotter.add_mesh(
                    dipole_dict["arrow_mesh"],
                    color=dipole_dict["color"],
                    culling="front",
                )
        self._update_arrows()

    def _get_helmet_coords(self, dip):
        """Compute the coordinate system used for drawing the big arrows on the helmet.

        In this coordinate system, Z is normal to the helmet surface, and XY
        are tangential to the helmet surface.
        """
        if "helmet" not in self._actors:
            return None, None

        # Get the closest vertex (=point) of the helmet mesh
        dip_pos = apply_trans(self._head_mri_t, dip.pos[0])
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
        """Compute (or re-compute) dipole timecourses.

        Called whenever something changes to the multi-dipole situation, i.e. a dipole
        is added, removed, (de-)activated or the "Fix pos" box is toggled.
        """
        self._save_button.set_enabled(len(self.dipoles) > 0)
        active_dips = [d for d in self._dipoles.values() if d["active"]]
        if len(active_dips) == 0:
            return

        if self._multi_dipole_method == "Multi dipole (MNE)":
            for d in active_dips:
                print(d["dip"], d["dip"].pos, d["dip"].ori)
            this_src = setup_volume_source_space(
                "sample",
                pos=dict(
                    rr=apply_trans(
                        self._head_mri_t,
                        np.vstack([d["dip"].pos[0] for d in active_dips]),
                    ),
                    nn=apply_trans(
                        self._head_mri_t,
                        np.vstack([d["dip"].ori[0] for d in active_dips]),
                    ),
                ),
            )
            this_fwd = self.fwd.compute(this_src)
            # this_fwd, _ = make_forward_dipole(
            #     [d["dip"] for d in active_dips],
            #     self._bem,
            #     self._evoked.info,
            #     trans=self._head_mri_t,
            #     n_jobs=self._n_jobs,
            # )
            this_fwd = convert_forward_solution(this_fwd, surf_ori=False)

            inv = make_inverse_operator(
                self._evoked.info,
                # fwd,
                this_fwd,
                self._cov,
                fixed=False,
                loose=1.0,
                depth=0,
                rank=self._rank,
            )
            stc = apply_inverse(
                self._evoked,
                inv,
                method="MNE",
                lambda2=1e-6,
                pick_ori="vector",
            )

            timecourses = stc.magnitude().data
            orientations = (stc.data / timecourses[:, np.newaxis, :]).transpose(0, 2, 1)
            fixed_timecourses = stc.project(
                np.array([dip["dip"].ori[0] for dip in active_dips])
            )[0].data

            for i, dip in enumerate(active_dips):
                if dip["fix_ori"]:
                    dip["timecourse"] = fixed_timecourses[i]
                    dip["orientation"] = dip["dip"].ori.repeat(len(stc.times), axis=0)
                else:
                    dip["timecourse"] = timecourses[i]
                    dip["orientation"] = orientations[i]
        elif self._multi_dipole_method == "Single dipole":
            for dip in active_dips:
                dip_with_timecourse, _ = fit_dipole(
                    self._evoked,
                    self._cov,
                    self._bem,
                    pos=dip["dip"].pos[0],  # position is always fixed
                    ori=dip["dip"].ori[0] if dip["fix_ori"] else None,
                    trans=self._head_mri_t,
                    rank=self._rank,
                    n_jobs=self._n_jobs,
                    verbose=True,
                )
                if dip["fix_ori"]:
                    dip["timecourse"] = dip_with_timecourse.data[0]
                    dip["orientation"] = dip["dip"].ori.repeat(
                        len(dip_with_timecourse.times), axis=0
                    )
                else:
                    dip["timecourse"] = dip_with_timecourse.amplitude
                    dip["orientation"] = dip_with_timecourse.ori

        # Update matplotlib canvas at the bottom of the window
        canvas = self._setup_mplcanvas()
        ymin, ymax = 0, 0
        for dip in active_dips:
            if "line_artist" in dip:
                dip["line_artist"].set_ydata(dip["timecourse"])
            else:
                dip["line_artist"] = canvas.plot(
                    self._evoked.times,
                    dip["timecourse"],
                    label=dip["dip"].name,
                    color=dip["color"],
                )
            ymin = min(ymin, 1.1 * dip["timecourse"].min())
            ymax = max(ymax, 1.1 * dip["timecourse"].max())
        canvas.axes.set_ylim(ymin, ymax)
        canvas.update_plot()
        self._update_arrows()

    @verbose
    @fill_doc
    def save(self, fname, verbose=None):
        """Save the fitted dipoles to a file.

        Parameters
        ----------
        fname : path-like
            The name of the file. Should end in ``'.dip'`` to save in plain text format,
            or in ``'.bdip'`` to save in binary format.
        %(verbose)s
        """
        if len(self.dipoles) == 0:
            logger.info("No dipoles to save.")
            return

        logger.info(f"Saving dipoles as: {fname}")
        fname = Path(fname)

        # Pack the dipoles into a single mne.Dipole object.
        if all(d.khi2 is not None for d in self.dipoles):
            khi2 = np.array([d.khi2[0] for d in self.dipoles])
        else:
            khi2 = None

        if all(d.nfree is not None for d in self.dipoles):
            nfree = np.array([d.nfree[0] for d in self.dipoles])
        else:
            nfree = None

        dip = Dipole(
            times=np.array([d.times[0] for d in self.dipoles]),
            pos=np.array([d.pos[0] for d in self.dipoles]),
            amplitude=np.array([d.amplitude[0] for d in self.dipoles]),
            ori=np.array([d.ori[0] for d in self.dipoles]),
            gof=np.array([d.gof[0] for d in self.dipoles]),
            khi2=khi2,
            nfree=nfree,
            conf={
                key: np.array([d.conf[key][0] for d in self.dipoles])
                for key in self.dipoles[0].conf.keys()
            },
            name=";".join(d.name if hasattr(d, "name") else "" for d in self.dipoles),
        )
        dip.save(fname, overwrite=True, verbose=verbose)

    def _update_arrows(self):
        """Update the arrows to have the correct size and orientation."""
        active_dips = [d for d in self._dipoles.values() if d["active"]]
        if len(active_dips) == 0:
            return
        orientations = [dip["orientation"] for dip in active_dips]
        timecourses = [dip["timecourse"] for dip in active_dips]
        arrow_scaling = 0.05 / np.max(np.abs(timecourses))
        for dip, ori, timecourse in zip(active_dips, orientations, timecourses):
            helmet_coords = dip["helmet_coords"]
            if helmet_coords is None:
                continue

            dip_ori = apply_trans(
                self._head_mri_t,
                [np.interp(self._current_time, self._evoked.times, o) for o in ori.T],
            )
            dip_moment = np.interp(self._current_time, self._evoked.times, timecourse)
            arrow_size = dip_moment * arrow_scaling
            arrow_mesh = dip["arrow_mesh"]

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
            arrow_mesh.points += dip["helmet_pos"]
        self._renderer._update()

    def _on_select_method(self, method):
        """Select the method to use for multi-dipole timecourse fitting."""
        self._multi_dipole_method = method
        self._fit_timecourses()

    def _on_dipole_toggle(self, active, dip_num):
        """Toggle a dipole on or off."""
        dipole = self._dipoles[dip_num]
        active = bool(active)
        dipole["active"] = active
        dipole["line_artist"].set_visible(active)
        # Labels starting with "_" are hidden from the legend.
        dipole["line_artist"].set_label(("" if active else "_") + dipole["dip"].name)
        dipole["brain_arrow_actor"].visibility = active
        dipole["helmet_arrow_actor"].visibility = active
        self._fit_timecourses()
        self._renderer._update()
        self._renderer._mplcanvas.update_plot()

    def _on_dipole_set_name(self, name, dip_num):
        """Set the name of a dipole."""
        self._dipoles[dip_num]["dip"].name = name
        self._dipoles[dip_num]["line_artist"].set_label(name)
        self._renderer._mplcanvas.update_plot()

    def _on_dipole_toggle_fix_orientation(self, fix, dip_num):
        """Fix dipole orientation when fitting timecourse."""
        self._dipoles[dip_num]["fix_ori"] = bool(fix)
        self._fit_timecourses()

    def _on_dipole_delete(self, dip_num):
        """Delete previously fitted dipole."""
        dipole = self._dipoles[dip_num]
        dipole["line_artist"].remove()
        dipole["brain_arrow_actor"].visibility = False
        if dipole["helmet_arrow_actor"] is not None:  # no helmet arrow for EEG
            dipole["helmet_arrow_actor"].visibility = False
        for widget in dipole["widgets"]:
            widget.hide()
        del self._dipoles[dip_num]
        self._fit_timecourses()
        self._renderer._update()
        self._renderer._mplcanvas.update_plot()

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
    """Obtain a mesh of an arrow."""
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
