from functools import partial

import numpy as np
import pyvista

from .. import pick_types
from ..bem import (
    ConductorModel,
    _ensure_bem_surfaces,
    fit_sphere_to_headshape,
    make_sphere_model,
)
from ..cov import make_ad_hoc_cov
from ..dipole import fit_dipole
from ..forward import make_field_map, make_forward_dipole
from ..minimum_norm import apply_inverse, make_inverse_operator
from ..transforms import _get_trans, _get_transforms_to_coord_frame
from ..utils import _check_option, fill_doc, verbose
from ..viz import EvokedField, create_3d_figure
from ..viz._3d import _plot_head_surface, _plot_sensors
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
        bem=None,
        initial_time=None,
        trans=None,
        show_density=True,
        subject=None,
        subjects_dir=None,
        n_jobs=None,
        verbose=None,
    ):
        field_map = make_field_map(
            evoked,
            ch_type="meg",
            trans=trans,
            subject=subject,
            subjects_dir=subjects_dir,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        if cov is None:
            cov = make_ad_hoc_cov(evoked.info)
        if bem is None:
            bem = make_sphere_model("auto", "auto", evoked.info)
        bem = _ensure_bem_surfaces(bem, extra_allow=(ConductorModel, None))

        if initial_time is None:
            data = evoked.copy().pick(field_map[0]["ch_names"]).data
            initial_time = evoked.times[np.argmax(np.mean(data**2, axis=0))]

        # Get transforms to convert all the various meshes to head space
        head_mri_t = _get_trans(trans, "head", "mri")[0]
        to_cf_t = _get_transforms_to_coord_frame(
            evoked.info, head_mri_t, coord_frame="head"
        )

        self._actors = dict()
        self._bem = bem
        self._cov = cov
        self._current_time = initial_time
        self._dips = dict()
        self._dips_active = set()
        self._dips_colors = dict()
        self._dips_timecourses = dict()
        self._dips_lines = dict()
        self._evoked = evoked
        self._field_map = field_map
        self._fig_sensors = None
        self._n_jobs = n_jobs
        self._show_density = show_density
        self._subject = subject
        self._subjects_dir = subjects_dir
        self._to_cf_t = to_cf_t
        self._trans = trans
        self._verbose = verbose

        # Configure the GUI
        self._renderer = self._configure_main_display()
        self._configure_dock()

    def _configure_main_display(self):
        """Configure main 3D display of the GUI."""
        fig = create_3d_figure((1900, 1020), bgcolor="white", show=True)
        fig = EvokedField(
            self._evoked,
            self._field_map,
            time=self._current_time,
            interpolation="linear",
            alpha=1,
            show_density=self._show_density,
            foreground="black",
            fig=fig,
        )
        fig.set_contour_line_width(2)
        fig._renderer.set_camera(
            focalpoint=fit_sphere_to_headshape(self._evoked.info)[1]
        )
        self._actors["helmet"] = fig._surf_maps[0]["mesh"]._actor

        self._actors["sensors"] = _plot_sensors(
            renderer=fig._renderer,
            info=self._evoked.info,
            to_cf_t=self._to_cf_t,
            picks=pick_types(self._evoked.info, meg=True),
            meg=True,
            eeg=False,
            fnirs=False,
            warn_meg=False,
            head_surf=None,
            units="m",
            sensor_opacity=0.1,
            orient_glyphs=False,
            scale_by_distance=False,
            project_points=False,
            surf=None,
            check_inside=None,
            nearest=None,
            sensor_colors="black",
        )["meg"]

        self._actors["head"], _, _ = _plot_head_surface(
            renderer=fig._renderer,
            head="head",
            subject=self._subject,
            subjects_dir=self._subjects_dir,
            bem=self._bem,
            coord_frame="head",
            to_cf_t=self._to_cf_t,
            alpha=1.0,
        )

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
        r._dock_add_button("Fit multi-dipole", self._on_fit_multi)
        r._dock_add_combo_box(
            "Method",
            value="MNE",
            rng=["MNE", "Single-dipole", "LCMV"],
            callback=self._on_select_method,
        )
        self._dipole_box = r._dock_add_group_box(name="Dipoles")
        r._dock_add_stretch()

    def toggle_mesh(self, name, show=None):
        """Toggle a mesh on or off.

        Parameters
        ----------
        name : "helmet"
            Name of the mesh to toggle.
        show : bool | None
            Whether to show the mesh. If None, the visibility of the mesh is toggled.
        """
        _check_option("name", name, self._actors.keys())
        actors = self._actors[name]
        # self._actors[name] is sometimes a list and sometimes not. Make it
        # always be a list to simplify the code.
        if isinstance(actors, list):
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

    def _on_sensor_data(self):
        """Show sensor data."""
        if self._fig_sensors is not None:
            return
        fig = self._evoked.plot_topo()
        fig.canvas.mpl_connect("close_event", self._on_sensor_data_close)
        subscribe(fig, "channels_select", self._on_channels_select)
        self._fig_sensors = fig

    def _on_sensor_data_close(self, event):
        self._fig_sensors = None

    def _on_channels_select(self, event):
        """Show selected channels."""
        print(event)

    def _on_fit_dipole(self):
        print("Fitting dipole...")
        evoked_picked = self._evoked
        cov_picked = self._cov
        if self._fig_sensors is not None:
            picks = self._fig_sensors[0].lasso.selection
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
        dip_name = f"dip{len(self._dips)}"
        self._dips[dip_name] = dip
        self._dips_active.add(dip_name)
        colors = _get_color_list()
        self._dips_colors[dip_name] = colors[(len(self._dips) - 1) % len(colors)]

    def _on_fit_multi(self):
        print("Fitting dipoles", self._dips_active)
        fwd, _ = make_forward_dipole(
            [self._dips[d] for d in self._dips_active], self._bem, self._evoked.info
        )

        inv = make_inverse_operator(
            self._evoked.info, fwd, self._cov, fixed=True, depth=0
        )
        stc = apply_inverse(self._evoked, inv, method="MNE", lambda2=0)
        timecourses = stc.data

        canvas = self._setup_mplcanvas()
        ymin, ymax = 0, 0
        for dip_name, timecourse in zip(self._dips_active, timecourses):
            self._dip_timecourses[dip_name] = timecourse
            if dip_name in self._dip_lines:
                self._dip_lines[dip_name].set_ydata(timecourse)
            else:
                self._dip_lines[dip_name] = canvas.plot(
                    self._evoked.times,
                    timecourse,
                    label=dip_name,
                    color=self._dips_colors[dip_name],
                )
            ymin = min(ymin, 1.1 * timecourse.min())
            ymax = max(ymax, 1.1 * timecourse.max())
        canvas.axes.set_ylim(ymin, ymax)
        canvas.update_plot()

    def _on_select_method(self):
        print("Select method")


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
    return pyvista.PolyData(vertices, faces)
