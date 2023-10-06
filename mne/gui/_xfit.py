import mne
import pyvista
import numpy as np
from mne.transforms import _get_trans, _get_transforms_to_coord_frame
from functools import partial

data_path = mne.datasets.sample.data_path()
evoked = mne.read_evokeds(
    f"{data_path}/MEG/sample/sample_audvis-ave.fif", condition="Left Auditory"
)
trans = mne.read_trans(f"{data_path}/MEG/sample/sample_audvis_raw-trans.fif")
head_mri_t = _get_trans(trans, "head", "mri")[0]
to_cf_t = _get_transforms_to_coord_frame(evoked.info, head_mri_t, coord_frame="head")

evoked.apply_baseline((-0.2, 0))
field_map = mne.make_field_map(evoked, trans=None)
# cov = mne.read_cov(
#     f"{data_path}/MEG/sample/sample_audvis-cov.fif"
# )
# bem = mne.read_bem_solution(
#     f"{data_path}/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif"
# )
cov = mne.make_ad_hoc_cov(evoked.info)
bem = mne.make_sphere_model("auto", "auto", evoked.info)

fig = mne.viz.create_3d_figure((1500, 1500), bgcolor="white", show=True)
fig = mne.viz.plot_alignment(
    evoked.info,
    surfaces=dict(seghead=0.2, pial=0.5),
    meg=False,
    eeg=False,
    subject="sample",
    subjects_dir=data_path / "subjects",
    trans=trans,
    coord_frame="head",
    fig=fig,
)
fig = mne.viz.EvokedField(
    evoked,
    field_map,
    time=0.096,
    interpolation="linear",
    alpha=0,
    show_density=False,
    foreground="black",
    fig=fig,
)
renderer = fig._renderer
sensor_actors = mne.viz._3d._plot_sensors(
    renderer=renderer,
    info=evoked.info,
    to_cf_t=to_cf_t,
    picks=mne.pick_types(evoked.info, meg=True),
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
fig.set_contour_line_width(2)
fig.separate_canvas = False

helmet = fig._surf_maps[0]["mesh"]._polydata
helmet.compute_normals(inplace=True)

fig_sensors = list()
dips = list()
dip_timecourses = list()
dip_lines = list()
dipole_actors = list()
green_arrows = list()
green_arrow_coords = list()
green_arrow_pos = list()
green_arrow_actors = list()
colors = mne.viz.utils._get_color_list()
time_line = list()
dipole_box = None

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


def setup_mplcanvas():
    if renderer._mplcanvas is None:
        renderer._mplcanvas = renderer._window_get_mplcanvas(fig, 0.5, False, False)
        renderer._window_adjust_mplcanvas_layout()
    if len(time_line) == 0:
        time_line.append(
            renderer._mplcanvas.plot_time_line(
                fig._current_time,
                label="time",
                color=fig._fg_color,
            )
        )
    return renderer._mplcanvas


def show_dipole(show, dip_num):
    show = bool(show)
    if dip_num >= len(dips):
        return
    dip_lines[dip_num].set_visible(show)
    green_arrow_actors[dip_num].visibility = show
    renderer._update()
    renderer._mplcanvas.update_plot()


def on_fit_dipole():
    print("Fitting dipole...")
    evoked_picked = evoked
    cov_picked = cov
    if len(fig_sensors) > 0:
        picks = fig_sensors[0].lasso.selection
        if len(picks) > 0:
            evoked_picked = evoked.copy().pick(picks)
            evoked_picked.info.normalize_proj()
            cov_picked = cov.copy().pick_channels(picks, ordered=False)
            cov_picked["projs"] = evoked_picked.info["projs"]

    dip = mne.fit_dipole(
        evoked_picked.copy().crop(fig._current_time, fig._current_time),
        cov_picked,
        bem,
        trans=trans,
        min_dist=0,
        verbose=False,
    )[0]
    dips.append(dip)
    dip_num = len(dips) - 1
    renderer.plotter.add_arrows(dip.pos, dip.ori, color=colors[dip_num], mag=0.05)
    dip_timecourse = mne.fit_dipole(
        evoked_picked,
        cov_picked,
        bem,
        pos=dip.pos[0],
        ori=dip.ori[0],
        trans=trans,
        verbose=False,
    )[0].data[0]
    dip_timecourses.append(dip_timecourse)
    draw_arrow(dip, dip_timecourse, color=colors[dip_num])

    canvas = setup_mplcanvas()
    dip_lines.append(
        canvas.plot(
            evoked.times, dip_timecourse, label=f"dip{dip_num}", color=colors[dip_num]
        )
    )
    renderer._dock_add_check_box(
        name=f"dip{dip_num}",
        value=True,
        callback=partial(show_dipole, dip_num=dip_num),
        layout=dipole_box,
    )


def on_channels_select(event):
    selected_channels = set(event.ch_names)
    for act, ch_name in zip(sensor_actors, evoked.ch_names):
        if ch_name in selected_channels:
            act.prop.SetColor(0, 1, 0)
            act.prop.SetOpacity(0.5)
        else:
            act.prop.SetColor(0, 0, 0)
            act.prop.SetOpacity(0.1)
    renderer._update()


def on_sensor_data():
    fig = evoked.plot_topo()
    mne.viz.ui_events.subscribe(fig, "channels_select", on_channels_select)
    fig_sensors[:] = [fig]


def on_time_change(event):
    new_time = (np.clip(event.time, evoked.times[0], evoked.times[-1]),)
    for i in range(len(green_arrows)):
        arrow = green_arrows[i]
        arrow_coords = green_arrow_coords[i]
        arrow_position = green_arrow_pos[i]
        dip_timecourse = dip_timecourses[i]
        scaling = (
            np.interp(
                new_time,
                evoked.times,
                dip_timecourse,
            )
            * 1e6
        )
        arrow.points = (vertices * scaling) @ arrow_coords + arrow_position
    if len(time_line) > 0:
        time_line[0].set_xdata([new_time])
        renderer._mplcanvas.update_plot()
    renderer._update()


def draw_arrow(dip, dip_timecourse, color):
    dip_position = dip.pos[0]

    # Get the closest vertex (=point) of the helmet mesh
    distances = ((helmet.points - dip_position) * helmet.point_normals).sum(axis=1)
    closest_point = np.argmin(distances)

    # Compute the position of the projected dipole
    norm = helmet.point_normals[closest_point]
    arrow_position = dip_position + (distances[closest_point] + 0.003) * norm

    # Create a cartesian coordinate system where X and Y are tangential to the helmet
    tan_coords = mne.surface._normal_orth(norm)

    # Project the orientation of the dipole tangential to the helmet
    dip_ori_tan = tan_coords[:2] @ dip.ori[0] @ tan_coords[:2]

    # Rotate the coordinate system such that Y lies along the dipole orientation
    arrow_coords = np.array([np.cross(dip_ori_tan, norm), dip_ori_tan, norm])
    arrow_coords /= np.linalg.norm(arrow_coords, axis=1, keepdims=True)

    # Place the arrow inside the new coordinate system
    scaling = np.interp(fig._current_time, evoked.times, dip_timecourse) * 1e6
    arrow = pyvista.PolyData(
        (vertices * scaling) @ arrow_coords + arrow_position, faces
    )
    green_arrows.append(arrow)
    green_arrow_coords.append(arrow_coords)
    green_arrow_pos.append(arrow_position)

    # Render the arrow
    green_arrow_actors.append(renderer.plotter.add_mesh(arrow, color=color))


def set_view(view):
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
    renderer.set_camera(**kwargs)


def add_view_buttons(r):
    layout = r._dock_add_group_box("Views")

    hlayout = r._dock_add_layout(vertical=False)
    r._dock_add_button(
        "🢆", callback=partial(set_view, view=7), layout=hlayout, style="pushbutton"
    )
    r._dock_add_button(
        "🢃", callback=partial(set_view, view=8), layout=hlayout, style="pushbutton"
    )
    r._dock_add_button(
        "🢇", callback=partial(set_view, view=9), layout=hlayout, style="pushbutton"
    )
    r._layout_add_widget(layout, hlayout)

    hlayout = r._dock_add_layout(vertical=False)
    r._dock_add_button(
        "🢂", callback=partial(set_view, view=4), layout=hlayout, style="pushbutton"
    )
    r._dock_add_button(
        "⊙", callback=partial(set_view, view=5), layout=hlayout, style="pushbutton"
    )
    r._dock_add_button(
        "🢀", callback=partial(set_view, view=6), layout=hlayout, style="pushbutton"
    )

    r._layout_add_widget(layout, hlayout)
    hlayout = r._dock_add_layout(vertical=False)
    r._dock_add_button(
        "🢅", callback=partial(set_view, view=1), layout=hlayout, style="pushbutton"
    )
    r._dock_add_button(
        "🢁", callback=partial(set_view, view=2), layout=hlayout, style="pushbutton"
    )
    r._dock_add_button(
        "🢄", callback=partial(set_view, view=3), layout=hlayout, style="pushbutton"
    )
    r._layout_add_widget(layout, hlayout)

    r.plotter.add_key_event("1", partial(set_view, view=1))
    r.plotter.add_key_event("2", partial(set_view, view=2))
    r.plotter.add_key_event("3", partial(set_view, view=3))
    r.plotter.add_key_event("4", partial(set_view, view=4))
    r.plotter.add_key_event("5", partial(set_view, view=5))
    r.plotter.add_key_event("6", partial(set_view, view=6))
    r.plotter.add_key_event("7", partial(set_view, view=7))
    r.plotter.add_key_event("8", partial(set_view, view=8))
    r.plotter.add_key_event("9", partial(set_view, view=9))


add_view_buttons(renderer)
renderer._dock_initialize(name="Dipole fitting", area="right")
renderer._dock_add_button("Sensor data", on_sensor_data)
renderer._dock_add_button("Fit dipole", on_fit_dipole)

dipole_box = renderer._dock_add_group_box(name="Dipoles")
renderer._dock_add_stretch()

renderer.set_camera(focalpoint=mne.bem.fit_sphere_to_headshape(evoked.info)[1])
mne.viz.ui_events.subscribe(fig, "time_change", on_time_change)


# gfp_ax = canvas.fig.axes[0].twinx()
# gfp_ax.plot(
#     evoked.times, np.mean(fig._surf_maps[0]['data'] ** 2, axis=0), color='maroon',
# )
# canvas.update_plot()
