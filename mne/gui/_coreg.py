from contextlib import contextmanager
from functools import partial
import os
import os.path as op
import time
import threading

import numpy as np
from traitlets import observe, HasTraits, Unicode, Bool, Float

from ..io.constants import FIFF
from ..defaults import DEFAULTS
from ..io import read_info, read_fiducials, read_raw
from ..io.pick import pick_types
from ..io.open import fiff_open, dir_tree_find
from ..io.meas_info import _empty_info
from ..io._read_raw import supported as raw_supported_types
from ..coreg import Coregistration, _is_mri_subject
from ..viz._3d import (_plot_head_surface, _plot_head_fiducials,
                       _plot_head_shape_points, _plot_mri_fiducials,
                       _plot_hpi_coils, _plot_sensors)
from ..transforms import (read_trans, write_trans, _ensure_trans,
                          rotation_angles, _get_transforms_to_coord_frame)
from ..utils import (get_subjects_dir, check_fname, _check_fname, fill_doc,
                     warn, verbose, logger)
from ..channels import read_dig_fif


@fill_doc
class CoregistrationUI(HasTraits):
    """Class for coregistration assisted by graphical interface.

    Parameters
    ----------
    info_file : None | str
        The FIFF file with digitizer data for coregistration.
    %(subject)s
    %(subjects_dir)s
    fiducials : list | dict | str
        The fiducials given in the MRI (surface RAS) coordinate
        system. If a dict is provided it must be a dict with 3 entries
        with keys 'lpa', 'rpa' and 'nasion' with as values coordinates in m.
        If a list it must be a list of DigPoint instances as returned
        by the read_fiducials function.
        If set to 'estimated', the fiducials are initialized
        automatically using fiducials defined in MNI space on fsaverage
        template. If set to 'auto', one tries to find the fiducials
        in a file with the canonical name (``bem/{subject}-fiducials.fif``)
        and if abstent one falls back to 'estimated'. Defaults to 'auto'.
    head_resolution : bool
        If True, use a high-resolution head surface. Defaults to False.
    head_transparency : bool
        If True, display the head surface with transparency. Defaults to False.
    hpi_coils : bool
        If True, display the HPI coils. Defaults to True.
    head_shape_points : bool
        If True, display the head shape points. Defaults to True.
    eeg_channels : bool
        If True, display the EEG channels. Defaults to True.
    orient_glyphs : bool
        If True, orient the sensors towards the head surface. Default to False.
    scale_by_distance : bool
        If True, scale the sensors based on their distance to the head surface.
        Defaults to True.
    project_eeg : bool
        If True, project the EEG channels onto the head surface. Defaults to
        False.
    mark_inside : bool
        If True, mark the head shape points that are inside the head surface
        with a different color. Defaults to True.
    sensor_opacity : float
        The opacity of the sensors between 0 and 1. Defaults to 1.0.
    trans : str
        The path to the Head<->MRI transform FIF file ("-trans.fif").
    size : tuple
        The dimensions (width, height) of the rendering view. The default is
        (800, 600).
    bgcolor : tuple | str
        The background color as a tuple (red, green, blue) of float
        values between 0 and 1 or a valid color name (i.e. 'white'
        or 'w'). Defaults to 'grey'.
    show : bool
        Display the window as soon as it is ready. Defaults to True.
    standalone : bool
        If True, start the Qt application event loop. Default to False.
    %(scene_interaction)s
        Defaults to ``'terrain'``.

        .. versionadded:: 1.0
    %(verbose)s
    """

    _subject = Unicode()
    _subjects_dir = Unicode()
    _lock_fids = Bool()
    _fiducials_file = Unicode()
    _current_fiducial = Unicode()
    _info_file = Unicode()
    _orient_glyphs = Bool()
    _scale_by_distance = Bool()
    _project_eeg = Bool()
    _mark_inside = Bool()
    _hpi_coils = Bool()
    _head_shape_points = Bool()
    _eeg_channels = Bool()
    _head_resolution = Bool()
    _head_transparency = Bool()
    _grow_hair = Float()
    _scale_mode = Unicode()
    _icp_fid_match = Unicode()

    @verbose
    def __init__(self, info_file, subject=None, subjects_dir=None,
                 fiducials='auto', head_resolution=None,
                 head_transparency=None, hpi_coils=None,
                 head_shape_points=None, eeg_channels=None, orient_glyphs=None,
                 scale_by_distance=None, project_eeg=None, mark_inside=None,
                 sensor_opacity=None, trans=None, size=None, bgcolor=None,
                 show=True, standalone=False, interaction='terrain',
                 verbose=None):
        from ..viz.backends.renderer import _get_renderer
        from ..viz.backends._utils import _qt_app_exec

        def _get_default(var, val):
            return var if var is not None else val
        self._actors = dict()
        self._surfaces = dict()
        self._widgets = dict()
        self._verbose = verbose
        self._plot_locked = False
        self._refresh_rate_ms = max(int(round(1000. / 60.)), 1)
        self._redraws_pending = set()
        self._parameter_mutex = threading.Lock()
        self._redraw_mutex = threading.Lock()
        self._head_geo = None
        self._coord_frame = "mri"
        self._mouse_no_mvt = -1
        self._to_cf_t = None
        self._omit_hsp_distance = 0.0
        self._head_opacity = 1.0
        self._fid_colors = tuple(
            DEFAULTS['coreg'][f'{key}_color'] for key in
            ('lpa', 'nasion', 'rpa'))
        self._defaults = dict(
            size=_get_default(size, (800, 600)),
            bgcolor=_get_default(bgcolor, "grey"),
            orient_glyphs=_get_default(orient_glyphs, True),
            scale_by_distance=_get_default(scale_by_distance, True),
            project_eeg=_get_default(project_eeg, False),
            mark_inside=_get_default(mark_inside, True),
            hpi_coils=_get_default(hpi_coils, True),
            head_shape_points=_get_default(head_shape_points, True),
            eeg_channels=_get_default(eeg_channels, True),
            head_resolution=_get_default(head_resolution, True),
            head_transparency=_get_default(head_transparency, False),
            head_opacity=0.5,
            sensor_opacity=_get_default(sensor_opacity, 1.0),
            fiducials=("LPA", "Nasion", "RPA"),
            fiducial="LPA",
            lock_fids=True,
            grow_hair=0.0,
            scale_modes=["None", "uniform", "3-axis"],
            scale_mode="None",
            icp_fid_matches=('nearest', 'matched'),
            icp_fid_match='matched',
            icp_n_iterations=20,
            omit_hsp_distance=10.0,
            lock_head_opacity=self._head_opacity < 1.0,
            weights=dict(
                lpa=1.0,
                nasion=10.0,
                rpa=1.0,
                hsp=1.0,
                eeg=1.0,
                hpi=1.0,
            ),
        )

        # process requirements
        info = None
        subjects_dir = get_subjects_dir(
            subjects_dir=subjects_dir, raise_error=True)
        subject = _get_default(subject, self._get_subjects(subjects_dir)[0])

        # setup the window
        self._renderer = _get_renderer(
            size=self._defaults["size"], bgcolor=self._defaults["bgcolor"])
        self._renderer._window_close_connect(self._clean)
        self._renderer.set_interaction(interaction)
        self._renderer._status_bar_initialize()

        # setup the model
        self._immediate_redraw = (self._renderer._kind != 'qt')
        self._info = info
        self._fiducials = fiducials
        self._coreg = Coregistration(
            self._info, subject, subjects_dir, fiducials)
        fid_accurate = self._coreg._fid_accurate
        for fid in self._defaults["weights"].keys():
            setattr(self, f"_{fid}_weight", self._defaults["weights"][fid])

        # set main traits
        self._set_subjects_dir(subjects_dir)
        self._set_subject(subject)
        self._set_info_file(info_file)
        self._set_orient_glyphs(self._defaults["orient_glyphs"])
        self._set_scale_by_distance(self._defaults["scale_by_distance"])
        self._set_project_eeg(self._defaults["project_eeg"])
        self._set_mark_inside(self._defaults["mark_inside"])
        self._set_hpi_coils(self._defaults["hpi_coils"])
        self._set_head_shape_points(self._defaults["head_shape_points"])
        self._set_eeg_channels(self._defaults["eeg_channels"])
        self._set_head_resolution(self._defaults["head_resolution"])
        self._set_head_transparency(self._defaults["head_transparency"])
        self._set_grow_hair(self._defaults["grow_hair"])
        self._set_omit_hsp_distance(self._defaults["omit_hsp_distance"])
        self._set_icp_n_iterations(self._defaults["icp_n_iterations"])
        self._set_icp_fid_match(self._defaults["icp_fid_match"])

        # configure UI
        self._reset_fitting_parameters()
        self._configure_status_bar()
        self._configure_dock()
        self._configure_picking()

        # once the docks are initialized
        self._set_current_fiducial(self._defaults["fiducial"])
        self._set_scale_mode(self._defaults["scale_mode"])
        if trans is not None:
            self._load_trans(trans)
        self._redraw()  # we need the elements to be present now
        if not fid_accurate:
            self._set_head_resolution('high')
            self._forward_widget_command('high_res_head', "set_value", True)
            self._set_lock_fids(True)  # hack to make the dig disappear
        self._set_lock_fids(fid_accurate)

        # must be done last
        if show:
            self._renderer.show()
        # update the view once shown
        views = {True: dict(azimuth=90, elevation=90),  # front
                 False: dict(azimuth=180, elevation=90)}  # left
        self._renderer.set_camera(distance=None, **views[self._lock_fids])
        self._redraw()
        # XXX: internal plotter/renderer should not be exposed
        if not self._immediate_redraw:
            self._renderer.plotter.add_callback(
                self._redraw, self._refresh_rate_ms)
        self._renderer.plotter.show_axes()
        if standalone:
            _qt_app_exec(self._renderer.figure.store["app"])

    def _set_subjects_dir(self, subjects_dir):
        self._subjects_dir = _check_fname(
            subjects_dir, overwrite=True, must_exist=True, need_dir=True)

    def _set_subject(self, subject):
        self._subject = subject

    def _set_lock_fids(self, state):
        self._lock_fids = bool(state)

    def _set_fiducials_file(self, fname):
        if not self._check_fif('fiducials', fname):
            return
        self._fiducials_file = _check_fname(
            fname, overwrite=True, must_exist=True, need_dir=False)

    def _set_current_fiducial(self, fid):
        self._current_fiducial = fid.lower()

    def _set_info_file(self, fname):
        if fname is None:
            return

        # info file can be anything supported by read_raw
        try:
            check_fname(fname, 'info', tuple(raw_supported_types.keys()),
                        endings_err=tuple(raw_supported_types.keys()))
        except IOError as e:
            warn(e)
            self._widgets["info_file"].set_value(0, '')
            return

        fname = _check_fname(fname, overwrite=True)  # convert to str

        # ctf ds `files` are actually directories
        if fname.endswith(('.ds',)):
            self._info_file = _check_fname(
                fname, overwrite=True, must_exist=True, need_dir=True)
        else:
            self._info_file = _check_fname(
                fname, overwrite=True, must_exist=True, need_dir=False)

    def _set_omit_hsp_distance(self, distance):
        self._omit_hsp_distance = distance

    def _set_orient_glyphs(self, state):
        self._orient_glyphs = bool(state)

    def _set_scale_by_distance(self, state):
        self._scale_by_distance = bool(state)

    def _set_project_eeg(self, state):
        self._project_eeg = bool(state)

    def _set_mark_inside(self, state):
        self._mark_inside = bool(state)

    def _set_hpi_coils(self, state):
        self._hpi_coils = bool(state)

    def _set_head_shape_points(self, state):
        self._head_shape_points = bool(state)

    def _set_eeg_channels(self, state):
        self._eeg_channels = bool(state)

    def _set_head_resolution(self, state):
        self._head_resolution = bool(state)

    def _set_head_transparency(self, state):
        self._head_transparency = bool(state)

    def _set_grow_hair(self, value):
        self._grow_hair = value

    def _set_scale_mode(self, mode):
        self._scale_mode = mode

    def _set_fiducial(self, value, coord):
        fid = self._current_fiducial.lower()
        coords = ["X", "Y", "Z"]
        idx = coords.index(coord)
        getattr(self._coreg, f"_{fid}")[0][idx] = value / 1e3
        self._update_plot("mri_fids")

    def _set_parameter(self, value, mode_name, coord):
        with self._parameter_mutex:
            self. _set_parameter_safe(value, mode_name, coord)
        self._update_plot("sensors")

    def _set_parameter_safe(self, value, mode_name, coord):
        params = dict(
            rotation=self._coreg._rotation,
            translation=self._coreg._translation,
            scale=self._coreg._scale,
        )
        idx = ["X", "Y", "Z"].index(coord)
        if mode_name == "rotation":
            params[mode_name][idx] = np.deg2rad(value)
        elif mode_name == "translation":
            params[mode_name][idx] = value / 1e3
        else:
            assert mode_name == "scale"
            params[mode_name][idx] = value / 1e2
        self._coreg._update_params(
            rot=params["rotation"],
            tra=params["translation"],
            sca=params["scale"],
        )

    def _set_icp_n_iterations(self, n_iterations):
        self._icp_n_iterations = n_iterations

    def _set_icp_fid_match(self, method):
        self._icp_fid_match = method

    def _set_point_weight(self, weight, point):
        funcs = {
            'hpi': '_set_hpi_coils',
            'hsp': '_set_head_shape_points',
            'eeg': '_set_eeg_channels',
        }
        if point in funcs.keys():
            getattr(self, funcs[point])(weight > 0)
        setattr(self, f"_{point}_weight", weight)
        setattr(self._coreg, f"_{point}_weight", weight)
        self._update_distance_estimation()

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
        self._coreg._setup_bem()
        self._coreg._setup_fiducials(self._fiducials)
        self._reset()
        rr = (self._coreg._processed_low_res_mri_points *
              self._coreg._scale)
        self._head_geo = dict(rr=rr, tris=self._coreg._bem_low_res["tris"],
                              nn=self._coreg._bem_low_res["nn"])

    @observe("_lock_fids")
    def _lock_fids_changed(self, change=None):
        view_widgets = ["project_eeg"]
        fid_widgets = ["fid_X", "fid_Y", "fid_Z", "fids_file", "fids"]
        self._set_head_transparency(self._lock_fids)
        if self._lock_fids:
            self._forward_widget_command(view_widgets, "set_enabled", True)
            self._display_message()
            self._update_distance_estimation()
        else:
            self._forward_widget_command(view_widgets, "set_enabled", False)
            self._display_message("Picking fiducials - "
                                  f"{self._current_fiducial.upper()}")
        self._set_sensors_visibility(self._lock_fids)
        self._forward_widget_command("lock_fids", "set_value", self._lock_fids)
        self._forward_widget_command(fid_widgets, "set_enabled",
                                     not self._lock_fids)

    @observe("_fiducials_file")
    def _fiducials_file_changed(self, change=None):
        fids, _ = read_fiducials(self._fiducials_file)
        self._coreg._setup_fiducials(fids)
        self._update_distance_estimation()
        self._reset()
        self._set_lock_fids(True)

    @observe("_current_fiducial")
    def _current_fiducial_changed(self, change=None):
        self._update_fiducials()
        self._follow_fiducial_view()
        if not self._lock_fids:
            self._display_message("Picking fiducials - "
                                  f"{self._current_fiducial.upper()}")

    @observe("_info_file")
    def _info_file_changed(self, change=None):
        if not self._info_file:
            return
        elif self._info_file.endswith(('.fif', '.fif.gz')):
            fid, tree, _ = fiff_open(self._info_file)
            fid.close()
            if len(dir_tree_find(tree, FIFF.FIFFB_MEAS_INFO)) > 0:
                self._info = read_info(self._info_file, verbose=False)
            elif len(dir_tree_find(tree, FIFF.FIFFB_ISOTRAK)) > 0:
                self._info = _empty_info(1)
                self._info['dig'] = read_dig_fif(fname=self._info_file).dig
                self._info._unlocked = False
        else:
            self._info = read_raw(self._info_file).info
        # XXX: add coreg.set_info()
        self._coreg._info = self._info
        self._coreg._setup_digs()
        self._reset()

    @observe("_orient_glyphs")
    def _orient_glyphs_changed(self, change=None):
        self._update_plot(["hpi", "hsp", "eeg"])

    @observe("_scale_by_distance")
    def _scale_by_distance_changed(self, change=None):
        self._update_plot(["hpi", "hsp", "eeg"])

    @observe("_project_eeg")
    def _project_eeg_changed(self, change=None):
        self._update_plot("eeg")

    @observe("_mark_inside")
    def _mark_inside_changed(self, change=None):
        self._update_plot("hsp")

    @observe("_hpi_coils")
    def _hpi_coils_changed(self, change=None):
        self._update_plot("hpi")

    @observe("_head_shape_points")
    def _head_shape_point_changed(self, change=None):
        self._update_plot("hsp")

    @observe("_eeg_channels")
    def _eeg_channels_changed(self, change=None):
        self._update_plot("eeg")

    @observe("_head_resolution")
    def _head_resolution_changed(self, change=None):
        self._update_plot(["head"])
        if self._grow_hair > 0:
            self._update_plot(["hair"])

    @observe("_head_transparency")
    def _head_transparency_changed(self, change=None):
        self._head_opacity = self._defaults["head_opacity"] \
            if self._head_transparency else 1.0
        self._actors["head"].GetProperty().SetOpacity(self._head_opacity)
        self._renderer._update()

    @observe("_grow_hair")
    def _grow_hair_changed(self, change=None):
        self._coreg.set_grow_hair(self._grow_hair)
        self._update_plot("hair")

    @observe("_scale_mode")
    def _scale_mode_changed(self, change=None):
        mode = None if self._scale_mode == "None" else self._scale_mode
        self._coreg.set_scale_mode(mode)
        self._forward_widget_command(["sX", "sY", "sZ"], "set_enabled",
                                     mode is not None)

    @observe("_icp_fid_match")
    def _icp_fid_match_changed(self, change=None):
        self._coreg.set_fid_match(self._icp_fid_match)

    def _configure_picking(self):
        self._renderer._update_picking_callback(
            self._on_mouse_move,
            self._on_button_press,
            self._on_button_release,
            self._on_pick
        )

    @verbose
    def _redraw(self, verbose=None):
        if not self._redraws_pending:
            return
        draw_map = dict(
            head=self._add_head_surface,
            hair=self._add_head_hair,
            mri_fids=self._add_mri_fiducials,
            hsp=self._add_head_shape_points,
            hpi=self._add_hpi_coils,
            eeg=self._add_eeg_channels,
            head_fids=self._add_head_fiducials,
        )
        with self._redraw_mutex:
            logger.debug(f'Redrawing {self._redraws_pending}')
            for key in self._redraws_pending:
                draw_map[key]()
            self._redraws_pending.clear()
            self._renderer._update()

    def _on_mouse_move(self, vtk_picker, event):
        if self._mouse_no_mvt:
            self._mouse_no_mvt -= 1

    def _on_button_press(self, vtk_picker, event):
        self._mouse_no_mvt = 2

    def _on_button_release(self, vtk_picker, event):
        if self._mouse_no_mvt > 0:
            x, y = vtk_picker.GetEventPosition()
            # XXX: internal plotter/renderer should not be exposed
            plotter = self._renderer.figure.plotter
            picked_renderer = self._renderer.figure.plotter.renderer
            # trigger the pick
            plotter.picker.Pick(x, y, 0, picked_renderer)
        self._mouse_no_mvt = 0

    def _on_pick(self, vtk_picker, event):
        if self._lock_fids:
            return
        # XXX: taken from Brain, can be refactored
        cell_id = vtk_picker.GetCellId()
        mesh = vtk_picker.GetDataSet()
        if mesh is None or cell_id == -1 or not self._mouse_no_mvt:
            return
        if not getattr(mesh, "_picking_target", False):
            return
        pos = np.array(vtk_picker.GetPickPosition())
        vtk_cell = mesh.GetCell(cell_id)
        cell = [vtk_cell.GetPointId(point_id) for point_id
                in range(vtk_cell.GetNumberOfPoints())]
        vertices = mesh.points[cell]
        idx = np.argmin(abs(vertices - pos), axis=0)
        vertex_id = cell[idx[0]]

        fiducials = [s.lower() for s in self._defaults["fiducials"]]
        idx = fiducials.index(self._current_fiducial.lower())
        # XXX: add coreg.set_fids
        self._coreg._fid_points[idx] = self._surfaces["head"].points[vertex_id]
        self._coreg._reset_fiducials()
        self._update_fiducials()
        self._update_plot("mri_fids")

    def _reset_fitting_parameters(self):
        self._forward_widget_command("icp_n_iterations", "set_value",
                                     self._defaults["icp_n_iterations"])
        self._forward_widget_command("icp_fid_match", "set_value",
                                     self._defaults["icp_fid_match"])
        weights_widgets = [f"{w}_weight"
                           for w in self._defaults["weights"].keys()]
        self._forward_widget_command(weights_widgets, "set_value",
                                     list(self._defaults["weights"].values()))

    def _reset_fiducials(self):
        self._set_current_fiducial(self._defaults["fiducial"])

    def _omit_hsp(self):
        self._coreg.omit_head_shape_points(self._omit_hsp_distance / 1e3)
        n_omitted = np.sum(~self._coreg._extra_points_filter)
        n_remaining = len(self._coreg._dig_dict['hsp']) - n_omitted
        self._update_plot("hsp")
        self._renderer._status_bar_show_message(
            f"{n_omitted} head shape points omitted, "
            f"{n_remaining} remaining.")

    def _reset_omit_hsp_filter(self):
        self._coreg._extra_points_filter = None
        self._update_plot("hsp")
        n_total = len(self._coreg._dig_dict['hsp'])
        self._renderer._status_bar_show_message(
            f"No head shape point is omitted, the total is {n_total}.")

    def _update_plot(self, changes="all"):
        # Update list of things that need to be updated/plotted (and maybe
        # draw them immediately)
        if self._plot_locked:
            return
        if self._info is None:
            changes = ["head", "mri_fids"]
            self._to_cf_t = dict(mri=dict(trans=np.eye(4)), head=None)
        else:
            self._to_cf_t = _get_transforms_to_coord_frame(
                self._info, self._coreg.trans, coord_frame=self._coord_frame)
        all_keys = (
            'head', 'mri_fids',  # MRI first
            'hair',  # then hair
            'hsp', 'hpi', 'eeg', 'head_fids',  # then dig
        )
        if changes == 'all':
            changes = list(all_keys)
        elif changes == 'sensors':
            changes = all_keys[2:]  # omit MRI ones
        elif isinstance(changes, str):
            changes = [changes]
        changes = set(changes)
        # ideally we would maybe have this in:
        # with self._redraw_mutex:
        # it would reduce "jerkiness" of the updates, but this should at least
        # work okay
        bad = changes.difference(set(all_keys))
        assert len(bad) == 0, f'Unknown changes: {bad}'
        self._redraws_pending.update(changes)
        if self._immediate_redraw:
            self._redraw()

    @contextmanager
    def _lock_plot(self):
        old_plot_locked = self._plot_locked
        self._plot_locked = True
        try:
            yield
        finally:
            self._plot_locked = old_plot_locked

    def _display_message(self, msg=""):
        if "msg" not in self._actors:
            self._actors["msg"] = self._renderer.text2d(0, 0, msg)
        else:
            self._actors["msg"].SetInput(msg)
        self._renderer._update()

    def _follow_fiducial_view(self):
        fid = self._current_fiducial.lower()
        view = dict(lpa='left', rpa='right', nasion='front')
        kwargs = dict(front=(90., 90.), left=(180, 90), right=(0., 90))
        kwargs = dict(zip(('azimuth', 'elevation'), kwargs[view[fid]]))
        if not self._lock_fids:
            self._renderer.set_camera(distance=None, **kwargs)

    def _update_fiducials(self):
        fid = self._current_fiducial.lower()
        val = getattr(self._coreg, f"_{fid}")[0] * 1e3
        with self._lock_plot():
            self._forward_widget_command(
                ["fid_X", "fid_Y", "fid_Z"], "set_value", val)

    def _update_distance_estimation(self):
        value = self._coreg._get_fiducials_distance_str() + '\n' + \
            self._coreg._get_point_distance_str()
        self._forward_widget_command("fit_label", "set_value", value)

    def _update_parameters(self):
        with self._lock_plot():
            # rotation
            self._forward_widget_command(["rX", "rY", "rZ"], "set_value",
                                         np.rad2deg(self._coreg._rotation))
            # translation
            self._forward_widget_command(["tX", "tY", "tZ"], "set_value",
                                         self._coreg._translation * 1e3)
            # scale
            self._forward_widget_command(["sX", "sY", "sZ"], "set_value",
                                         self._coreg._scale * 1e2)

    def _reset(self):
        self._reset_fitting_parameters()
        self._coreg.reset()
        self._update_plot()
        self._update_parameters()

    def _forward_widget_command(self, names, command, value):
        names = [names] if not isinstance(names, list) else names
        value = list(value) if isinstance(value, np.ndarray) else value
        for idx, name in enumerate(names):
            val = value[idx] if isinstance(value, list) else value
            if name in self._widgets:
                getattr(self._widgets[name], command)(val)

    def _set_sensors_visibility(self, state):
        sensors = ["head_fiducials", "hpi_coils", "head_shape_points",
                   "eeg_channels"]
        for sensor in sensors:
            if sensor in self._actors and self._actors[sensor] is not None:
                actors = self._actors[sensor]
                actors = actors if isinstance(actors, list) else [actors]
                for actor in actors:
                    actor.SetVisibility(state)
        self._renderer._update()

    def _update_actor(self, actor_name, actor):
        # XXX: internal plotter/renderer should not be exposed
        self._renderer.plotter.remove_actor(self._actors.get(actor_name))
        self._actors[actor_name] = actor

    def _add_mri_fiducials(self):
        mri_fids_actors = _plot_mri_fiducials(
            self._renderer, self._coreg._fid_points, self._subjects_dir,
            self._subject, self._to_cf_t, self._fid_colors)
        # disable picking on the markers
        for actor in mri_fids_actors:
            actor.SetPickable(False)
        self._update_actor("mri_fiducials", mri_fids_actors)

    def _add_head_fiducials(self):
        head_fids_actors = _plot_head_fiducials(
            self._renderer, self._info, self._to_cf_t, self._fid_colors)
        self._update_actor("head_fiducials", head_fids_actors)

    def _add_hpi_coils(self):
        if self._hpi_coils:
            hpi_actors = _plot_hpi_coils(
                self._renderer, self._info, self._to_cf_t,
                opacity=self._defaults["sensor_opacity"],
                scale=DEFAULTS["coreg"]["extra_scale"],
                orient_glyphs=self._orient_glyphs,
                scale_by_distance=self._scale_by_distance,
                surf=self._head_geo)
        else:
            hpi_actors = None
        self._update_actor("hpi_coils", hpi_actors)

    def _add_head_shape_points(self):
        if self._head_shape_points:
            hsp_actors = _plot_head_shape_points(
                self._renderer, self._info, self._to_cf_t,
                opacity=self._defaults["sensor_opacity"],
                orient_glyphs=self._orient_glyphs,
                scale_by_distance=self._scale_by_distance,
                mark_inside=self._mark_inside, surf=self._head_geo,
                mask=self._coreg._extra_points_filter)
        else:
            hsp_actors = None
        self._update_actor("head_shape_points", hsp_actors)

    def _add_eeg_channels(self):
        if self._eeg_channels:
            eeg = ["original"]
            picks = pick_types(self._info, eeg=(len(eeg) > 0))
            if len(picks) > 0:
                eeg_actors = _plot_sensors(
                    self._renderer, self._info, self._to_cf_t, picks,
                    meg=False, eeg=eeg, fnirs=False, warn_meg=False,
                    head_surf=self._head_geo, units='m',
                    sensor_opacity=self._defaults["sensor_opacity"],
                    orient_glyphs=self._orient_glyphs,
                    scale_by_distance=self._scale_by_distance,
                    project_points=self._project_eeg,
                    surf=self._head_geo)
                eeg_actors = eeg_actors["eeg"]
            else:
                eeg_actors = None
        else:
            eeg_actors = None
        self._update_actor("eeg_channels", eeg_actors)

    def _add_head_surface(self):
        bem = None
        surface = "head-dense" if self._head_resolution else "head"
        try:
            head_actor, head_surf, _ = _plot_head_surface(
                self._renderer, surface, self._subject,
                self._subjects_dir, bem, self._coord_frame, self._to_cf_t,
                alpha=self._head_opacity)
        except IOError:
            head_actor, head_surf, _ = _plot_head_surface(
                self._renderer, "head", self._subject, self._subjects_dir,
                bem, self._coord_frame, self._to_cf_t,
                alpha=self._head_opacity)
        # mark head surface mesh to restrict picking
        head_surf._picking_target = True
        self._update_actor("head", head_actor)
        self._surfaces["head"] = head_surf

    def _add_head_hair(self):
        if "head" in self._surfaces:
            res = "high" if self._head_resolution else "low"
            self._surfaces["head"].points = \
                self._coreg._get_processed_mri_points(res)

    def _fit_fiducials(self):
        start = time.time()
        self._coreg.fit_fiducials(
            lpa_weight=self._lpa_weight,
            nasion_weight=self._nasion_weight,
            rpa_weight=self._rpa_weight,
            verbose=self._verbose,
        )
        end = time.time()
        self._renderer._status_bar_show_message(
            f"Fitting fiducials finished in {end - start:.2f} seconds.")
        self._update_plot("sensors")
        self._update_parameters()
        self._update_distance_estimation()

    def _fit_icp(self):
        self._current_icp_iterations = 0
        self._last_log = None

        def callback(iteration, n_iterations):
            self._display_message(f"Fitting ICP - iteration {iteration + 1}")
            self._update_plot("sensors")
            self._current_icp_iterations = iteration
            dists = self._coreg.compute_dig_mri_distances() * 1e3
            self._last_log = "Distance between HSP and MRI (mean/min/max): "\
                f"{np.mean(dists):.2f} mm "\
                f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
            self._status_msg.set_value(self._last_log)
            self._status_msg.show()
            self._status_msg.update()
            self._renderer._status_bar_update()
            self._update_distance_estimation()
            self._renderer._process_events()  # allow a draw or cancel

        start = time.time()
        self._coreg.fit_icp(
            n_iterations=self._icp_n_iterations,
            lpa_weight=self._lpa_weight,
            nasion_weight=self._nasion_weight,
            rpa_weight=self._rpa_weight,
            callback=callback,
            verbose=self._verbose,
        )
        end = time.time()
        self._status_msg.set_value("")
        self._status_msg.hide()
        self._display_message()
        self._renderer._status_bar_show_message(
            self._last_log +
            f" - Fitting ICP finished in {end - start:.2f} seconds and "
            f"{self._current_icp_iterations} iterations.")
        self._update_parameters()
        del self._current_icp_iterations
        del self._last_log

    def _save_trans(self, fname):
        write_trans(fname, self._coreg.trans)
        self._renderer._status_bar_show_message(
            f"{fname} transform file is saved.")

    def _load_trans(self, fname):
        mri_head_t = _ensure_trans(read_trans(fname, return_all=True),
                                   'mri', 'head')['trans']
        rot_x, rot_y, rot_z = rotation_angles(mri_head_t)
        x, y, z = mri_head_t[:3, 3]
        self._coreg._update_params(
            rot=np.array([rot_x, rot_y, rot_z]),
            tra=np.array([x, y, z]),
        )
        self._update_parameters()
        self._renderer._status_bar_show_message(
            f"{fname} transform file is loaded.")

    def _get_subjects(self, sdir=None):
        # XXX: would be nice to move this function to util
        sdir = sdir if sdir is not None else self._subjects_dir
        is_dir = sdir and op.isdir(sdir)
        if is_dir:
            dir_content = os.listdir(sdir)
            subjects = [s for s in dir_content if _is_mri_subject(s, sdir)]
            if len(subjects) == 0:
                subjects.append('')
        else:
            subjects = ['']
        return sorted(subjects)

    def _check_fif(self, filetype, fname):
        try:
            check_fname(fname, filetype, ('.fif'), ('.fif'))
        except IOError:
            warn(f"The filename {fname} for {filetype} must end with '.fif'.")
            self._widgets[f"{filetype}_file"].set_value(0, '')
            return False
        return True

    def _configure_dock(self):
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
            value=self._lock_fids,
            callback=self._set_lock_fids,
            layout=layout
        )
        self._widgets["fiducials_file"] = self._renderer._dock_add_file_button(
            name="fiducials_file",
            desc="Load",
            func=self._set_fiducials_file,
            value=self._fiducials_file,
            placeholder="Path to fiducials",
            layout=layout,
        )
        self._widgets["fids"] = self._renderer._dock_add_radio_buttons(
            value=self._defaults["fiducial"],
            rng=self._defaults["fiducials"],
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
                rng=[-1e3, 1e3],
                callback=partial(
                    self._set_fiducial,
                    coord=coord,
                ),
                compact=True,
                double=True,
                layout=hlayout
            )
        self._renderer._layout_add_widget(layout, hlayout)

        layout = self._renderer._dock_add_group_box("Digitization Source")
        self._widgets["info_file"] = self._renderer._dock_add_file_button(
            name="info_file",
            desc="Load",
            func=self._set_info_file,
            value=self._info_file,
            placeholder="Path to info",
            layout=layout,
        )
        self._widgets["grow_hair"] = self._renderer._dock_add_spin_box(
            name="Grow Hair (mm)",
            value=self._grow_hair,
            rng=[0.0, 10.0],
            callback=self._set_grow_hair,
            layout=layout,
        )
        hlayout = self._renderer._dock_add_layout(vertical=False)
        self._widgets["omit_distance"] = self._renderer._dock_add_spin_box(
            name="Omit Distance (mm)",
            value=self._omit_hsp_distance,
            rng=[0.0, 100.0],
            callback=self._set_omit_hsp_distance,
            layout=hlayout,
        )
        self._widgets["omit"] = self._renderer._dock_add_button(
            name="Omit",
            callback=self._omit_hsp,
            layout=hlayout,
        )
        self._widgets["reset_omit"] = self._renderer._dock_add_button(
            name="Reset",
            callback=self._reset_omit_hsp_filter,
            layout=hlayout,
        )
        self._renderer._layout_add_widget(layout, hlayout)

        layout = self._renderer._dock_add_group_box("View options")
        self._widgets["project_eeg"] = self._renderer._dock_add_check_box(
            name="Project EEG",
            value=self._project_eeg,
            callback=self._set_project_eeg,
            layout=layout
        )
        self._widgets["high_res_head"] = self._renderer._dock_add_check_box(
            name="Show High Resolution Head",
            value=self._head_resolution,
            callback=self._set_head_resolution,
            layout=layout
        )
        self._renderer._dock_add_stretch()

        self._renderer._dock_initialize(name="Parameters", area="right")
        self._widgets["scaling_mode"] = self._renderer._dock_add_combo_box(
            name="Scaling Mode",
            value=self._defaults["scale_mode"],
            rng=self._defaults["scale_modes"],
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
                rng=[-1e3, 1e3],
                callback=partial(
                    self._set_parameter,
                    mode_name="scale",
                    coord=coord,
                ),
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
                    rng=[-1e3, 1e3],
                    callback=partial(
                        self._set_parameter,
                        mode_name=mode_name.lower(),
                        coord=coord,
                    ),
                    compact=True,
                    double=True,
                    step=1,
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
        self._widgets["fit_label"] = self._renderer._dock_add_label(
            value="",
            layout=layout,
        )
        self._widgets["icp_n_iterations"] = self._renderer._dock_add_spin_box(
            name="Number Of ICP Iterations",
            value=self._defaults["icp_n_iterations"],
            rng=[1, 100],
            callback=self._set_icp_n_iterations,
            compact=True,
            double=False,
            layout=layout,
        )
        self._widgets["icp_fid_match"] = self._renderer._dock_add_combo_box(
            name="Fiducial point matching",
            value=self._defaults["icp_fid_match"],
            rng=self._defaults["icp_fid_matches"],
            callback=self._set_icp_fid_match,
            compact=True,
            layout=layout
        )
        layout = self._renderer._dock_add_group_box(
            name="Weights",
            layout=layout,
        )
        for point, fid in zip(("HSP", "EEG", "HPI"),
                              self._defaults["fiducials"]):
            hlayout = self._renderer._dock_add_layout(vertical=False)
            point_lower = point.lower()
            name = f"{point_lower}_weight"
            self._widgets[name] = self._renderer._dock_add_spin_box(
                name=point,
                value=getattr(self, f"_{point_lower}_weight"),
                rng=[0., 100.],
                callback=partial(self._set_point_weight, point=point_lower),
                compact=True,
                double=True,
                layout=hlayout
            )

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
            save=True,
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

    def _configure_status_bar(self):
        self._status_msg = self._renderer._status_bar_add_label("", stretch=1)
        self._status_msg.hide()

    def _clean(self):
        self._renderer = None
        self._coreg = None
        self._widgets.clear()
        self._actors.clear()
        self._surfaces.clear()
        self._defaults.clear()
        self._head_geo = None
        self._redraw = None
        self._status_msg = None

    def close(self):
        """Close interface and cleanup data structure."""
        self._renderer.close()
