# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import inspect
import os
import os.path as op
import platform
import queue
import re
import threading
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
from traitlets import Bool, Float, HasTraits, Instance, Unicode, observe

from .._fiff.constants import FIFF
from .._fiff.meas_info import _empty_info, read_fiducials, read_info, write_fiducials
from .._fiff.open import dir_tree_find, fiff_open
from .._fiff.pick import pick_types
from ..bem import make_bem_solution, write_bem_solution
from ..channels import read_dig_fif
from ..coreg import (
    Coregistration,
    _find_head_bem,
    _is_mri_subject,
    _map_fid_name_to_idx,
    _mri_subject_has_bem,
    bem_fname,
    fid_fname,
    scale_mri,
)
from ..defaults import DEFAULTS
from ..io._read_raw import _get_supported, read_raw
from ..surface import _CheckInside, _DistanceQuery
from ..transforms import (
    Transform,
    _ensure_trans,
    _get_trans,
    _get_transforms_to_coord_frame,
    read_trans,
    rotation_angles,
    write_trans,
)
from ..utils import (
    _check_fname,
    _validate_type,
    check_fname,
    fill_doc,
    get_subjects_dir,
    logger,
    verbose,
)
from ..viz._3d import (
    _plot_head_fiducials,
    _plot_head_shape_points,
    _plot_head_surface,
    _plot_helmet,
    _plot_hpi_coils,
    _plot_mri_fiducials,
    _plot_sensors_3d,
)
from ..viz.backends._utils import _qt_app_exec, _qt_safe_window
from ..viz.utils import safe_event


class _WorkerData:
    def __init__(self, name, params=None):
        self._name = name
        self._params = params


def _get_subjects(sdir):
    # XXX: would be nice to move this function to util
    is_dir = sdir and op.isdir(sdir)
    if is_dir:
        dir_content = os.listdir(sdir)
        subjects = [s for s in dir_content if _is_mri_subject(s, sdir)]
        if len(subjects) == 0:
            subjects.append("")
    else:
        subjects = [""]
    return sorted(subjects)


@fill_doc
class CoregistrationUI(HasTraits):
    """Class for coregistration assisted by graphical interface.

    Parameters
    ----------
    info_file : None | path-like
        The FIFF file with digitizer data for coregistration.
    %(subject)s
    %(subjects_dir)s
    %(fiducials)s
    head_resolution : bool
        If ``True``, use a high-resolution head surface. Defaults to ``False``.
    head_opacity : float
        The opacity of the head surface. Defaults to ``0.8``.
    hpi_coils : bool
        If ``True``, display the HPI coils. Defaults to ``True``.
    head_shape_points : bool
        If ``True``, display the head shape points. Defaults to ``True``.
    eeg_channels : bool
        If ``True``, display the EEG channels. Defaults to ``True``.
    meg_channels : bool
        If ``True``, display the MEG channels. Defaults to ``False``.
    fnirs_channels : bool
        If ``True``, display the fNIRS channels. Defaults to ``True``.
    orient_glyphs : bool
        If ``True``, orient the sensors towards the head surface. Default to ``False``.
    scale_by_distance : bool
        If ``True``, scale the sensors based on their distance to the head surface.
        Defaults to ``True``.
    mark_inside : bool
        If ``True``, mark the head shape points that are inside the head surface
        with a different color. Defaults to ``True``.
    sensor_opacity : float
        The opacity of the sensors between ``0`` and ``1``. Defaults to ``1.``.
    trans : path-like | Transform
        The Head<->MRI transform or the path to its FIF file (``"-trans.fif"``).
    size : tuple
        The dimensions (width, height) of the rendering view. The default is
        ``(800, 600)``.
    bgcolor : tuple of float | str
        The background color as a tuple (red, green, blue) of float
        values between ``0`` and ``1`` or a valid color name (i.e. ``'white'``
        or ``'w'``). Defaults to ``'grey'``.
    show : bool
        Display the window as soon as it is ready. Defaults to ``True``.
    block : bool
        Whether to halt program execution until the GUI has been closed
        (``True``) or not (``False``, default).
    %(fullscreen)s
        The default is ``False``.

        .. versionadded:: 1.1
    %(interaction_scene)s
        Defaults to ``'terrain'``.

        .. versionadded:: 1.0
    %(verbose)s

    Attributes
    ----------
    coreg : mne.coreg.Coregistration
        The coregistration instance used by the graphical interface.
    """

    _subject = Unicode()
    _subjects_dir = Unicode()
    _lock_fids = Bool()
    _current_fiducial = Unicode()
    _info_file = Instance(Path, default_value=Path("."))
    _orient_glyphs = Bool()
    _scale_by_distance = Bool()
    _mark_inside = Bool()
    _hpi_coils = Bool()
    _head_shape_points = Bool()
    _eeg_channels = Bool()
    _meg_channels = Bool()
    _fnirs_channels = Bool()
    _head_resolution = Bool()
    _head_opacity = Float()
    _helmet = Bool()
    _grow_hair = Float()
    _subject_to = Unicode()
    _scale_mode = Unicode()
    _icp_fid_match = Unicode()

    @_qt_safe_window(
        splash="_renderer.figure.splash", window="_renderer.figure.plotter"
    )
    @verbose
    def __init__(
        self,
        info_file,
        *,
        subject=None,
        subjects_dir=None,
        fiducials="auto",
        head_resolution=None,
        head_opacity=None,
        hpi_coils=None,
        head_shape_points=None,
        eeg_channels=None,
        meg_channels=None,
        fnirs_channels=None,
        orient_glyphs=None,
        scale_by_distance=None,
        mark_inside=None,
        sensor_opacity=None,
        trans=None,
        size=None,
        bgcolor=None,
        show=True,
        block=False,
        fullscreen=False,
        interaction="terrain",
        verbose=None,
    ):
        from ..viz.backends.renderer import _get_renderer

        def _get_default(var, val):
            return var if var is not None else val

        self._actors = dict()
        self._surfaces = dict()
        self._widgets = dict()
        self._verbose = verbose
        self._plot_locked = False
        self._params_locked = False
        self._refresh_rate_ms = max(int(round(1000.0 / 60.0)), 1)
        self._redraws_pending = set()
        self._parameter_mutex = threading.Lock()
        self._redraw_mutex = threading.Lock()
        self._job_queue = queue.Queue()
        self._parameter_queue = queue.Queue()
        self._head_geo = None
        self._check_inside = None
        self._nearest = None
        self._coord_frame = "mri"
        self._mouse_no_mvt = -1
        self._to_cf_t = None
        self._omit_hsp_distance = 0.0
        self._fiducials_file = None
        self._trans_modified = False
        self._mri_fids_modified = False
        self._mri_scale_modified = False
        self._accept_close_event = True
        self._fid_colors = tuple(
            DEFAULTS["coreg"][f"{key}_color"] for key in ("lpa", "nasion", "rpa")
        )
        self._defaults = dict(
            size=_get_default(size, (800, 600)),
            bgcolor=_get_default(bgcolor, "grey"),
            orient_glyphs=_get_default(orient_glyphs, True),
            scale_by_distance=_get_default(scale_by_distance, True),
            mark_inside=_get_default(mark_inside, True),
            hpi_coils=_get_default(hpi_coils, True),
            head_shape_points=_get_default(head_shape_points, True),
            eeg_channels=_get_default(eeg_channels, True),
            meg_channels=_get_default(meg_channels, False),
            fnirs_channels=_get_default(fnirs_channels, True),
            head_resolution=_get_default(head_resolution, True),
            head_opacity=_get_default(head_opacity, 0.8),
            helmet=False,
            sensor_opacity=_get_default(sensor_opacity, 1.0),
            fiducials=("LPA", "Nasion", "RPA"),
            fiducial="LPA",
            lock_fids=True,
            grow_hair=0.0,
            subject_to="",
            scale_modes=["None", "uniform", "3-axis"],
            scale_mode="None",
            icp_fid_matches=("nearest", "matched"),
            icp_fid_match="matched",
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
        subjects_dir = str(
            get_subjects_dir(subjects_dir=subjects_dir, raise_error=True)
        )
        subject = _get_default(subject, _get_subjects(subjects_dir)[0])

        # setup the window
        splash = "Initializing coregistration GUI..." if show else False
        self._renderer = _get_renderer(
            size=self._defaults["size"],
            bgcolor=self._defaults["bgcolor"],
            splash=splash,
            fullscreen=fullscreen,
        )
        self._renderer._window_close_connect(self._clean)
        self._renderer._window_close_connect(self._close_callback, after=False)
        self._renderer.set_interaction(interaction)

        # coregistration model setup
        self._immediate_redraw = self._renderer._kind != "qt"
        self._info = info
        self._fiducials = fiducials
        self.coreg = Coregistration(
            info=self._info,
            subject=subject,
            subjects_dir=subjects_dir,
            fiducials=fiducials,
            on_defects="ignore",  # safe due to interactive visual inspection
        )
        fid_accurate = self.coreg._fid_accurate
        for fid in self._defaults["weights"].keys():
            setattr(self, f"_{fid}_weight", self._defaults["weights"][fid])

        # set main traits
        self._set_head_opacity(self._defaults["head_opacity"])
        self._old_head_opacity = self._head_opacity
        self._set_subjects_dir(subjects_dir)
        self._set_subject(subject)
        self._set_info_file(info_file)
        self._set_orient_glyphs(self._defaults["orient_glyphs"])
        self._set_scale_by_distance(self._defaults["scale_by_distance"])
        self._set_mark_inside(self._defaults["mark_inside"])
        self._set_hpi_coils(self._defaults["hpi_coils"])
        self._set_head_shape_points(self._defaults["head_shape_points"])
        self._set_eeg_channels(self._defaults["eeg_channels"])
        self._set_meg_channels(self._defaults["meg_channels"])
        self._set_fnirs_channels(self._defaults["fnirs_channels"])
        self._set_head_resolution(self._defaults["head_resolution"])
        self._set_helmet(self._defaults["helmet"])
        self._set_grow_hair(self._defaults["grow_hair"])
        self._set_omit_hsp_distance(self._defaults["omit_hsp_distance"])
        self._set_icp_n_iterations(self._defaults["icp_n_iterations"])
        self._set_icp_fid_match(self._defaults["icp_fid_match"])

        # configure UI
        self._reset_fitting_parameters()
        self._configure_dialogs()
        self._configure_status_bar()
        self._configure_dock()
        self._configure_picking()
        self._configure_legend()

        # once the docks are initialized
        self._set_current_fiducial(self._defaults["fiducial"])
        self._set_scale_mode(self._defaults["scale_mode"])
        self._set_subject_to(self._defaults["subject_to"])
        if trans is not None:
            self._load_trans(trans)
        self._redraw()  # we need the elements to be present now

        if fid_accurate:
            assert self.coreg._fid_filename is not None
            # _set_fiducials_file() calls _update_fiducials_label()
            # internally
            self._set_fiducials_file(self.coreg._fid_filename)
        else:
            self._set_head_resolution("high")
            self._forward_widget_command("high_res_head", "set_value", True)
            self._set_lock_fids(True)  # hack to make the dig disappear
            self._update_fiducials_label()
            self._update_fiducials()

        self._set_lock_fids(fid_accurate)

        # configure worker
        self._configure_worker()

        # must be done last
        if show:
            self._renderer.show()
        # update the view once shown
        views = {
            True: dict(azimuth=90, elevation=90),  # front
            False: dict(azimuth=180, elevation=90),
        }  # left
        self._renderer.set_camera(distance="auto", **views[self._lock_fids])
        self._redraw()
        # XXX: internal plotter/renderer should not be exposed
        if not self._immediate_redraw:
            self._renderer.plotter.add_callback(self._redraw, self._refresh_rate_ms)
        self._renderer.plotter.show_axes()
        # initialization does not count as modification by the user
        self._trans_modified = False
        self._mri_fids_modified = False
        self._mri_scale_modified = False
        if block and self._renderer._kind != "notebook":
            _qt_app_exec(self._renderer.figure.store["app"])

    def _set_subjects_dir(self, subjects_dir):
        if subjects_dir is None or not subjects_dir:
            return
        try:
            subjects_dir = str(
                _check_fname(
                    subjects_dir,
                    overwrite="read",
                    must_exist=True,
                    need_dir=True,
                )
            )
            subjects = _get_subjects(subjects_dir)
            low_res_path = _find_head_bem(subjects[0], subjects_dir, high_res=False)
            high_res_path = _find_head_bem(subjects[0], subjects_dir, high_res=True)
            valid = low_res_path is not None or high_res_path is not None
        except Exception:
            valid = False
        if valid:
            style = dict(border="initial")
            self._subjects_dir = subjects_dir
        else:
            style = dict(border="2px solid #ff0000")
        self._forward_widget_command("subjects_dir_field", "set_style", style)

    def _set_subject(self, subject):
        self._subject = subject

    def _set_lock_fids(self, state):
        self._lock_fids = bool(state)

    def _set_fiducials_file(self, fname):
        if fname is None:
            fids = "auto"
        else:
            fname = str(
                _check_fname(
                    fname,
                    overwrite="read",
                    must_exist=True,
                    need_dir=False,
                )
            )
            fids, _ = read_fiducials(fname)

        self._fiducials_file = fname
        self.coreg._setup_fiducials(fids)
        self._update_distance_estimation()
        self._update_fiducials_label()
        self._update_fiducials()
        self._reset(keep_trans=True)

        if fname is None:
            self._set_lock_fids(False)
            self._forward_widget_command("reload_mri_fids", "set_enabled", False)
        else:
            self._set_lock_fids(True)
            self._forward_widget_command("reload_mri_fids", "set_enabled", True)
            self._display_message(f"Loading MRI fiducials from {fname}... Done!")

    def _set_current_fiducial(self, fid):
        self._current_fiducial = fid.lower()

    def _set_info_file(self, fname):
        if fname is None:
            return

        # info file can be anything supported by read_raw
        supported = _get_supported()
        try:
            check_fname(
                fname,
                "info",
                tuple(supported),
                endings_err=tuple(supported),
            )
            fname = Path(fname)
            # ctf ds `files` are actually directories
            if fname.suffix == ".ds":
                info_file = _check_fname(
                    fname, overwrite="read", must_exist=True, need_dir=True
                )
            else:
                info_file = _check_fname(
                    fname, overwrite="read", must_exist=True, need_dir=False
                )
            valid = True
        except OSError:
            valid = False
        if valid:
            style = dict(border="initial")
            self._info_file = info_file
        else:
            style = dict(border="2px solid #ff0000")
        self._forward_widget_command("info_file_field", "set_style", style)

    def _set_omit_hsp_distance(self, distance):
        self._omit_hsp_distance = distance

    def _set_orient_glyphs(self, state):
        self._orient_glyphs = bool(state)

    def _set_scale_by_distance(self, state):
        self._scale_by_distance = bool(state)

    def _set_mark_inside(self, state):
        self._mark_inside = bool(state)

    def _set_hpi_coils(self, state):
        self._hpi_coils = bool(state)

    def _set_head_shape_points(self, state):
        self._head_shape_points = bool(state)

    def _set_eeg_channels(self, state):
        self._eeg_channels = bool(state)

    def _set_meg_channels(self, state):
        self._meg_channels = bool(state)

    def _set_fnirs_channels(self, state):
        self._fnirs_channels = bool(state)

    def _set_head_resolution(self, state):
        self._head_resolution = bool(state)

    def _set_head_opacity(self, value):
        self._head_opacity = value

    def _set_helmet(self, state):
        self._helmet = bool(state)

    def _set_grow_hair(self, value):
        self._grow_hair = value

    def _set_subject_to(self, value):
        self._subject_to = value
        self._forward_widget_command("save_subject", "set_enabled", len(value) > 0)
        if self._check_subject_exists():
            style = dict(border="2px solid #ff0000")
        else:
            style = dict(border="initial")
        self._forward_widget_command("subject_to", "set_style", style)

    def _set_scale_mode(self, mode):
        self._scale_mode = mode

    def _set_fiducial(self, value, coord):
        self._mri_fids_modified = True
        fid = self._current_fiducial
        fid_idx = _map_fid_name_to_idx(name=fid)

        coords = ["X", "Y", "Z"]
        coord_idx = coords.index(coord)

        self.coreg.fiducials.dig[fid_idx]["r"][coord_idx] = value / 1e3
        self._update_plot("mri_fids")

    def _set_parameter(self, value, mode_name, coord, plot_locked=False):
        if mode_name == "scale":
            self._mri_scale_modified = True
        else:
            self._trans_modified = True
        if self._params_locked:
            return
        if mode_name == "scale" and self._scale_mode == "uniform":
            with self._lock(params=True):
                self._forward_widget_command(["sY", "sZ"], "set_value", value)
        with self._parameter_mutex:
            self._set_parameter_safe(value, mode_name, coord)
        if not plot_locked:
            self._update_plot("sensors")

    def _set_parameter_safe(self, value, mode_name, coord):
        params = dict(
            rotation=self.coreg._rotation,
            translation=self.coreg._translation,
            scale=self.coreg._scale,
        )
        idx = ["X", "Y", "Z"].index(coord)
        if mode_name == "rotation":
            params[mode_name][idx] = np.deg2rad(value)
        elif mode_name == "translation":
            params[mode_name][idx] = value / 1e3
        else:
            assert mode_name == "scale"
            if self._scale_mode == "uniform":
                params[mode_name][:] = value / 1e2
            else:
                params[mode_name][idx] = value / 1e2
            self._update_plot("head")
        self.coreg._update_params(
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
            "hpi": "_set_hpi_coils",
            "hsp": "_set_head_shape_points",
            "eeg": "_set_eeg_channels",
            "meg": "_set_meg_channels",
            "fnirs": "_set_fnirs_channels",
        }
        if point in funcs.keys():
            getattr(self, funcs[point])(weight > 0)
        setattr(self, f"_{point}_weight", weight)
        setattr(self.coreg, f"_{point}_weight", weight)
        self._update_distance_estimation()

    @observe("_subjects_dir")
    def _subjects_dir_changed(self, change=None):
        # XXX: add coreg.set_subjects_dir
        self.coreg._subjects_dir = self._subjects_dir
        subjects = _get_subjects(self._subjects_dir)

        if self._subject not in subjects:  # Just pick the first available one
            self._subject = subjects[0]

        self._reset()

    @observe("_subject")
    def _subject_changed(self, change=None):
        # XXX: add coreg.set_subject()
        self.coreg._subject = self._subject
        self.coreg._setup_bem()
        self.coreg._setup_fiducials(self._fiducials)
        self._reset()

        default_fid_fname = fid_fname.format(
            subjects_dir=self._subjects_dir, subject=self._subject
        )
        if Path(default_fid_fname).exists():
            fname = default_fid_fname
        else:
            fname = None

        self._set_fiducials_file(fname)
        self._reset_fiducials()

    @observe("_lock_fids")
    def _lock_fids_changed(self, change=None):
        locked_widgets = [
            # MRI fiducials
            "save_mri_fids",
            # View options
            "helmet",
            "meg",
            "head_opacity",
            "high_res_head",
            # Digitization source
            "info_file",
            "grow_hair",
            "omit_distance",
            "omit",
            "reset_omit",
            # Scaling
            "scaling_mode",
            "sX",
            "sY",
            "sZ",
            # Transformation
            "tX",
            "tY",
            "tZ",
            "rX",
            "rY",
            "rZ",
            # Fitting buttons
            "fit_fiducials",
            "fit_icp",
            # Transformation I/O
            "save_trans",
            "load_trans",
            "reset_trans",
            # ICP
            "icp_n_iterations",
            "icp_fid_match",
            "reset_fitting_options",
            # Weights
            "hsp_weight",
            "eeg_weight",
            "hpi_weight",
            "lpa_weight",
            "nasion_weight",
            "rpa_weight",
        ]
        fits_widgets = ["fits_fiducials", "fits_icp"]
        fid_widgets = ["fid_X", "fid_Y", "fid_Z", "fids_file", "fids"]
        if self._lock_fids:
            self._forward_widget_command(locked_widgets, "set_enabled", True)
            self._forward_widget_command(
                "head_opacity", "set_value", self._old_head_opacity
            )
            self._scale_mode_changed()
            self._display_message()
            self._update_distance_estimation()
        else:
            self._old_head_opacity = self._head_opacity
            self._forward_widget_command("head_opacity", "set_value", 1.0)
            self._forward_widget_command(locked_widgets, "set_enabled", False)
            self._forward_widget_command(fits_widgets, "set_enabled", False)
            self._display_message(
                f"Placing MRI fiducials - {self._current_fiducial.upper()}"
            )

        self._set_sensors_visibility(self._lock_fids)
        self._forward_widget_command("lock_fids", "set_value", self._lock_fids)
        self._forward_widget_command(fid_widgets, "set_enabled", not self._lock_fids)

    @observe("_current_fiducial")
    def _current_fiducial_changed(self, change=None):
        self._update_fiducials()
        self._follow_fiducial_view()
        if not self._lock_fids:
            self._display_message(
                f"Placing MRI fiducials - {self._current_fiducial.upper()}"
            )

    @observe("_info_file")
    def _info_file_changed(self, change=None):
        if not self._info_file:
            return
        elif self._info_file.name.endswith((".fif", ".fif.gz")):
            fid, tree, _ = fiff_open(self._info_file)
            fid.close()
            if len(dir_tree_find(tree, FIFF.FIFFB_MEAS_INFO)) > 0:
                self._info = read_info(self._info_file, verbose=False)
            elif len(dir_tree_find(tree, FIFF.FIFFB_ISOTRAK)) > 0:
                self._info = _empty_info(1)
                self._info["dig"] = read_dig_fif(fname=self._info_file).dig
                self._info._unlocked = False
        else:
            self._info = read_raw(self._info_file).info
        # XXX: add coreg.set_info()
        self.coreg._info = self._info
        self.coreg._setup_digs()
        self._reset()

    @observe("_orient_glyphs")
    def _orient_glyphs_changed(self, change=None):
        self._update_plot(["hpi", "hsp", "sensors"])

    @observe("_scale_by_distance")
    def _scale_by_distance_changed(self, change=None):
        self._update_plot(["hpi", "hsp", "sensors"])

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
        self._update_plot("sensors")

    @observe("_meg_channels")
    def _meg_channels_changed(self, change=None):
        self._update_plot("sensors")

    @observe("_fnirs_channels")
    def _fnirs_channels_changed(self, change=None):
        self._update_plot("sensors")

    @observe("_head_resolution")
    def _head_resolution_changed(self, change=None):
        self._update_plot(["head", "hsp"])

    @observe("_head_opacity")
    def _head_opacity_changed(self, change=None):
        if "head" in self._actors:
            self._actors["head"].GetProperty().SetOpacity(self._head_opacity)
            self._renderer._update()

    @observe("_helmet")
    def _helmet_changed(self, change=None):
        self._update_plot("helmet")

    @observe("_grow_hair")
    def _grow_hair_changed(self, change=None):
        self.coreg.set_grow_hair(self._grow_hair)
        self._update_plot("head")
        self._update_plot("hsp")  # inside/outside could change

    @observe("_scale_mode")
    def _scale_mode_changed(self, change=None):
        locked_widgets = ["sX", "sY", "sZ", "fits_icp", "subject_to"]
        mode = None if self._scale_mode == "None" else self._scale_mode
        self.coreg.set_scale_mode(mode)
        if self._lock_fids:
            self._forward_widget_command(
                locked_widgets, "set_enabled", mode is not None
            )
            self._forward_widget_command(
                "fits_fiducials", "set_enabled", mode not in (None, "3-axis")
            )
        if self._scale_mode == "uniform":
            self._forward_widget_command(["sY", "sZ"], "set_enabled", False)

    @observe("_icp_fid_match")
    def _icp_fid_match_changed(self, change=None):
        self.coreg.set_fid_match(self._icp_fid_match)

    def _run_worker(self, queue, jobs):
        while True:
            data = queue.get()
            func = jobs[data._name]
            if data._params is not None:
                func(**data._params)
            else:
                func()
            queue.task_done()

    def _configure_dialogs(self):
        from ..viz.backends.renderer import MNE_3D_BACKEND_TESTING

        for name, buttons in zip(
            ["overwrite_subject", "overwrite_subject_exit"],
            [["Yes", "No"], ["Yes", "Discard", "Cancel"]],
        ):
            self._widgets[name] = self._renderer._dialog_create(
                title="CoregistrationUI",
                text="The name of the output subject used to "
                "save the scaled anatomy already exists.",
                info_text="Do you want to overwrite?",
                callback=self._overwrite_subject_callback,
                buttons=buttons,
                modal=not MNE_3D_BACKEND_TESTING,
            )

    def _configure_worker(self):
        work_plan = {
            "_job_queue": dict(save_subject=self._save_subject),
            "_parameter_queue": dict(set_parameter=self._set_parameter),
        }
        for queue_name, jobs in work_plan.items():
            t = threading.Thread(
                target=partial(
                    self._run_worker,
                    queue=getattr(self, queue_name),
                    jobs=jobs,
                )
            )
            t.daemon = True
            t.start()

    def _configure_picking(self):
        self._renderer._update_picking_callback(
            self._on_mouse_move,
            self._on_button_press,
            self._on_button_release,
            self._on_pick,
        )

    def _configure_legend(self):
        colors = [
            np.array(DEFAULTS["coreg"][f"{fid.lower()}_color"]).astype(float)
            for fid in self._defaults["fiducials"]
        ]
        labels = list(zip(self._defaults["fiducials"], colors))
        mri_fids_legend_actor = self._renderer.legend(labels=labels)
        self._update_actor("mri_fids_legend", mri_fids_legend_actor)

    @safe_event
    @verbose
    def _redraw(self, *, verbose=None):
        if not self._redraws_pending:
            return
        draw_map = dict(
            head=self._add_head_surface,
            mri_fids=self._add_mri_fiducials,
            hsp=self._add_head_shape_points,
            hpi=self._add_hpi_coils,
            sensors=self._add_channels,
            head_fids=self._add_head_fiducials,
            helmet=self._add_helmet,
        )
        with self._redraw_mutex:
            # We need at least "head" before "hsp", because the grow_hair param
            # for head sets the rr that are used for inside/outside hsp
            redraws_ordered = sorted(
                self._redraws_pending, key=lambda key: list(draw_map).index(key)
            )
            logger.debug(f"Redrawing {redraws_ordered}")
            for ki, key in enumerate(redraws_ordered):
                logger.debug(f"{ki}. Drawing {repr(key)}")
                draw_map[key]()
            self._redraws_pending.clear()
            self._renderer._update()
            # necessary for MacOS
            if platform.system() == "Darwin":
                self._renderer._process_events()

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
        cell = [
            vtk_cell.GetPointId(point_id)
            for point_id in range(vtk_cell.GetNumberOfPoints())
        ]
        vertices = mesh.points[cell]
        idx = np.argmin(abs(vertices - pos), axis=0)
        vertex_id = cell[idx[0]]

        fiducials = [s.lower() for s in self._defaults["fiducials"]]
        idx = fiducials.index(self._current_fiducial.lower())
        # XXX: add coreg.set_fids
        self.coreg._fid_points[idx] = self._surfaces["head"].points[vertex_id]
        self.coreg._reset_fiducials()
        self._update_fiducials()
        self._update_plot("mri_fids")

    def _reset_fitting_parameters(self):
        self._forward_widget_command(
            "icp_n_iterations", "set_value", self._defaults["icp_n_iterations"]
        )
        self._forward_widget_command(
            "icp_fid_match", "set_value", self._defaults["icp_fid_match"]
        )
        weights_widgets = [f"{w}_weight" for w in self._defaults["weights"].keys()]
        self._forward_widget_command(
            weights_widgets, "set_value", list(self._defaults["weights"].values())
        )

    def _reset_fiducials(self):
        self._set_current_fiducial(self._defaults["fiducial"])

    def _omit_hsp(self):
        self.coreg.omit_head_shape_points(self._omit_hsp_distance / 1e3)
        n_omitted = np.sum(~self.coreg._extra_points_filter)
        n_remaining = len(self.coreg._dig_dict["hsp"]) - n_omitted
        self._update_plot("hsp")
        self._update_distance_estimation()
        self._display_message(
            f"{n_omitted} head shape points omitted, {n_remaining} remaining."
        )

    def _reset_omit_hsp_filter(self):
        self.coreg._extra_points_filter = None
        self.coreg._update_params(force_update=True)
        self._update_plot("hsp")
        self._update_distance_estimation()
        n_total = len(self.coreg._dig_dict["hsp"])
        self._display_message(
            f"No head shape point is omitted, the total is {n_total}."
        )

    @verbose
    def _update_plot(self, changes="all", verbose=None):
        # Update list of things that need to be updated/plotted (and maybe
        # draw them immediately)
        try:
            fun_name = inspect.currentframe().f_back.f_back.f_code.co_name
        except Exception:  # just in case one of these attrs is missing
            fun_name = "unknown"
        logger.debug(f"Updating plots based on {fun_name}: {repr(changes)}")
        if self._plot_locked:
            return
        if self._info is None:
            changes = ["head", "mri_fids"]
            self._to_cf_t = dict(mri=dict(trans=np.eye(4)), head=None)
        else:
            self._to_cf_t = _get_transforms_to_coord_frame(
                self._info, self.coreg.trans, coord_frame=self._coord_frame
            )
        all_keys = (
            "head",
            "mri_fids",  # MRI first
            "hsp",
            "hpi",
            "sensors",
            "head_fids",  # then dig
            "helmet",
        )
        if changes == "all":
            changes = list(all_keys)
        elif changes == "sensors":
            changes = all_keys[2:]  # omit MRI ones
        elif isinstance(changes, str):
            changes = [changes]
        changes = set(changes)
        # ideally we would maybe have this in:
        # with self._redraw_mutex:
        # it would reduce "jerkiness" of the updates, but this should at least
        # work okay
        bad = changes.difference(set(all_keys))
        assert len(bad) == 0, f"Unknown changes: {bad}"
        self._redraws_pending.update(changes)
        if self._immediate_redraw:
            self._redraw()

    @contextmanager
    def _lock(self, plot=False, params=False, scale_mode=False, fitting=False):
        """Which part of the UI to temporarily disable."""
        if plot:
            old_plot_locked = self._plot_locked
            self._plot_locked = True
        if params:
            old_params_locked = self._params_locked
            self._params_locked = True
        if scale_mode:
            old_scale_mode = self.coreg._scale_mode
            self.coreg._scale_mode = None
        if fitting:
            widgets = [
                "sX",
                "sY",
                "sZ",
                "tX",
                "tY",
                "tZ",
                "rX",
                "rY",
                "rZ",
                "fit_icp",
                "fit_fiducials",
                "fits_icp",
                "fits_fiducials",
            ]
            states = [
                self._forward_widget_command(
                    w, "is_enabled", None, input_value=False, output_value=True
                )
                for w in widgets
            ]
            self._forward_widget_command(widgets, "set_enabled", False)
        try:
            yield
        finally:
            if plot:
                self._plot_locked = old_plot_locked
            if params:
                self._params_locked = old_params_locked
            if scale_mode:
                self.coreg._scale_mode = old_scale_mode
            if fitting:
                for idx, w in enumerate(widgets):
                    self._forward_widget_command(w, "set_enabled", states[idx])

    def _display_message(self, msg=""):
        self._forward_widget_command("status_message", "set_value", msg)
        self._forward_widget_command("status_message", "show", None, input_value=False)
        self._forward_widget_command(
            "status_message", "update", None, input_value=False
        )
        if msg:
            logger.info(msg)

    def _follow_fiducial_view(self):
        fid = self._current_fiducial.lower()
        view = dict(lpa="left", rpa="right", nasion="front")
        kwargs = dict(front=(90.0, 90.0), left=(180, 90), right=(0.0, 90))
        kwargs = dict(zip(("azimuth", "elevation"), kwargs[view[fid]]))
        if not self._lock_fids:
            self._renderer.set_camera(distance="auto", **kwargs)

    def _update_fiducials(self):
        fid = self._current_fiducial
        if not fid:
            return

        idx = _map_fid_name_to_idx(name=fid)
        val = self.coreg.fiducials.dig[idx]["r"] * 1e3

        with self._lock(plot=True):
            self._forward_widget_command(["fid_X", "fid_Y", "fid_Z"], "set_value", val)

    def _update_distance_estimation(self):
        value = (
            self.coreg._get_fiducials_distance_str()
            + "\n"
            + self.coreg._get_point_distance_str()
        )
        dists = self.coreg.compute_dig_mri_distances() * 1e3
        if self._hsp_weight > 0:
            if len(dists) == 0:
                value += "\nNo head shape points found."
            else:
                value += (
                    "\nHSP <-> MRI (mean/min/max): "
                    f"{np.mean(dists):.2f} "
                    f"/ {np.min(dists):.2f} / {np.max(dists):.2f} mm"
                )
        self._forward_widget_command("fit_label", "set_value", value)

    def _update_parameters(self):
        with self._lock(plot=True, params=True):
            # rotation
            deg = np.rad2deg(self.coreg._rotation)
            logger.debug(f"  Rotation:    {deg}")
            self._forward_widget_command(["rX", "rY", "rZ"], "set_value", deg)
            # translation
            mm = self.coreg._translation * 1e3
            logger.debug(f"  Translation: {mm}")
            self._forward_widget_command(["tX", "tY", "tZ"], "set_value", mm)
            # scale
            sc = self.coreg._scale * 1e2
            logger.debug(f"  Scale:       {sc}")
            self._forward_widget_command(["sX", "sY", "sZ"], "set_value", sc)

    def _reset(self, keep_trans=False):
        """Refresh the scene, and optionally reset transformation & scaling.

        Parameters
        ----------
        keep_trans : bool
            Whether to retain translation, rotation, and scaling; or reset them
            to their default values (no translation, no rotation, no scaling).
        """
        if not keep_trans:
            self.coreg.set_scale(self.coreg._default_parameters[6:9])
            self.coreg.set_rotation(self.coreg._default_parameters[:3])
            self.coreg.set_translation(self.coreg._default_parameters[3:6])
        self._update_plot()
        self._update_parameters()
        self._update_distance_estimation()

    def _forward_widget_command(
        self, names, command, value, input_value=True, output_value=False
    ):
        """Invoke a method of one or more widgets if the widgets exist.

        Parameters
        ----------
        names : str | array-like of str
            The widget names to operate on.
        command : str
            The method to invoke.
        value : object | array-like
            The value(s) to pass to the method.
        input_value : bool
            Whether the ``command`` accepts a ``value``. If ``False``, no
            ``value`` will be passed to ``command``.
        output_value : bool
            Whether to return the return value of ``command``.

        Returns
        -------
        ret : object | None
            ``None`` if ``output_value`` is ``False``, and the return value of
            ``command`` otherwise.
        """
        _validate_type(item=names, types=(str, list), item_name="names")
        if isinstance(names, str):
            names = [names]

        if not isinstance(value, str | float | int | dict | type(None)):
            value = list(value)
            assert len(names) == len(value)

        for idx, name in enumerate(names):
            val = value[idx] if isinstance(value, list) else value
            if name in self._widgets and self._widgets[name] is not None:
                if input_value:
                    ret = getattr(self._widgets[name], command)(val)
                else:
                    ret = getattr(self._widgets[name], command)()
                if output_value:
                    return ret

    def _set_sensors_visibility(self, state):
        sensors = [
            "head_fiducials",
            "hpi_coils",
            "head_shape_points",
            "sensors",
            "helmet",
        ]
        for sensor in sensors:
            if sensor in self._actors and self._actors[sensor] is not None:
                actors = self._actors[sensor]
                actors = actors if isinstance(actors, list) else [actors]
                for actor in actors:
                    actor.SetVisibility(state)
        self._renderer._update()

    def _update_actor(self, actor_name, actor):
        # XXX: internal plotter/renderer should not be exposed
        # Work around PyVista sequential update bug with iterable until > 0.42.3 is req
        # https://github.com/pyvista/pyvista/pull/5046
        actors = self._actors.get(actor_name) or []  # convert None to list
        if not isinstance(actors, list):
            actors = [actors]
        for this_actor in actors:
            self._renderer.plotter.remove_actor(this_actor, render=False)
        self._actors[actor_name] = actor

    def _add_mri_fiducials(self):
        mri_fids_actors = _plot_mri_fiducials(
            self._renderer,
            self.coreg._fid_points,
            self._subjects_dir,
            self._subject,
            self._to_cf_t,
            self._fid_colors,
        )
        # disable picking on the markers
        for actor in mri_fids_actors:
            actor.SetPickable(False)
        self._update_actor("mri_fiducials", mri_fids_actors)

    def _add_head_fiducials(self):
        head_fids_actors = _plot_head_fiducials(
            self._renderer, self._info, self._to_cf_t, self._fid_colors
        )
        self._update_actor("head_fiducials", head_fids_actors)

    def _add_hpi_coils(self):
        if self._hpi_coils:
            hpi_actors = _plot_hpi_coils(
                self._renderer,
                self._info,
                self._to_cf_t,
                opacity=self._defaults["sensor_opacity"],
                scale=DEFAULTS["coreg"]["extra_scale"],
                orient_glyphs=self._orient_glyphs,
                scale_by_distance=self._scale_by_distance,
                surf=self._head_geo,
                check_inside=self._check_inside,
                nearest=self._nearest,
            )
        else:
            hpi_actors = None
        self._update_actor("hpi_coils", hpi_actors)

    def _add_head_shape_points(self):
        if self._head_shape_points:
            hsp_actors = _plot_head_shape_points(
                self._renderer,
                self._info,
                self._to_cf_t,
                opacity=self._defaults["sensor_opacity"],
                orient_glyphs=self._orient_glyphs,
                scale_by_distance=self._scale_by_distance,
                mark_inside=self._mark_inside,
                surf=self._head_geo,
                mask=self.coreg._extra_points_filter,
                check_inside=self._check_inside,
                nearest=self._nearest,
            )
        else:
            hsp_actors = None
        self._update_actor("head_shape_points", hsp_actors)

    def _add_channels(self):
        plot_types = dict(eeg=False, meg=False, fnirs=False)
        if self._eeg_channels:
            plot_types["eeg"] = ["original"]
        if self._meg_channels:
            plot_types["meg"] = ["sensors"]
        if self._fnirs_channels:
            plot_types["fnirs"] = ["sources", "detectors"]
        sensor_alpha = dict(
            eeg=self._defaults["sensor_opacity"],
            fnirs=self._defaults["sensor_opacity"],
            meg=0.25,
        )
        picks = pick_types(
            self._info,
            ref_meg=False,
            meg=True,
            eeg=True,
            fnirs=True,
            exclude=(),
        )
        these_actors = _plot_sensors_3d(
            self._renderer,
            self._info,
            self._to_cf_t,
            picks=picks,
            warn_meg=False,
            head_surf=self._head_geo,
            units="m",
            sensor_alpha=sensor_alpha,
            orient_glyphs=self._orient_glyphs,
            scale_by_distance=self._scale_by_distance,
            surf=self._head_geo,
            check_inside=self._check_inside,
            nearest=self._nearest,
            **plot_types,
        )
        sens_actors = sum((these_actors or {}).values(), list())
        self._update_actor("sensors", sens_actors)

    def _add_head_surface(self):
        bem = None
        if self._head_resolution:
            surface = "head-dense"
            key = "high"
        else:
            surface = "head"
            key = "low"
        try:
            head_actor, head_surf, _ = _plot_head_surface(
                self._renderer,
                surface,
                self._subject,
                self._subjects_dir,
                bem,
                self._coord_frame,
                self._to_cf_t,
                alpha=self._head_opacity,
            )
        except OSError:
            head_actor, head_surf, _ = _plot_head_surface(
                self._renderer,
                "head",
                self._subject,
                self._subjects_dir,
                bem,
                self._coord_frame,
                self._to_cf_t,
                alpha=self._head_opacity,
            )
            key = "low"
        self._update_actor("head", head_actor)
        # mark head surface mesh to restrict picking
        head_surf._picking_target = True
        # We need to use _get_processed_mri_points to incorporate grow_hair
        rr = self.coreg._get_processed_mri_points(key) * self.coreg._scale.T
        head_surf.points = rr
        head_surf.compute_normals()
        self._surfaces["head"] = head_surf
        tris = self._surfaces["head"].faces.reshape(-1, 4)[:, 1:]
        assert tris.ndim == 2 and tris.shape[1] == 3, tris.shape
        nn = self._surfaces["head"].point_normals
        assert nn.shape == (len(rr), 3), nn.shape
        self._head_geo = dict(rr=rr, tris=tris, nn=nn)
        self._check_inside = _CheckInside(head_surf, mode="pyvista")
        self._nearest = _DistanceQuery(rr)

    def _add_helmet(self):
        if self._helmet:
            logger.debug("Drawing helmet")
            head_mri_t = _get_trans(self.coreg.trans, "head", "mri")[0]
            helmet_actor, _, _ = _plot_helmet(
                self._renderer, self._info, self._to_cf_t, head_mri_t, self._coord_frame
            )
        else:
            helmet_actor = None
        self._update_actor("helmet", helmet_actor)

    def _fit_fiducials(self):
        with self._lock(scale_mode=True):
            self._fits_fiducials()

    def _fits_fiducials(self):
        with self._lock(params=True, fitting=True):
            start = time.time()
            self.coreg.fit_fiducials(
                lpa_weight=self._lpa_weight,
                nasion_weight=self._nasion_weight,
                rpa_weight=self._rpa_weight,
                verbose=self._verbose,
            )
            end = time.time()
            self._display_message(
                f"Fitting fiducials finished in {end - start:.2f} seconds."
            )
            self._update_plot("sensors")
            self._update_parameters()
            self._update_distance_estimation()

    def _fit_icp(self):
        with self._lock(scale_mode=True):
            self._fit_icp_real(update_head=False)

    def _fits_icp(self):
        self._fit_icp_real(update_head=True)

    def _fit_icp_real(self, *, update_head):
        with self._lock(params=True, fitting=True):
            self._current_icp_iterations = 0
            updates = ["hsp", "hpi", "sensors", "head_fids", "helmet"]
            if update_head:
                updates.insert(0, "head")

            def callback(iteration, n_iterations):
                self._display_message(f"Fitting ICP - iteration {iteration + 1}")
                self._update_plot(updates)
                self._current_icp_iterations += 1
                self._update_distance_estimation()
                self._update_parameters()
                self._renderer._process_events()  # allow a draw or cancel

            start = time.time()
            self.coreg.fit_icp(
                n_iterations=self._icp_n_iterations,
                lpa_weight=self._lpa_weight,
                nasion_weight=self._nasion_weight,
                rpa_weight=self._rpa_weight,
                callback=callback,
                verbose=self._verbose,
            )
            end = time.time()
            self._display_message()
            self._display_message(
                f"Fitting ICP finished in {end - start:.2f} seconds and "
                f"{self._current_icp_iterations} iterations."
            )
            del self._current_icp_iterations

    def _task_save_subject(self):
        from ..viz.backends.renderer import MNE_3D_BACKEND_TESTING

        if MNE_3D_BACKEND_TESTING:
            self._save_subject()
        else:
            self._job_queue.put(_WorkerData("save_subject", None))

    def _task_set_parameter(self, value, mode_name, coord):
        from ..viz.backends.renderer import MNE_3D_BACKEND_TESTING

        if MNE_3D_BACKEND_TESTING:
            self._set_parameter(value, mode_name, coord, self._plot_locked)
        else:
            self._parameter_queue.put(
                _WorkerData(
                    "set_parameter",
                    dict(
                        value=value,
                        mode_name=mode_name,
                        coord=coord,
                        plot_locked=self._plot_locked,
                    ),
                )
            )

    def _overwrite_subject_callback(self, button_name):
        if button_name == "Yes":
            self._save_subject_callback(overwrite=True)
        elif button_name == "Cancel":
            self._accept_close_event = False
        else:
            assert button_name == "No" or button_name == "Discard"

    def _check_subject_exists(self):
        if not self._subject_to:
            return False
        subject_dirname = os.path.join("{subjects_dir}", "{subject}")
        dest = subject_dirname.format(
            subject=self._subject_to, subjects_dir=self._subjects_dir
        )
        return os.path.exists(dest)

    def _save_subject(self, exit_mode=False):
        dialog = "overwrite_subject_exit" if exit_mode else "overwrite_subject"
        if self._check_subject_exists():
            self._forward_widget_command(dialog, "show", True)
        else:
            self._save_subject_callback()

    def _save_subject_callback(self, overwrite=False):
        self._display_message(f"Saving {self._subject_to}...")
        default_cursor = self._renderer._window_get_cursor()
        self._renderer._window_set_cursor(
            self._renderer._window_new_cursor("WaitCursor")
        )

        # prepare bem
        bem_names = []
        if self._scale_mode != "None":
            can_prepare_bem = _mri_subject_has_bem(self._subject, self._subjects_dir)
        else:
            can_prepare_bem = False
        if can_prepare_bem:
            pattern = bem_fname.format(
                subjects_dir=self._subjects_dir, subject=self._subject, name="(.+-bem)"
            )
            bem_dir, pattern = os.path.split(pattern)
            for filename in os.listdir(bem_dir):
                match = re.match(pattern, filename)
                if match:
                    bem_names.append(match.group(1))

        # save the scaled MRI
        try:
            self._display_message(f"Scaling {self._subject_to}...")
            scale_mri(
                subject_from=self._subject,
                subject_to=self._subject_to,
                scale=self.coreg._scale,
                overwrite=overwrite,
                subjects_dir=self._subjects_dir,
                skip_fiducials=True,
                labels=True,
                annot=True,
                on_defects="ignore",
            )
        except Exception:
            logger.error(f"Error scaling {self._subject_to}")
            bem_names = []
        else:
            self._display_message(f"Scaling {self._subject_to}... Done!")

        # Precompute BEM solutions
        for bem_name in bem_names:
            try:
                self._display_message(f"Computing {bem_name} solution...")
                bem_file = bem_fname.format(
                    subjects_dir=self._subjects_dir,
                    subject=self._subject_to,
                    name=bem_name,
                )
                bemsol = make_bem_solution(bem_file)
                write_bem_solution(bem_file[:-4] + "-sol.fif", bemsol)
            except Exception:
                logger.error(f"Error computing {bem_name} solution")
            else:
                self._display_message(f"Computing {bem_name} solution... Done!")
        self._display_message(f"Saving {self._subject_to}... Done!")
        self._renderer._window_set_cursor(default_cursor)
        self._mri_scale_modified = False

    def _save_mri_fiducials(self, fname):
        self._display_message(f"Saving {fname}...")
        dig_montage = self.coreg.fiducials
        write_fiducials(
            fname=fname, pts=dig_montage.dig, coord_frame="mri", overwrite=True
        )
        self._set_fiducials_file(fname)
        self._display_message(f"Saving {fname}... Done!")
        self._mri_fids_modified = False

    def _save_trans(self, fname):
        write_trans(fname, self.coreg.trans, overwrite=True)
        self._display_message(f"{fname} transform file is saved.")
        self._trans_modified = False

    def _load_trans(self, trans):
        if not isinstance(trans, Transform):
            trans = read_trans(trans, return_all=True)
        mri_head_t = _ensure_trans(trans, "mri", "head")["trans"]
        rot_x, rot_y, rot_z = rotation_angles(mri_head_t)
        x, y, z = mri_head_t[:3, 3]
        self.coreg._update_params(
            rot=np.array([rot_x, rot_y, rot_z]),
            tra=np.array([x, y, z]),
        )
        self._update_parameters()
        self._update_distance_estimation()
        self._update_plot()
        self._display_message(f"{trans} transform file is loaded.")

    def _update_fiducials_label(self):
        if self._fiducials_file is None:
            text = (
                "<p><strong>No custom MRI fiducials loaded!</strong></p>"
                "<p>MRI fiducials could not be found in the standard "
                "location. The displayed initial MRI fiducial locations "
                "(diamonds) were derived from fsaverage. Place, lock, and "
                "save fiducials to discard this message.</p>"
            )
        else:
            assert self._fiducials_file == fid_fname.format(
                subjects_dir=self._subjects_dir, subject=self._subject
            )
            assert self.coreg._fid_accurate is True
            text = (
                f"<p><strong>MRI fiducials (diamonds) loaded from "
                f"standard location:</strong></p>"
                f"<p>{self._fiducials_file}</p>"
            )

        self._forward_widget_command("mri_fiducials_label", "set_value", text)

    def _configure_dock(self):
        if self._renderer._kind == "notebook":
            collapse = True  # collapsible and collapsed
        else:
            collapse = None  # not collapsible
        self._renderer._dock_initialize(name="Input", area="left", max_width="375px")
        mri_subject_layout = self._renderer._dock_add_group_box(
            name="MRI Subject",
            collapse=collapse,
        )
        subjects_dir_layout = self._renderer._dock_add_layout(vertical=False)
        self._widgets["subjects_dir_field"] = self._renderer._dock_add_text(
            name="subjects_dir_field",
            value=self._subjects_dir,
            placeholder="Subjects Directory",
            callback=self._set_subjects_dir,
            layout=subjects_dir_layout,
        )
        self._widgets["subjects_dir"] = self._renderer._dock_add_file_button(
            name="subjects_dir",
            desc="Load",
            func=self._set_subjects_dir,
            is_directory=True,
            icon=True,
            tooltip="Load the path to the directory containing the "
            "FreeSurfer subjects",
            layout=subjects_dir_layout,
        )
        self._renderer._layout_add_widget(
            layout=mri_subject_layout,
            widget=subjects_dir_layout,
        )
        self._widgets["subject"] = self._renderer._dock_add_combo_box(
            name="Subject",
            value=self._subject,
            rng=_get_subjects(self._subjects_dir),
            callback=self._set_subject,
            compact=True,
            tooltip="Select the FreeSurfer subject name",
            layout=mri_subject_layout,
        )

        mri_fiducials_layout = self._renderer._dock_add_group_box(
            name="MRI Fiducials",
            collapse=collapse,
        )
        # Add MRI fiducials I/O widgets
        self._widgets["mri_fiducials_label"] = self._renderer._dock_add_label(
            value="",  # Will be filled via _update_fiducials_label()
            layout=mri_fiducials_layout,
            selectable=True,
        )
        # Reload & Save buttons go into their own layout widget
        mri_fiducials_button_layout = self._renderer._dock_add_layout(vertical=False)
        self._renderer._layout_add_widget(
            layout=mri_fiducials_layout, widget=mri_fiducials_button_layout
        )
        self._widgets["reload_mri_fids"] = self._renderer._dock_add_button(
            name="Reload MRI Fid.",
            callback=lambda: self._set_fiducials_file(self._fiducials_file),
            tooltip="Reload MRI fiducials from the standard location",
            layout=mri_fiducials_button_layout,
        )
        # Disable reload button until we've actually loaded a fiducial file
        # (happens in _set_fiducials_file method)
        self._forward_widget_command("reload_mri_fids", "set_enabled", False)

        self._widgets["save_mri_fids"] = self._renderer._dock_add_button(
            name="Save MRI Fid.",
            callback=lambda: self._save_mri_fiducials(
                fid_fname.format(subjects_dir=self._subjects_dir, subject=self._subject)
            ),
            tooltip="Save MRI fiducials to the standard location. Fiducials "
            "must be locked first!",
            layout=mri_fiducials_button_layout,
        )
        self._widgets["lock_fids"] = self._renderer._dock_add_check_box(
            name="Lock fiducials",
            value=self._lock_fids,
            callback=self._set_lock_fids,
            tooltip="Lock/Unlock interactive fiducial editing",
            layout=mri_fiducials_layout,
        )
        self._widgets["fids"] = self._renderer._dock_add_radio_buttons(
            value=self._defaults["fiducial"],
            rng=self._defaults["fiducials"],
            callback=self._set_current_fiducial,
            vertical=False,
            layout=mri_fiducials_layout,
        )
        fiducial_coords_layout = self._renderer._dock_add_layout()
        for coord in ("X", "Y", "Z"):
            name = f"fid_{coord}"
            self._widgets[name] = self._renderer._dock_add_spin_box(
                name=coord,
                value=0.0,
                rng=[-1e3, 1e3],
                callback=partial(
                    self._set_fiducial,
                    coord=coord,
                ),
                compact=True,
                double=True,
                step=1,
                tooltip=f"Set the {coord} fiducial coordinate",
                layout=fiducial_coords_layout,
            )
        self._renderer._layout_add_widget(mri_fiducials_layout, fiducial_coords_layout)

        dig_source_layout = self._renderer._dock_add_group_box(
            name="Info source with digitization",
            collapse=collapse,
        )
        info_file_layout = self._renderer._dock_add_layout(vertical=False)
        self._widgets["info_file_field"] = self._renderer._dock_add_text(
            name="info_file_field",
            value=self._info_file,
            placeholder="Path to info",
            callback=self._set_info_file,
            layout=info_file_layout,
        )
        self._widgets["info_file"] = self._renderer._dock_add_file_button(
            name="info_file",
            desc="Load",
            func=self._set_info_file,
            icon=True,
            tooltip="Load the FIFF file with digitization data for coregistration",
            layout=info_file_layout,
        )
        self._renderer._layout_add_widget(
            layout=dig_source_layout,
            widget=info_file_layout,
        )
        self._widgets["grow_hair"] = self._renderer._dock_add_spin_box(
            name="Grow Hair (mm)",
            value=self._grow_hair,
            rng=[0.0, 10.0],
            callback=self._set_grow_hair,
            tooltip="Compensate for hair on the digitizer head shape",
            layout=dig_source_layout,
        )
        omit_hsp_layout_1 = self._renderer._dock_add_layout(vertical=False)
        omit_hsp_layout_2 = self._renderer._dock_add_layout(vertical=False)
        self._widgets["omit_distance"] = self._renderer._dock_add_spin_box(
            name="Omit Distance (mm)",
            value=self._omit_hsp_distance,
            rng=[0.0, 100.0],
            callback=self._set_omit_hsp_distance,
            tooltip="Set the head shape points exclusion distance",
            layout=omit_hsp_layout_1,
        )
        self._widgets["omit"] = self._renderer._dock_add_button(
            name="Omit",
            callback=self._omit_hsp,
            tooltip="Exclude the head shape points that are far away from "
            "the MRI head",
            layout=omit_hsp_layout_2,
        )
        self._widgets["reset_omit"] = self._renderer._dock_add_button(
            name="Reset",
            callback=self._reset_omit_hsp_filter,
            tooltip="Reset all excluded head shape points",
            layout=omit_hsp_layout_2,
        )
        self._renderer._layout_add_widget(dig_source_layout, omit_hsp_layout_1)
        self._renderer._layout_add_widget(dig_source_layout, omit_hsp_layout_2)

        view_options_layout = self._renderer._dock_add_group_box(
            name="View Options",
            collapse=collapse,
        )
        self._widgets["helmet"] = self._renderer._dock_add_check_box(
            name="Show MEG helmet",
            value=self._helmet,
            callback=self._set_helmet,
            tooltip="Enable/Disable MEG helmet",
            layout=view_options_layout,
        )
        self._widgets["meg"] = self._renderer._dock_add_check_box(
            name="Show MEG sensors",
            value=self._meg_channels,
            callback=self._set_meg_channels,
            tooltip="Enable/Disable MEG sensors",
            layout=view_options_layout,
        )
        self._widgets["high_res_head"] = self._renderer._dock_add_check_box(
            name="Show high-resolution head",
            value=self._head_resolution,
            callback=self._set_head_resolution,
            tooltip="Enable/Disable high resolution head surface",
            layout=view_options_layout,
        )
        self._widgets["head_opacity"] = self._renderer._dock_add_slider(
            name="Head opacity",
            value=self._head_opacity,
            rng=[0.25, 1.0],
            callback=self._set_head_opacity,
            compact=True,
            double=True,
            layout=view_options_layout,
        )
        self._renderer._dock_add_stretch()

        self._renderer._dock_initialize(
            name="Parameters", area="right", max_width="375px"
        )
        mri_scaling_layout = self._renderer._dock_add_group_box(
            name="MRI Scaling",
            collapse=collapse,
        )
        self._widgets["scaling_mode"] = self._renderer._dock_add_combo_box(
            name="Scaling Mode",
            value=self._defaults["scale_mode"],
            rng=self._defaults["scale_modes"],
            callback=self._set_scale_mode,
            tooltip="Select the scaling mode",
            compact=True,
            layout=mri_scaling_layout,
        )
        scale_params_layout = self._renderer._dock_add_group_box(
            name="Scaling Parameters",
            layout=mri_scaling_layout,
        )
        coords = ["X", "Y", "Z"]
        for coord in coords:
            name = f"s{coord}"
            attr = getattr(self.coreg, "_scale")
            self._widgets[name] = self._renderer._dock_add_spin_box(
                name=name,
                value=attr[coords.index(coord)] * 1e2,
                rng=[1.0, 10000.0],  # percent
                callback=partial(
                    self._set_parameter,
                    mode_name="scale",
                    coord=coord,
                ),
                compact=True,
                double=True,
                step=1,
                tooltip=f"Set the {coord} scaling parameter (in %)",
                layout=scale_params_layout,
            )

        fit_scale_layout = self._renderer._dock_add_layout(vertical=False)
        self._widgets["fits_fiducials"] = self._renderer._dock_add_button(
            name="Fit fiducials with scaling",
            callback=self._fits_fiducials,
            tooltip="Find MRI scaling, rotation, and translation to fit all "
            "3 fiducials",
            layout=fit_scale_layout,
        )
        self._widgets["fits_icp"] = self._renderer._dock_add_button(
            name="Fit ICP with scaling",
            callback=self._fits_icp,
            tooltip="Find MRI scaling, rotation, and translation to match the "
            "head shape points",
            layout=fit_scale_layout,
        )
        self._renderer._layout_add_widget(scale_params_layout, fit_scale_layout)
        subject_to_layout = self._renderer._dock_add_layout(vertical=False)
        self._widgets["subject_to"] = self._renderer._dock_add_text(
            name="subject-to",
            value=self._subject_to,
            placeholder="subject name",
            callback=self._set_subject_to,
            layout=subject_to_layout,
        )
        self._widgets["save_subject"] = self._renderer._dock_add_button(
            name="Save scaled anatomy",
            callback=self._task_save_subject,
            tooltip="Save scaled anatomy",
            layout=subject_to_layout,
        )
        self._renderer._layout_add_widget(mri_scaling_layout, subject_to_layout)
        param_layout = self._renderer._dock_add_group_box(
            name="Translation (t) and Rotation (r)",
            collapse=collapse,
        )
        for coord in coords:
            coord_layout = self._renderer._dock_add_layout(vertical=False)
            for mode, mode_name in (("t", "Translation"), ("r", "Rotation")):
                name = f"{mode}{coord}"
                attr = getattr(self.coreg, f"_{mode_name.lower()}")
                rng = [-360, 360] if mode_name == "Rotation" else [-100, 100]
                unit = "°" if mode_name == "Rotation" else "mm"
                self._widgets[name] = self._renderer._dock_add_spin_box(
                    name=name,
                    value=attr[coords.index(coord)] * 1e3,
                    rng=np.array(rng),
                    callback=partial(
                        self._task_set_parameter,
                        mode_name=mode_name.lower(),
                        coord=coord,
                    ),
                    compact=True,
                    double=True,
                    step=1,
                    tooltip=f"Set the {coord} {mode_name.lower()}"
                    f" parameter (in {unit})",
                    layout=coord_layout,
                )
            self._renderer._layout_add_widget(param_layout, coord_layout)

        fit_layout = self._renderer._dock_add_layout(vertical=False)
        self._widgets["fit_fiducials"] = self._renderer._dock_add_button(
            name="Fit fiducials",
            callback=self._fit_fiducials,
            tooltip="Find rotation and translation to fit all 3 fiducials",
            layout=fit_layout,
        )
        self._widgets["fit_icp"] = self._renderer._dock_add_button(
            name="Fit ICP",
            callback=self._fit_icp,
            tooltip="Find rotation and translation to match the head shape points",
            layout=fit_layout,
        )
        self._renderer._layout_add_widget(param_layout, fit_layout)
        trans_layout = self._renderer._dock_add_group_box(
            name="HEAD <> MRI Transform",
            collapse=collapse,
        )
        save_trans_layout = self._renderer._dock_add_layout(vertical=False)
        self._widgets["save_trans"] = self._renderer._dock_add_file_button(
            name="save_trans",
            desc="Save...",
            save=True,
            func=self._save_trans,
            tooltip="Save the transform file to disk",
            layout=save_trans_layout,
            filter_="Head->MRI transformation (*-trans.fif *_trans.fif)",
            initial_directory=self._info_file.parent,
        )
        self._widgets["load_trans"] = self._renderer._dock_add_file_button(
            name="load_trans",
            desc="Load...",
            func=self._load_trans,
            tooltip="Load the transform file from disk",
            layout=save_trans_layout,
            filter_="Head->MRI transformation (*-trans.fif *_trans.fif)",
            initial_directory=self._info_file.parent,
        )
        self._renderer._layout_add_widget(trans_layout, save_trans_layout)
        self._widgets["reset_trans"] = self._renderer._dock_add_button(
            name="Reset Parameters",
            callback=self._reset,
            tooltip="Reset all the parameters affecting the coregistration",
            layout=trans_layout,
        )

        fitting_options_layout = self._renderer._dock_add_group_box(
            name="Fitting Options",
            collapse=collapse,
        )
        self._widgets["fit_label"] = self._renderer._dock_add_label(
            value="",
            layout=fitting_options_layout,
        )
        self._widgets["icp_n_iterations"] = self._renderer._dock_add_spin_box(
            name="Number Of ICP Iterations",
            value=self._defaults["icp_n_iterations"],
            rng=[1, 100],
            callback=self._set_icp_n_iterations,
            compact=True,
            double=False,
            tooltip="Set the number of ICP iterations",
            layout=fitting_options_layout,
        )
        self._widgets["icp_fid_match"] = self._renderer._dock_add_combo_box(
            name="Fiducial point matching",
            value=self._defaults["icp_fid_match"],
            rng=self._defaults["icp_fid_matches"],
            callback=self._set_icp_fid_match,
            compact=True,
            tooltip="Select the fiducial point matching method",
            layout=fitting_options_layout,
        )
        weights_layout = self._renderer._dock_add_group_box(
            name="Weights",
            layout=fitting_options_layout,
        )
        for point, fid in zip(("HSP", "EEG", "HPI"), self._defaults["fiducials"]):
            weight_layout = self._renderer._dock_add_layout(vertical=False)
            point_lower = point.lower()
            name = f"{point_lower}_weight"
            self._widgets[name] = self._renderer._dock_add_spin_box(
                name=point,
                value=getattr(self, f"_{point_lower}_weight"),
                rng=[0.0, 100.0],
                callback=partial(self._set_point_weight, point=point_lower),
                compact=True,
                double=True,
                tooltip=f"Set the {point} weight",
                layout=weight_layout,
            )

            fid_lower = fid.lower()
            name = f"{fid_lower}_weight"
            self._widgets[name] = self._renderer._dock_add_spin_box(
                name=fid,
                value=getattr(self, f"_{fid_lower}_weight"),
                rng=[0.0, 100.0],
                callback=partial(self._set_point_weight, point=fid_lower),
                compact=True,
                double=True,
                tooltip=f"Set the {fid} weight",
                layout=weight_layout,
            )
            self._renderer._layout_add_widget(weights_layout, weight_layout)
        self._widgets["reset_fitting_options"] = self._renderer._dock_add_button(
            name="Reset Fitting Options",
            callback=self._reset_fitting_parameters,
            tooltip="Reset all the fitting parameters to default value",
            layout=fitting_options_layout,
        )
        self._renderer._dock_add_stretch()

    def _configure_status_bar(self):
        self._renderer._status_bar_initialize()
        self._widgets["status_message"] = self._renderer._status_bar_add_label(
            "", stretch=1
        )
        self._forward_widget_command(
            "status_message", "hide", value=None, input_value=False
        )

    def _clean(self):
        if not self._accept_close_event:
            return
        self._renderer = None
        self._widgets.clear()
        self._actors.clear()
        self._surfaces.clear()
        self._defaults.clear()
        self._head_geo = None
        self._check_inside = None
        self._nearest = None
        self._redraw = None

    @safe_event
    def close(self):
        """Close interface and cleanup data structure."""
        if self._renderer is not None:
            self._renderer.close()

    def _close_dialog_callback(self, button_name):
        from ..viz.backends.renderer import MNE_3D_BACKEND_TESTING

        self._accept_close_event = True
        if button_name == "Save":
            if self._trans_modified:
                self._forward_widget_command("save_trans", "set_value", None)
                # cancel means _save_trans is not called
                if self._trans_modified:
                    self._accept_close_event = False
            if self._mri_fids_modified:
                self._forward_widget_command("save_mri_fids", "set_value", None)
            if self._mri_scale_modified:
                if self._subject_to:
                    self._save_subject(exit_mode=True)
                else:
                    dialog = self._renderer._dialog_create(
                        title="CoregistrationUI",
                        text="The name of the output subject used to "
                        "save the scaled anatomy is not set.",
                        info_text="Please set a subject name",
                        callback=lambda x: None,
                        buttons=["Ok"],
                        modal=not MNE_3D_BACKEND_TESTING,
                    )
                    dialog.show()
                    self._accept_close_event = False
        elif button_name == "Cancel":
            self._accept_close_event = False
        else:
            assert button_name == "Discard"

    def _close_callback(self):
        if self._trans_modified or self._mri_fids_modified or self._mri_scale_modified:
            from ..viz.backends.renderer import MNE_3D_BACKEND_TESTING

            # prepare the dialog's text
            text = "The following is/are not saved:"
            text += "<ul>"
            if self._trans_modified:
                text += "<li>Head&lt;&gt;MRI transform</li>"
            if self._mri_fids_modified:
                text += "<li>MRI fiducials</li>"
            if self._mri_scale_modified:
                text += "<li>scaled subject MRI</li>"
            text += "</ul>"
            self._widgets["close_dialog"] = self._renderer._dialog_create(
                title="CoregistrationUI",
                text=text,
                info_text="Do you want to save?",
                callback=self._close_dialog_callback,
                buttons=["Save", "Discard", "Cancel"],
                # modal=True means that the dialog blocks the application
                # when show() is called, until one of the buttons is clicked
                modal=not MNE_3D_BACKEND_TESTING,
            )
            self._widgets["close_dialog"].show()
        return self._accept_close_event
