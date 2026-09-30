# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from ..utils import get_config, verbose_static


@verbose_static("subjects_dir", "interaction_scene_none", "fullscreen")
def coregistration(
    *,
    width=None,
    height=None,
    inst=None,
    subject=None,
    subjects_dir=None,
    head_opacity=None,
    head_high_res=None,
    trans=None,
    orient_to_surface=None,
    scale_by_distance=None,
    mark_inside=None,
    interaction=None,
    fullscreen=None,
    show=True,
    block=False,
    verbose=None,
):
    """Coregister an MRI with a subject's head shape.

    The GUI can be launched through the command line interface:

    .. code-block::  bash

        $ mne coreg

    or using a python interpreter as shown in :ref:`tut-source-alignment`.

    Parameters
    ----------
    width : int | None
        Specify the width for window (in logical pixels).
        Default is None, which uses ``MNE_COREG_WINDOW_WIDTH`` config value
        (which defaults to ``800``).
    height : int | None
        Specify a height for window (in logical pixels).
        Default is None, which uses ``MNE_COREG_WINDOW_WIDTH`` config value
        (which defaults to ``400``).
    inst : None | path-like
        Path to an instance file containing the digitizer data. Compatible for
        Raw, Epochs, and Evoked files.
    subject : None | str
        Name of the mri subject.
    subjects_dir : path-like | None
        The path to the directory containing the FreeSurfer subjects
        reconstructions. If ``None``, defaults to the ``SUBJECTS_DIR`` environment
        variable.
    head_opacity : float | None
        The opacity of the head surface in the range ``[0., 1.]``.
        Default is None, which uses ``MNE_COREG_HEAD_OPACITY`` config value
        (which defaults to ``1.``).
    head_high_res : bool | None
        Use a high resolution head surface.
        Default is None, which uses ``MNE_COREG_HEAD_HIGH_RES`` config value
        (which defaults to True).
    trans : path-like | Transform | None
        The Head<->MRI transform or the path to its FIF file (``"-trans.fif"``).
    orient_to_surface : bool | None
        If True (default), orient EEG electrode and head shape points to the head
        surface.

        .. versionadded:: 0.16
    scale_by_distance : bool | None
        If True (default), scale the digitization points by their distance from the
        scalp surface.

        .. versionadded:: 0.16
    mark_inside : bool | None
        If True (default), mark points inside the head surface in a
        different color.

        .. versionadded:: 0.16
    interaction : 'trackball' | 'terrain' | None
        How interactions with the scene via an input device (e.g., mouse or
        trackpad) modify the camera position. If ``'terrain'``, one axis is
        fixed, enabling "turntable-style" rotations. If ``'trackball'``,
        movement along all axes is possible, which provides more freedom of
        movement, but you may incidentally perform unintentional rotations along
        some axes.
        If ``None``, the setting stored in the MNE-Python configuration file is
        used.
        Defaults to ``'terrain'``.

        .. versionadded:: 0.16
        .. versionchanged:: 1.0
           Default interaction mode if ``None`` and no config setting found
           changed from ``'trackball'`` to ``'terrain'``.
    fullscreen : bool
        Whether to start in fullscreen (``True``) or windowed mode
        (``False``).
        Default is ``None``, which uses ``MNE_COREG_FULLSCREEN`` config value
        (which defaults to ``False``).

        .. versionadded:: 1.1
    show : bool
        Show the GUI if True.
    block : bool
        Whether to halt program execution until the figure is closed.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    frame : instance of CoregistrationUI
        The coregistration frame.

    Notes
    -----
    Many parameters (e.g., ``head_opacity``) take None as a parameter,
    which means that the default will be read from the MNE-Python
    configuration file (which gets saved when exiting).

    Step by step instructions for the coregistrations are shown below:

    .. youtube:: ALV5qqMHLlQ
    """
    config = get_config()
    if head_high_res is None:
        head_high_res = config.get("MNE_COREG_HEAD_HIGH_RES", "true") == "true"
    if head_opacity is None:
        head_opacity = config.get("MNE_COREG_HEAD_OPACITY", 0.8)
    if width is None:
        width = config.get("MNE_COREG_WINDOW_WIDTH", 800)
    if height is None:
        height = config.get("MNE_COREG_WINDOW_HEIGHT", 600)
    if subjects_dir is None:
        if "SUBJECTS_DIR" in config:
            subjects_dir = config["SUBJECTS_DIR"]
        elif "MNE_COREG_SUBJECTS_DIR" in config:
            subjects_dir = config["MNE_COREG_SUBJECTS_DIR"]
    false_like = ("false", "0")
    if orient_to_surface is None:
        orient_to_surface = config.get("MNE_COREG_ORIENT_TO_SURFACE", "true").lower()
        orient_to_surface = orient_to_surface not in false_like
    if scale_by_distance is None:
        scale_by_distance = config.get("MNE_COREG_SCALE_BY_DISTANCE", "true").lower()
        scale_by_distance = scale_by_distance not in false_like
    if interaction is None:
        interaction = config.get("MNE_COREG_INTERACTION", "terrain")
    if mark_inside is None:
        mark_inside = config.get("MNE_COREG_MARK_INSIDE", "true").lower()
        mark_inside = mark_inside not in false_like
    if fullscreen is None:
        fullscreen = config.get("MNE_COREG_FULLSCREEN", "") == "true"
    head_opacity = float(head_opacity)
    width = int(width)
    height = int(height)

    from ..viz.backends.renderer import MNE_3D_BACKEND_TESTING
    from ._coreg import CoregistrationUI

    if MNE_3D_BACKEND_TESTING:
        show = block = False
    return CoregistrationUI(
        info_file=inst,
        subject=subject,
        subjects_dir=subjects_dir,
        head_resolution=head_high_res,
        head_opacity=head_opacity,
        orient_glyphs=orient_to_surface,
        scale_by_distance=scale_by_distance,
        mark_inside=mark_inside,
        trans=trans,
        size=(width, height),
        show=show,
        block=block,
        interaction=interaction,
        fullscreen=fullscreen,
        verbose=verbose,
    )


@verbose_static("baseline_evoked", "subjects_dir", "rank", "n_jobs")
def dipolefit(
    evoked,
    *,
    baseline=None,
    cov=None,
    bem=None,
    initial_time=None,
    trans=None,
    stc=None,
    subject=None,
    subjects_dir=None,
    surf_maps=None,
    rank="info",
    show_density=True,
    ch_type=None,
    show_sensors=True,
    n_jobs=None,
    show=True,
    block=False,
    verbose=None,
):
    """GUI for interactive dipole fitting, inspired by MEGIN's XFit program.

    Parameters
    ----------
    evoked : instance of Evoked | path-like | None
        Evoked data to show fieldmap of and fit dipoles to.
    baseline : None | tuple of length 2
        The time interval to consider as "baseline" when applying baseline
        correction. If ``None``, do not apply baseline correction.
        If a tuple ``(a, b)``, the interval is between ``a`` and ``b``
        (in seconds), including the endpoints.
        If ``a`` is ``None``, the **beginning** of the data is used; and if ``b``
        is ``None``, it is set to the **end** of the data.
        If ``(None, None)``, the entire time interval is used.

        .. note::
            The baseline ``(a, b)`` includes both endpoints, i.e. all timepoints
            ``t`` such that ``a <= t <= b``.

        Correction is applied **to each channel individually** in the following
        way:

        1. Calculate the mean signal of the baseline period.
        2. Subtract this mean from the **entire** ``Evoked``.
    cov : instance of Covariance | path-like | "baseline" | None
        Noise covariance matrix. If ``None``, an ad-hoc covariance matrix is used with
        default values for the diagonal elements (see Notes). If ``"baseline"``, the
        diagonal elements is estimated from the baseline period of the evoked data.
    bem : instance of ConductorModel | path-like | None
        Boundary element model to use in forward calculations, or a path to the BEM
        solution file (``"-bem-sol.fif"``) to read it from. If ``None``, a spherical
        model is used.
    initial_time : float | None
        Initial time point to show. If ``None``, the time point of the maximum field
        strength is used.
    trans : instance of Transform | path-like | None
        The transformation from head coordinates to MRI coordinates. If ``None``,
        the identity matrix is used and everything will be done in head coordinates.
    stc : instance of SourceEstimate | path-like | None
        An optional distributed source estimate to show alongside the fieldmap. The time
        samples need to match those of the evoked data.
    subject : str | None
        The subject name. If ``None``, no MRI data is shown.
    subjects_dir : path-like | None
        The path to the directory containing the FreeSurfer subjects
        reconstructions. If ``None``, defaults to the ``SUBJECTS_DIR`` environment
        variable.
    surf_maps : list | None
        The surface mapping information obtained with make_field_map. If ``None``, one
        will be generated based on the given data.
    rank : None | 'info' | 'full' | dict
        This controls the rank computation that can be read from the
        measurement info or estimated from the data. When a noise covariance
        is used for whitening, this should reflect the rank of that covariance,
        otherwise amplification of noise components can occur in whitening (e.g.,
        often during source localization).

        :data:`python:None`
            The rank will be estimated from the data after proper scaling of
            different channel types.
        ``'info'``
            The rank is inferred from ``info``. If data have been processed
            with Maxwell filtering, the Maxwell filtering header is used.
            Otherwise, the channel counts themselves are used.
            In both cases, the number of projectors is subtracted from
            the (effective) number of channels in the data.
            For example, if Maxwell filtering reduces the rank to 68, with
            two projectors the returned value will be 66.
        ``'full'``
            The rank is assumed to be full, i.e. equal to the
            number of good channels. If a `~mne.Covariance` is passed, this can
            make sense if it has been (possibly improperly) regularized without
            taking into account the true data rank.
        :class:`dict`
            Calculate the rank only for a subset of channel types, and explicitly
            specify the rank for the remaining channel types. This can be
            extremely useful if you already **know** the rank of (part of) your
            data, for instance in case you have calculated it earlier.

            This parameter must be a dictionary whose **keys** correspond to
            channel types in the data (e.g. ``'meg'``, ``'mag'``, ``'grad'``,
            ``'eeg'``), and whose **values** are integers representing the
            respective ranks. For example, ``{'mag': 90, 'eeg': 45}`` will assume
            a rank of ``90`` and ``45`` for magnetometer data and EEG data,
            respectively.

            The ranks for all channel types present in the data, but
            **not** specified in the dictionary will be estimated empirically.
            That is, if you passed a dataset containing magnetometer, gradiometer,
            and EEG data together with the dictionary from the previous example,
            only the gradiometer rank would be determined, while the specified
            magnetometer and EEG ranks would be taken for granted.
    show_density : bool
        Whether to show the density of the fieldmap.
    ch_type : "meg" | "eeg" | None
        Type of channels to use for the dipole fitting. By default (``None``) both MEG
        and EEG channels will be used.
    show_sensors : bool
        Whether to show the sensors in the 3D view.
    n_jobs : int | None
        The number of jobs to run in parallel. If ``-1``, it is set
        to the number of CPU cores. Requires the :mod:`joblib` package.
        ``None`` (default) is a marker for 'unset' that will be interpreted
        as ``n_jobs=1`` (sequential execution) unless the call is performed under
        a :class:`joblib:joblib.parallel_config` context manager that sets another
        value for ``n_jobs``.
    show : bool
        Show the GUI if True.
    block : bool
        Whether to halt program execution until the figure is closed.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    fitter : instance of DipoleFitUI
        The dipole fitting GUI. The ``.dipoles`` attribute contains the fitted dipoles.

    Notes
    -----
    .. versionadded:: 1.12

    When using ``cov=None`` the default noise values are 5 fT/cm, 20 fT, and 0.2 µV for
    gradiometers, magnetometers, and EEG channels respectively.

    Here is an incomplete comparison between the features of the MEGIN XFit™ 5.5.18
    software and the MNE interactive dipole fitting GUI:

    .. table::
       :widths: auto

       +-----------------------------------------------------------------------------+-----+------+
       | Feature                                                                     | MNE | Xfit |
       +=============================================================================+=====+======+
       |                                                                                          |
       +-----------------------------------------------------------------------------+-----+------+
       | **Head model**                                                                           |
       +-----------------------------------------------------------------------------+-----+------+
       | Use spherical head model                                                    | ✓   | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
       | Use BEM head model based on MRI                                             | ✓   | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
       | Use ad-hoc covariance matrix                                                | ✓   | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
       | Estimate covariance from baseline period                                    | ✓   | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
       | **Dipole fitting**                                                                       |
       +-----------------------------------------------------------------------------+-----+------+
       | Fit a dipole at the current time                                            | ✓   | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
       | Fit a dipole by averaging the signal over a time range                      |     | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
       | Fit dipoles on a subset of sensors, selected from the sensor view           | ✓   | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
       | Fit multiple dipoles and construct a multi-dipole model                     | ✓   | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
       | Toggle individual dipoles on or off in the multi-dipole model               | ✓   | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
       | Give names to dipoles                                                       | ✓   |      |
       +-----------------------------------------------------------------------------+-----+------+
       | Save dipoles to .dip or .bdip file                                          | ✓   | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
       | View source timecourses                                                     | ✓   | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
       | View total variance explained by the dipole model                           |     | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
       | View detailed dipole information (coordinates, goodness of fit, etc.)       |     | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
       | Toggle dipoles to have a free or fixed orientation                          | ✓   | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
       | Project-out signals from dipoles currently in the model                     |     | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
       | **3D view**                                                                              |
       +-----------------------------------------------------------------------------+-----+------+
       | View magnetic field patterns for Evokeds                                    | ✓   | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
       | Show head surface in relation to the MEG helmet                             | ✓   | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
       | Show brain surfaces inside the head surface                                 | ✓   |      |
       +-----------------------------------------------------------------------------+-----+------+
       | Show MNE source estimate to guide dipole fitting                            | ✓   |      |
       +-----------------------------------------------------------------------------+-----+------+
       | Show location of the dipoles in the 3D view                                 | ✓   |      |
       +-----------------------------------------------------------------------------+-----+------+
       | Show dipole projected to the helmet (big arrows)                            | ✓   | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
       | Show currently selected sensors in the 3D view                              | ✓   |      |
       +-----------------------------------------------------------------------------+-----+------+
       | Only show magnetic field patterns as seen by the currently selected sensors |     | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
       | **Sensor view**                                                                          |
       +-----------------------------------------------------------------------------+-----+------+
       | View sensor-level timecourses                                               | ✓   | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
       | Switch layouts (all, grads, mags, eeg) in the sensor display                |     | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
       | Overlay reconstructed sensor time courses from the dipole model             |     | ✓    |
       +-----------------------------------------------------------------------------+-----+------+
    """  # noqa E501
    from ..viz.backends.renderer import MNE_3D_BACKEND_TESTING
    from ._dipolefit import DipoleFitUI

    if MNE_3D_BACKEND_TESTING:
        show = block = False

    return DipoleFitUI(
        evoked=evoked,
        baseline=baseline,
        cov=cov,
        bem=bem,
        initial_time=initial_time,
        trans=trans,
        stc=stc,
        subject=subject,
        subjects_dir=subjects_dir,
        surf_maps=surf_maps,
        rank=rank,
        show_density=show_density,
        ch_type=ch_type,
        show_sensors=show_sensors,
        n_jobs=n_jobs,
        show=show,
        block=block,
        verbose=verbose,
    )


def _gui_closed(gui):
    """Check whether a GUI's 3D renderer has already been closed."""
    if not hasattr(gui, "_renderer"):  # nothing to close
        return False
    try:
        plotter = gui._renderer.plotter
    except Exception:
        return True
    return bool(getattr(plotter, "_closed", False))


def _close_gui(gui):
    """Close a GUI, tolerating GUIs that are already (partially) closed."""
    try:  # for compatibility with both GUIs, will be refactored
        gui._renderer.close()  # TODO should be triggered by close
    except Exception:
        pass
    gui.close()


class _GUIScraper:
    """Scrape GUI outputs.

    By default a GUI is scraped once and then closed, so each GUI shows up as a single
    image in the rendered example. An example can instead add a file-level

    .. code-block:: python

        # sphinx_gallery_preserve_gui = True

    comment, in which case the GUI is scraped after *every* code block and left open, so
    that successive code blocks can keep operating on it. The scraper then holds the
    only strong reference to those GUIs, and ``close_preserved`` (called from the doc
    build's ``reset_modules``) closes them once the example is done.
    """

    def __init__(self):
        self._preserved_guis = list()

    def __repr__(self):
        return "<GUIScraper>"

    def close_preserved(self):
        """Close (and forget about) all GUIs preserved across code blocks."""
        guis, self._preserved_guis = self._preserved_guis, list()
        for gui in guis:
            _close_gui(gui)

    def __call__(self, block, block_vars, gallery_conf):
        from ._coreg import CoregistrationUI
        from ._dipolefit import DipoleFitUI

        gui_classes = (CoregistrationUI, DipoleFitUI)
        try:
            from mne_gui_addons._ieeg_locate import IntracranialElectrodeLocator
        except Exception:
            pass
        else:
            gui_classes = gui_classes + (IntracranialElectrodeLocator,)
        from qtpy import QtGui
        from sphinx_gallery.scrapers import figure_rst

        preserve = bool((block_vars.get("file_conf") or {}).get("preserve_gui", False))
        for gui in block_vars["example_globals"].values():
            if (
                isinstance(gui, gui_classes)
                and gallery_conf["builder_name"] == "html"
                and (
                    # A preserved GUI is scraped for every code block; any other one
                    # only the first time we see it.
                    not _gui_closed(gui)
                    if preserve
                    else not getattr(gui, "_scraped", False)
                )
            ):
                gui._scraped = True  # monkey-patch but it's easy enough
                img_fname = next(block_vars["image_path_iterator"])
                # TODO fix in window refactor
                window = gui if hasattr(gui, "grab") else gui._renderer._window
                # window is QWindow
                # https://doc.qt.io/qt-5/qwidget.html#grab
                pixmap = window.grab()
                if hasattr(gui, "_renderer"):  # if no renderer, no need
                    # Now the tricky part: we need to get the 3D renderer,
                    # extract the image from it, and put it in the correct
                    # place in the pixmap. The easiest way to do this is
                    # actually to save the 3D image first, then load it
                    # using QPixmap and Qt geometry.
                    plotter = gui._renderer.plotter
                    plotter.screenshot(img_fname)
                    sub_pixmap = QtGui.QPixmap(img_fname)
                    # The screenshot is in physical pixels, but QPainter works in
                    # logical pixels, so on HiDPI displays (e.g., Retina, where
                    # devicePixelRatio == 2) the screenshot must be marked with the
                    # window's scale factor or it is composited at twice its size.
                    sub_pixmap.setDevicePixelRatio(window.devicePixelRatio())
                    # https://doc.qt.io/qt-5/qwidget.html#mapTo
                    # https://doc.qt.io/qt-5/qpainter.html#drawPixmap-1
                    QtGui.QPainter(pixmap).drawPixmap(
                        plotter.mapTo(window, plotter.rect().topLeft()), sub_pixmap
                    )
                # https://doc.qt.io/qt-5/qpixmap.html#save
                pixmap.save(img_fname)
                if preserve:
                    # Keep it open (and alive) so the next code block can use it.
                    if not any(gui is known for known in self._preserved_guis):
                        self._preserved_guis.append(gui)
                    if hasattr(gui, "_renderer"):
                        # The PyVista scraper runs after us and screenshots *and then
                        # closes* every plotter it knows about, which would take our GUI
                        # down with it. Deregister ours from it, just like closing a
                        # figure does (see _pyvista._close_3d_figure).
                        from ..viz.backends._pyvista import _ALL_PLOTTERS

                        _ALL_PLOTTERS.pop(plotter._id_name, None)
                else:
                    _close_gui(gui)
                return figure_rst([img_fname], gallery_conf["src_dir"], "GUI")
        return ""
