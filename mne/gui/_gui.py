# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from ..utils import get_config, verbose


@verbose
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
    %(subjects_dir)s
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
    %(interaction_scene_none)s
        Defaults to ``'terrain'``.

        .. versionadded:: 0.16
        .. versionchanged:: 1.0
           Default interaction mode if ``None`` and no config setting found
           changed from ``'trackball'`` to ``'terrain'``.
    %(fullscreen)s
        Default is ``None``, which uses ``MNE_COREG_FULLSCREEN`` config value
        (which defaults to ``False``).

        .. versionadded:: 1.1
    show : bool
        Show the GUI if True.
    block : bool
        Whether to halt program execution until the figure is closed.
    %(verbose)s

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


class _GUIScraper:
    """Scrape GUI outputs."""

    def __repr__(self):
        return "<GUIScraper>"

    def __call__(self, block, block_vars, gallery_conf):
        from ._coreg import CoregistrationUI

        gui_classes = (CoregistrationUI,)
        try:
            from mne_gui_addons._ieeg_locate import IntracranialElectrodeLocator
        except Exception:
            pass
        else:
            gui_classes = gui_classes + (IntracranialElectrodeLocator,)
        from qtpy import QtGui
        from sphinx_gallery.scrapers import figure_rst

        for gui in block_vars["example_globals"].values():
            if (
                isinstance(gui, gui_classes)
                and not getattr(gui, "_scraped", False)
                and gallery_conf["builder_name"] == "html"
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
                    # https://doc.qt.io/qt-5/qwidget.html#mapTo
                    # https://doc.qt.io/qt-5/qpainter.html#drawPixmap-1
                    QtGui.QPainter(pixmap).drawPixmap(
                        plotter.mapTo(window, plotter.rect().topLeft()), sub_pixmap
                    )
                # https://doc.qt.io/qt-5/qpixmap.html#save
                pixmap.save(img_fname)
                try:  # for compatibility with both GUIs, will be refactored
                    gui._renderer.close()  # TODO should be triggered by close
                except Exception:
                    pass
                gui.close()
                return figure_rst([img_fname], gallery_conf["src_dir"], "GUI")
        return ""
