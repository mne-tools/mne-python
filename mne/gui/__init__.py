"""Convenience functions for opening GUIs."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

from ..utils import verbose, get_config, warn


@verbose
def coregistration(tabbed=False, split=True, width=None, inst=None,
                   subject=None, subjects_dir=None, guess_mri_subject=None,
                   height=None, head_opacity=None, head_high_res=None,
                   trans=None, scrollable=True, *,
                   orient_to_surface=True, scale_by_distance=True,
                   mark_inside=True, interaction=None, scale=None,
                   advanced_rendering=None, head_inside=True,
                   fullscreen=None, show=True, block=False, verbose=None):
    """Coregister an MRI with a subject's head shape.

    The GUI can be launched through the command line interface:

    .. code-block::  bash

        $ mne coreg

    or using a python interpreter as shown in :ref:`tut-source-alignment`.

    Parameters
    ----------
    tabbed : bool
        Combine the data source panel and the coregistration panel into a
        single panel with tabs.
    split : bool
        Split the main panels with a movable splitter (good for QT4 but
        unnecessary for wx backend).
    width : int | None
        Specify the width for window (in logical pixels).
        Default is None, which uses ``MNE_COREG_WINDOW_WIDTH`` config value
        (which defaults to 800).
    inst : None | str
        Path to an instance file containing the digitizer data. Compatible for
        Raw, Epochs, and Evoked files.
    subject : None | str
        Name of the mri subject.
    %(subjects_dir)s
    guess_mri_subject : bool
        When selecting a new head shape file, guess the subject's name based
        on the filename and change the MRI subject accordingly (default True).
    height : int | None
        Specify a height for window (in logical pixels).
        Default is None, which uses ``MNE_COREG_WINDOW_WIDTH`` config value
        (which defaults to 400).
    head_opacity : float | None
        The opacity of the head surface in the range [0., 1.].
        Default is None, which uses ``MNE_COREG_HEAD_OPACITY`` config value
        (which defaults to 1.).
    head_high_res : bool | None
        Use a high resolution head surface.
        Default is None, which uses ``MNE_COREG_HEAD_HIGH_RES`` config value
        (which defaults to True).
    trans : str | None
        The transform file to use.
    scrollable : bool
        Make the coregistration panel vertically scrollable (default True).
    orient_to_surface : bool | None
        If True (default), orient EEG electrode and head shape points
        to the head surface.

        .. versionadded:: 0.16
    scale_by_distance : bool | None
        If True (default), scale the digitization points by their
        distance from the scalp surface.

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
    scale : float | None
        The scaling for the scene.

        .. versionadded:: 0.16
    advanced_rendering : bool
        Use advanced OpenGL rendering techniques (default True).
        For some renderers (such as MESA software) this can cause rendering
        bugs.

        .. versionadded:: 0.18
    head_inside : bool
        If True (default), add opaque inner scalp head surface to help occlude
        points behind the head.

        .. versionadded:: 0.23
    %(fullscreen)s
        Default is None, which uses ``MNE_COREG_FULLSCREEN`` config value
        (which defaults to False).

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

    .. youtube:: uK4n5g6DBcg
    """
    unsupported_params = {
        'tabbed': (tabbed, False),
        'split': (split, True),
        'scrollable': (scrollable, True),
        'head_inside': (head_inside, True),
        'guess_mri_subject': guess_mri_subject,
        'scale': scale,
        'advanced_rendering': advanced_rendering,
    }
    for key, val in unsupported_params.items():
        if isinstance(val, tuple):
            to_raise = val[0] != val[1]
        else:
            to_raise = val is not None
        if to_raise:
            warn(f"The parameter {key} is not supported with"
                 " the pyvistaqt 3d backend. It will be ignored.")
    config = get_config()
    if guess_mri_subject is None:
        guess_mri_subject = config.get(
            'MNE_COREG_GUESS_MRI_SUBJECT', 'true') == 'true'
    if head_high_res is None:
        head_high_res = config.get('MNE_COREG_HEAD_HIGH_RES', 'true') == 'true'
    if advanced_rendering is None:
        advanced_rendering = \
            config.get('MNE_COREG_ADVANCED_RENDERING', 'true') == 'true'
    if head_opacity is None:
        head_opacity = config.get('MNE_COREG_HEAD_OPACITY', 0.8)
    if head_inside is None:
        head_inside = \
            config.get('MNE_COREG_HEAD_INSIDE', 'true').lower() == 'true'
    if width is None:
        width = config.get('MNE_COREG_WINDOW_WIDTH', 800)
    if height is None:
        height = config.get('MNE_COREG_WINDOW_HEIGHT', 600)
    if subjects_dir is None:
        if 'SUBJECTS_DIR' in config:
            subjects_dir = config['SUBJECTS_DIR']
        elif 'MNE_COREG_SUBJECTS_DIR' in config:
            subjects_dir = config['MNE_COREG_SUBJECTS_DIR']
    if orient_to_surface is None:
        orient_to_surface = (config.get('MNE_COREG_ORIENT_TO_SURFACE', '') ==
                             'true')
    if scale_by_distance is None:
        scale_by_distance = (config.get('MNE_COREG_SCALE_BY_DISTANCE', '') ==
                             'true')
    if interaction is None:
        interaction = config.get('MNE_COREG_INTERACTION', 'terrain')
    if mark_inside is None:
        mark_inside = config.get('MNE_COREG_MARK_INSIDE', '') == 'true'
    if scale is None:
        scale = config.get('MNE_COREG_SCENE_SCALE', 0.16)
    if fullscreen is None:
        fullscreen = config.get('MNE_COREG_FULLSCREEN', '') == 'true'
    head_opacity = float(head_opacity)
    head_inside = bool(head_inside)
    width = int(width)
    height = int(height)
    scale = float(scale)

    from ..viz.backends.renderer import MNE_3D_BACKEND_TESTING
    from ._coreg import CoregistrationUI
    if MNE_3D_BACKEND_TESTING:
        show = block = False
    return CoregistrationUI(
        info_file=inst, subject=subject, subjects_dir=subjects_dir,
        head_resolution=head_high_res, head_opacity=head_opacity,
        orient_glyphs=orient_to_surface, scale_by_distance=scale_by_distance,
        mark_inside=mark_inside, trans=trans, size=(width, height), show=show,
        block=block, interaction=interaction, fullscreen=fullscreen,
        verbose=verbose
    )


@verbose
def locate_ieeg(info, trans, aligned_ct, subject=None, subjects_dir=None,
                groups=None, show=True, block=False, verbose=None):
    """Locate intracranial electrode contacts.

    Parameters
    ----------
    %(info_not_none)s
    %(trans_not_none)s
    aligned_ct : path-like | nibabel.spatialimages.SpatialImage
        The CT image that has been aligned to the Freesurfer T1. Path-like
        inputs and nibabel image objects are supported.
    %(subject)s
    %(subjects_dir)s
    groups : dict | None
        A dictionary with channels as keys and their group index as values.
        If None, the groups will be inferred by the channel names. Channel
        names must have a format like ``LAMY 7`` where a string prefix
        like ``LAMY`` precedes a numeric index like ``7``. If the channels
        are formatted improperly, group plotting will work incorrectly.
        Group assignments can be adjusted in the GUI.
    show : bool
        Show the GUI if True.
    block : bool
        Whether to halt program execution until the figure is closed.
    %(verbose)s

    Returns
    -------
    gui : instance of IntracranialElectrodeLocator
        The graphical user interface (GUI) window.
    """
    from ..viz.backends._utils import _qt_app_exec
    from ._ieeg_locate import IntracranialElectrodeLocator
    from qtpy.QtWidgets import QApplication
    # get application
    app = QApplication.instance()
    if app is None:
        app = QApplication(["Intracranial Electrode Locator"])
    gui = IntracranialElectrodeLocator(
        info, trans, aligned_ct, subject=subject, subjects_dir=subjects_dir,
        groups=groups, show=show, verbose=verbose)
    if block:
        _qt_app_exec(app)
    return gui


class _GUIScraper(object):
    """Scrape GUI outputs."""

    def __repr__(self):
        return '<GUIScraper>'

    def __call__(self, block, block_vars, gallery_conf):
        from ._ieeg_locate import IntracranialElectrodeLocator
        from ._coreg import CoregistrationUI
        from sphinx_gallery.scrapers import figure_rst
        from qtpy import QtGui
        for gui in block_vars['example_globals'].values():
            if (isinstance(gui, (IntracranialElectrodeLocator,
                                 CoregistrationUI)) and
                    not getattr(gui, '_scraped', False) and
                    gallery_conf['builder_name'] == 'html'):
                gui._scraped = True  # monkey-patch but it's easy enough
                img_fname = next(block_vars['image_path_iterator'])
                # TODO fix in window refactor
                window = gui if hasattr(gui, 'grab') else gui._renderer._window
                # window is QWindow
                # https://doc.qt.io/qt-5/qwidget.html#grab
                pixmap = window.grab()
                if hasattr(gui, '_renderer'):  # if no renderer, no need
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
                        plotter.mapTo(window, plotter.rect().topLeft()),
                        sub_pixmap)
                # https://doc.qt.io/qt-5/qpixmap.html#save
                pixmap.save(img_fname)
                try:  # for compatibility with both GUIs, will be refactored
                    gui._renderer.close()  # TODO should be triggered by close
                except Exception:
                    pass
                gui.close()
                return figure_rst(
                    [img_fname], gallery_conf['src_dir'], 'GUI')
        return ''
