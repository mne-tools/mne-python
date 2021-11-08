"""Convenience functions for opening GUIs."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

import os

from ..utils import (_check_mayavi_version, verbose, get_config, warn,
                     deprecated)
from ._backend import _testing_mode


def _initialize_gui(frame, view=None):
    """Initialize GUI depending on testing mode."""
    if _testing_mode():  # open without entering mainloop
        return frame.edit_traits(view=view), frame
    else:
        frame.configure_traits(view=view)
        return frame


@verbose
def coregistration(tabbed=False, split=True, width=None, inst=None,
                   subject=None, subjects_dir=None, guess_mri_subject=None,
                   height=None, head_opacity=None, head_high_res=None,
                   trans=None, scrollable=True, project_eeg=None,
                   orient_to_surface=None, scale_by_distance=None,
                   mark_inside=None, interaction=None, scale=None,
                   advanced_rendering=None, head_inside=True, verbose=None):
    """Coregister an MRI with a subject's head shape.

    The recommended way to use the GUI is through bash with:

    .. code-block::  bash

        $ mne coreg

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
    project_eeg : bool | None
        If True (default None), project EEG electrodes to the head surface.
        This is only for visualization purposes and does not affect fitting.

        .. versionadded:: 0.16
    orient_to_surface : bool | None
        If True (default None), orient EEG electrode and head shape points
        to the head surface.

        .. versionadded:: 0.16
    scale_by_distance : bool | None
        If True (default None), scale the digitization points by their
        distance from the scalp surface.

        .. versionadded:: 0.16
    mark_inside : bool | None
        If True (default None), mark points inside the head surface in a
        different color.

        .. versionadded:: 0.16
    %(scene_interaction_None)s
        Defaults to ``'trackball'``.

        .. versionadded:: 0.16
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
    %(verbose)s

    Returns
    -------
    frame : instance of CoregFrame or CoregistrationUI
        The coregistration frame.

    Notes
    -----
    Many parameters (e.g., ``project_eeg``) take None as a parameter,
    which means that the default will be read from the MNE-Python
    configuration file (which gets saved when exiting).

    Step by step instructions for the coregistrations can be accessed as
    slides, `for subjects with structural MRI
    <https://www.slideshare.net/mne-python/mnepython-coregistration>`_ and `for
    subjects for which no MRI is available
    <https://www.slideshare.net/mne-python/mnepython-scale-mri>`_.
    """
    from ..viz.backends.renderer import _get_3d_backend
    pyvistaqt = _get_3d_backend() == 'pyvistaqt'
    if pyvistaqt:
        # unsupported parameters
        params = {
            'tabbed': (tabbed, False),
            'split': (split, True),
            'scrollable': (scrollable, True),
            'head_inside': (head_inside, True),
            'guess_mri_subject': guess_mri_subject,
            'head_opacity': head_opacity,
            'project_eeg': project_eeg,
            'scale_by_distance': scale_by_distance,
            'mark_inside': mark_inside,
            'scale': scale,
            'advanced_rendering': advanced_rendering,
        }
        for key, val in params.items():
            if isinstance(val, tuple):
                to_raise = val[0] != val[1]
            else:
                to_raise = val is not None
            if to_raise:
                warn(f"The parameter {key} is not supported with"
                      " the pyvistaqt 3d backend. It will be ignored.")
    config = get_config(home_dir=os.environ.get('_MNE_FAKE_HOME_DIR'))
    if guess_mri_subject is None:
        guess_mri_subject = config.get(
            'MNE_COREG_GUESS_MRI_SUBJECT', 'true') == 'true'
    if head_high_res is None:
        head_high_res = config.get('MNE_COREG_HEAD_HIGH_RES', 'true') == 'true'
    if advanced_rendering is None:
        advanced_rendering = \
            config.get('MNE_COREG_ADVANCED_RENDERING', 'true') == 'true'
    if head_opacity is None:
        head_opacity = config.get('MNE_COREG_HEAD_OPACITY', 1.)
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
    if project_eeg is None:
        project_eeg = config.get('MNE_COREG_PROJECT_EEG', '') == 'true'
    if orient_to_surface is None:
        orient_to_surface = (config.get('MNE_COREG_ORIENT_TO_SURFACE', '') ==
                             'true')
    if scale_by_distance is None:
        scale_by_distance = (config.get('MNE_COREG_SCALE_BY_DISTANCE', '') ==
                             'true')
    if interaction is None:
        interaction = config.get('MNE_COREG_INTERACTION', 'trackball')
    if mark_inside is None:
        mark_inside = config.get('MNE_COREG_MARK_INSIDE', '') == 'true'
    if scale is None:
        scale = config.get('MNE_COREG_SCENE_SCALE', 0.16)
    head_opacity = float(head_opacity)
    head_inside = bool(head_inside)
    width = int(width)
    height = int(height)
    scale = float(scale)
    if pyvistaqt:
        from ..viz.backends.renderer import MNE_3D_BACKEND_TESTING
        from ._coreg import CoregistrationUI
        show = not MNE_3D_BACKEND_TESTING
        standalone = not MNE_3D_BACKEND_TESTING
        return CoregistrationUI(
            info_file=inst, subject=subject, subjects_dir=subjects_dir,
            head_resolution=head_high_res, orient_glyphs=orient_to_surface,
            trans=trans, size=(width, height), show=show, standalone=standalone,
            verbose=verbose
        )
    else:
        _check_mayavi_version()
        from ._backend import _check_backend
        _check_backend()
        from ._coreg_gui import CoregFrame, _make_view
        view = _make_view(tabbed, split, width, height, scrollable)
        frame = CoregFrame(inst, subject, subjects_dir, guess_mri_subject,
                           head_opacity, head_high_res, trans, config,
                           project_eeg=project_eeg,
                           orient_to_surface=orient_to_surface,
                           scale_by_distance=scale_by_distance,
                           mark_inside=mark_inside, interaction=interaction,
                           scale=scale, advanced_rendering=advanced_rendering,
                           head_inside=head_inside)
        return _initialize_gui(frame, view)


@deprecated('The `fiducials` function has moved to the separate mne-kit-gui '
            'module and will be removed from mne after 0.24.')
def fiducials(subject=None, fid_file=None, subjects_dir=None):
    """Set the fiducials for an MRI subject.

    Parameters
    ----------
    subject : str
        Name of the mri subject.
    fid_file : None | str
        Load a fiducials file different form the subject's default
        ("{subjects_dir}/{subject}/bem/{subject}-fiducials.fif").
    subjects_dir : None | str
        Overrule the subjects_dir environment variable.

    Returns
    -------
    frame : instance of FiducialsFrame
        The GUI frame.

    Notes
    -----
    All parameters are optional, since they can be set through the GUI.
    The functionality in this GUI is also part of :func:`coregistration`.
    """
    _check_mayavi_version()
    from ._backend import _check_backend
    _check_backend()
    from ._fiducials_gui import FiducialsFrame
    frame = FiducialsFrame(subject, subjects_dir, fid_file=fid_file)
    return _initialize_gui(frame)


@deprecated('The `mne kit2fiff` command will require the mne-kit-gui '
            'module after 0.24, install it using conda-forge or pip to '
            'continue using this utility.')
def kit2fiff():
    """Convert KIT files to the fiff format.

    The recommended way to use the GUI is through bash with::

        $ mne kit2fiff

    Returns
    -------
    frame : instance of Kit2FiffFrame
        The GUI frame.
    """
    _check_mayavi_version()
    from ._backend import _check_backend
    _check_backend()
    from ._kit2fiff_gui import Kit2FiffFrame
    frame = Kit2FiffFrame()
    return _initialize_gui(frame)


@verbose
def locate_ieeg(info, trans, aligned_ct, subject=None, subjects_dir=None,
                groups=None, verbose=None):
    """Locate intracranial electrode contacts.

    Parameters
    ----------
    %(info_not_none)s
    %(trans_not_none)s
    aligned_ct : str | pathlib.Path | nibabel.spatialimages.SpatialImage
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
    %(verbose)s

    Returns
    -------
    gui : instance of IntracranialElectrodeLocator
        The graphical user interface (GUI) window.
    """
    from ._ieeg_locate_gui import IntracranialElectrodeLocator
    from PyQt5.QtWidgets import QApplication
    # get application
    app = QApplication.instance()
    if app is None:
        app = QApplication(["Intracranial Electrode Locator"])
    gui = IntracranialElectrodeLocator(
        info, trans, aligned_ct, subject=subject,
        subjects_dir=subjects_dir, groups=groups, verbose=verbose)
    gui.show()
    return gui


class _LocateScraper(object):
    """Scrape locate_ieeg outputs."""

    def __repr__(self):
        return '<LocateScraper>'

    def __call__(self, block, block_vars, gallery_conf):
        from ._ieeg_locate_gui import IntracranialElectrodeLocator
        from sphinx_gallery.scrapers import figure_rst
        from PyQt5 import QtGui
        for gui in block_vars['example_globals'].values():
            if (isinstance(gui, IntracranialElectrodeLocator) and
                    not getattr(gui, '_scraped', False) and
                    gallery_conf['builder_name'] == 'html'):
                gui._scraped = True  # monkey-patch but it's easy enough
                img_fname = next(block_vars['image_path_iterator'])
                # gui is QWindow
                # https://doc.qt.io/qt-5/qwidget.html#grab
                pixmap = gui.grab()
                # Now the tricky part: we need to get the 3D renderer, extract
                # the image from it, and put it in the correct place in the
                # pixmap. The easiest way to do this is actually to save the
                # 3D image first, then load it using QPixmap and Qt geometry.
                plotter = gui._renderer.plotter
                plotter.screenshot(img_fname)
                sub_pixmap = QtGui.QPixmap(img_fname)
                # https://doc.qt.io/qt-5/qwidget.html#mapTo
                # https://doc.qt.io/qt-5/qpainter.html#drawPixmap-1
                QtGui.QPainter(pixmap).drawPixmap(
                    plotter.mapTo(gui, plotter.rect().topLeft()),
                    sub_pixmap)
                # https://doc.qt.io/qt-5/qpixmap.html#save
                pixmap.save(img_fname)
                gui._renderer.close()  # TODO should be triggered by close...
                gui.close()
                return figure_rst(
                    [img_fname], gallery_conf['src_dir'], 'iEEG GUI')
        return ''
