"""Convenience functions for opening GUIs."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os

from ..utils import _check_mayavi_version, verbose, get_config
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
                   advanced_rendering=None, verbose=None):
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
    interaction : str | None
        Can be 'terrain' (default None), use terrain-style interaction (where
        "up" is the Z/superior direction), or 'trackball' to use
        orientationless interactions.

        .. versionadded:: 0.16
    scale : float | None
        The scaling for the scene.

        .. versionadded:: 0.16
    advanced_rendering : bool
        Use advanced OpenGL rendering techniques (default True).
        For some renderers (such as MESA software) this can cause rendering
        bugs.

        .. versionadded:: 0.18
    %(verbose)s

    Returns
    -------
    frame : instance of CoregFrame
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
    width = int(width)
    height = int(height)
    scale = float(scale)
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
                       scale=scale, advanced_rendering=advanced_rendering)
    return _initialize_gui(frame, view)


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
