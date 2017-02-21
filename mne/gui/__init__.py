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


def combine_kit_markers():
    """Create a new KIT marker file by interpolating two marker files.

    Notes
    -----
    The functionality in this GUI is also part of :func:`kit2fiff`.
    """
    _check_mayavi_version()
    from ._backend import _check_backend
    _check_backend()
    from ._marker_gui import CombineMarkersFrame
    frame = CombineMarkersFrame()
    return _initialize_gui(frame)


@verbose
def coregistration(tabbed=False, split=True, scene_width=None, inst=None,
                   subject=None, subjects_dir=None, guess_mri_subject=None,
                   scene_height=None, head_opacity=None, head_high_res=None,
                   trans=None, verbose=None):
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
    scene_width : int | None
        Specify a minimum width for the 3d scene (in pixels).
        Default is None, which uses ``MNE_COREG_SCENE_WIDTH`` config value
        (which defaults to 500).
    inst : None | str
        Path to an instance file containing the digitizer data. Compatible for
        Raw, Epochs, and Evoked files.
    subject : None | str
        Name of the mri subject.
    subjects_dir : None | path
        Override the SUBJECTS_DIR environment variable
        (sys.environ['SUBJECTS_DIR'])
    guess_mri_subject : bool
        When selecting a new head shape file, guess the subject's name based
        on the filename and change the MRI subject accordingly (default True).
    scene_height : int | None
        Specify a minimum height for the 3d scene (in pixels).
        Default is None, which uses ``MNE_COREG_SCENE_WIDTH`` config value
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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Notes
    -----
    Step by step instructions for the coregistrations can be accessed as
    slides, `for subjects with structural MRI
    <http://www.slideshare.net/mne-python/mnepython-coregistration>`_ and `for
    subjects for which no MRI is available
    <http://www.slideshare.net/mne-python/mnepython-scale-mri>`_.
    """
    config = get_config(home_dir=os.environ.get('_MNE_FAKE_HOME_DIR'))
    if guess_mri_subject is None:
        guess_mri_subject = config.get(
            'MNE_COREG_GUESS_MRI_SUBJECT', 'true') == 'true'
    if head_high_res is None:
        head_high_res = config.get('MNE_COREG_HEAD_HIGH_RES', 'true') == 'true'
    if head_opacity is None:
        head_opacity = config.get('MNE_COREG_HEAD_OPACITY', 1.)
    if scene_width is None:
        scene_width = config.get('MNE_COREG_SCENE_WIDTH', 500)
    if scene_height is None:
        scene_height = config.get('MNE_COREG_SCENE_HEIGHT', 400)
    if subjects_dir is None:
        if 'SUBJECTS_DIR' in config:
            subjects_dir = config['SUBJECTS_DIR']
        elif 'MNE_COREG_SUBJECTS_DIR' in config:
            subjects_dir = config['MNE_COREG_SUBJECTS_DIR']
    head_opacity = float(head_opacity)
    scene_width = int(scene_width)
    scene_height = int(scene_height)
    _check_mayavi_version()
    from ._backend import _check_backend
    _check_backend()
    from ._coreg_gui import CoregFrame, _make_view
    view = _make_view(tabbed, split, scene_width, scene_height)
    frame = CoregFrame(inst, subject, subjects_dir, guess_mri_subject,
                       head_opacity, head_high_res, trans, config)
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

    """
    _check_mayavi_version()
    from ._backend import _check_backend
    _check_backend()
    from ._kit2fiff_gui import Kit2FiffFrame
    frame = Kit2FiffFrame()
    return _initialize_gui(frame)
