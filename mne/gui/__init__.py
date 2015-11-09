"""Convenience functions for opening GUIs."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

from ..utils import _check_mayavi_version


def combine_kit_markers():
    """Create a new KIT marker file by interpolating two marker files

    Notes
    -----
    The functionality in this GUI is also part of :func:`kit2fiff`.
    """
    _check_mayavi_version()
    from ._marker_gui import CombineMarkersFrame
    gui = CombineMarkersFrame()
    gui.configure_traits()
    return gui


def coregistration(tabbed=False, split=True, scene_width=0o1, inst=None,
                   subject=None, subjects_dir=None, raw=None):
    """Coregister an MRI with a subject's head shape

    Parameters
    ----------
    tabbed : bool
        Combine the data source panel and the coregistration panel into a
        single panel with tabs.
    split : bool
        Split the main panels with a movable splitter (good for QT4 but
        unnecessary for wx backend).
    scene_width : int
        Specify a minimum width for the 3d scene (in pixels).
    inst : None | str
        Path to an instance file containing the digitizer data. Compatible for
        Raw, Epochs, and Evoked files.
    subject : None | str
        Name of the mri subject.
    subjects_dir : None | path
        Override the SUBJECTS_DIR environment variable
        (sys.environ['SUBJECTS_DIR'])

    Notes
    -----
    All parameters are optional, since they can be set through the GUI.
    Step by step instructions for the coregistrations can be accessed as
    slides, `for subjects with structural MRI
    <http://www.slideshare.net/mne-python/mnepython-coregistration>`_ and `for
    subjects for which no MRI is available
    <http://www.slideshare.net/mne-python/mnepython-scale-mri>`_.
    """
    _check_mayavi_version()
    if raw is not None:
        raise DeprecationWarning('The `raw` argument has been deprecated for '
                                 'the `inst` argument. Will be removed '
                                 'in 0.11. Use `inst` instead.')
        inst = raw
    from ._coreg_gui import CoregFrame, _make_view
    view = _make_view(tabbed, split, scene_width)
    gui = CoregFrame(inst, subject, subjects_dir)
    gui.configure_traits(view=view)
    return gui


def fiducials(subject=None, fid_file=None, subjects_dir=None):
    """Set the fiducials for an MRI subject

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
    from ._fiducials_gui import FiducialsFrame
    gui = FiducialsFrame(subject, subjects_dir, fid_file=fid_file)
    gui.configure_traits()
    return gui


def kit2fiff():
    """Convert KIT files to the fiff format
    """
    _check_mayavi_version()
    from ._kit2fiff_gui import Kit2FiffFrame
    gui = Kit2FiffFrame()
    gui.configure_traits()
    return gui
