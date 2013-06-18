"""Convenience functions for opening GUIs."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)


def combine_markers(mrk1='', mrk2=''):
    """Create a new KIT marker file by interpolating two marker files

    All parameters are optional, since they can be set through the GUI.

    Parameters
    ----------
    mrk1, mrk2 : str
        Path to source marker files (*.sqd; can be empty str, in which case the
        files can be loaded in GUI)
    """
    from .marker_gui import CombineMarkersFrame
    gui = CombineMarkersFrame(mrk1=mrk1, mrk2=mrk2)
    gui.configure_traits()
    return gui


def coregistration(raw=None, subject=None, subjects_dir=None):
    """Coregister an MRI with a subject's head shape

    All parameters are optional, since they can be set through the GUI.

    Parameters
    ----------
    raw : None | str(path)
        Path to a raw file containing the digitizer data.
    subject : None | str
        Name of the mri subject.
    subjects_dir : None | path
        Override the SUBJECTS_DIR environment variable
        (sys.environ['SUBJECTS_DIR'])
    """
    from .coreg_gui import CoregFrame
    gui = CoregFrame(raw, subject, subjects_dir)
    gui.configure_traits()
    return gui


def fiducials(subject=None, fid_file=None, subjects_dir=None):
    """Set the fiducials for an MRI subject

    All parameters are optional, since they can be set through the GUI.

    Parameters
    ----------
    subject : str
        Name of the mri subject.
    fid_file : None | str
        Load a fiducials file different form the subject's default
        ("{subjects_dir}/{subject}/bem/{subject}-fiducials.fif").
    subjects_dir : None | str
        Overrule the subjects_dir environment variable.
    """
    from .fiducials_gui import FiducialsFrame
    gui = FiducialsFrame(subject, subjects_dir, fid_file=fid_file)
    gui.configure_traits()
    return gui


def kit2fiff():
    """Convert KIT files to the fiff format
    """
    from .kit2fiff_gui import Kit2FiffFrame
    gui = Kit2FiffFrame()
    gui.configure_traits()
    return gui
