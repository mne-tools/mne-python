"""Convenience functions for opening GUIs."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)


def combine_markers(mrk1='', mrk2=''):
    """Create a new KIT marker file by interpolating two marker files

    Parameters
    ----------
    mrk1, mrk2 : str
        Path to source marker files (*.sqd; can be empty str, in which case the
        files can be loaded in GUI)
    """
    from .transforms.marker_gui import CombineMarkersFrame
    gui = CombineMarkersFrame(mrk1=mrk1, mrk2=mrk2)
    gui.configure_traits()
    return gui


def coregistration(raw=None, subject=None, subjects_dir=None):
    """Open a gui for scaling an mri to fit a subject's head shape

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
    from .transforms.coreg_gui import CoregFrame
    gui = CoregFrame(raw, subject, subjects_dir)
    gui.configure_traits()
    return gui


def fiducials(subject=None, subjects_dir=None, fid_file=None):
    """Open a gui to set the fiducials for an mri subject

    Parameters
    ----------
    subject : str
        Name of the mri subject.
    subjects_dir : None | str
        Overrule the subjects_dir environment variable.
    fid_file : None | str
        Load a fiducials file different form the subject's default
        ("{subjects_dir}/{subject}/bem/{subject}-fiducials.fif").
    """
    from .transforms.fiducials_gui import MainWindow
    gui = MainWindow(subject, subjects_dir, fid_file=fid_file)
    gui.configure_traits()
    return gui


def kit2fiff():
    from .transforms.kit2fiff_gui import MainWindow
    gui = MainWindow()
    gui.configure_traits()
    return gui
