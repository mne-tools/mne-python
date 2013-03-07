"""Convenience functions for opening GUIs."""

from .transforms.coreg import trans_fname as _trans


def combine_markers(mrk1='', mrk2=''):
    """Create a new KIT marker file by interpolating two marker files

    Parameters
    ----------
    mrk1, mrk2 : str
        Path to source marker files (*.sqd; can be empty str, in which case the
        files can be loaded in GUI)
    """
    from .transforms.marker_gui import MainWindow
    gui = MainWindow(mrk1=mrk1, mrk2=mrk2)
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


def fit_mri_to_head(raw, s_from='fsaverage', s_to=None, trans_fname=_trans,
                    subjects_dir=None):
    """Open a gui for head-mri coregistration

    Parameters
    ----------
    raw : str(path)
        path to a raw file containing the digitizer data.
    s_from : str
        name of the source subject (e.g., 'fsaverage').
    s_to : str | None
        Name of the the subject for which the MRI is destined (used to
        save MRI and in the trans file's file name).
        Can be None if the raw file-name starts with "{subject}_".
    trans_fname : str
        Filename pattern for the trans file. "{raw_dir}" will be formatted to
        the directory containing the raw file, and "{subject}" will be
        formatted to s_to.
    subjects_dir : None | path
        Override the SUBJECTS_DIR environment variable
        (sys.environ['SUBJECTS_DIR'])
    """
    from .transforms.coreg_gui import MriHeadCoreg
    gui = MriHeadCoreg(raw, s_from, s_to, trans_fname, subjects_dir)
    gui.configure_traits()
    return gui


def kit2fiff():
    from .transforms.kit2fiff_gui import MainWindow
    gui = MainWindow()
    gui.configure_traits()
    return gui
