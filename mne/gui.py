"""Convenience functions for opening guis."""

from .transforms.coreg import trans_fname as _trans


def coregistration(raw, subject=None, trans_fname=_trans, subjects_dir=None):
    """Open a gui for scaling an mri to fit a subject's head shape

    Parameters
    ----------
    raw : str(path)
        path to a raw file containing the digitizer data.
    subject : str
        name of the mri subject.
        Can be None if the raw file-name starts with "{subject}_".
    trans_fname : str
        Filename pattern for the trans file. "{raw_dir}" will be formatted to
        the directory containing the raw file, and "{subject}" will be
        formatted to the subject name.
    subjects_dir : None | path
        Override the SUBJECTS_DIR environment variable
        (sys.environ['SUBJECTS_DIR'])
    """
    from .transforms.coreg_gui import HeadMriCoreg
    gui = HeadMriCoreg(raw, subject, trans_fname, subjects_dir)
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


def set_fiducials(subject, fid=None, subjects_dir=None):
    """Open a gui for creating a fiducials file for an mri

    Parameters
    ----------
    subject : str
        The mri subject.
    fid : None | str
        Fiducials file for initial positions.
    subjects_dir : None | str
        Overrule the subjects_dir environment variable.
    """
    from .transforms.coreg_gui import Fiducials
    gui = Fiducials(subject, fid, subjects_dir)
    gui.configure_traits()
    return gui
