"""Convenience functions for opening guis."""



def coregistration(raw, subject=None, subjects_dir=None):
    """Open a gui for scaling an mri to fit a subject's head shape.

    Parameters
    ----------
    raw : str(path)
        path to a raw file containing the digitizer data.
    subject : str
        name of the mri subject.
        Can be None if the raw file-name starts with "{subject}_".
    subjects_dir : None | path
        Override the SUBJECTS_DIR environment variable
        (sys.environ['SUBJECTS_DIR'])

    """
    from .transforms.coreg_gui import HeadMriCoreg
    gui = HeadMriCoreg(raw, subject, subjects_dir)
    gui.configure_traits()
    return gui


def fit_mri_to_head(raw, s_from=None, s_to=None, subjects_dir=None):
    """Open a gui for head-mri coregistration.

    Parameters
    ----------
    raw : str(path)
        path to a raw file containing the digitizer data.
    s_from : str
        name of the source subject (e.g., 'fsaverage').
        Can be None if the raw file-name starts with "{subject}_".
    s_to : str | None
        Name of the the subject for which the MRI is destined (used to
        save MRI and in the trans file's file name).
        Can be None if the raw file-name starts with "{subject}_".
    subjects_dir : None | path
        Override the SUBJECTS_DIR environment variable
        (sys.environ['SUBJECTS_DIR'])

    """
    from .transforms.coreg_gui import MriHeadCoreg
    gui = MriHeadCoreg(raw, s_from, s_to, subjects_dir)
    gui.configure_traits()
    return gui


def set_fiducials(subject, fid=None, subjects_dir=None):
    """Open a gui for creating a fiducials file for an mri.

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
