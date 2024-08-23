# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

from ...utils import get_subjects_dir, set_config, verbose
from ..utils import _get_path, _manifest_check_download

FSAVERAGE_MANIFEST_PATH = Path(__file__).parent


@verbose
def fetch_fsaverage(subjects_dir=None, *, verbose=None):
    """Fetch and update fsaverage.

    Parameters
    ----------
    subjects_dir : str | None
        The path to use as the subjects directory in the MNE-Python
        config file. None will use the existing config variable (i.e.,
        will not change anything), and if it does not exist, will use
        ``~/mne_data/MNE-fsaverage-data``.
    %(verbose)s

    Returns
    -------
    fs_dir : Path
        The fsaverage directory.
        (essentially ``subjects_dir / 'fsaverage'``).

        .. versionchanged:: 1.8
           A :class:`pathlib.Path` object is returned instead of a string.

    Notes
    -----
    This function is designed to provide

    1. All modern (Freesurfer 6) fsaverage subject files
    2. All MNE fsaverage parcellations
    3. fsaverage head surface, fiducials, head<->MRI trans, 1- and 3-layer
       BEMs (and surfaces)

    This function will compare the contents of ``subjects_dir/fsaverage``
    to the ones provided in the remote zip file. If any are missing,
    the zip file is downloaded and files are updated. No files will
    be overwritten.

    .. versionadded:: 0.18
    """
    # Code used to create the BEM (other files taken from MNE-sample-data):
    #
    # $ mne watershed_bem -s fsaverage -d $PWD --verbose info --copy
    # $ python
    # >>> bem = mne.make_bem_model('fsaverage', subjects_dir='.', verbose=True)
    # >>> mne.write_bem_surfaces(
    # ...     'fsaverage/bem/fsaverage-5120-5120-5120-bem.fif', bem)
    # >>> sol = mne.make_bem_solution(bem, verbose=True)
    # >>> mne.write_bem_solution(
    # ...     'fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif', sol)
    # >>> import os
    # >>> import os.path as op
    # >>> names = sorted(op.join(r, f)
    # ...                for r, d, files in os.walk('fsaverage')
    # ...                for f in files)
    # with open('fsaverage.txt', 'w') as fid:
    #     fid.write('\n'.join(names))
    #
    subjects_dir = _set_montage_coreg_path(subjects_dir)
    subjects_dir = subjects_dir.expanduser().absolute()
    fs_dir = subjects_dir / "fsaverage"
    fs_dir.mkdir(parents=True, exist_ok=True)
    _manifest_check_download(
        manifest_path=FSAVERAGE_MANIFEST_PATH / "root.txt",
        destination=subjects_dir,
        url="https://osf.io/3bxqt/download?version=2",
        hash_="5133fe92b7b8f03ae19219d5f46e4177",
    )
    _manifest_check_download(
        manifest_path=FSAVERAGE_MANIFEST_PATH / "bem.txt",
        destination=subjects_dir / "fsaverage",
        url="https://osf.io/7ve8g/download?version=4",
        hash_="b31509cdcf7908af6a83dc5ee8f49fb1",
    )
    return fs_dir


def _get_create_subjects_dir(subjects_dir):
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=False)
    if subjects_dir is None:
        subjects_dir = _get_path(None, "MNE_DATA", "montage coregistration")
        subjects_dir = subjects_dir / "MNE-fsaverage-data"
        subjects_dir.mkdir(parents=True, exist_ok=True)
    return subjects_dir


def _set_montage_coreg_path(subjects_dir=None):
    """Set a subject directory suitable for montage(-only) coregistration."""
    subjects_dir = _get_create_subjects_dir(subjects_dir)
    old_subjects_dir = get_subjects_dir(None, raise_error=False)
    if old_subjects_dir is None:
        set_config("SUBJECTS_DIR", subjects_dir)
    return subjects_dir
