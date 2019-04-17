# -*- coding: utf-8 -*-
# Authors: Eric Larson <larson.eric.d@gmail.com>
# License: BSD Style.

import os
import os.path as op
import tempfile
import zipfile

from ..utils import (_fetch_file, logger, verbose, get_subjects_dir, _pl,
                     set_config)


@verbose
def fetch_fsaverage(subjects_dir=None, verbose=None):
    """Fetch and update fsaverage.

    Parameters
    ----------
    subjects_dir : str | None
        The subjects directory to use. The ``subjects_dir + '/fsaverage'``
        directory will be created or updated as necessary.
    %(verbose)s

    Returns
    -------
    fs_dir : str
        The fsaverage directory.

    See Also
    --------
    mne.datasets.set_montage_coreg_path

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
    # Code used to create this dataset:
    #
    # $ tar xzf freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz
    # $ cp -a freesurfer/subjects/fsaverage .
    # $ mkdir fsaverage/bem
    # $ cp ~/python/mne-python/mne/data/fsaverage/* fsaverage/bem/
    # $ mne watershed_bem -s fsaverage -d $PWD --verbose info --copy
    # $ python
    # >>> src = mne.setup_source_space('fsaverage', spacing='ico5',
    # ...                              add_dist=False, subjects_dir='.')
    # >>> mne.write_source_spaces('fsaverage/bem/fsaverage-5-src.fif', src)
    # >>> mne.datasets.fetch_hcp_mmp_parcellation('.', verbose=True)
    # >>> mne.datasets.fetch_aparc_sub_parcellation('.', verbose=True)
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
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    subjects_dir = op.abspath(subjects_dir)
    fs_dir = op.join(subjects_dir, 'fsaverage')
    os.makedirs(fs_dir, exist_ok=True)
    with open(op.join(op.dirname(__file__), 'fsaverage.txt'), 'r') as fid:
        names = [name.strip() for name in fid.readlines()]
    need = list()
    for name in names:
        if not op.isfile(op.join(subjects_dir, name)):
            need.append(name)
    logger.info('%d file%s missing from fsaverage in %s'
                % (len(need), _pl(need), fs_dir))
    if len(need) > 0:
        with tempfile.TemporaryDirectory() as path:
            url = 'https://osf.io/j5htk/download?revision=1'
            hash_ = '614a3680dcfcebd5653b892cc1234a4a'
            fname = op.join(path, 'fsaverage.zip')
            logger.info('Downloading missing files remotely')
            _fetch_file(url, fname, hash_=hash_)
            logger.info('Extracting missing files')
            with zipfile.ZipFile(fname, 'r') as ff:
                members = set(f for f in ff.namelist()
                              if not f.endswith(op.sep))
                missing = sorted(members.symmetric_difference(set(names)))
                if len(missing):
                    raise RuntimeError('Zip file did not have correct names:'
                                       '\n%s' % ('\n'.join(missing)))
                for name in need:
                    ff.extract(name, path=subjects_dir)
        logger.info('Successfully extracted %d file%s'
                    % (len(need), _pl(need)))
    return fs_dir


def _get_create_subjects_dir(subjects_dir):
    from .utils import _get_path
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=False)
    if subjects_dir is None:
        subjects_dir = _get_path(None, '', 'montage coregistration')
        subjects_dir = op.join(subjects_dir, 'MNE-fsaverage-data')
        os.makedirs(subjects_dir, exist_ok=True)
    return subjects_dir


def set_montage_coreg_path(subjects_dir=None):
    """Set a subject directory suitable for montage(-only) coregistration.

    Parameters
    ----------
    subjects_dir : str | None
        The path to use as the subjects directory in the MNE-Python
        config file. None will use the existing config variable (i.e.,
        will not change anything), and if it does not exist, will use
        ``~/mne_data/MNE-fsaverage-data``.

    Returns
    -------
    subjects_dir : str
        The subjects directory that was used.

    See Also
    --------
    mne.datasets.fetch_fsaverage
    mne.get_config
    mne.set_config

    Notes
    -----
    If you plan to only do EEG-montage based coregistrations with fsaverage
    without any MRI warping, this function can facilitate the process.
    Essentially it sets the default value for ``subjects_dir`` in MNE
    functions to be ``~/mne_data/MNE-fsaverage-data`` (assuming it has
    not already been set to some other value).

    .. versionadded:: 0.18
    """
    subjects_dir = _get_create_subjects_dir(subjects_dir)
    old_subjects_dir = get_subjects_dir(None, raise_error=False)
    if old_subjects_dir is not None and old_subjects_dir != subjects_dir:
        raise ValueError('The subjects dir is already set to %r, which does '
                         'not match the provided subjects_dir=%r'
                         % (old_subjects_dir, subjects_dir))
    set_config('SUBJECTS_DIR', subjects_dir)
    return subjects_dir
