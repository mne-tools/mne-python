# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
# License: BSD Style.

import os
import os.path as op
import shutil
from warnings import warn

import logging
logger = logging.getLogger('mne')

from ... import __version__ as mne_version
from ...utils import get_config, set_config, _fetch_file


def _sample_version(path):
    """Get the version of the Sample dataset"""
    ver_fname = op.join(path, 'version.txt')
    if op.exists(ver_fname):
        fid = open(ver_fname, 'r')
        version = fid.readline().strip()  # version is on first line
        fid.close()
    else:
        # Sample dataset versioning was introduced after 0.3
        version = '0.3'

    return version


def data_path(path=None, force_update=False, update_path=True):
    """Get path to local copy of Sample dataset

    Parameters
    ----------
    path : None | str
        Location of where to look for the sample dataset.
        If None, the environment variable or config parameter
        MNE_DATASETS_SAMPLE_PATH is used. If it doesn't exist, the
        "mne-python/examples" directory is used. If the sample dataset
        is not found under the given path (e.g., as
        "mne-python/examples/MNE-sample-data"), the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the sample dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_SAMPLE_PATH in mne-python
        config to the given path. If None, the user is prompted.
    """
    if path is None:
        # use an intelligent guess if it's not defined
        def_path = op.abspath(op.join(op.dirname(__file__), '..', '..',
                                      '..', 'examples'))
        path = get_config('MNE_DATASETS_SAMPLE_PATH', def_path)

    if not isinstance(path, basestring):
        raise ValueError('path must be a string or None')

    archive_name = "MNE-sample-data-processed.tar.gz"
    url = "ftp://surfer.nmr.mgh.harvard.edu/pub/data/" + archive_name
    folder_name = "MNE-sample-data"
    folder_path = op.join(path, folder_name)
    rm_archive = False

    martinos_path = '/cluster/fusion/sample_data/' + archive_name
    neurospin_path = '/neurospin/tmp/gramfort/' + archive_name

    if not op.exists(folder_path) or force_update:
        logger.info('Sample data archive %s not found at:\n%s\n'
                    'It will be downloaded and extracted at this location.'
                    % (archive_name, folder_path))

        if op.exists(martinos_path):
            archive_name = martinos_path
        elif op.exists(neurospin_path):
            archive_name = neurospin_path
        else:
            archive_name = op.join(path, archive_name)
            rm_archive = True
            if op.exists(archive_name):
                msg = ('Archive already exists at %r. Overwrite it '
                       '(y/[n])? ' % archive_name)
                answer = raw_input(msg)
                if answer.lower() == 'y':
                    os.remove(archive_name)
                else:
                    raise IOError('Archive file already exists at target '
                                  'location %r.' % archive_name)

            _fetch_file(url, archive_name, print_destination=False)

        if op.exists(folder_path):
            shutil.rmtree(folder_path)

        import tarfile
        # note that we use print statements here because these processes
        # are interactive
        logger.info('Decompressiong the archive: ' + archive_name)
        tarfile.open(archive_name, 'r:gz').extractall(path=path)
        if rm_archive:
            os.remove(archive_name)

    path = op.abspath(path)
    if update_path is None:
        if get_config('MNE_DATASETS_SAMPLE_PATH', '') != path:
            update_path = True
            msg = ('Do you want to set the path:\n    %s\nas the default '
                   'sample dataset path in the mne-python config [y]/n? '
                   % path)
            answer = raw_input(msg)
            if answer.lower() == 'n':
                update_path = False
        else:
            update_path = False
    if update_path is True:
        set_config('MNE_DATASETS_SAMPLE_PATH', path)

    path = op.join(path, folder_name)

    # compare the version of the Sample dataset and mne
    sample_version = _sample_version(path)
    try:
        from distutils.version import LooseVersion
    except:
        warn('Could not determine sample dataset version; dataset could\n'
             'be out of date. Please install the "distutils" package.')
    else:
        if LooseVersion(sample_version) < LooseVersion(mne_version):
            warn('Sample dataset (version %s) is older than mne-python '
                 '(version %s). If the examples fail, you may need to update '
                 'the sample dataset by using force_update=True'
                 % (sample_version, mne_version))

    return path
