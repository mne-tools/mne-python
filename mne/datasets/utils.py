# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Denis Egnemann <denis.engemann@gmail.com>
# License: BSD Style.

import os
import os.path as op
import shutil
import tarfile
from warnings import warn
import stat

from .. import __version__ as mne_version
from ..utils import get_config, set_config, _fetch_file, logger
from ..externals.six import string_types
from ..externals.six.moves import input


_data_path_doc = """Get path to local copy of {name} dataset

    Parameters
    ----------
    path : None | str
        Location of where to look for the {name} dataset.
        If None, the environment variable or config parameter
        {conf} is used. If it doesn't exist, the
        "mne-python/examples" directory is used. If the {name} dataset
        is not found under the given path (e.g., as
        "mne-python/examples/MNE-{name}-data"), the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the {name} dataset even if a local copy exists.
    update_path : bool | None
        If True, set the {conf} in mne-python
        config to the given path. If None, the user is prompted.
    download : bool
        If False and the {name} dataset has not been downloaded yet,
        it will not be downloaded and the path will be returned as
        '' (empty string). This is mostly used for debugging purposes
        and can be safely ignored by most users.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    path : str
        Path to {name} dataset directory.
"""


_version_doc = """Get version of the local {name} dataset

    Returns
    -------
    version : str | None
        Version of the {name} local dataset, or None if the dataset
        does not exist locally.
"""


def _dataset_version(path, name):
    """Get the version of the dataset"""
    ver_fname = op.join(path, 'version.txt')
    if op.exists(ver_fname):
        with open(ver_fname, 'r') as fid:
            version = fid.readline().strip()  # version is on first line
    else:
        # Sample dataset versioning was introduced after 0.3
        # SPM dataset was introduced with 0.7
        version = '0.3' if name == 'sample' else '0.7'

    return version


def _get_path(path, key, name):
    """Helper to get a dataset path"""
    if path is None:
            # use an intelligent guess if it's not defined
            def_path = op.realpath(op.join(op.dirname(__file__), '..', '..',
                                           '..', 'examples'))
            if get_config(key) is None:
                key = 'MNE_DATA'
            path = get_config(key, def_path)

            # use the same for all datasets
            if not op.exists(path) or not os.access(path, os.W_OK):
                try:
                    os.mkdir(path)
                except OSError:
                    try:
                        logger.info('Checking for %s data in '
                                    '"~/mne_data"...' % name)
                        path = op.join(op.expanduser("~"), "mne_data")
                        if not op.exists(path):
                            logger.info("Trying to create "
                                        "'~/mne_data' in home directory")
                            os.mkdir(path)
                    except OSError:
                        raise OSError("User does not have write permissions "
                                      "at '%s', try giving the path as an "
                                      "argument to data_path() where user has "
                                      "write permissions, for ex:data_path"
                                      "('/home/xyz/me2/')" % (path))
    if not isinstance(path, string_types):
        raise ValueError('path must be a string or None')
    return path


def _do_path_update(path, update_path, key, name):
    """Helper to update path"""
    path = op.abspath(path)
    if update_path is None:
        if get_config(key, '') != path:
            update_path = True
            msg = ('Do you want to set the path:\n    %s\nas the default '
                   '%s dataset path in the mne-python config [y]/n? '
                   % (path, name))
            answer = input(msg)
            if answer.lower() == 'n':
                update_path = False
        else:
            update_path = False

    if update_path is True:
        set_config(key, path)
    return path


def _data_path(path=None, force_update=False, update_path=True, download=True,
               name=None, check_version=False, return_version=False):
    """Aux function
    """
    key = {'sample': 'MNE_DATASETS_SAMPLE_PATH',
           'spm': 'MNE_DATASETS_SPM_FACE_PATH',
           'somato': 'MNE_DATASETS_SOMATO_PATH',
           'testing': 'MNE_DATASETS_TESTING_PATH',
           'fake': 'MNE_DATASETS_FAKE_PATH',
           }[name]

    path = _get_path(path, key, name)
    archive_names = dict(
        sample='MNE-sample-data-processed.tar.gz',
        spm='MNE-spm-face.tar.bz2',
        somato='MNE-somato-data.tar.gz',
        testing='mne-testing-data-master.tar.gz',
        fake='foo.tgz',
    )
    folder_names = dict(
        sample='MNE-sample-data',
        spm='MNE-spm-face',
        somato='MNE-somato-data',
        testing='MNE-testing-data',
        fake='foo',
    )
    urls = dict(
        sample="https://s3.amazonaws.com/mne-python/datasets/%s",
        spm='https://s3.amazonaws.com/mne-python/datasets/%s',
        somato='https://s3.amazonaws.com/mne-python/datasets/%s',
        testing='https://github.com/mne-tools/mne-testing-data/archive/'
                'master.tar.gz',
        fake='https://github.com/mne-tools/mne-testing-data/raw/master/'
             'datasets/%s',
    )
    hashes = dict(
        sample='f73186795af820428e5e8e779ce5bfcf',
        spm='3e9e83c642136e5b720e2ecc5dcc3244',
        somato='f3e3a8441477bb5bacae1d0c6e0964fb',
        testing=None,
        fake='3194e9f7b46039bb050a74f3e1ae9908',
    )
    folder_origs = dict(  # not listed means None
        testing='mne-testing-data-master',
    )
    folder_name = folder_names[name]
    archive_name = archive_names[name]
    hash_ = hashes[name]
    url = urls[name]
    folder_orig = folder_origs.get(name, None)
    if '%s' in url:
        url = url % archive_name

    folder_path = op.join(path, folder_name)

    rm_archive = False
    martinos_path = '/cluster/fusion/sample_data/' + archive_name
    neurospin_path = '/neurospin/tmp/gramfort/' + archive_name
    if not op.exists(folder_path) and not download:
        return ''
    if not op.exists(folder_path) or force_update:
        logger.info('Downloading or reinstalling '
                    'data archive %s at location %s' % (archive_name, path))

        if op.exists(martinos_path):
            archive_name = martinos_path
        elif op.exists(neurospin_path):
            archive_name = neurospin_path
        else:
            archive_name = op.join(path, archive_name)
            rm_archive = True
            fetch_archive = True
            if op.exists(archive_name):
                msg = ('Archive already exists. Overwrite it (y/[n])? ')
                answer = input(msg)
                if answer.lower() == 'y':
                    os.remove(archive_name)
                else:
                    fetch_archive = False

            if fetch_archive:
                _fetch_file(url, archive_name, print_destination=False,
                            hash_=hash_)

        if op.exists(folder_path):
            def onerror(func, path, exc_info):
                """Deal with access errors (e.g. testing dataset read-only)"""
                # Is the error an access error ?
                do = False
                if not os.access(path, os.W_OK):
                    perm = os.stat(path).st_mode | stat.S_IWUSR
                    os.chmod(path, perm)
                    do = True
                if not os.access(op.dirname(path), os.W_OK):
                    dir_perm = (os.stat(op.dirname(path)).st_mode |
                                stat.S_IWUSR)
                    os.chmod(op.dirname(path), dir_perm)
                    do = True
                if do:
                    func(path)
                else:
                    raise
            shutil.rmtree(folder_path, onerror=onerror)

        logger.info('Decompressing the archive: %s' % archive_name)
        logger.info('(please be patient, this can take some time)')
        for ext in ['gz', 'bz2']:  # informed guess (and the only 2 options).
            try:
                tarfile.open(archive_name, 'r:%s' % ext).extractall(path=path)
                break
            except tarfile.ReadError as err:
                logger.info('%s is %s trying "bz2"' % (archive_name, err))
        if folder_orig is not None:
            shutil.move(op.join(path, folder_orig), folder_path)

        if rm_archive:
            os.remove(archive_name)

    path = _do_path_update(path, update_path, key, name)
    path = op.join(path, folder_name)

    # compare the version of the dataset and mne
    data_version = _dataset_version(path, name)
    try:
        from distutils.version import LooseVersion as LV
    except:
        warn('Could not determine %s dataset version; dataset could\n'
             'be out of date. Please install the "distutils" package.'
             % name)
    else:  # 0.7 < 0.7.git shoud be False, therefore strip
        if check_version and LV(data_version) < LV(mne_version.strip('.git')):
            warn('The {name} dataset (version {current}) is older than '
                 'mne-python (version {newest}). If the examples fail, '
                 'you may need to update the {name} dataset by using '
                 'mne.datasets.{name}.data_path(force_update=True)'.format(
                     name=name, current=data_version, newest=mne_version))
    return (path, data_version) if return_version else path


def _get_version(name):
    """Helper to get a dataset version"""
    if not has_dataset(name):
        return None
    return _data_path(name=name, return_version=True)[1]


def has_dataset(name):
    """Helper for dataset presence"""
    endswith = {'sample': 'MNE-sample-data',
                'spm': 'MNE-spm-face',
                'somato': 'MNE-somato-data',
                'testing': 'MNE-testing-data',
                'fake': 'foo',
                }[name]
    dp = _data_path(download=False, name=name, check_version=False)
    return dp.endswith(endswith)
