# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Denis Egnemann <denis.engemann@gmail.com>
# License: BSD Style.

import os
import os.path as op
import shutil
import tarfile
import stat
import sys

from .. import __version__ as mne_version
from ..utils import get_config, set_config, _fetch_file, logger, warn, verbose
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


_bst_license_text = """
License
-------
This tutorial dataset (EEG and MRI data) remains a property of the MEG Lab,
McConnell Brain Imaging Center, Montreal Neurological Institute,
McGill University, Canada. Its use and transfer outside the Brainstorm
tutorial, e.g. for research purposes, is prohibited without written consent
from the MEG Lab.

If you reference this dataset in your publications, please:
1) aknowledge its authors: Elizabeth Bock, Esther Florin, Francois Tadel and
Sylvain Baillet
2) cite Brainstorm as indicated on the website:
http://neuroimage.usc.edu/brainstorm

For questions, please contact Francois Tadel (francois.tadel@mcgill.ca).
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
                                       'examples'))
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
            if '--update-dataset-path' in sys.argv:
                answer = 'y'
            else:
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
               name=None, check_version=False, return_version=False,
               archive_name=None):
    """Aux function
    """
    key = {
        'fake': 'MNE_DATASETS_FAKE_PATH',
        'misc': 'MNE_DATASETS_MISC_PATH',
        'sample': 'MNE_DATASETS_SAMPLE_PATH',
        'spm': 'MNE_DATASETS_SPM_FACE_PATH',
        'somato': 'MNE_DATASETS_SOMATO_PATH',
        'brainstorm': 'MNE_DATASETS_BRAINSTORM_PATH',
        'testing': 'MNE_DATASETS_TESTING_PATH',
    }[name]

    path = _get_path(path, key, name)
    # To update the testing or misc dataset, push commits, then make a new
    # release on GitHub. Then update the "releases" variable:
    releases = dict(testing='0.19', misc='0.1')
    # And also update the "hashes['testing']" variable below.

    # To update any other dataset, update the data archive itself (upload
    # an updated version) and update the hash.
    archive_names = dict(
        misc='mne-misc-data-%s.tar.gz' % releases['misc'],
        sample='MNE-sample-data-processed.tar.gz',
        somato='MNE-somato-data.tar.gz',
        spm='MNE-spm-face.tar.gz',
        testing='mne-testing-data-%s.tar.gz' % releases['testing'],
        fake='foo.tgz',
    )
    if archive_name is not None:
        archive_names.update(archive_name)
    folder_names = dict(
        brainstorm='MNE-brainstorm-data',
        fake='foo',
        misc='MNE-misc-data',
        sample='MNE-sample-data',
        somato='MNE-somato-data',
        spm='MNE-spm-face',
        testing='MNE-testing-data',
    )
    urls = dict(
        brainstorm='https://mne-tools.s3.amazonaws.com/datasets/'
                   'MNE-brainstorm-data/%s',
        fake='https://github.com/mne-tools/mne-testing-data/raw/master/'
             'datasets/%s',
        misc='https://codeload.github.com/mne-tools/mne-misc-data/'
             'tar.gz/%s' % releases['misc'],
        sample="https://mne-tools.s3.amazonaws.com/datasets/%s",
        somato='https://mne-tools.s3.amazonaws.com/datasets/%s',
        spm='https://mne-tools.s3.amazonaws.com/datasets/%s',
        testing='https://codeload.github.com/mne-tools/mne-testing-data/'
                'tar.gz/%s' % releases['testing'],
    )
    hashes = dict(
        brainstorm=None,
        fake='3194e9f7b46039bb050a74f3e1ae9908',
        misc='f0708d8914cf2692fee7b6c9f105e71c',
        sample='1d5da3a809fded1ef5734444ab5bf857',
        somato='f3e3a8441477bb5bacae1d0c6e0964fb',
        spm='f61041e3f3f2ba0def8a2ca71592cc41',
        testing='77b2a435d80adb23cbe7e19144e7bc47'
    )
    folder_origs = dict(  # not listed means None
        misc='mne-misc-data-%s' % releases['misc'],
        testing='mne-testing-data-%s' % releases['testing'],
    )
    folder_name = folder_names[name]
    archive_name = archive_names[name]
    hash_ = hashes[name]
    url = urls[name]
    folder_orig = folder_origs.get(name, None)
    if '%s' in url:
        url = url % archive_name

    folder_path = op.join(path, folder_name)
    if name == 'brainstorm':
        extract_path = folder_path
        folder_path = op.join(folder_path, archive_names[name].split('.')[0])

    rm_archive = False
    martinos_path = '/cluster/fusion/sample_data/' + archive_name
    neurospin_path = '/neurospin/tmp/gramfort/' + archive_name

    if not op.exists(folder_path) and not download:
        return ''
    if not op.exists(folder_path) or force_update:
        if name == 'brainstorm':
            if '--accept-brainstorm-license' in sys.argv:
                answer = 'y'
            else:
                answer = input('%sAgree (y/[n])? ' % _bst_license_text)
            if answer.lower() != 'y':
                raise RuntimeError('You must agree to the license to use this '
                                   'dataset')
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
                if name != 'brainstorm':
                    extract_path = path
                tf = tarfile.open(archive_name, 'r:%s' % ext)
                tf.extractall(path=extract_path)
                tf.close()
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
        warn('Could not determine %s dataset version; dataset could '
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
    endswith = {
        'brainstorm': 'MNE_brainstorm-data',
        'fake': 'foo',
        'misc': 'MNE-misc-data',
        'sample': 'MNE-sample-data',
        'somato': 'MNE-somato-data',
        'spm': 'MNE-spm-face',
        'testing': 'MNE-testing-data',
    }[name]
    archive_name = None
    if name == 'brainstorm':
        archive_name = dict(brainstorm='bst_raw')
    dp = _data_path(download=False, name=name, check_version=False,
                    archive_name=archive_name)
    return dp.endswith(endswith)


@verbose
def _download_all_example_data(verbose=True):
    """Helper to download all datasets used in examples and tutorials"""
    # This function is designed primarily to be used by CircleCI. It has
    # verbose=True by default so we get nice status messages
    from . import (sample, testing, misc, spm_face, somato, brainstorm, megsim,
                   eegbci)
    sample.data_path()
    testing.data_path()
    misc.data_path()
    spm_face.data_path()
    somato.data_path()
    sys.argv += ['--accept-brainstorm-license']
    try:
        brainstorm.bst_raw.data_path()
        brainstorm.bst_auditory.data_path()
    finally:
        sys.argv.pop(-1)
    sys.argv += ['--update-dataset-path']
    try:
        megsim.load_data(condition='visual', data_format='single-trial',
                         data_type='simulation')
        megsim.load_data(condition='visual', data_format='raw',
                         data_type='experimental')
        megsim.load_data(condition='visual', data_format='evoked',
                         data_type='simulation')
    finally:
        sys.argv.pop(-1)
    url_root = 'http://www.physionet.org/physiobank/database/eegmmidb/'
    eegbci.data_path(url_root + 'S001/S001R06.edf', update_path=True)
    eegbci.data_path(url_root + 'S001/S001R10.edf', update_path=True)
    eegbci.data_path(url_root + 'S001/S001R14.edf', update_path=True)
