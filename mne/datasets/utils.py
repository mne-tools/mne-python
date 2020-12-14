# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Denis Egnemann <denis.engemann@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
# License: BSD Style.

from collections import OrderedDict
import os
import os.path as op
import shutil
import tarfile
import stat
import sys
import zipfile
import tempfile
from distutils.version import LooseVersion

import numpy as np

from ._fsaverage.base import fetch_fsaverage
from .. import __version__ as mne_version
from ..label import read_labels_from_annot, Label, write_labels_to_annot
from ..utils import (get_config, set_config, _fetch_file, logger, warn,
                     verbose, get_subjects_dir, hashfunc, _pl, _safe_input)
from ..utils.docs import docdict
from ..externals.doccer import docformat


_data_path_doc = """Get path to local copy of {name} dataset.

    Parameters
    ----------
    path : None | str
        Location of where to look for the {name} dataset.
        If None, the environment variable or config parameter
        ``{conf}`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the {name} dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the {name} dataset even if a local copy exists.
    update_path : bool | None
        If True, set the ``{conf}`` in mne-python
        config to the given path. If None, the user is prompted.
    download : bool
        If False and the {name} dataset has not been downloaded yet,
        it will not be downloaded and the path will be returned as
        '' (empty string). This is mostly used for debugging purposes
        and can be safely ignored by most users.
    %(verbose)s

    Returns
    -------
    path : str
        Path to {name} dataset directory.
"""
_data_path_doc_accept = _data_path_doc.split('%(verbose)s')
_data_path_doc_accept[-1] = '%(verbose)s' + _data_path_doc_accept[-1]
_data_path_doc_accept.insert(1, '    %(accept)s')
_data_path_doc_accept = ''.join(_data_path_doc_accept)
_data_path_doc = docformat(_data_path_doc, docdict)
_data_path_doc_accept = docformat(_data_path_doc_accept, docdict)

_version_doc = """Get version of the local {name} dataset.

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

    1) acknowledge its authors: Elizabeth Bock, Esther Florin, Francois Tadel
       and Sylvain Baillet, and
    2) cite Brainstorm as indicated on the website:
       http://neuroimage.usc.edu/brainstorm

For questions, please contact Francois Tadel (francois.tadel@mcgill.ca).
"""

_hcp_mmp_license_text = """
License
-------
I request access to data collected by the Washington University - University
of Minnesota Consortium of the Human Connectome Project (WU-Minn HCP), and
I agree to the following:

1. I will not attempt to establish the identity of or attempt to contact any
   of the included human subjects.

2. I understand that under no circumstances will the code that would link
   these data to Protected Health Information be given to me, nor will any
   additional information about individual human subjects be released to me
   under these Open Access Data Use Terms.

3. I will comply with all relevant rules and regulations imposed by my
   institution. This may mean that I need my research to be approved or
   declared exempt by a committee that oversees research on human subjects,
   e.g. my IRB or Ethics Committee. The released HCP data are not considered
   de-identified, insofar as certain combinations of HCP Restricted Data
   (available through a separate process) might allow identification of
   individuals.  Different committees operate under different national, state
   and local laws and may interpret regulations differently, so it is
   important to ask about this. If needed and upon request, the HCP will
   provide a certificate stating that you have accepted the HCP Open Access
   Data Use Terms.

4. I may redistribute original WU-Minn HCP Open Access data and any derived
   data as long as the data are redistributed under these same Data Use Terms.

5. I will acknowledge the use of WU-Minn HCP data and data derived from
   WU-Minn HCP data when publicly presenting any results or algorithms
   that benefitted from their use.

   1. Papers, book chapters, books, posters, oral presentations, and all
      other printed and digital presentations of results derived from HCP
      data should contain the following wording in the acknowledgments
      section: "Data were provided [in part] by the Human Connectome
      Project, WU-Minn Consortium (Principal Investigators: David Van Essen
      and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and
      Centers that support the NIH Blueprint for Neuroscience Research; and
      by the McDonnell Center for Systems Neuroscience at Washington
      University."

   2. Authors of publications or presentations using WU-Minn HCP data
      should cite relevant publications describing the methods used by the
      HCP to acquire and process the data. The specific publications that
      are appropriate to cite in any given study will depend on what HCP
      data were used and for what purposes. An annotated and appropriately
      up-to-date list of publications that may warrant consideration is
      available at http://www.humanconnectome.org/about/acknowledgehcp.html

   3. The WU-Minn HCP Consortium as a whole should not be included as an
      author of publications or presentations if this authorship would be
      based solely on the use of WU-Minn HCP data.

6. Failure to abide by these guidelines will result in termination of my
   privileges to access WU-Minn HCP data.
"""


def _dataset_version(path, name):
    """Get the version of the dataset."""
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
    """Get a dataset path."""
    # 1. Input
    if path is not None:
        if not isinstance(path, str):
            raise ValueError('path must be a string or None')
        return path
    # 2. get_config(key)
    # 3. get_config('MNE_DATA')
    path = get_config(key, get_config('MNE_DATA'))
    if path is not None:
        return path
    # 4. ~/mne_data (but use a fake home during testing so we don't
    #    unnecessarily create ~/mne_data)
    logger.info('Using default location ~/mne_data for %s...' % name)
    path = op.join(os.getenv('_MNE_FAKE_HOME_DIR',
                             op.expanduser("~")), 'mne_data')
    if not op.exists(path):
        logger.info('Creating ~/mne_data')
        try:
            os.mkdir(path)
        except OSError:
            raise OSError("User does not have write permissions "
                          "at '%s', try giving the path as an "
                          "argument to data_path() where user has "
                          "write permissions, for ex:data_path"
                          "('/home/xyz/me2/')" % (path))
    return path


def _do_path_update(path, update_path, key, name):
    """Update path."""
    path = op.abspath(path)
    identical = get_config(key, '', use_env=False) == path
    if not identical:
        if update_path is None:
            update_path = True
            if '--update-dataset-path' in sys.argv:
                answer = 'y'
            else:
                msg = ('Do you want to set the path:\n    %s\nas the default '
                       '%s dataset path in the mne-python config [y]/n? '
                       % (path, name))
                answer = _safe_input(msg, alt='pass update_path=True')
            if answer.lower() == 'n':
                update_path = False

        if update_path:
            set_config(key, path, set_env=False)
    return path


def _data_path(path=None, force_update=False, update_path=True, download=True,
               name=None, check_version=False, return_version=False,
               archive_name=None, accept=False):
    """Aux function."""
    key = {
        'fake': 'MNE_DATASETS_FAKE_PATH',
        'misc': 'MNE_DATASETS_MISC_PATH',
        'sample': 'MNE_DATASETS_SAMPLE_PATH',
        'spm': 'MNE_DATASETS_SPM_FACE_PATH',
        'somato': 'MNE_DATASETS_SOMATO_PATH',
        'brainstorm': 'MNE_DATASETS_BRAINSTORM_PATH',
        'testing': 'MNE_DATASETS_TESTING_PATH',
        'multimodal': 'MNE_DATASETS_MULTIMODAL_PATH',
        'fnirs_motor': 'MNE_DATASETS_FNIRS_MOTOR_PATH',
        'opm': 'MNE_DATASETS_OPM_PATH',
        'visual_92_categories': 'MNE_DATASETS_VISUAL_92_CATEGORIES_PATH',
        'kiloword': 'MNE_DATASETS_KILOWORD_PATH',
        'mtrf': 'MNE_DATASETS_MTRF_PATH',
        'fieldtrip_cmc': 'MNE_DATASETS_FIELDTRIP_CMC_PATH',
        'phantom_4dbti': 'MNE_DATASETS_PHANTOM_4DBTI_PATH',
        'limo': 'MNE_DATASETS_LIMO_PATH',
        'refmeg_noise': 'MNE_DATASETS_REFMEG_NOISE_PATH',
    }[name]

    path = _get_path(path, key, name)
    # To update the testing or misc dataset, push commits, then make a new
    # release on GitHub. Then update the "releases" variable:
    releases = dict(testing='0.112', misc='0.7')
    # And also update the "md5_hashes['testing']" variable below.
    # To update any other dataset, update the data archive itself (upload
    # an updated version) and update the md5 hash.

    # try to match url->archive_name->folder_name
    urls = dict(  # the URLs to use
        brainstorm=dict(
            bst_auditory='https://osf.io/5t9n8/download?version=1',
            bst_phantom_ctf='https://osf.io/sxr8y/download?version=1',
            bst_phantom_elekta='https://osf.io/dpcku/download?version=1',
            bst_raw='https://osf.io/9675n/download?version=2',
            bst_resting='https://osf.io/m7bd3/download?version=3'),
        fake='https://github.com/mne-tools/mne-testing-data/raw/master/'
             'datasets/foo.tgz',
        misc='https://codeload.github.com/mne-tools/mne-misc-data/'
             'tar.gz/%s' % releases['misc'],
        sample='https://osf.io/86qa2/download?version=5',
        somato='https://osf.io/tp4sg/download?version=7',
        spm='https://osf.io/je4s8/download?version=2',
        testing='https://codeload.github.com/mne-tools/mne-testing-data/'
                'tar.gz/%s' % releases['testing'],
        multimodal='https://ndownloader.figshare.com/files/5999598',
        fnirs_motor='https://osf.io/dj3eh/download?version=1',
        opm='https://osf.io/p6ae7/download?version=2',
        visual_92_categories=[
            'https://osf.io/8ejrs/download?version=1',
            'https://osf.io/t4yjp/download?version=1'],
        mtrf='https://osf.io/h85s2/download?version=1',
        kiloword='https://osf.io/qkvf9/download?version=1',
        fieldtrip_cmc='https://osf.io/j9b6s/download?version=1',
        phantom_4dbti='https://osf.io/v2brw/download?version=2',
        refmeg_noise='https://osf.io/drt6v/download?version=1',
    )
    # filename of the resulting downloaded archive (only needed if the URL
    # name does not match resulting filename)
    archive_names = dict(
        fieldtrip_cmc='SubjectCMC.zip',
        kiloword='MNE-kiloword-data.tar.gz',
        misc='mne-misc-data-%s.tar.gz' % releases['misc'],
        mtrf='mTRF_1.5.zip',
        multimodal='MNE-multimodal-data.tar.gz',
        fnirs_motor='MNE-fNIRS-motor-data.tgz',
        opm='MNE-OPM-data.tar.gz',
        sample='MNE-sample-data-processed.tar.gz',
        somato='MNE-somato-data.tar.gz',
        spm='MNE-spm-face.tar.gz',
        testing='mne-testing-data-%s.tar.gz' % releases['testing'],
        visual_92_categories=['MNE-visual_92_categories-data-part1.tar.gz',
                              'MNE-visual_92_categories-data-part2.tar.gz'],
        phantom_4dbti='MNE-phantom-4DBTi.zip',
        refmeg_noise='sample_reference_MEG_noise-raw.zip'
    )
    # original folder names that get extracted (only needed if the
    # archive does not extract the right folder name; e.g., usually GitHub)
    folder_origs = dict(  # not listed means None (no need to move)
        misc='mne-misc-data-%s' % releases['misc'],
        testing='mne-testing-data-%s' % releases['testing'],
    )
    # finally, where we want them to extract to (only needed if the folder name
    # is not the same as the last bit of the archive name without the file
    # extension)
    folder_names = dict(
        brainstorm='MNE-brainstorm-data',
        fake='foo',
        misc='MNE-misc-data',
        mtrf='mTRF_1.5',
        sample='MNE-sample-data',
        testing='MNE-testing-data',
        visual_92_categories='MNE-visual_92_categories-data',
        fieldtrip_cmc='MNE-fieldtrip_cmc-data',
        phantom_4dbti='MNE-phantom-4DBTi',
        refmeg_noise='MNE-refmeg-noise-data'
    )
    md5_hashes = dict(
        brainstorm=dict(
            bst_auditory='fa371a889a5688258896bfa29dd1700b',
            bst_phantom_ctf='80819cb7f5b92d1a5289db3fb6acb33c',
            bst_phantom_elekta='1badccbe17998d18cc373526e86a7aaf',
            bst_raw='fa2efaaec3f3d462b319bc24898f440c',
            bst_resting='70fc7bf9c3b97c4f2eab6260ee4a0430'),
        fake='3194e9f7b46039bb050a74f3e1ae9908',
        misc='2b2f2fec9d1197ed459117db1c6341ee',
        sample='12b75d1cb7df9dfb4ad73ed82f61094f',
        somato='32fd2f6c8c7eb0784a1de6435273c48b',
        spm='9f43f67150e3b694b523a21eb929ea75',
        testing='8eabd73532dd7df7c155983962c5b1fd',
        multimodal='26ec847ae9ab80f58f204d09e2c08367',
        fnirs_motor='c4935d19ddab35422a69f3326a01fef8',
        opm='370ad1dcfd5c47e029e692c85358a374',
        visual_92_categories=['74f50bbeb65740903eadc229c9fa759f',
                              '203410a98afc9df9ae8ba9f933370e20'],
        kiloword='3a124170795abbd2e48aae8727e719a8',
        mtrf='273a390ebbc48da2c3184b01a82e4636',
        fieldtrip_cmc='6f9fd6520f9a66e20994423808d2528c',
        phantom_4dbti='938a601440f3ffa780d20a17bae039ff',
        refmeg_noise='779fecd890d98b73a4832e717d7c7c45'
    )
    assert set(md5_hashes.keys()) == set(urls.keys())
    url = urls[name]
    hash_ = md5_hashes[name]
    folder_orig = folder_origs.get(name, None)
    if name == 'brainstorm':
        assert archive_name is not None
        url = [url[archive_name.split('.')[0]]]
        folder_path = [op.join(path, folder_names[name],
                               archive_name.split('.')[0])]
        hash_ = [hash_[archive_name.split('.')[0]]]
        archive_name = [archive_name]
    else:
        url = [url] if not isinstance(url, list) else url
        hash_ = [hash_] if not isinstance(hash_, list) else hash_
        archive_name = archive_names.get(name)
        if archive_name is None:
            archive_name = [u.split('/')[-1] for u in url]
        if not isinstance(archive_name, list):
            archive_name = [archive_name]
        folder_path = [op.join(path, folder_names.get(name, a.split('.')[0]))
                       for a in archive_name]
    if not isinstance(folder_orig, list):
        folder_orig = [folder_orig] * len(url)
    folder_path = [op.abspath(f) for f in folder_path]
    assert hash_ is not None
    assert all(isinstance(x, list) for x in (url, archive_name, hash_,
                                             folder_path))
    assert len(url) == len(archive_name) == len(hash_) == len(folder_path)
    logger.debug('URL:          %s' % (url,))
    logger.debug('archive_name: %s' % (archive_name,))
    logger.debug('hash:         %s' % (hash_,))
    logger.debug('folder_path:  %s' % (folder_path,))

    need_download = any(not op.exists(f) for f in folder_path)
    if need_download and not download:
        return ''

    if need_download or force_update:
        logger.debug('Downloading: need_download=%s, force_update=%s'
                     % (need_download, force_update))
        for f in folder_path:
            logger.debug('  Exists: %s: %s' % (f, op.exists(f)))
        if name == 'brainstorm':
            if accept or '--accept-brainstorm-license' in sys.argv:
                answer = 'y'
            else:
                # If they don't have stdin, just accept the license
                # https://github.com/mne-tools/mne-python/issues/8513#issuecomment-726823724  # noqa: E501
                answer = _safe_input(
                    '%sAgree (y/[n])? ' % _bst_license_text, use='y')
            if answer.lower() != 'y':
                raise RuntimeError('You must agree to the license to use this '
                                   'dataset')
        assert len(url) == len(hash_)
        assert len(url) == len(archive_name)
        assert len(url) == len(folder_orig)
        assert len(url) == len(folder_path)
        assert len(url) > 0
        # 1. Get all the archives
        full_name = list()
        for u, an, h, fo in zip(url, archive_name, hash_, folder_orig):
            remove_archive, full = _download(path, u, an, h)
            full_name.append(full)
        del archive_name
        # 2. Extract all of the files
        remove_dir = True
        for u, fp, an, h, fo in zip(url, folder_path, full_name, hash_,
                                    folder_orig):
            _extract(path, name, fp, an, fo, remove_dir)
            remove_dir = False  # only do on first iteration
        # 3. Remove all of the archives
        if remove_archive:
            for an in full_name:
                os.remove(op.join(path, an))

        logger.info('Successfully extracted to: %s' % folder_path)

    _do_path_update(path, update_path, key, name)
    path = folder_path[0]

    # compare the version of the dataset and mne
    data_version = _dataset_version(path, name)
    # 0.7 < 0.7.git should be False, therefore strip
    if check_version and (LooseVersion(data_version) <
                          LooseVersion(mne_version.strip('.git'))):
        warn('The {name} dataset (version {current}) is older than '
             'mne-python (version {newest}). If the examples fail, '
             'you may need to update the {name} dataset by using '
             'mne.datasets.{name}.data_path(force_update=True)'.format(
                 name=name, current=data_version, newest=mne_version))
    return (path, data_version) if return_version else path


def _download(path, url, archive_name, hash_, hash_type='md5'):
    """Download and extract an archive, completing the filename."""
    martinos_path = '/cluster/fusion/sample_data/' + archive_name
    neurospin_path = '/neurospin/tmp/gramfort/' + archive_name
    remove_archive = False
    if op.exists(martinos_path):
        full_name = martinos_path
    elif op.exists(neurospin_path):
        full_name = neurospin_path
    else:
        full_name = op.join(path, archive_name)
        remove_archive = True
        fetch_archive = True
        if op.exists(full_name):
            logger.info('Archive exists (%s), checking hash %s.'
                        % (archive_name, hash_,))
            fetch_archive = False
            if hashfunc(full_name, hash_type=hash_type) != hash_:
                if input('Archive already exists but the hash does not match: '
                         '%s\nOverwrite (y/[n])?'
                         % (archive_name,)).lower() == 'y':
                    os.remove(full_name)
                    fetch_archive = True
        if fetch_archive:
            logger.info('Downloading archive %s to %s' % (archive_name, path))
            _fetch_file(url, full_name, print_destination=False,
                        hash_=hash_, hash_type=hash_type)
    return remove_archive, full_name


def _extract(path, name, folder_path, archive_name, folder_orig, remove_dir):
    if op.exists(folder_path) and remove_dir:
        logger.info('Removing old directory: %s' % (folder_path,))

        def onerror(func, path, exc_info):
            """Deal with access errors (e.g. testing dataset read-only)."""
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
                raise exc_info[1]
        shutil.rmtree(folder_path, onerror=onerror)

    logger.info('Decompressing the archive: %s' % archive_name)
    logger.info('(please be patient, this can take some time)')
    if name == 'fieldtrip_cmc':
        extract_path = folder_path
    elif name == 'brainstorm':
        extract_path = op.join(*op.split(folder_path)[:-1])
    else:
        extract_path = path
    if archive_name.endswith('.zip'):
        with zipfile.ZipFile(archive_name, 'r') as ff:
            ff.extractall(extract_path)
    else:
        if archive_name.endswith('.bz2'):
            ext = 'bz2'
        else:
            ext = 'gz'
        with tarfile.open(archive_name, 'r:%s' % ext) as tf:
            tf.extractall(path=extract_path)

    if folder_orig is not None:
        shutil.move(op.join(path, folder_orig), folder_path)


def _get_version(name):
    """Get a dataset version."""
    if not has_dataset(name):
        return None
    if name.startswith('brainstorm'):
        name, archive_name = name.split('.')
    else:
        archive_name = None
    return _data_path(name=name, archive_name=archive_name,
                      return_version=True)[1]


def has_dataset(name):
    """Check for dataset presence.

    Parameters
    ----------
    name : str
        The dataset name.
        For brainstorm datasets, should be formatted like
        "brainstorm.bst_raw".

    Returns
    -------
    has : bool
        True if the dataset is present.
    """
    name = 'spm' if name == 'spm_face' else name
    if name.startswith('brainstorm'):
        name, archive_name = name.split('.')
        endswith = archive_name
    else:
        archive_name = None
        # XXX eventually should be refactored with data_path
        endswith = {
            'fieldtrip_cmc': 'MNE-fieldtrip_cmc-data',
            'fake': 'foo',
            'misc': 'MNE-misc-data',
            'sample': 'MNE-sample-data',
            'somato': 'MNE-somato-data',
            'spm': 'MNE-spm-face',
            'multimodal': 'MNE-multimodal-data',
            'fnirs_motor': 'MNE-fNIRS-motor-data',
            'opm': 'MNE-OPM-data',
            'testing': 'MNE-testing-data',
            'visual_92_categories': 'MNE-visual_92_categories-data',
            'kiloword': 'MNE-kiloword-data',
            'phantom_4dbti': 'MNE-phantom-4DBTi',
            'mtrf': 'mTRF_1.5',
            'refmeg_noise': 'MNE-refmeg-noise-data'
        }[name]
    dp = _data_path(download=False, name=name, check_version=False,
                    archive_name=archive_name)
    return dp.endswith(endswith)


@verbose
def _download_all_example_data(verbose=True):
    """Download all datasets used in examples and tutorials."""
    # This function is designed primarily to be used by CircleCI, to:
    #
    # 1. Streamline data downloading
    # 2. Make CircleCI fail early (rather than later) if some necessary data
    #    cannot be retrieved.
    # 3. Avoid download statuses and timing biases in rendered examples.
    #
    # verbose=True by default so we get nice status messages.
    # Consider adding datasets from here to CircleCI for PR-auto-build
    from . import (sample, testing, misc, spm_face, somato, brainstorm,
                   eegbci, multimodal, opm, hf_sef, mtrf, fieldtrip_cmc,
                   kiloword, phantom_4dbti, sleep_physionet, limo,
                   fnirs_motor, refmeg_noise)
    sample_path = sample.data_path()
    testing.data_path()
    misc.data_path()
    spm_face.data_path()
    somato.data_path()
    hf_sef.data_path()
    multimodal.data_path()
    fnirs_motor.data_path()
    opm.data_path()
    mtrf.data_path()
    fieldtrip_cmc.data_path()
    kiloword.data_path()
    phantom_4dbti.data_path()
    refmeg_noise.data_path()
    brainstorm.bst_raw.data_path(accept=True)
    brainstorm.bst_auditory.data_path(accept=True)
    brainstorm.bst_resting.data_path(accept=True)
    brainstorm.bst_phantom_elekta.data_path(accept=True)
    brainstorm.bst_phantom_ctf.data_path(accept=True)
    eegbci.load_data(1, [6, 10, 14], update_path=True)
    for subj in range(4):
        eegbci.load_data(subj + 1, runs=[3], update_path=True)
    sleep_physionet.age.fetch_data(subjects=[0, 1], recording=[1],
                                   update_path=True)
    # If the user has SUBJECTS_DIR, respect it, if not, set it to the EEG one
    # (probably on CircleCI, or otherwise advanced user)
    fetch_fsaverage(None)
    fetch_hcp_mmp_parcellation(
        subjects_dir=sample_path + '/subjects', accept=True)
    limo.load_data(subject=1, update_path=True)


@verbose
def fetch_aparc_sub_parcellation(subjects_dir=None, verbose=None):
    """Fetch the modified subdivided aparc parcellation.

    This will download and install the subdivided aparc parcellation
    :footcite:'KhanEtAl2018' files for
    FreeSurfer's fsaverage to the specified directory.

    Parameters
    ----------
    subjects_dir : str | None
        The subjects directory to use. The file will be placed in
        ``subjects_dir + '/fsaverage/label'``.
    %(verbose)s

    References
    ----------
    .. footbibliography::
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    destination = op.join(subjects_dir, 'fsaverage', 'label')
    urls = dict(lh='https://osf.io/p92yb/download',
                rh='https://osf.io/4kxny/download')
    hashes = dict(lh='9e4d8d6b90242b7e4b0145353436ef77',
                  rh='dd6464db8e7762d969fc1d8087cd211b')
    for hemi in ('lh', 'rh'):
        fname = op.join(destination, '%s.aparc_sub.annot' % hemi)
        if not op.isfile(fname):
            _fetch_file(urls[hemi], fname, hash_=hashes[hemi])


@verbose
def fetch_hcp_mmp_parcellation(subjects_dir=None, combine=True, *,
                               accept=False, verbose=None):
    """Fetch the HCP-MMP parcellation.

    This will download and install the HCP-MMP parcellation
    :footcite:`GlasserEtAl2016` files for FreeSurfer's fsaverage
    :footcite:`Mills2016` to the specified directory.

    Parameters
    ----------
    subjects_dir : str | None
        The subjects directory to use. The file will be placed in
        ``subjects_dir + '/fsaverage/label'``.
    combine : bool
        If True, also produce the combined/reduced set of 23 labels per
        hemisphere as ``HCPMMP1_combined.annot``
        :footcite:`GlasserEtAl2016supp`.
    %(accept)s
    %(verbose)s

    Notes
    -----
    Use of this parcellation is subject to terms of use on the
    `HCP-MMP webpage <https://balsa.wustl.edu/WN56>`_.

    References
    ----------
    .. footbibliography::
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    destination = op.join(subjects_dir, 'fsaverage', 'label')
    fnames = [op.join(destination, '%s.HCPMMP1.annot' % hemi)
              for hemi in ('lh', 'rh')]
    urls = dict(lh='https://ndownloader.figshare.com/files/5528816',
                rh='https://ndownloader.figshare.com/files/5528819')
    hashes = dict(lh='46a102b59b2fb1bb4bd62d51bf02e975',
                  rh='75e96b331940227bbcb07c1c791c2463')
    if not all(op.isfile(fname) for fname in fnames):
        if accept or '--accept-hcpmmp-license' in sys.argv:
            answer = 'y'
        else:
            answer = _safe_input('%s\nAgree (y/[n])? ' % _hcp_mmp_license_text)
        if answer.lower() != 'y':
            raise RuntimeError('You must agree to the license to use this '
                               'dataset')
    for hemi, fname in zip(('lh', 'rh'), fnames):
        if not op.isfile(fname):
            _fetch_file(urls[hemi], fname, hash_=hashes[hemi])
    if combine:
        fnames = [op.join(destination, '%s.HCPMMP1_combined.annot' % hemi)
                  for hemi in ('lh', 'rh')]
        if all(op.isfile(fname) for fname in fnames):
            return
        # otherwise, let's make them
        logger.info('Creating combined labels')
        groups = OrderedDict([
            ('Primary Visual Cortex (V1)',
             ('V1',)),
            ('Early Visual Cortex',
             ('V2', 'V3', 'V4')),
            ('Dorsal Stream Visual Cortex',
             ('V3A', 'V3B', 'V6', 'V6A', 'V7', 'IPS1')),
            ('Ventral Stream Visual Cortex',
             ('V8', 'VVC', 'PIT', 'FFC', 'VMV1', 'VMV2', 'VMV3')),
            ('MT+ Complex and Neighboring Visual Areas',
             ('V3CD', 'LO1', 'LO2', 'LO3', 'V4t', 'FST', 'MT', 'MST', 'PH')),
            ('Somatosensory and Motor Cortex',
             ('4', '3a', '3b', '1', '2')),
            ('Paracentral Lobular and Mid Cingulate Cortex',
             ('24dd', '24dv', '6mp', '6ma', 'SCEF', '5m', '5L', '5mv',)),
            ('Premotor Cortex',
             ('55b', '6d', '6a', 'FEF', '6v', '6r', 'PEF')),
            ('Posterior Opercular Cortex',
             ('43', 'FOP1', 'OP4', 'OP1', 'OP2-3', 'PFcm')),
            ('Early Auditory Cortex',
             ('A1', 'LBelt', 'MBelt', 'PBelt', 'RI')),
            ('Auditory Association Cortex',
             ('A4', 'A5', 'STSdp', 'STSda', 'STSvp', 'STSva', 'STGa', 'TA2',)),
            ('Insular and Frontal Opercular Cortex',
             ('52', 'PI', 'Ig', 'PoI1', 'PoI2', 'FOP2', 'FOP3',
              'MI', 'AVI', 'AAIC', 'Pir', 'FOP4', 'FOP5')),
            ('Medial Temporal Cortex',
             ('H', 'PreS', 'EC', 'PeEc', 'PHA1', 'PHA2', 'PHA3',)),
            ('Lateral Temporal Cortex',
             ('PHT', 'TE1p', 'TE1m', 'TE1a', 'TE2p', 'TE2a',
              'TGv', 'TGd', 'TF',)),
            ('Temporo-Parieto-Occipital Junction',
             ('TPOJ1', 'TPOJ2', 'TPOJ3', 'STV', 'PSL',)),
            ('Superior Parietal Cortex',
             ('LIPv', 'LIPd', 'VIP', 'AIP', 'MIP',
              '7PC', '7AL', '7Am', '7PL', '7Pm',)),
            ('Inferior Parietal Cortex',
             ('PGp', 'PGs', 'PGi', 'PFm', 'PF', 'PFt', 'PFop',
              'IP0', 'IP1', 'IP2',)),
            ('Posterior Cingulate Cortex',
             ('DVT', 'ProS', 'POS1', 'POS2', 'RSC', 'v23ab', 'd23ab',
              '31pv', '31pd', '31a', '23d', '23c', 'PCV', '7m',)),
            ('Anterior Cingulate and Medial Prefrontal Cortex',
             ('33pr', 'p24pr', 'a24pr', 'p24', 'a24', 'p32pr', 'a32pr', 'd32',
              'p32', 's32', '8BM', '9m', '10v', '10r', '25',)),
            ('Orbital and Polar Frontal Cortex',
             ('47s', '47m', 'a47r', '11l', '13l',
              'a10p', 'p10p', '10pp', '10d', 'OFC', 'pOFC',)),
            ('Inferior Frontal Cortex',
             ('44', '45', 'IFJp', 'IFJa', 'IFSp', 'IFSa', '47l', 'p47r',)),
            ('DorsoLateral Prefrontal Cortex',
             ('8C', '8Av', 'i6-8', 's6-8', 'SFL', '8BL', '9p', '9a', '8Ad',
              'p9-46v', 'a9-46v', '46', '9-46d',)),
            ('???',
             ('???',))])
        assert len(groups) == 23
        labels_out = list()

        for hemi in ('lh', 'rh'):
            labels = read_labels_from_annot('fsaverage', 'HCPMMP1', hemi=hemi,
                                            subjects_dir=subjects_dir,
                                            sort=False)
            label_names = [
                '???' if label.name.startswith('???') else
                label.name.split('_')[1] for label in labels]
            used = np.zeros(len(labels), bool)
            for key, want in groups.items():
                assert '\t' not in key
                these_labels = [li for li, label_name in enumerate(label_names)
                                if label_name in want]
                assert not used[these_labels].any()
                assert len(these_labels) == len(want)
                used[these_labels] = True
                these_labels = [labels[li] for li in these_labels]
                # take a weighted average to get the color
                # (here color == task activation)
                w = np.array([len(label.vertices) for label in these_labels])
                w = w / float(w.sum())
                color = np.dot(w, [label.color for label in these_labels])
                these_labels = sum(these_labels,
                                   Label([], subject='fsaverage', hemi=hemi))
                these_labels.name = key
                these_labels.color = color
                labels_out.append(these_labels)
            assert used.all()
        assert len(labels_out) == 46
        for hemi, side in (('lh', 'left'), ('rh', 'right')):
            table_name = './%s.fsaverage164.label.gii' % (side,)
            write_labels_to_annot(labels_out, 'fsaverage', 'HCPMMP1_combined',
                                  hemi=hemi, subjects_dir=subjects_dir,
                                  sort=False, table_name=table_name)


def _manifest_check_download(manifest_path, destination, url, hash_):
    with open(manifest_path, 'r') as fid:
        names = [name.strip() for name in fid.readlines()]
    need = list()
    for name in names:
        if not op.isfile(op.join(destination, name)):
            need.append(name)
    logger.info('%d file%s missing from %s in %s'
                % (len(need), _pl(need), manifest_path, destination))
    if len(need) > 0:
        with tempfile.TemporaryDirectory() as path:
            logger.info('Downloading missing files remotely')

            fname_path = op.join(path, 'temp.zip')
            _fetch_file(url, fname_path, hash_=hash_)
            logger.info('Extracting missing file%s' % (_pl(need),))
            with zipfile.ZipFile(fname_path, 'r') as ff:
                members = set(f for f in ff.namelist() if not f.endswith('/'))
                missing = sorted(members.symmetric_difference(set(names)))
                if len(missing):
                    raise RuntimeError('Zip file did not have correct names:'
                                       '\n%s' % ('\n'.join(missing)))
                for name in need:
                    ff.extract(name, path=destination)
        logger.info('Successfully extracted %d file%s'
                    % (len(need), _pl(need)))
