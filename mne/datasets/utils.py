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
import zipfile
from distutils.version import LooseVersion

from .. import __version__ as mne_version
from ..utils import (get_config, set_config, _fetch_file, logger, warn,
                     verbose, get_subjects_dir)
from ..externals.six import string_types
from ..externals.six.moves import input


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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`).

    Returns
    -------
    path : str
        Path to {name} dataset directory.
"""


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
        if not isinstance(path, string_types):
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
        set_config(key, path, set_env=False)
    return path


def _data_path(path=None, force_update=False, update_path=True, download=True,
               name=None, check_version=False, return_version=False,
               archive_name=None):
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
        'visual_92_categories': 'MNE_DATASETS_VISUAL_92_CATEGORIES_PATH',
        'mtrf': 'MNE_DATASETS_MTRF_PATH',
        'fieldtrip_cmc': 'MNE_DATASETS_FIELDTRIP_CMC_PATH'
    }[name]

    path = _get_path(path, key, name)
    # To update the testing or misc dataset, push commits, then make a new
    # release on GitHub. Then update the "releases" variable:
    releases = dict(testing='0.40', misc='0.3')
    # And also update the "hashes['testing']" variable below.

    # To update any other dataset, update the data archive itself (upload
    # an updated version) and update the hash.

    # try to match url->archive_name->folder_name
    urls = dict(  # the URLs to use
        brainstorm=dict(
            bst_auditory='https://osf.io/5t9n8/download',
            bst_phantom_ctf='https://osf.io/sxr8y/download',
            bst_phantom_elekta='https://osf.io/dpcku/download',
            bst_raw='https://osf.io/9675n/download',
            bst_resting='https://osf.io/m7bd3/download'),
        fake='https://github.com/mne-tools/mne-testing-data/raw/master/'
             'datasets/foo.tgz',
        misc='https://codeload.github.com/mne-tools/mne-misc-data/'
             'tar.gz/%s' % releases['misc'],
        sample="https://osf.io/86qa2/download",
        somato='https://osf.io/tp4sg/download',
        spm='https://osf.io/je4s8/download',
        testing='https://codeload.github.com/mne-tools/mne-testing-data/'
                'tar.gz/%s' % releases['testing'],
        multimodal='https://ndownloader.figshare.com/files/5999598',
        visual_92_categories=[
            'https://osf.io/8ejrs/download',
            'https://osf.io/t4yjp/download'],
        mtrf="https://superb-dca2.dl.sourceforge.net/project/aespa/"
             "mTRF_1.5.zip",
        fieldtrip_cmc='ftp://ftp.fieldtriptoolbox.org/pub/fieldtrip/'
                      'tutorial/SubjectCMC.zip',
    )
    # filename of the resulting downloaded archive (only needed if the URL
    # name does not match resulting filename)
    archive_names = dict(
        misc='mne-misc-data-%s.tar.gz' % releases['misc'],
        multimodal='MNE-multimodal-data.tar.gz',
        sample='MNE-sample-data-processed.tar.gz',
        somato='MNE-somato-data.tar.gz',
        spm='MNE-spm-face.tar.gz',
        testing='mne-testing-data-%s.tar.gz' % releases['testing'],
        visual_92_categories=['MNE-visual_92_categories-data-part1.tar.gz',
                              'MNE-visual_92_categories-data-part2.tar.gz'],
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
        fieldtrip_cmc='MNE-fieldtrip_cmc-data'
    )
    hashes = dict(
        brainstorm=dict(
            bst_auditory='fa371a889a5688258896bfa29dd1700b',
            bst_phantom_ctf='80819cb7f5b92d1a5289db3fb6acb33c',
            bst_phantom_elekta='1badccbe17998d18cc373526e86a7aaf',
            bst_raw='f82ba1f17b2e7a2d96995c1c08e1cc8d',
            bst_resting='a14186aebe7bd2aaa2d28db43aa6587e'),
        fake='3194e9f7b46039bb050a74f3e1ae9908',
        misc='d822a720ef94302467cb6ad1d320b669',
        sample='fc2d5b9eb0a144b1d6ba84dc3b983602',
        somato='f3e3a8441477bb5bacae1d0c6e0964fb',
        spm='9f43f67150e3b694b523a21eb929ea75',
        testing='02796b3ab145ee9cad680a545563beb5',
        multimodal='26ec847ae9ab80f58f204d09e2c08367',
        visual_92_categories=['74f50bbeb65740903eadc229c9fa759f',
                              '203410a98afc9df9ae8ba9f933370e20'],
        mtrf='273a390ebbc48da2c3184b01a82e4636',
        fieldtrip_cmc='6f9fd6520f9a66e20994423808d2528c'
    )
    assert set(hashes.keys()) == set(urls.keys())
    url = urls[name]
    hash_ = hashes[name]
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
            if '--accept-brainstorm-license' in sys.argv:
                answer = 'y'
            else:
                answer = input('%sAgree (y/[n])? ' % _bst_license_text)
            if answer.lower() != 'y':
                raise RuntimeError('You must agree to the license to use this '
                                   'dataset')
        assert len(url) == len(hash_)
        assert len(url) == len(archive_name)
        assert len(url) == len(folder_orig)
        assert len(url) == len(folder_path)
        for u, fp, an, h, fo in zip(url, folder_path, archive_name, hash_,
                                    folder_orig):
            _download_and_extract(path, name, u, fp, an, h, fo)

    _do_path_update(path, update_path, key, name)
    path = folder_path[0]

    # compare the version of the dataset and mne
    data_version = _dataset_version(path, name)
    # 0.7 < 0.7.git shoud be False, therefore strip
    if check_version and (LooseVersion(data_version) <
                          LooseVersion(mne_version.strip('.git'))):
        warn('The {name} dataset (version {current}) is older than '
             'mne-python (version {newest}). If the examples fail, '
             'you may need to update the {name} dataset by using '
             'mne.datasets.{name}.data_path(force_update=True)'.format(
                 name=name, current=data_version, newest=mne_version))
    return (path, data_version) if return_version else path


def _download_and_extract(path, name, url, folder_path, archive_name, hash_,
                          folder_orig):
    """Download and extract an archive."""
    logger.info('Downloading or reinstalling '
                'data archive %s at location %s' % (archive_name, path))
    rm_archive = False
    martinos_path = '/cluster/fusion/sample_data/' + archive_name
    neurospin_path = '/neurospin/tmp/gramfort/' + archive_name
    if op.exists(martinos_path):
        archive_name = martinos_path
    elif op.exists(neurospin_path):
        archive_name = neurospin_path
    else:
        archive_name = op.join(path, archive_name)
        rm_archive = False  # XXX put back
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
                raise
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

    if rm_archive:
        os.remove(archive_name)
    logger.info('Successfully extracted to: %s' % folder_path)


def _get_version(name):
    """Get a dataset version."""
    if not has_dataset(name):
        return None
    return _data_path(name=name, return_version=True)[1]


def has_dataset(name):
    """Check for dataset presence."""
    endswith = {
        'brainstorm': 'MNE_brainstorm-data',
        'fieldtrip_cmc': 'MNE-fieldtrip_cmc-data',
        'fake': 'foo',
        'misc': 'MNE-misc-data',
        'sample': 'MNE-sample-data',
        'somato': 'MNE-somato-data',
        'spm': 'MNE-spm-face',
        'testing': 'MNE-testing-data',
        'visual_92_categories': 'visual_92_categories-data',
    }[name]
    archive_name = None
    if name == 'brainstorm':
        archive_name = dict(brainstorm='bst_raw')
    dp = _data_path(download=False, name=name, check_version=False,
                    archive_name=archive_name)
    return dp.endswith(endswith)


@verbose
def _download_all_example_data(verbose=True):
    """Download all datasets used in examples and tutorials."""
    # This function is designed primarily to be used by CircleCI. It has
    # verbose=True by default so we get nice status messages
    # Consider adding datasets from here to CircleCI for PR-auto-build
    from . import (sample, testing, misc, spm_face, somato, brainstorm, megsim,
                   eegbci, multimodal, mtrf, fieldtrip_cmc)
    sample.data_path()
    testing.data_path()
    misc.data_path()
    spm_face.data_path()
    somato.data_path()
    multimodal.data_path()
    mtrf.data_path()
    fieldtrip_cmc.data_path()
    sys.argv += ['--accept-brainstorm-license']
    try:
        brainstorm.bst_raw.data_path()
        brainstorm.bst_auditory.data_path()
        brainstorm.bst_phantom_elekta.data_path()
        brainstorm.bst_phantom_ctf.data_path()
    finally:
        sys.argv.pop(-1)
    megsim.load_data(condition='visual', data_format='single-trial',
                     data_type='simulation', update_path=True)
    megsim.load_data(condition='visual', data_format='raw',
                     data_type='experimental', update_path=True)
    megsim.load_data(condition='visual', data_format='evoked',
                     data_type='simulation', update_path=True)
    eegbci.load_data(1, [6, 10, 14], update_path=True)
    sys.argv += ['--accept-hcpmmp-license']
    try:
        fetch_hcp_mmp_parcellation()
    finally:
        sys.argv.pop(-1)


@verbose
def fetch_hcp_mmp_parcellation(subjects_dir=None, verbose=None):
    """Fetch the HCP-MMP parcellation.

    This will download and install the HCP-MMP parcellation [1]_ files for
    FreeSurfer's fsaverage [2]_ to the specified directory.

    Parameters
    ----------
    subjects_dir : str | None
        The subjects directory to use. The file will be placed in
        ``subjects_dir + '/fsaverage/label'``.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Notes
    -----
    Use of this parcellation is subject to terms of use on the
    `HCP-MMP webpage <https://balsa.wustl.edu/WN56>`_.

    References
    ----------
    .. [1] Glasser MF et al. (2016) A multi-modal parcellation of human
           cerebral cortex. Nature 536:171-178.
    .. [2] Mills K (2016) HCP-MMP1.0 projected on fsaverage.
           https://figshare.com/articles/HCP-MMP1_0_projected_on_fsaverage/3498446/2
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    destination = op.join(subjects_dir, 'fsaverage', 'label')
    fnames = [op.join(destination, 'lh.HCPMMP1.annot'),
              op.join(destination, 'rh.HCPMMP1.annot')]
    if all(op.isfile(fname) for fname in fnames):
        return
    if '--accept-hcpmmp-license' in sys.argv:
        answer = 'y'
    else:
        answer = input('%s\nAgree (y/[n])? ' % _hcp_mmp_license_text)
    if answer.lower() != 'y':
        raise RuntimeError('You must agree to the license to use this '
                           'dataset')
    _fetch_file('https://ndownloader.figshare.com/files/5528816',
                fnames[0], hash_='46a102b59b2fb1bb4bd62d51bf02e975')
    _fetch_file('https://ndownloader.figshare.com/files/5528819',
                fnames[1], hash_='75e96b331940227bbcb07c1c791c2463')
