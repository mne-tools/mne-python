# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Denis Egnemann <denis.engemann@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Daniel McCloy <dan@mccloy.info>
# License: BSD Style.

from collections import OrderedDict
import os
import os.path as op
import sys
import zipfile
import tempfile
import pkg_resources
from distutils.version import LooseVersion

import numpy as np

from .. import __version__ as mne_version
from ..label import read_labels_from_annot, Label, write_labels_to_annot
from ..utils import (get_config, set_config, _fetch_file, logger, warn,
                     verbose, get_subjects_dir, _pl, _safe_input)
from ..utils.docs import docdict
from ..utils.check import _soft_import
from ..externals.doccer import docformat

pooch = _soft_import('pooch', 'dataset downloading', True)

_FAKE_VERSION = None  # used for monkeypatching while testing versioning

# To update the `testing` or `misc` datasets, push or merge commits to their
# respective repos, and make a new release of the dataset on GitHub. Then
# update the checksum in `mne/data/dataset_checksums.txt` and change version
# here:                  ↓↓↓↓↓         ↓↓↓
RELEASES = dict(testing='0.112', misc='0.8')
# To update any other dataset besides `testing` or `misc`, upload the new
# version of the data archive itself (e.g., to https://osf.io or wherever) and
# then update the corresponding checksum in `mne/data/dataset_checksums.txt`.
TESTING_VERSIONED = f'mne-testing-data-{RELEASES["testing"]}'
MISC_VERSIONED = f'mne-misc-data-{RELEASES["misc"]}'
# remote locations of the various datasets
URLS = dict(
    bst_auditory='https://osf.io/5t9n8/download?version=1',
    bst_phantom_ctf='https://osf.io/sxr8y/download?version=1',
    bst_phantom_elekta='https://osf.io/dpcku/download?version=1',
    bst_raw='https://osf.io/9675n/download?version=2',
    bst_resting='https://osf.io/m7bd3/download?version=3',
    fake=('https://github.com/mne-tools/mne-testing-data/raw/master/'
          'datasets/foo.tgz'),
    misc=('https://codeload.github.com/mne-tools/mne-misc-data/tar.gz/'
          f'{RELEASES["misc"]}'),
    sample='https://osf.io/86qa2/download?version=5',
    somato='https://osf.io/tp4sg/download?version=7',
    spm='https://osf.io/je4s8/download?version=2',
    testing=('https://codeload.github.com/mne-tools/mne-testing-data/'
             f'tar.gz/{RELEASES["testing"]}'),
    multimodal='https://ndownloader.figshare.com/files/5999598',
    fnirs_motor='https://osf.io/dj3eh/download?version=1',
    opm='https://osf.io/p6ae7/download?version=2',
    visual_92_categories_1='https://osf.io/8ejrs/download?version=1',
    visual_92_categories_2='https://osf.io/t4yjp/download?version=1',
    mtrf='https://osf.io/h85s2/download?version=1',
    kiloword='https://osf.io/qkvf9/download?version=1',
    fieldtrip_cmc='https://osf.io/j9b6s/download?version=1',
    phantom_4dbti='https://osf.io/v2brw/download?version=2',
    refmeg_noise='https://osf.io/drt6v/download?version=1',
    hf_sef_raw='https://zenodo.org/record/889296/files/hf_sef_raw.tar.gz',
    hf_sef_evoked=('https://zenodo.org/record/3523071/files/'
                   'hf_sef_evoked.tar.gz'),
)
ARCHIVE_NAMES = dict(
    bst_auditory='bst_auditory.tar.gz',
    bst_phantom_ctf='bst_phantom_ctf.tar.gz',
    bst_phantom_elekta='bst_phantom_elekta.tar.gz',
    bst_raw='bst_raw.tar.gz',
    bst_resting='bst_resting.tar.gz',
    fake='foo.tgz',
    fieldtrip_cmc='SubjectCMC.zip',
    kiloword='MNE-kiloword-data.tar.gz',
    misc=f'{MISC_VERSIONED}.tar.gz',
    mtrf='mTRF_1.5.zip',
    multimodal='MNE-multimodal-data.tar.gz',
    fnirs_motor='MNE-fNIRS-motor-data.tgz',
    opm='MNE-OPM-data.tar.gz',
    sample='MNE-sample-data-processed.tar.gz',
    somato='MNE-somato-data.tar.gz',
    spm='MNE-spm-face.tar.gz',
    testing=f'{TESTING_VERSIONED}.tar.gz',
    visual_92_categories_1='MNE-visual_92_categories-data-part1.tar.gz',
    visual_92_categories_2='MNE-visual_92_categories-data-part2.tar.gz',
    phantom_4dbti='MNE-phantom-4DBTi.zip',
    refmeg_noise='sample_reference_MEG_noise-raw.zip',
    hf_sef_raw='hf_sef_raw.tar.gz',
    hf_sef_evoked='hf_sef_evoked.tar.gz',
)
FOLDER_NAMES = dict(
    bst_auditory='MNE-brainstorm-data',
    bst_phantom_ctf='MNE-brainstorm-data',
    bst_phantom_elekta='MNE-brainstorm-data',
    bst_raw='MNE-brainstorm-data',
    bst_resting='MNE-brainstorm-data',
    fake='foo',
    fieldtrip_cmc='MNE-fieldtrip_cmc-data',
    kiloword='MNE-kiloword-data',
    misc='MNE-misc-data',
    mtrf='mTRF_1.5',
    multimodal='MNE-multimodal-data',
    fnirs_motor='MNE-fNIRS-motor-data',
    opm='MNE-OPM-data',
    sample='MNE-sample-data',
    somato='MNE-somato-data',
    spm='MNE-spm-face',
    testing='MNE-testing-data',
    visual_92_categories='MNE-visual_92_categories-data',
    phantom_4dbti='MNE-phantom-4DBTi',
    refmeg_noise='MNE-refmeg-noise-data',
    hf_sef_raw='HF_SEF',
    hf_sef_evoked='HF_SEF',
)
CONFIG_KEYS = dict(
    fake='MNE_DATASETS_FAKE_PATH',
    misc='MNE_DATASETS_MISC_PATH',
    sample='MNE_DATASETS_SAMPLE_PATH',
    spm='MNE_DATASETS_SPM_FACE_PATH',
    somato='MNE_DATASETS_SOMATO_PATH',
    bst_auditory='MNE_DATASETS_BRAINSTORM_PATH',
    bst_phantom_ctf='MNE_DATASETS_BRAINSTORM_PATH',
    bst_phantom_elekta='MNE_DATASETS_BRAINSTORM_PATH',
    bst_raw='MNE_DATASETS_BRAINSTORM_PATH',
    bst_resting='MNE_DATASETS_BRAINSTORM_PATH',
    testing='MNE_DATASETS_TESTING_PATH',
    multimodal='MNE_DATASETS_MULTIMODAL_PATH',
    fnirs_motor='MNE_DATASETS_FNIRS_MOTOR_PATH',
    opm='MNE_DATASETS_OPM_PATH',
    visual_92_categories='MNE_DATASETS_VISUAL_92_CATEGORIES_PATH',
    kiloword='MNE_DATASETS_KILOWORD_PATH',
    mtrf='MNE_DATASETS_MTRF_PATH',
    fieldtrip_cmc='MNE_DATASETS_FIELDTRIP_CMC_PATH',
    phantom_4dbti='MNE_DATASETS_PHANTOM_4DBTI_PATH',
    refmeg_noise='MNE_DATASETS_REFMEG_NOISE_PATH',
    hf_sef_raw='MNE_DATASETS_HF_SEF_PATH',
    hf_sef_evoked='MNE_DATASETS_HF_SEF_PATH',
)
assert set(ARCHIVE_NAMES) == set(URLS)

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
        if not op.exists(path):
            msg = (f"Download location {path} as specified by MNE_DATA does "
                   f"not exist. Either create this directory manually and try "
                   f"again, or set MNE_DATA to an existing directory.")
            raise FileNotFoundError(msg)
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
               accept=False):
    """Aux function."""
    # update the path
    path = _get_path(path, CONFIG_KEYS[name], name)
    final_path = op.join(path, FOLDER_NAMES[name])
    if name.startswith('bst_'):
        final_path = op.join(final_path, name)
    # check if testing or misc data is outdated; if so, redownload it
    want_version = RELEASES.get(name, None)
    want_version = _FAKE_VERSION if name == 'fake' else want_version
    data_version = _dataset_version(final_path, name)
    outdated = want_version is not None and want_version != data_version
    if outdated:
        logger.info(f'Dataset {name} version {data_version} out of date, '
                    f'latest version is {want_version}')
    # reasons to bail early (hf_sef has separate code for this):
    if not force_update and not outdated and not name.startswith('hf_sef_'):
        # if target folder exists (otherwise pooch downloads every time,
        # because we don't save the archive files after unpacking)
        if op.isdir(final_path):
            return final_path
        # if download=False (useful for debugging)
        elif not download:
            return ''
        # if user didn't accept the license
        elif name.startswith('bst_'):
            if accept or '--accept-brainstorm-license' in sys.argv:
                answer = 'y'
            else:
                # If they don't have stdin, just accept the license
                # https://github.com/mne-tools/mne-python/issues/8513#issuecomment-726823724  # noqa: E501
                answer = _safe_input(
                    '%sAgree (y/[n])? ' % _bst_license_text, use='y')
            if answer.lower() != 'y':
                raise RuntimeError(
                    'You must agree to the license to use this dataset')
    # downloader & processors
    downloader = pooch.HTTPDownloader(progressbar=True)  # use tqdm
    unzip = pooch.Unzip(extract_dir=path)
    untar = pooch.Untar(extract_dir=path)
    nested_untar = pooch.Untar(extract_dir=op.join(path, FOLDER_NAMES[name]))
    processors = dict(
        bst_auditory=nested_untar,
        bst_phantom_ctf=nested_untar,
        bst_phantom_elekta=nested_untar,
        bst_raw=nested_untar,
        bst_resting=nested_untar,
        fake=untar,
        fieldtrip_cmc=nested_untar,
        kiloword=untar,
        misc=untar,
        mtrf=unzip,
        multimodal=untar,
        fnirs_motor=untar,
        opm=untar,
        sample=untar,
        somato=untar,
        spm=untar,
        testing=untar,
        visual_92_categories=untar,
        phantom_4dbti=unzip,
        refmeg_noise=unzip,
        hf_sef_raw=pooch.Untar(
            extract_dir=path, members=[f'hf_sef/{subdir}' for subdir in
                                       ('MEG', 'SSS', 'subjects')]),
        hf_sef_evoked=pooch.Untar(
            extract_dir=path, members=[f'hf_sef/{subdir}' for subdir in
                                       ('MEG', 'SSS', 'subjects')]),
    )
    # construct the mapping needed by pooch
    pooch_urls = {ARCHIVE_NAMES[key]: URLS[key] for key in URLS}
    # create the download manager
    fetcher = pooch.create(
        path=path,
        base_url='',    # Full URLs are given in the `urls` dict.
        version=None,   # Data versioning is decoupled from MNE-Python version.
        registry=None,  # Registry is loaded from file, below.
        urls=pooch_urls,
        retry_if_failed=2  # 2 retries = 3 total attempts
    )
    # load the checksum registry
    registry = pkg_resources.resource_stream(
        'mne', op.join('data', 'dataset_checksums.txt'))
    fetcher.load_registry(registry)
    # update the keys that are versioned
    versioned_keys = {
        f'{TESTING_VERSIONED}.tar.gz': fetcher.registry['mne-testing-data'],
        f'{MISC_VERSIONED}.tar.gz': fetcher.registry['mne-misc-data']}
    fetcher.registry.update(versioned_keys)
    for key in ('testing', 'misc'):
        del fetcher.registry[f'mne-{key}-data']
    # use our logger level for pooch's logger too
    pooch.get_logger().setLevel(logger.getEffectiveLevel())
    # fetch and unpack the data
    if name == 'visual_92_categories':
        names = [f'visual_92_categories_{n}' for n in (1, 2)]
    else:
        names = [name]
    for this_name in names:
        archive_name = ARCHIVE_NAMES[this_name]
        fetcher.fetch(fname=archive_name, downloader=downloader,
                      processor=processors[name])
        # after unpacking, remove the archive file
        os.remove(op.join(path, archive_name))
    # remove version number from "misc" and "testing" datasets folder names
    if name == 'misc':
        os.replace(op.join(path, MISC_VERSIONED), final_path)
    elif name == 'testing':
        os.replace(op.join(path, TESTING_VERSIONED), final_path)
    # maybe update the config
    old_name = 'brainstorm' if name.startswith('bst_') else name
    _do_path_update(path, update_path, CONFIG_KEYS[name], old_name)
    # compare the version of the dataset and mne
    data_version = _dataset_version(final_path, old_name)
    # 0.7 < 0.7.git should be False, therefore strip
    if check_version and (LooseVersion(data_version) <
                          LooseVersion(mne_version.strip('.git'))):
        warn('The {name} dataset (version {current}) is older than '
             'mne-python (version {newest}). If the examples fail, '
             'you may need to update the {name} dataset by using '
             'mne.datasets.{name}.data_path(force_update=True)'.format(
                 name=name, current=data_version, newest=mne_version))
    return (final_path, data_version) if return_version else final_path


def _get_version(name):
    """Get a dataset version."""
    if not has_dataset(name):
        return None
    return _data_path(name=name, return_version=True)[1]


def has_dataset(name):
    """Check for dataset presence.

    Parameters
    ----------
    name : str
        The dataset name.

    Returns
    -------
    has : bool
        True if the dataset is present.
    """
    name = 'spm' if name == 'spm_face' else name
    dp = _data_path(download=False, name=name, check_version=False)
    return dp.endswith(FOLDER_NAMES[name])


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
                   fnirs_motor, refmeg_noise, fetch_infant_template,
                   fetch_fsaverage)
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
    fetch_infant_template('6mo')
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
    manifest_path = op.basename(manifest_path)
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
