# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Denis Egnemann <denis.engemann@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Adam Li <adam2392@gmail.com>
#          Daniel McCloy <dan@mccloy.info>
#
# License: BSD Style.

from collections import OrderedDict
from importlib_resources import files
import os
import os.path as op
import sys
import zipfile
import tempfile
from distutils.version import LooseVersion
from shutil import rmtree

import numpy as np

from .config import (_bst_license_text, _hcp_mmp_license_text, URLS,
                     CONFIG_KEYS, ARCHIVE_NAMES, FOLDER_NAMES, RELEASES,
                     TESTING_VERSIONED, MISC_VERSIONED)
from .. import __version__ as mne_version
from ..label import read_labels_from_annot, Label, write_labels_to_annot
from ..utils import (get_config, set_config, logger, warn,
                     verbose, get_subjects_dir, _pl, _safe_input)
from ..utils.docs import docdict
from ..utils.check import _soft_import
from ..externals.doccer import docformat


# import pooch library for handling the dataset downloading
pooch = _soft_import('pooch', 'dataset downloading', strict=True)

_FAKE_VERSION = None  # used for monkeypatching while testing versioning


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
        Default is False.
    update_path : bool | None
        If True (default), set the ``{conf}`` in mne-python
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
    """Aux function.

    This is a general function for fetching MNE datasets. In order
    to define an MNE dataset, one needs to define a URL, archive name,
    folder name, and environment configuration key. They also need
    to define a ``pooch`` registry txt file mapping the dataset
    archive name to a hash (i.e. md5, or sha).

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
        Default is False.
    update_path : bool | None
        If True (default), set the ``{conf}`` in mne-python
        config to the given path. If None, the user is prompted.
    download : bool
        If False and the {name} dataset has not been downloaded yet,
        it will not be downloaded and the path will be returned as
        '' (empty string). This is mostly used for debugging purposes
        and can be safely ignored by most users.
    name : str | None
        The name of the dataset, which should correspond to the
        URL, archive name, folder names and configuration key mappings
        for pooch.
    check_version : bool
        Whether to check the version of the dataset or not. Each version
        of the dataset is stored in the root with a ``version.txt`` file.
    return_version : bool
        Whether or not to return the version of the dataset or not.
        Defaults to False.
    accept : bool
        Some datasets require an acceptance of an additional license.
        Default to False. If this is True, then license text should
        be passed into key word argument ``license_text``.
    license_text : str | None
        The text of a license agreement. Only used if ``accept`` is True.

    Returns
    -------
    path : str
        Path to {name} dataset directory.
    """
    # get download path for specific dataset
    path = _get_path(path=path, key=CONFIG_KEYS[name], name=name)

    # get the actual path to each dataset folder name
    final_path = op.join(path, FOLDER_NAMES[name])

    # handle BrainStorm datasets with nested folders for datasets
    if name.startswith('bst_'):
        final_path = op.join(final_path, name)

    # additional condition: check for version.txt and parse it
    # check if testing or misc data is outdated; if so, redownload it
    want_version = RELEASES.get(name, None)
    want_version = _FAKE_VERSION if name == 'fake' else want_version

    # get the version of the dataset and then check if the version is outdated
    data_version = _dataset_version(final_path, name)
    outdated_dataset = want_version is not None and \
        LooseVersion(want_version) > LooseVersion(data_version)

    if outdated_dataset:
        logger.info(f'Dataset {name} version {data_version} out of date, '
                    f'latest version is {want_version}')

    # return empty string if outdated dataset and we don't want
    # to download
    if (not force_update) and outdated_dataset and not download:
        return ('', data_version) if return_version else ''

    # reasons to bail early (hf_sef has separate code for this):
    if (not force_update) and (not outdated_dataset) and \
            (not name.startswith('hf_sef_')):
        # if target folder exists (otherwise pooch downloads every time,
        # because we don't save the archive files after unpacking)
        if op.isdir(final_path):
            _do_path_update(path, update_path, CONFIG_KEYS[name], name)
            return (final_path, data_version) if return_version else final_path
        # if download=False (useful for debugging)
        elif not download:
            return ('', data_version) if return_version else ''
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
                raise RuntimeError('You must agree to the license to use this '
                                   'dataset')

    # downloader & processors TODO: may want to skip using tqdm during tests?
    progressbar = True
    if name == 'fake':
        progressbar = False
    downloader = pooch.HTTPDownloader(progressbar=progressbar)  # use tqdm
    unzip = pooch.Unzip(extract_dir=path)  # to unzip downloaded file
    untar = pooch.Untar(extract_dir=path)  # to untar downloaded file

    # this processor handles nested tar files
    nested_untar = pooch.Untar(extract_dir=op.join(path, FOLDER_NAMES[name]))

    # create a map for each dataset name to its corresponding processor
    # Note: when adding a new dataset, a new line must be added here.
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
        ssvep=unzip,
        erp_core=untar,
        epilepsy_ecog=untar,
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
    registry = files('mne.data').joinpath('dataset_checksums.txt')
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
        rmtree(final_path, ignore_errors=True)
        os.replace(op.join(path, MISC_VERSIONED), final_path)
    elif name == 'testing':
        rmtree(final_path, ignore_errors=True)
        os.replace(op.join(path, TESTING_VERSIONED), final_path)
    # maybe update the config
    old_name = 'brainstorm' if name.startswith('bst_') else name
    _do_path_update(path, update_path, CONFIG_KEYS[name], old_name)

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
        For brainstorm datasets, should be formatted like
        "brainstorm.bst_raw".

    Returns
    -------
    has : bool
        True if the dataset is present.
    """
    name = 'spm' if name == 'spm_face' else name
    dp = _data_path(download=False, name=name, check_version=False)
    check = name if name.startswith('bst_') else FOLDER_NAMES[name]
    return dp.endswith(check)


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
                   fetch_fsaverage, ssvep, erp_core, epilepsy_ecog)
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
    ssvep.data_path()
    epilepsy_ecog.data_path()
    brainstorm.bst_raw.data_path(accept=True)
    brainstorm.bst_auditory.data_path(accept=True)
    brainstorm.bst_resting.data_path(accept=True)
    brainstorm.bst_phantom_elekta.data_path(accept=True)
    brainstorm.bst_phantom_ctf.data_path(accept=True)
    eegbci.load_data(1, [6, 10, 14], update_path=True)
    for subj in range(4):
        eegbci.load_data(subj + 1, runs=[3], update_path=True)
    sleep_physionet.age.fetch_data(subjects=[0, 1], recording=[1])
    # If the user has SUBJECTS_DIR, respect it, if not, set it to the EEG one
    # (probably on CircleCI, or otherwise advanced user)
    fetch_fsaverage(None)
    fetch_infant_template('6mo')
    fetch_hcp_mmp_parcellation(
        subjects_dir=sample_path + '/subjects', accept=True)
    limo.load_data(subject=1, update_path=True)

    erp_core.data_path()


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
        fname = f'{hemi}.aparc_sub.annot'
        fpath = op.join(destination, fname)
        if not op.isfile(fpath):
            pooch.retrieve(
                url=urls[hemi],
                known_hash=f"md5:{hashes[hemi]}",
                path=destination,
                fname=fname
            )


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
    for hemi, fpath in zip(('lh', 'rh'), fnames):
        if not op.isfile(fpath):
            fname = op.basename(fpath)
            pooch.retrieve(
                url=urls[hemi],
                known_hash=f"md5:{hashes[hemi]}",
                path=destination,
                fname=fname
            )

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
            pooch.retrieve(
                url=url,
                known_hash=f"md5:{hash_}",
                path=path,
                fname=op.basename(fname_path)
            )

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
