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
import importlib
import inspect
import os
import os.path as op
from pathlib import Path
import sys
import zipfile
import tempfile

import numpy as np

from .config import _hcp_mmp_license_text, MNE_DATASETS
from ..label import read_labels_from_annot, Label, write_labels_to_annot
from ..utils import (get_config, set_config, logger, _validate_type,
                     verbose, get_subjects_dir, _pl, _safe_input)
from ..utils.docs import docdict, _docformat


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
    path : instance of Path
        Path to {name} dataset directory.
"""
_data_path_doc_accept = _data_path_doc.split('%(verbose)s')
_data_path_doc_accept[-1] = '%(verbose)s' + _data_path_doc_accept[-1]
_data_path_doc_accept.insert(1, '    %(accept)s')
_data_path_doc_accept = ''.join(_data_path_doc_accept)
_data_path_doc = _docformat(_data_path_doc, docdict)
_data_path_doc_accept = _docformat(_data_path_doc_accept, docdict)

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
        logger.debug(f'Version file missing: {ver_fname}')
        # Sample dataset versioning was introduced after 0.3
        # SPM dataset was introduced with 0.7
        versions = dict(sample='0.7', spm='0.3')
        version = versions.get(name, '0.0')
    return version


def _get_path(path, key, name):
    """Get a dataset path."""
    # 1. Input
    _validate_type(path, ('path-like', None), path)
    if path is not None:
        return path
    # 2. get_config(key) — unless key is None or "" (special get_config values)
    # 3. get_config('MNE_DATA')
    path = get_config(key or 'MNE_DATA', get_config('MNE_DATA'))
    if path is not None:
        if not op.exists(path):
            msg = (f"Download location {path} as specified by MNE_DATA does "
                   f"not exist. Either create this directory manually and try "
                   f"again, or set MNE_DATA to an existing directory.")
            raise FileNotFoundError(msg)
        return Path(path)
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
    return Path(path)


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
            set_config(key, str(path), set_env=False)
    return path


# This is meant to be semi-public: let packages like mne-bids use it to make
# sure they don't accidentally set download=True in their tests, too
_MODULES_TO_ENSURE_DOWNLOAD_IS_FALSE_IN_TESTS = ('mne',)


def _check_in_testing_and_raise(name, download):
    """Check if we're in an MNE test and raise an error if download!=False."""
    root_dirs = [
        importlib.import_module(ns)
        for ns in _MODULES_TO_ENSURE_DOWNLOAD_IS_FALSE_IN_TESTS]
    root_dirs = [str(Path(ns.__file__).parent) for ns in root_dirs]
    check = False
    func = None
    frame = inspect.currentframe()
    try:
        # First, traverse out of the data_path() call
        while frame:
            if frame.f_code.co_name in ('data_path', 'load_data'):
                func = frame.f_code.co_name
                frame = frame.f_back.f_back  # out of verbose decorator
                break
            frame = frame.f_back
        # Next, see what the caller was
        while frame:
            fname = frame.f_code.co_filename
            if fname is not None:
                fname = Path(fname)
                # in mne namespace, and
                # (can't use is_relative_to here until 3.9)
                if any(str(fname).startswith(rd) for rd in root_dirs) and (
                        # in tests/*.py
                        fname.parent.stem == 'tests' or
                        # or in a conftest.py
                        fname.stem == 'conftest.py'):
                    check = True
                    break
            frame = frame.f_back
    finally:
        del frame
    if check and download is not False:
        raise RuntimeError(
            f'Do not download dataset {repr(name)} in tests, pass '
            f'{func}(download=False) to prevent accidental downloads')


def _download_mne_dataset(name, processor, path, force_update,
                          update_path, download, accept=False):
    """Aux function for downloading internal MNE datasets."""
    import pooch
    from mne.datasets._fetch import fetch_dataset

    _check_in_testing_and_raise(name, download)

    # import pooch library for handling the dataset downloading
    dataset_params = MNE_DATASETS[name]
    dataset_params['dataset_name'] = name
    config_key = MNE_DATASETS[name]['config_key']
    folder_name = MNE_DATASETS[name]['folder_name']

    # get download path for specific dataset
    path = _get_path(path=path, key=config_key, name=name)

    # instantiate processor that unzips file
    if processor == 'nested_untar':
        processor_ = pooch.Untar(extract_dir=op.join(path, folder_name))
    elif processor == 'nested_unzip':
        processor_ = pooch.Unzip(extract_dir=op.join(path, folder_name))
    else:
        processor_ = processor

    # handle case of multiple sub-datasets with different urls
    if name == 'visual_92_categories':
        dataset_params = []
        for name in ['visual_92_categories_1', 'visual_92_categories_2']:
            this_dataset = MNE_DATASETS[name]
            this_dataset['dataset_name'] = name
            dataset_params.append(this_dataset)

    return fetch_dataset(dataset_params=dataset_params, processor=processor_,
                         path=path, force_update=force_update,
                         update_path=update_path, download=download,
                         accept=accept)


def _get_version(name):
    """Get a dataset version."""
    from mne.datasets._fetch import fetch_dataset

    if not has_dataset(name):
        return None
    dataset_params = MNE_DATASETS[name]
    dataset_params['dataset_name'] = name
    config_key = MNE_DATASETS[name]['config_key']

    # get download path for specific dataset
    path = _get_path(path=None, key=config_key, name=name)

    return fetch_dataset(dataset_params, path=path,
                         return_version=True)[1]


def has_dataset(name):
    """Check for presence of a dataset.

    Parameters
    ----------
    name : str | dict
        The dataset to check. Strings refer to one of the supported datasets
        listed :ref:`here <datasets>`. A :class:`dict` can be used to check for
        user-defined datasets (see the Notes section of :func:`fetch_dataset`),
        and must contain keys ``dataset_name``, ``archive_name``, ``url``,
        ``folder_name``, ``hash``.

    Returns
    -------
    has : bool
        True if the dataset is present.
    """
    from mne.datasets._fetch import fetch_dataset

    if isinstance(name, dict):
        dataset_name = name['dataset_name']
        dataset_params = name
    else:
        dataset_name = 'spm' if name == 'spm_face' else name
        dataset_params = MNE_DATASETS[dataset_name]
        dataset_params['dataset_name'] = dataset_name

    config_key = dataset_params['config_key']

    # get download path for specific dataset
    path = _get_path(path=None, key=config_key, name=dataset_name)

    dp = fetch_dataset(dataset_params, path=path, download=False,
                       check_version=False)
    if dataset_name.startswith('bst_'):
        check = dataset_name
    else:
        check = MNE_DATASETS[dataset_name]['folder_name']
    return str(dp).endswith(check)


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
                   fetch_fsaverage, ssvep, erp_core, epilepsy_ecog,
                   fetch_phantom)
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
    phantom_path = brainstorm.bst_phantom_elekta.data_path(accept=True)
    fetch_phantom('otaniemi', subjects_dir=phantom_path)
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
        subjects_dir=sample_path / 'subjects', accept=True)
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
    import pooch

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
    import pooch

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
    import pooch

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
