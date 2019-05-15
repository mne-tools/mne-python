# -*- coding: utf-8 -*-
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD Style.

import os
import os.path as op
import numpy as np
from distutils.version import LooseVersion

from ...utils import _fetch_file, verbose, _TempDir, _check_pandas_installed
from ..utils import _get_path

AGE_SLEEP_RECORDS = op.join(op.dirname(__file__), 'age_records.csv')
TEMAZEPAM_SLEEP_RECORDS = op.join(op.dirname(__file__),
                                  'temazepam_records.csv')

TEMAZEPAM_RECORDS_URL = 'https://physionet.org/physiobank/database/sleep-edfx/ST-subjects.xls'  # noqa: E501
TEMAZEPAM_RECORDS_URL_SHA1 = 'f52fffe5c18826a2bd4c5d5cb375bb4a9008c885'

AGE_RECORDS_URL = 'https://physionet.org/physiobank/database/sleep-edfx/SC-subjects.xls'  # noqa: E501
AGE_RECORDS_URL_SHA1 = '0ba6650892c5d33a8e2b3f62ce1cc9f30438c54f'

sha1sums_fname = op.join(op.dirname(__file__), 'SHA1SUMS')


def _fetch_one(fname, hashsum, path, force_update, base_url):
    # Fetch the file
    url = base_url + '/' + fname
    destination = op.join(path, fname)
    if not op.isfile(destination) or force_update:
        if op.isfile(destination):
            os.remove(destination)
        if not op.isdir(op.dirname(destination)):
            os.makedirs(op.dirname(destination))
        _fetch_file(url, destination, print_destination=False,
                    hash_=hashsum, hash_type='sha1')
    return destination


@verbose
def _data_path(path=None, force_update=False, update_path=None, verbose=None):
    """Get path to local copy of EEG Physionet age Polysomnography dataset URL.

    This is a low-level function useful for getting a local copy of a
    remote Polysomnography dataset [1]_ which is available at PhysioNet [2]_.

    Parameters
    ----------
    path : None | str
        Location of where to look for the data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_PHYSIONET_SLEEP_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_PHYSIONET_SLEEP_PATH in mne-python
        config to the given path. If None, the user is prompted.
    %(verbose)s

    Returns
    -------
    path : list of str
        Local path to the given data file. This path is contained inside a list
        of length one, for compatibility.

    References
    ----------
    .. [1] B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of
           a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity
           of the EEG. IEEE-BME 47(9):1185-1194 (2000).
    .. [2] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh,
           Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000)
           PhysioBank, PhysioToolkit, and PhysioNet: Components of a New
           Research Resource for Complex Physiologic Signals.
           Circulation 101(23):e215-e220
    """  # noqa: E501
    key = 'PHYSIONET_SLEEP_PATH'
    name = 'PHYSIONET_SLEEP'
    path = _get_path(path, key, name)
    return op.join(path, 'physionet-sleep-data')


def _update_sleep_temazepam_records(fname=TEMAZEPAM_SLEEP_RECORDS):
    """Help function to download Physionet's temazepam dataset records."""
    pd = _check_pandas_installed()
    tmp = _TempDir()

    # Download subjects info.
    subjects_fname = op.join(tmp, 'ST-subjects.xls')
    _fetch_file(url=TEMAZEPAM_RECORDS_URL,
                file_name=subjects_fname,
                hash_=TEMAZEPAM_RECORDS_URL_SHA1,
                hash_type='sha1')

    # Load and Massage the checksums.
    sha1_df = pd.read_csv(sha1sums_fname, sep='  ', header=None,
                          names=['sha', 'fname'], engine='python')
    select_age_records = (sha1_df.fname.str.startswith('ST') &
                          sha1_df.fname.str.endswith('edf'))
    sha1_df = sha1_df[select_age_records]
    sha1_df['id'] = [name[:6] for name in sha1_df.fname]

    # Load and massage the data.
    data = pd.read_excel(subjects_fname, header=[0, 1])
    if LooseVersion(pd.__version__) >= LooseVersion('0.24.0'):
        data = data.set_index(('Subject - age - sex', 'Nr'))
    data.index.name = 'subject'
    data.columns.names = [None, None]
    data = (data.set_index([('Subject - age - sex', 'Age'),
                            ('Subject - age - sex', 'M1/F2')], append=True)
            .stack(level=0).reset_index())

    data = data.rename(columns={('Subject - age - sex', 'Age'): 'age',
                                ('Subject - age - sex', 'M1/F2'): 'sex',
                                'level_3': 'drug'})
    data['id'] = ['ST7{:02d}{:1d}'.format(s, n)
                  for s, n in zip(data.subject, data['night nr'])]

    data = pd.merge(sha1_df, data, how='outer', on='id')
    data['record type'] = (data.fname.str.split('-', expand=True)[1]
                                     .str.split('.', expand=True)[0]
                                     .astype('category'))

    data = data.set_index(['id', 'subject', 'age', 'sex', 'drug',
                           'lights off', 'night nr', 'record type']).unstack()
    data.columns = [l1 + '_' + l2 for l1, l2 in data.columns]
    if LooseVersion(pd.__version__) < LooseVersion('0.21.0'):
        data = data.reset_index().drop(labels=['id'], axis=1)
    else:
        data = data.reset_index().drop(columns=['id'])

    data['sex'] = (data.sex.astype('category')
                       .cat.rename_categories({1: 'male', 2: 'female'}))

    data['drug'] = data['drug'].str.split(expand=True)[0]
    data['subject_orig'] = data['subject']
    data['subject'] = data.index // 2  # to make sure index is from 0 to 21

    # Save the data.
    data.to_csv(fname, index=False)


def _update_sleep_age_records(fname=AGE_SLEEP_RECORDS):
    """Help function to download Physionet's age dataset records."""
    pd = _check_pandas_installed()
    tmp = _TempDir()

    # Download subjects info.
    subjects_fname = op.join(tmp, 'SC-subjects.xls')
    _fetch_file(url=AGE_RECORDS_URL,
                file_name=subjects_fname,
                hash_=AGE_RECORDS_URL_SHA1,
                hash_type='sha1')

    # Load and Massage the checksums.
    sha1_df = pd.read_csv(sha1sums_fname, sep='  ', header=None,
                          names=['sha', 'fname'], engine='python')
    select_age_records = (sha1_df.fname.str.startswith('SC') &
                          sha1_df.fname.str.endswith('edf'))
    sha1_df = sha1_df[select_age_records]
    sha1_df['id'] = [name[:6] for name in sha1_df.fname]

    # Load and massage the data.
    data = pd.read_excel(subjects_fname)
    data = data.rename(index=str, columns={'sex (F=1)': 'sex',
                                           'LightsOff': 'lights off'})
    data['sex'] = (data.sex.astype('category')
                       .cat.rename_categories({1: 'female', 2: 'male'}))

    data['id'] = ['SC4{:02d}{:1d}'.format(s, n)
                  for s, n in zip(data.subject, data.night)]

    data = data.set_index('id').join(sha1_df.set_index('id')).dropna()

    data['record type'] = (data.fname.str.split('-', expand=True)[1]
                                     .str.split('.', expand=True)[0]
                                     .astype('category'))

    if LooseVersion(pd.__version__) < LooseVersion('0.21.0'):
        data = data.reset_index().drop(labels=['id'], axis=1)
    else:
        data = data.reset_index().drop(columns=['id'])
    data = data[['subject', 'night', 'record type', 'age', 'sex', 'lights off',
                 'sha', 'fname']]

    # Save the data.
    data.to_csv(fname, index=False)


def _check_subjects(subjects, n_subjects):
    valid_subjects = np.arange(n_subjects)
    unknown_subjects = np.setdiff1d(subjects, valid_subjects)
    if unknown_subjects.size > 0:
        subjects_list = ', '.join([str(s) for s in unknown_subjects])
        raise ValueError('Only subjects 0 to {} are'
                         ' available from this dataset.'
                         ' Unknown subjects: {}'.format(n_subjects - 1,
                                                        subjects_list))
