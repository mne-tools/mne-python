# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD Style.

import os
from os import path as op

import numpy as np

from ...utils import _get_path
from ....utils import _fetch_file, verbose, _TempDir

BASE_URL = 'https://physionet.org/pn4/sleep-edfx/'
SLEEP_RECORDS = 'physionet_sleep_records.npy'


def _update_sleep_records():
    import pandas as pd

    SLEEP_RECORDS = 'records.csv'
    tmp = _TempDir()

    # Download files checksum.
    sha1sums_url = BASE_URL + "SHA1SUMS"
    sha1sums_fname = op.join(tmp, 'sha1sums')
    _fetch_file(sha1sums_url, sha1sums_fname)

    # Download subjects info.
    subjects_url = BASE_URL + 'ST-subjects.xls'
    subjects_fname = op.join(tmp, 'ST-subjects.xls')
    _fetch_file(url=subjects_url, file_name=subjects_fname,
                hash_='f52fffe5c18826a2bd4c5d5cb375bb4a9008c885',
                hash_type='sha1')

    # Load and Massage the checksums.
    sha1_df = pd.read_csv(sha1sums_fname, sep='  ', header=None,
                          names=['sha', 'fname'], engine='python')
    select_age_records = (sha1_df.fname.str.startswith('ST') &
                          sha1_df.fname.str.endswith('edf'))
    sha1_df = sha1_df[select_age_records]
    sha1_df['id'] = [name[:6] for name in sha1_df.fname]

    # Load and massage the data.
    data = pd.read_excel(subjects_fname)
    data.index.name = 'subject'
    data.columns.names = [None, None]
    data = (data.set_index([('Subject - age - sex', 'Age'),
                            ('Subject - age - sex', 'M1/F2')], append=True)
                .stack(level=0).reset_index())

    data = data.rename(columns={('Subject - age - sex', 'Age'): 'Age',
                                ('Subject - age - sex', 'M1/F2'): 'sex',
                                'level_3': 'record'})
    data['id'] = ['ST7{0:02d}{1:1d}'.format(s, n)
                  for s, n in zip(data.subject, data['night nr'])]

    xx = data.set_index(['id', 'subject', 'Age', 'sex', 'record',
                         'lights off', 'night nr', 'record type']).unstack()
    xx = xx.drop(columns=[('sha', np.nan), ('fname', np.nan)])
    xx.columns = [l1 + '_' + l2 for l1, l2 in xx.columns]
    xx.reset_index()

    # import pdb; pdb.set_trace()

    print('done')

    # Save the data.
    # data.to_csv(op.join(op.dirname(__file__), SLEEP_RECORDS),
    #             index=False)


@verbose
def data_path(path=None, force_update=False, update_path=None, verbose=None):
    """Get path to local copy of EEG Physionet Polysomnography dataset URL.

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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`).

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


def _fetch_one(fname, hashsum, path, force_update):
    # Fetch the file
    url = BASE_URL + '/' + fname
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
def fetch_data(subjects, path=None, force_update=False, update_path=None,
               base_url=BASE_URL, verbose=None):  # noqa: D301
    """Get paths to local copies of PhysioNet Polysomnography dataset files.

    This will fetch data for the EEGBCI dataset [1]_, which is also
    available at PhysioNet [2]_.

    Parameters
    ----------
    subject : list of int
        The subjects to use. Can be in the range of 0-60 (inclusive).
    path : None | str
        Location of where to look for the EEGBCI data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_PHYSIONET_SLEEP_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the Polysomnography dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_EEGBCI_PATH in mne-python
        config to the given path. If None, the user is prompted.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    paths : list
        List of local data paths of the given type.

    Notes
    -----
    For example, one could do:

        >>> from mne.datasets import sleep_physionet
        >>> sleep_physionet.temazepam.fetch_data(subjects=[0]) # doctest: +SKIP

    This would download data for subject 0 if it isn't there already.

    References
    ----------
    .. [1] B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis
           of a sleep-dependent neuronal feedback loop: the slow-wave
           microcontinuity of the EEG. IEEE-BME 47(9):1185-1194 (2000).
    .. [2] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh,
           Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000)
           PhysioBank, PhysioToolkit, and PhysioNet: Components of a New
           Research Resource for Complex Physiologic Signals.
           Circulation 101(23):e215-e220
    """
    records = np.load(op.join(op.dirname(__file__), SLEEP_RECORDS))
    path = data_path(path=path, update_path=update_path)
    params = [path, force_update]

    fnames = []
    for subject in subjects:
        assert 0 <= subject <= 60
        idx_psg, idx_hyp = np.where(records['index'] == subject)[0]
        psg_fname, sha_psg = records['fname'][idx_psg], records['sha'][idx_psg]
        hyp_fname, sha_hyp = records['fname'][idx_hyp], records['sha'][idx_hyp]

        psg_fname = _fetch_one(psg_fname, sha_psg, *params)
        hyp_fname = _fetch_one(hyp_fname, sha_hyp, *params)
        fnames.append([psg_fname, hyp_fname])

    return fnames
