# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD Style.

from os import path as op

import numpy as np

from .._utils import _fetch_one, BASE_URL
from ...utils import _get_path
from ....utils import _fetch_file, verbose, _TempDir

SLEEP_RECORDS = 'physionet_sleep_records.npy'


def _update_sleep_records():
    # XXX: use requrie pandas
    import pandas as pd

    SLEEP_RECORDS = 'records.csv'
    tmp = _TempDir()

    # Download files checksum.
    sha1sums_url = BASE_URL + "SHA1SUMS"
    sha1sums_fname = op.join(tmp, 'sha1sums')
    _fetch_file(sha1sums_url, sha1sums_fname)

    # Download subjects info.
    subjects_url = BASE_URL + 'SC-subjects.xls'
    subjects_fname = op.join(tmp, 'SC-subjects.xls')
    _fetch_file(url=subjects_url, file_name=subjects_fname,
                hash_='0ba6650892c5d33a8e2b3f62ce1cc9f30438c54f',
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

    data['id'] = ['SC4{0:02d}{1:1d}'.format(s, n)
                  for s, n in zip(data.subject, data.night)]

    data = data.set_index('id').join(sha1_df.set_index('id')).dropna()

    data['record type'] = (data.fname.str.split('-', expand=True)[1]
                                     .str.split('.', expand=True)[0]
                                     .astype('category'))

    # data = data.set_index(['subject', 'night', 'record type'])
    data = data.reset_index().drop(columns=['id'])
    data = data[['subject', 'night', 'record type', 'age', 'sex', 'lights off',
                 'sha', 'fname']]

    # Save the data.
    data.to_csv(op.join(op.dirname(__file__), SLEEP_RECORDS),
                index=False)


@verbose
def data_path(path=None, force_update=False, update_path=None, verbose=None):
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


@verbose
def fetch_data(subjects, record=[1, 2], path=None, force_update=False,
               update_path=None, base_url=BASE_URL,
               verbose=None):  # noqa: D301
    """Get paths to local copies of PhysioNet Polysomnography dataset files.

    This will fetch data from the publicly available subjects from PhysioNet's
    study of age effects on sleep in healthy subjects [1]_[2]_. This
    corresponds to a subset of 20 subjects, 10 males and 10 females that were
    25-34 years old at the time of the recordings. There are two night
    recordings per subject except for subject 13 since the second record was
    lost.

    See more details in `physionet website <https://physionet.org/pn4/sleep-edfx/#data-from-a-study-of-age-effects-on-sleep-in-healt>`_.

    Parameters
    ----------
    subject : list of int
        The subjects to use. Can be in the range of 0-19 (inclusive).
    records : list of int
        The night record. Valid values are : [1], [2], or [1, 2].
    path : None | str
        Location of where to look for the PhysioNet data storing location.
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
        >>> sleep_physionet.age.fetch_data(subjects=[0])  # doctest: +SKIP

    This would download data for subject 0 if it isn't there already.

    References
    ----------
    .. [1] MS Mourtazaev, B Kemp, AH Zwinderman, HAC Kamphuisen. Age and gender
           affect different characteristics of slow waves in the sleep EEG.
           Sleep 18(7):557–564 (1995).
    .. [2] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh,
           Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000)
           PhysioBank, PhysioToolkit, and PhysioNet: Components of a New
           Research Resource for Complex Physiologic Signals.
           Circulation 101(23):e215-e220
    """

    records = np.loadtxt(op.join(op.dirname(__file__), 'records.csv'),
                         skiprows=1,
                         delimiter=',',
                         usecols=(0, 1, 2, 6, 7),
                         encoding='latin1',
                         dtype={'names': ('subject', 'record', 'type', 'sha',
                                          'fname'),
                                'formats': ('<i2', 'i1', '<S9', 'S40', '<S22')}
                         )
    psg_records = records[np.where(records['type'] == b'PSG')]
    hyp_records = records[np.where(records['type'] == b'Hypnogram')]

    path = data_path(path=path, update_path=update_path)
    params = [path, force_update]

    fnames = []
    for subject in subjects:
        assert 0 <= subject <= 19  # there are only 20 of 82 records available

        for idx in np.where(psg_records['subject'] == subject)[0]:
            if psg_records['record'][idx] in record:
                psg_fname = _fetch_one(psg_records['fname'][idx].decode(),
                                       psg_records['sha'][idx].decode(),
                                       *params)
                hyp_fname = _fetch_one(hyp_records['fname'][idx].decode(),
                                       hyp_records['sha'][idx].decode(),
                                       *params)
                fnames.append([psg_fname, hyp_fname])

    return fnames
