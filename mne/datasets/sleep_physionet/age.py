# -*- coding: utf-8 -*-
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD Style.

import numpy as np

from ...utils import verbose
from ._utils import _fetch_one, _data_path, AGE_SLEEP_RECORDS
from ._utils import _check_subjects

data_path = _data_path  # expose _data_path(..) as data_path(..)

BASE_URL = 'https://physionet.org/physiobank/database/sleep-edfx/sleep-cassette/'  # noqa: E501


@verbose
def fetch_data(subjects, recording=[1, 2], path=None, force_update=False,
               update_path=None, base_url=BASE_URL,
               verbose=None):  # noqa: D301
    """Get paths to local copies of PhysioNet Polysomnography dataset files.

    This will fetch data from the publicly available subjects from PhysioNet's
    study of age effects on sleep in healthy subjects [1]_[2]_. This
    corresponds to a subset of 20 subjects, 10 males and 10 females that were
    25-34 years old at the time of the recordings. There are two night
    recordings per subject except for subject 13 since the second record was
    lost.

    See more details in
    `physionet website <https://physionet.org/physiobank/database/sleep-edfx/sleep-cassette/>`_.

    Parameters
    ----------
    subjects : list of int
        The subjects to use. Can be in the range of 0-19 (inclusive).
    recording : list of int
        The night recording indices. Valid values are : [1], [2], or [1, 2].
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
    %(verbose)s

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

    See Also
    --------
    :func:`mne.datasets.sleep_physionet.temazepam.fetch_data`
    """  # noqa: E501
    records = np.loadtxt(AGE_SLEEP_RECORDS,
                         skiprows=1,
                         delimiter=',',
                         usecols=(0, 1, 2, 6, 7),
                         dtype={'names': ('subject', 'record', 'type', 'sha',
                                          'fname'),
                                'formats': ('<i2', 'i1', '<S9', 'S40', '<S22')}
                         )
    psg_records = records[np.where(records['type'] == b'PSG')]
    hyp_records = records[np.where(records['type'] == b'Hypnogram')]

    path = data_path(path=path, update_path=update_path)
    params = [path, force_update, base_url]

    _check_subjects(subjects, 20)

    fnames = []
    for subject in subjects:
        for idx in np.where(psg_records['subject'] == subject)[0]:
            if psg_records['record'][idx] in recording:
                psg_fname = _fetch_one(psg_records['fname'][idx].decode(),
                                       psg_records['sha'][idx].decode(),
                                       *params)
                hyp_fname = _fetch_one(hyp_records['fname'][idx].decode(),
                                       hyp_records['sha'][idx].decode(),
                                       *params)
                fnames.append([psg_fname, hyp_fname])

    return fnames
