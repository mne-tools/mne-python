# -*- coding: utf-8 -*-
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD Style.

import numpy as np

from ...utils import verbose
from ._utils import _fetch_one, _data_path, TEMAZEPAM_SLEEP_RECORDS
from ._utils import _check_subjects

data_path = _data_path  # expose _data_path(..) as data_path(..)

BASE_URL = 'https://physionet.org/physiobank/database/sleep-edfx/sleep-telemetry/'  # noqa: E501


@verbose
def fetch_data(subjects, path=None, force_update=False, base_url=BASE_URL, *,
               verbose=None):
    """Get paths to local copies of PhysioNet Polysomnography dataset files.

    This will fetch data from the publicly available subjects from PhysioNet's
    study of Temazepam effects on sleep :footcite:`KempEtAl2000`. This
    corresponds to a set of 22 subjects. Subjects had mild difficulty falling
    asleep but were otherwise healthy.

    See more details in the `physionet website
    <https://physionet.org/physiobank/database/sleep-edfx/>`_
    :footcite:`GoldbergerEtAl2000`.

    Parameters
    ----------
    subjects : list of int
        The subjects to use. Can be in the range of 0-21 (inclusive).
    path : None | str
        Location of where to look for the PhysioNet data storing location.
        If None, the environment variable or config parameter
        ``PHYSIONET_SLEEP_PATH`` is used. If it doesn't exist, the "~/mne_data"
        directory is used. If the Polysomnography dataset is not found under
        the given path, the data will be automatically downloaded to the
        specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    base_url : str
        The base URL to download from.
    %(verbose)s

    Returns
    -------
    paths : list
        List of local data paths of the given type.

    See Also
    --------
    mne.datasets.sleep_physionet.age.fetch_data

    Notes
    -----
    For example, one could do:

        >>> from mne.datasets import sleep_physionet
        >>> sleep_physionet.temazepam.fetch_data(subjects=[1]) # doctest: +SKIP

    This would download data for subject 0 if it isn't there already.

    References
    ----------
    .. footbibliography::
    """
    records = np.loadtxt(TEMAZEPAM_SLEEP_RECORDS,
                         skiprows=1,
                         delimiter=',',
                         usecols=(0, 3, 6, 7, 8, 9),
                         dtype={'names': ('subject', 'record', 'hyp sha',
                                          'psg sha', 'hyp fname', 'psg fname'),
                                'formats': ('<i2', '<S15', 'S40', 'S40',
                                            '<S22', '<S16')}
                         )

    _check_subjects(subjects, 22)

    path = data_path(path=path)
    params = [path, force_update, base_url]

    fnames = []
    for subject in subjects:  # all the subjects are present at this point
        for idx in np.where(records['subject'] == subject)[0]:
            if records['record'][idx] == b'Placebo':
                psg_fname = _fetch_one(records['psg fname'][idx].decode(),
                                       records['psg sha'][idx].decode(),
                                       *params)
                hyp_fname = _fetch_one(records['hyp fname'][idx].decode(),
                                       records['hyp sha'][idx].decode(),
                                       *params)
                fnames.append([psg_fname, hyp_fname])

    return fnames
