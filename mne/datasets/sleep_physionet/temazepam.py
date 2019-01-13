# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD Style.

import numpy as np

from ...utils import verbose
from ._utils import _fetch_one, _data_path, BASE_URL, TEMAZEPAM_SLEEP_RECORDS
from ._utils import _check_subjects

SLEEP_RECORDS = 'physionet_sleep_records.npy'

data_path = _data_path  # expose _data_path(..) as data_path(..)


@verbose
def fetch_data(subjects, path=None, force_update=False,
               update_path=None, base_url=BASE_URL, verbose=None):
    """Get paths to local copies of PhysioNet Polysomnography dataset files.

    This will fetch data from the publicly available subjects from PhysioNet's
    study of Temazepam effects on sleep [1]_. This corresponds to
    a set of 22 subjects. Subjects had mild difficulty falling asleep
    but were otherwise healthy.

    Only the data with Placebo injection are available. The data with
    Temazepam have so far not been made public.

    See more details in
    `physionet website <https://physionet.org/pn4/sleep-edfx/>`_.

    Parameters
    ----------
    subjects : list of int
        The subjects to use. Can be in the range of 0-21 (inclusive).
    drug : bool
        If True it's the data with the Temazepam and if False it's
        the placebo.
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
        >>> sleep_physionet.temazepam.fetch_data(subjects=[1]) # doctest: +SKIP

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

    See Also
    --------
    :func:`mne.datasets.sleep_physionet.age.fetch_data`
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

    path = data_path(path=path, update_path=update_path)
    params = [path, force_update]

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
