# Author: Alexandre Barachant <alexandre.barachant@gmail.com>
# License: BSD Style.

import os
from os import path as op
import numpy as np

from scipy.io import loadmat

from ... import create_info
from ...io import RawArray
from ...channels import read_montage

from ..utils import _get_path, _do_path_update
from ...utils import _fetch_file, _url_to_local_path, verbose

BNCI2014001_URL = 'http://bnci-horizon-2020.eu/database/data-sets/001-2014/'


@verbose
def data_path(url, path=None, force_update=False, update_path=None,
              verbose=None):
    """Get path to local copy of BNCI 2014-001 dataset URL.

    This is a low-level function useful for getting a local copy of a
    remote BNCI 2014 001 dataset [1].

    Parameters
    ----------
    url : str
        The dataset to use.
    path : None | str
        Location of where to look for the BNCI2014001 data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_BNCI2014001_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the BNCI2014001 dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_BNCI2014001_PATH in mne-python
        config to the given path. If None, the user is prompted.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`).

    Returns
    -------
    path : list of str
        Local path to the given data file. This path is contained inside a list
        of length one, for compatibility.

    Notes
    -----
    For example, one could do:

        >>> from mne.datasets import eegbci
        >>> url = 'http://bnci-horizon-2020.eu/database/data-sets/001-2014/'
        >>> bnci_2014_001.data_path(url, os.getenv('HOME') + '/datasets') # doctest:+SKIP

    This would download the given BNCI2014001 data file to the 'datasets' folder,
    and prompt the user to save the 'datasets' path to the mne-python config,
    if it isn't there already.

    References
    ----------
    .. [1] Tangermann, Michael, et al. "Review of the BCI competition IV."
           Frontiers in neuroscience 6 (2012): 55.
    """  # noqa: E501
    key = 'MNE_DATASETS_BNCI2014001_PATH'
    name = 'BNCI_2014_001'
    path = _get_path(path, key, name)
    destination = _url_to_local_path(url, op.join(path,
                                     'MNE-bnci2014001-data'))
    destinations = [destination]

    # Fetch the file
    if not op.isfile(destination) or force_update:
        if op.isfile(destination):
            os.remove(destination)
        if not op.isdir(op.dirname(destination)):
            os.makedirs(op.dirname(destination))
        _fetch_file(url, destination, print_destination=False)

    # Offer to update the path
    _do_path_update(path, update_path, key, name)
    return destinations


@verbose
def load_data(subject, sessions, path=None, force_update=False,
              update_path=None, base_url=BNCI2014001_URL,
              verbose=None):  # noqa: D301
    """Get paths to local copies of BNCI2014001 dataset files.

    This will fetch data for the BNCI2014001 dataset [1], a 4-class motor
    imagery dataset also known as the dataset IIa of the BCI competition IV.

    Events are encoded in the stim channel:

        - left hand: 1
        - right hand: 2
        - feet: 3
        - tongue: 4

    The experimental paradigm is the following. At t=0, a beep mark the
    begining of the trial. the Cue is given at t=2s and the imagination of
    movement occurs from t=3s to t=6s.

    Each session is composed of 6 run of 48 trial each (12 for each classes).
    the total number of trial per subject is 544.

    Parameters
    ----------
    subject : int
        The subject to use. Can be in the range of 1-109 (inclusive).
    sessions : string | list of string
        the session to load, could be 'T' for the training session or
        'E' for the test session.
    path : None | str
        Location of where to look for the BNCI2014001 data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_BNCI2014001_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the BNCI2014001 dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_BNCI2014001_PATH in mne-python
        config to the given path. If None, the user is prompted.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    raws : list
        List of raw instances for each run and sessions.


    References
    ----------
    .. [1] Tangermann, Michael, et al. "Review of the BCI competition IV."
           Frontiers in neuroscience 6 (2012): 55.
    """
    if not hasattr(sessions, '__iter__'):
        sessions = [sessions]

    data_paths = []
    for r in sessions:
        url = '{u}A{s:02d}{r}.mat'.format(u=base_url, s=subject, r=r)
        data_paths.extend(data_path(url, path, force_update, update_path))

    raws = []
    for filename in data_paths:
        data = loadmat(filename, struct_as_record=False, squeeze_me=True)

        for run in data['data'][3:]:
            ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3',
                        'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz',
                        'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
            ch_names += ['eog%d' % d for d in range(1, 4)] + ['stim']
            ch_types = ['eeg'] * 22 + ['eog'] * 3 + ['stim']
            montage = read_montage('standard_1005')
            eeg_data = 1e-6 * run.X
            sfreq = run.fs
            trigger = np.zeros((len(eeg_data), 1))
            trigger[run.trial, 0] = run.y
            eeg_data = np.c_[eeg_data, trigger]
            info = create_info(ch_names=ch_names, ch_types=ch_types,
                               sfreq=sfreq, montage=montage)
            raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
            raws.append(raw)

    return raws
