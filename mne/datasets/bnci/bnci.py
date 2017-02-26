# Author: Alexandre Barachant <alexandre.barachant@gmail.com>
# License: BSD Style.

import os
from os import path as op
import numpy as np

from scipy.io import loadmat

from ... import create_info, concatenate_raws
from ...io import RawArray
from ...channels import read_montage

from ..utils import _get_path, _do_path_update
from ...utils import _fetch_file, _url_to_local_path, verbose

BNCI_URL = 'http://bnci-horizon-2020.eu/database/data-sets/'


@verbose
def data_path(url, path=None, force_update=False, update_path=None,
              verbose=None):
    """Get path to local copy of BNCI dataset URL.

    This is a low-level function useful for getting a local copy of a
    remote BNCI dataset : http://bnci-horizon-2020.eu/database/data-sets.

    Parameters
    ----------
    url : str
        The dataset to use.
    path : None | str
        Location of where to look for the BNCI data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_BNCI_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the BNCI dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_BNCI_PATH in mne-python
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
        >>> url = 'http://bnci-horizon-2020.eu/database/data-sets/'
        >>> bnci.data_path(url, os.getenv('HOME') + '/datasets') # doctest:+SKIP

    This would download the given BNCI data file to the 'datasets' folder,
    and prompt the user to save the 'datasets' path to the mne-python config,
    if it isn't there already.

    """  # noqa: E501
    key = 'MNE_DATASETS_BNCI_PATH'
    name = 'BNCI'
    path = _get_path(path, key, name)
    destination = _url_to_local_path(url, op.join(path,
                                     'MNE-bnci-data'))
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
def load_data(subject, dataset='001-2014', path=None, force_update=False,
              update_path=None, base_url=BNCI_URL,
              verbose=None):  # noqa: D301
    """Get paths to local copies of a BNCI dataset files.

    This will fetch data for a given BNCI dataset. Report to the bnci website
    for a complete description of the experimental setup of each dataset.

    Parameters
    ----------
    subject : int
        The subject to use.
    dataset : string
        The bnci dataset name
    path : None | str
        Location of where to look for the BNCI data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_BNCI_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the BNCI2014001 dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_BNCI_PATH in mne-python
        config to the given path. If None, the user is prompted.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    raws : list
        List of raw instances for each run or sessions.
    event_id: dict
        dictonary containing events and their code.
    """
    dataset_list = {'001-2014': _load_data_001_2014,
                    '002-2014': _load_data_002_2014,
                    '004-2014': _load_data_004_2014,
                    '008-2014': _load_data_008_2014,
                    '009-2014': _load_data_009_2014,
                    '001-2015': _load_data_001_2015,
                    '004-2015': _load_data_004_2015}

    if dataset not in dataset_list.keys():
        raise ValueError("Dataset '%s' is not a valid BNCI dataset ID."
                         % dataset)

    return dataset_list[dataset](subject, path, force_update, update_path,
                                 base_url, verbose)


@verbose
def _load_data_001_2014(subject, path=None, force_update=False,
                        update_path=None, base_url=BNCI_URL,
                        verbose=None):
    """Load data for 001-2014 dataset."""
    if (subject < 1) or (subject > 9):
        raise ValueError("Subject must be between 1 and 9.")

    ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3',
                'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz',
                'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz', 'EOG1', 'EOG2', 'EOG3']
    ch_types = ['eeg'] * 22 + ['eog'] * 3

    data_paths = []
    for r in ['T', 'E']:
        url = '{u}001-2014/A{s:02d}{r}.mat'.format(u=base_url, s=subject, r=r)
        data_paths.extend(data_path(url, path, force_update, update_path))

    raws = []
    event_id = {}
    for filename in data_paths:
        data = loadmat(filename, struct_as_record=False, squeeze_me=True)

        runs = []
        for run in data['data']:
            raw, evd = _convert_run(run, ch_names, ch_types, verbose)
            runs.append(raw)
            event_id.update(evd)
        raws.append(concatenate_raws(runs))
    return raws, event_id


@verbose
def _load_data_002_2014(subject, path=None, force_update=False,
                        update_path=None, base_url=BNCI_URL,
                        verbose=None):
    """Load data for 002-2014 dataset."""
    if (subject < 1) or (subject > 14):
        raise ValueError("Subject must be between 1 and 14.")

    data_paths = []
    for r in ['T', 'E']:
        url = '{u}002-2014/S{s:02d}{r}.mat'.format(u=base_url, s=subject, r=r)
        data_paths.extend(data_path(url, path, force_update, update_path))

    raws = []
    event_id = {}
    for filename in data_paths:
        data = loadmat(filename, struct_as_record=False, squeeze_me=True)
        for run in data['data']:
            raw, evd = _convert_run(run, verbose=verbose)
            raws.append(raw)
            event_id.update(evd)
    return raws, event_id


@verbose
def _load_data_004_2014(subject, path=None, force_update=False,
                        update_path=None, base_url=BNCI_URL,
                        verbose=None):
    """Load data for 004-2014 dataset."""
    if (subject < 1) or (subject > 9):
        raise ValueError("Subject must be between 1 and 9.")

    ch_names = ['C3', 'Cz', 'C4', 'EOG1', 'EOG2', 'EOG3']
    ch_types = ['eeg'] * 3 + ['eog'] * 3

    data_paths = []
    for r in ['T', 'E']:
        url = '{u}004-2014/B{s:02d}{r}.mat'.format(u=base_url, s=subject, r=r)
        data_paths.extend(data_path(url, path, force_update, update_path))

    raws = []
    event_id = {}
    for filename in data_paths:
        data = loadmat(filename, struct_as_record=False, squeeze_me=True)
        for run in data['data']:
            raw, evd = _convert_run(run, ch_names, ch_types, verbose)
            raws.append(raw)
            event_id.update(evd)
    return raws, event_id


@verbose
def _load_data_008_2014(subject, path=None, force_update=False,
                        update_path=None, base_url=BNCI_URL,
                        verbose=None):
    """Load data for 008-2014 dataset."""
    if (subject < 1) or (subject > 8):
        raise ValueError("Subject must be between 1 and 8.")

    url = '{u}008-2014/A{s:02d}.mat'.format(u=base_url, s=subject)
    filename = data_path(url, path, force_update, update_path)[0]
    run = loadmat(filename, struct_as_record=False, squeeze_me=True)['data']
    raw, event_id = _convert_run_p300_sl(run, verbose=verbose)

    return [raw], event_id


@verbose
def _load_data_009_2014(subject, path=None, force_update=False,
                        update_path=None, base_url=BNCI_URL,
                        verbose=None):
    """Load data for 009-2014 dataset."""
    if (subject < 1) or (subject > 10):
        raise ValueError("Subject must be between 1 and 10.")

    # FIXME there is two type of speller, grid speller and geo-speller.
    # we load only grid speller data
    url = '{u}009-2014/A{s:02d}S.mat'.format(u=base_url, s=subject)
    filename = data_path(url, path, force_update, update_path)[0]
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)['data']
    raws = []
    event_id = {}
    for run in data:
        raw, ev = _convert_run_p300_sl(run, verbose=verbose)
        raws.append(raw)
        event_id.update(ev)

    return raws, event_id


@verbose
def _load_data_001_2015(subject, path=None, force_update=False,
                        update_path=None, base_url=BNCI_URL,
                        verbose=None):
    """Load data for 001-2015 dataset."""
    if (subject < 1) or (subject > 12):
        raise ValueError("Subject must be between 1 and 12.")

    if subject in [8, 9, 10, 11]:
        sessions = ['A', 'B', 'C']  # 3 sessions for those subjects
    else:
        sessions = ['A', 'B']

    data_paths = []
    for r in sessions:
        url = '{u}001-2015/S{s:02d}{r}.mat'.format(u=base_url, s=subject, r=r)
        data_paths.extend(data_path(url, path, force_update, update_path))

    ch_names = ['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2',
                'C4', 'C6', 'CP3', 'CPz', 'CP4']
    ch_types = ['eeg'] * 13

    raws = []
    event_id = {}
    for filename in data_paths:
        data = loadmat(filename, struct_as_record=False, squeeze_me=True)
        raw, evd = _convert_run(data['data'], ch_names, ch_types, verbose)
        raws.append(raw)
        event_id.update(evd)
    return raws, event_id


@verbose
def _load_data_004_2015(subject, path=None, force_update=False,
                        update_path=None, base_url=BNCI_URL,
                        verbose=None):
    """Load data for 004-2015 dataset."""
    if (subject < 1) or (subject > 9):
        raise ValueError("Subject must be between 1 and 9.")

    subjects = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'L']

    url = '{u}004-2015/{s}.mat'.format(u=base_url, s=subjects[subject - 1])
    filename = data_path(url, path, force_update, update_path)[0]

    ch_names = ['AFz', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC3', 'FCz',
                'FC4', 'T3', 'C3', 'Cz', 'C4', 'T4', 'CP3', 'CPz',
                'CP4', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4',
                'P6', 'P8', 'PO3', 'PO4', 'O1', 'O2']
    ch_types = ['eeg'] * 30

    raws = []
    event_id = {}
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    for run in data['data']:
        raw, evd = _convert_run(run, ch_names, ch_types, verbose)
        raws.append(raw)
        event_id.update(evd)
    return raws, event_id


@verbose
def _convert_run(run, ch_names=None, ch_types=None, verbose=None):
    """Convert one  run to raw."""
    # parse eeg data
    event_id = {}
    n_chan = run.X.shape[1]
    montage = read_montage('standard_1005')
    eeg_data = 1e-6 * run.X
    sfreq = run.fs

    if not ch_names:
        ch_names = ['EEG%d' % ch for ch in range(1, n_chan + 1)]
        montage = None  # no montage

    if not ch_types:
        ch_types = ['eeg'] * n_chan

    trigger = np.zeros((len(eeg_data), 1))
    # some runs does not contains trials i.e baseline runs
    if len(run.trial) > 0:
        trigger[run.trial - 1, 0] = run.y

    eeg_data = np.c_[eeg_data, trigger]
    ch_names = ch_names + ['stim']
    ch_types = ch_types + ['stim']
    event_id = {ev: (ii + 1) for ii, ev in enumerate(run.classes)}

    info = create_info(ch_names=ch_names, ch_types=ch_types,
                       sfreq=sfreq, montage=montage)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
    return raw, event_id


@verbose
def _convert_run_p300_sl(run, verbose=None):
    """convert one p300 run from santa lucia file format"""
    montage = read_montage('standard_1005')
    eeg_data = 1e-6 * run.X
    sfreq = 256
    ch_names = list(run.channels) + ['Target stim', 'Flash stim']
    ch_types = ['eeg'] * len(run.channels) + ['stim'] * 2

    flash_stim = run.y_stim
    flash_stim[flash_stim > 0] += 2
    eeg_data = np.c_[eeg_data, run.y, flash_stim]
    event_id = {ev: (ii + 1) for ii, ev in enumerate(run.classes)}
    event_id.update({ev: (ii + 3) for ii, ev in enumerate(run.classes_stim)})
    info = create_info(ch_names=ch_names, ch_types=ch_types,
                       sfreq=sfreq, montage=montage)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
    return raw, event_id
