# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD (3-clause)

import os
import os.path as op
import shutil
import zipfile
from sys import stdout

import numpy as np

from ...channels import make_standard_montage
from ...epochs import EpochsArray
from ...io.meas_info import create_info
from ...utils import _fetch_file, _check_pandas_installed, verbose
from ..utils import _get_path, _do_path_update

# root url for LIMO files
root_url = 'https://files.de-1.osf.io/v1/resources/52rea/providers/osfstorage/'

# subject identifier
subject_ids = {'S1': '5cde823c8d6e050018595862',
               'S2': '5cde825e23fec40017e0561a',
               'S3': '5cf7eedee650a2001ad560f2',
               'S4': '5cf7eee7d4c7d700193defcb',
               'S5': '5cf7eeece650a20017d5b153',

               'S6': '5cf8300fe650a20018d59cef',
               'S7': '5cf83018a542b8001bc7c75f',
               'S8': '5cf8301ea542b8001ac7cc47',
               'S9': '5cf830243a4d9500178a692b',
               'S10': '5cf83029e650a20017d600b1',

               'S11': '5cf834bfa542b8001bc7cae0',
               'S12': '5cf834c53a4d9500188a6311',
               'S13': '5cf834caa542b8001cc8149b',
               'S14': '5cf834cf3a4d9500178a6c6c',
               'S15': '5cf834d63a4d9500168ae5d6',

               'S16': '5cf834dbe650a20018d5a123',
               'S17': '5cf834e23a4d9500198a911f',
               'S18': '5cf834e73a4d9500198a9122'}


@verbose
def data_path(subject, path=None, force_update=False, update_path=None,
              verbose=None):
    """Get path to local copy of LIMO dataset URL.

    This is a low-level function useful for getting a local copy of the
    remote LIMO dataset [1]_. The complete dataset is available at
    datashare.is.ed.ac.uk/ [2]_.

    Parameters
    ----------
    subject : int
        Subject to download. Must be of class ìnt in the range from 1 to 18.
    path : None | str
        Location of where to look for the LIMO data storing directory.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_LIMO_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the LIMO dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_LIMO_PATH in mne-python
        config to the given path. If None, the user is prompted.
    %(verbose)s

    Returns
    -------
    path : str
        Local path to the given data file.

    Notes
    -----
    For example, one could do:

        >>> from mne.datasets import limo
        >>> limo.data_path(subject=1, path=os.getenv('HOME') + '/datasets') # doctest:+SKIP

    This would download the LIMO data file to the 'datasets' folder,
    and prompt the user to save the 'datasets' path to the mne-python config,
    if it isn't there already.

    References
    ----------
    .. [1] Guillaume, Rousselet. (2016). LIMO EEG Dataset, [dataset].
           University of Edinburgh, Centre for Clinical Brain Sciences.
           https://doi.org/10.7488/ds/1556.
    .. [2] https://datashare.is.ed.ac.uk/handle/10283/2189?show=full
    """  # noqa: E501
    # set destination path for download
    key = 'MNE_DATASETS_LIMO_PATH'
    name = 'LIMO'
    path = _get_path(path, key, name)
    limo_dir = op.join(path, 'MNE-limo-data')
    subject_id = 'S%s' % subject
    destination = op.join(limo_dir, '%s.zip') % subject_id

    # url for subject in question
    url = root_url + subject_ids[subject_id] + '/?zip='

    # check if LIMO directory exists; update if desired
    if not op.isdir(limo_dir) or force_update:
        if op.isdir(limo_dir):
            shutil.rmtree(limo_dir)
        if not op.isdir(limo_dir):
            os.makedirs(limo_dir)

    # check if subject in question exists
    if not op.isdir(op.join(limo_dir, subject_id)):
        os.makedirs(op.join(limo_dir, subject_id))
        _fetch_file(url, destination, print_destination=False)

        # check if download is a zip-folder
        if any(group.endswith(".zip") for group in op.splitext(destination)):
            if not op.isdir(op.join(limo_dir, subject_id)):
                os.makedirs(op.join(limo_dir, subject_id))
            with zipfile.ZipFile(destination) as z1:
                files = [op.join(limo_dir, file) for file in z1.namelist()]
                stdout.write('Decompressing %g files from\n'
                             '"%s" ...' % (len(files), destination))
                z1.extractall(op.join(limo_dir, subject_id))
                stdout.write(' [done]\n')
                z1.close()
                os.remove(destination)

    # update path if desired
    _do_path_update(path, update_path, key, name)

    return limo_dir


@verbose
def load_data(subject, path=None, force_update=False, update_path=None,
              verbose=None):
    """Fetch subjects epochs data for the LIMO data set.

    Parameters
    ----------
    subject : int
        Subject to use. Must be of class ìnt in the range from 1 to 18.
    path : str
        Location of where to look for the LIMO data.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_LIMO_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_LIMO_PATH in mne-python
        config to the given path. If None, the user is prompted.
    %(verbose)s

    Returns
    -------
    epochs : instance of Epochs
        The epochs.
    """  # noqa: E501
    pd = _check_pandas_installed()
    from scipy.io import loadmat

    # subject in question
    if isinstance(subject, int) and 1 <= subject <= 18:
        subj = 'S%i' % subject
    else:
        raise ValueError('subject must be an int in the range from 1 to 18')

    # set limo path, download and decompress files if not found
    limo_path = data_path(subject, path, force_update, update_path)

    # -- 1) import .mat files
    # epochs info
    fname_info = op.join(limo_path, subj, 'LIMO.mat')
    data_info = loadmat(fname_info)
    # number of epochs per condition
    design = data_info['LIMO']['design'][0][0]['X'][0][0]
    data_info = data_info['LIMO']['data'][0][0][0][0]
    # epochs data
    fname_eeg = op.join(limo_path, subj, 'Yr.mat')
    data = loadmat(fname_eeg)

    # -- 2) get epochs information from structure
    # sampling rate
    sfreq = data_info['sampling_rate'][0][0]
    # tmin and tmax
    tmin = data_info['start'][0][0]
    # create events matrix
    sample = np.arange(len(design))
    prev_id = np.zeros(len(design))
    ev_id = design[:, 1]
    events = np.array([sample, prev_id, ev_id]).astype(int).T
    # event ids, such that Face B == 1
    event_id = {'Face/A': 0, 'Face/B': 1}

    # -- 3) extract channel labels from LIMO structure
    # get individual labels
    labels = data_info['chanlocs']['labels']
    labels = [label for label, *_ in labels[0]]
    # get montage
    montage = make_standard_montage('biosemi128')
    # add external electrodes (e.g., eogs)
    ch_names = montage.ch_names + ['EXG1', 'EXG2', 'EXG3', 'EXG4']
    # match individual labels to labels in montage
    found_inds = [ind for ind, name in enumerate(ch_names) if name in labels]
    missing_chans = [name for name in ch_names if name not in labels]
    assert labels == [ch_names[ind] for ind in found_inds]

    # -- 4) extract data from subjects Yr structure
    # data is stored as channels x time points x epochs
    # data['Yr'].shape  # <-- see here
    # transpose to epochs x channels time points
    data = np.transpose(data['Yr'], (2, 0, 1))
    # initialize data in expected order
    temp_data = np.empty((data.shape[0], len(ch_names), data.shape[2]))
    # copy over the non-missing data
    for source, target in enumerate(found_inds):
        # avoid copy when fancy indexing
        temp_data[:, target, :] = data[:, source, :]
    # data to V (to match MNE's format)
    data = temp_data / 1e6
    # create list containing channel types
    types = ["eog" if ch.startswith("EXG") else "eeg" for ch in ch_names]

    # -- 5) Create custom info for mne epochs structure
    # create info
    info = create_info(ch_names, sfreq, types).set_montage(montage)
    # get faces and noise variables from design matrix
    event_list = list(events[:, 2])
    faces = ['B' if event else 'A' for event in event_list]
    noise = list(design[:, 2])
    # create epochs metadata
    metadata = {'face': faces, 'phase-coherence': noise}
    metadata = pd.DataFrame(metadata)

    # -- 6) Create custom epochs array
    epochs = EpochsArray(data, info, events, tmin, event_id, metadata=metadata)
    epochs.info['bads'] = missing_chans  # missing channels are marked as bad.

    return epochs
