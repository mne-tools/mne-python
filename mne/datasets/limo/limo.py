# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD-3-Clause

import os
import os.path as op

import numpy as np

from ...channels import make_standard_montage
from ...epochs import EpochsArray
from ...io.meas_info import create_info
from ...utils import _check_pandas_installed, verbose
from ..utils import _get_path, _do_path_update, logger


# root url for LIMO files
root_url = 'https://files.de-1.osf.io/v1/resources/52rea/providers/osfstorage/'


@verbose
def data_path(subject, path=None, force_update=False, update_path=None, *,
              verbose=None):
    """Get path to local copy of LIMO dataset URL.

    This is a low-level function useful for getting a local copy of the
    remote LIMO dataset :footcite:`Rousselet2016`. The complete dataset is
    available at datashare.is.ed.ac.uk/.

    Parameters
    ----------
    subject : int
        Subject to download. Must be of :class:`ìnt` in the range from 1
        to 18 (inclusive).
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
    .. footbibliography::
    """  # noqa: E501
    import pooch

    downloader = pooch.HTTPDownloader(progressbar=True)  # use tqdm

    # local storage patch
    config_key = 'MNE_DATASETS_LIMO_PATH'
    name = 'LIMO'
    subj = f'S{subject}'
    path = _get_path(path, config_key, name)
    base_path = op.join(path, 'MNE-limo-data')
    subject_path = op.join(base_path, subj)
    # the remote URLs are in the form of UUIDs:
    urls = dict(
        S18={'Yr.mat': '5cf839833a4d9500178a6ff8',
             'LIMO.mat': '5cf83907e650a2001ad592e4'},
        S17={'Yr.mat': '5cf838e83a4d9500168aeb76',
             'LIMO.mat': '5cf83867a542b80019c87602'},
        S16={'Yr.mat': '5cf83857e650a20019d5778f',
             'LIMO.mat': '5cf837dc3a4d9500188a64fe'},
        S15={'Yr.mat': '5cf837cce650a2001ad591e8',
             'LIMO.mat': '5cf83758a542b8001ac7d11d'},
        S14={'Yr.mat': '5cf837493a4d9500198a938f',
             'LIMO.mat': '5cf836e4a542b8001bc7cc53'},
        S13={'Yr.mat': '5cf836d23a4d9500178a6df7',
             'LIMO.mat': '5cf836543a4d9500168ae7cb'},
        S12={'Yr.mat': '5cf83643d4c7d700193e5954',
             'LIMO.mat': '5cf835193a4d9500178a6c92'},
        S11={'Yr.mat': '5cf8356ea542b8001cc81517',
             'LIMO.mat': '5cf834f7d4c7d700163daab8'},
        S10={'Yr.mat': '5cf833b0e650a20019d57454',
             'LIMO.mat': '5cf83204e650a20018d59eb2'},
        S9={'Yr.mat': '5cf83201a542b8001cc811cf',
            'LIMO.mat': '5cf8316c3a4d9500168ae13b'},
        S8={'Yr.mat': '5cf8326ce650a20017d60373',
            'LIMO.mat': '5cf8316d3a4d9500198a8dc5'},
        S7={'Yr.mat': '5cf834a03a4d9500168ae59b',
            'LIMO.mat': '5cf83069e650a20017d600d7'},
        S6={'Yr.mat': '5cf830e6a542b80019c86a70',
            'LIMO.mat': '5cf83057a542b80019c869ca'},
        S5={'Yr.mat': '5cf8115be650a20018d58041',
            'LIMO.mat': '5cf80c0bd4c7d700193e213c'},
        S4={'Yr.mat': '5cf810c9a542b80019c8450a',
            'LIMO.mat': '5cf80bf83a4d9500198a6eb4'},
        S3={'Yr.mat': '5cf80c55d4c7d700163d8f52',
            'LIMO.mat': '5cf80bdea542b80019c83cab'},
        S2={'Yr.mat': '5cde827123fec40019e01300',
            'LIMO.mat': '5cde82682a50c4001677c259'},
        S1={'Yr.mat': '5d6d3071536cf5001a8b0c78',
            'LIMO.mat': '5d6d305f6f41fc001a3151d8'},
    )
    # these can't be in the registry file (mne/data/dataset_checksums.txt)
    # because of filename duplication
    hashes = dict(
        S18={'Yr.mat': 'md5:87f883d442737971a80fc0a35d057e51',
             'LIMO.mat': 'md5:8b4879646f65d7876fa4adf2e40162c5'},
        S17={'Yr.mat': 'md5:7b667ec9eefd7a9996f61ae270e295ee',
             'LIMO.mat': 'md5:22eaca4e6fad54431fd61b307fc426b8'},
        S16={'Yr.mat': 'md5:c877afdb4897426421577e863a45921a',
             'LIMO.mat': 'md5:86672d7afbea1e8c39305bc3f852c8c2'},
        S15={'Yr.mat': 'md5:eea9e0140af598fefc08c886a6f05de5',
             'LIMO.mat': 'md5:aed5cb71ddbfd27c6a3ac7d3e613d07f'},
        S14={'Yr.mat': 'md5:8bd842cfd8588bd5d32e72fdbe70b66e',
             'LIMO.mat': 'md5:1e07d1f36f2eefad435a77530daf2680'},
        S13={'Yr.mat': 'md5:d7925d2af7288b8a5186dfb5dbb63d34',
             'LIMO.mat': 'md5:ba891015d2f9e447955fffa9833404ca'},
        S12={'Yr.mat': 'md5:0e1d05beaa4bf2726e0d0671b78fe41e',
             'LIMO.mat': 'md5:423fd479d71097995b6614ecb11df9ad'},
        S11={'Yr.mat': 'md5:1b0016fb9832e43b71f79c1992fcbbb1',
             'LIMO.mat': 'md5:1a281348c2a41ee899f42731d30cda70'},
        S10={'Yr.mat': 'md5:13c66f60e241b9a9cc576eaf1b55a417',
             'LIMO.mat': 'md5:3c4b41e221eb352a21bbef1a7e006f06'},
        S9={'Yr.mat': 'md5:3ae1d9c3a1d9325deea2f2dddd1ab507',
            'LIMO.mat': 'md5:5e204e2a4bcfe4f535b4b1af469b37f7'},
        S8={'Yr.mat': 'md5:7e9adbca4e03d8d7ce8ea07ccecdc8fd',
            'LIMO.mat': 'md5:88313c21d34428863590e586b2bc3408'},
        S7={'Yr.mat': 'md5:6b5290a6725ecebf1022d5d2789b186d',
            'LIMO.mat': 'md5:8c769219ebc14ce3f595063e84bfc0a9'},
        S6={'Yr.mat': 'md5:420c858a8340bf7c28910b7b0425dc5d',
            'LIMO.mat': 'md5:9cf4e1a405366d6bd0cc6d996e32fd63'},
        S5={'Yr.mat': 'md5:946436cfb474c8debae56ffb1685ecf3',
            'LIMO.mat': 'md5:241fac95d3a79d2cea081391fb7078bd'},
        S4={'Yr.mat': 'md5:c8216af78ac87b739e86e57b345cafdd',
            'LIMO.mat': 'md5:8e10ef36c2e075edc2f787581ba33459'},
        S3={'Yr.mat': 'md5:ff02e885b65b7b807146f259a30b1b5e',
            'LIMO.mat': 'md5:59b5fb3a9749003133608b5871309e2c'},
        S2={'Yr.mat': 'md5:a4329022e57fd07ceceb7d1735fd2718',
            'LIMO.mat': 'md5:98b284b567f2dd395c936366e404f2c6'},
        S1={'Yr.mat': 'md5:076c0ae78fb71d43409c1877707df30e',
            'LIMO.mat': 'md5:136c8cf89f8f111a11f531bd9fa6ae69'},
    )
    # create the download manager
    fetcher = pooch.create(
        path=subject_path,
        base_url='',
        version=None,   # Data versioning is decoupled from MNE-Python version.
        registry=hashes[subj],
        urls={key: f'{root_url}{uuid}' for key, uuid in urls[subj].items()},
        retry_if_failed=2  # 2 retries = 3 total attempts
    )
    # use our logger level for pooch's logger too
    pooch.get_logger().setLevel(logger.getEffectiveLevel())
    # fetch the data
    for fname in ('LIMO.mat', 'Yr.mat'):
        destination = op.join(subject_path, fname)
        if force_update and op.isfile(destination):
            os.remove(destination)
        # fetch the remote file (if local file missing or has hash mismatch)
        fetcher.fetch(fname=fname, downloader=downloader)
    # update path in config if desired
    _do_path_update(path, update_path, config_key, name)
    return base_path


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
