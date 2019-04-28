# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD (3-clause)

import os
from os import path as op

import re
import zipfile
import tarfile
from sys import stdout
import scipy.io
import numpy as np
import pandas as pd
import mne

from ..utils import _get_path, _do_path_update
from ...utils import _fetch_file, _url_to_local_path, verbose


@verbose
def data_path(url, path=None, force_update=False, update_path=None,
              verbose=None):
    """Get path to local copy of LIMO dataset URL.

    This is a low-level function useful for getting a local copy of the
    remote LIMO dataset [1]_ which is available at datashare.is.ed.ac.uk/ [2]_.

    Parameters
    ----------
    url : str
        The location from where the dataset should be downloaded.
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
    path : list of str
        Local path to the given data file. This path is contained inside a list
        of length one, for compatibility.

    Notes
    -----
    For example, one could do:

        >>> from mne.datasets import limo
        >>> url = 'http://datashare.is.ed.ac.uk/download/DS_10283_2189.zip'
        >>> limo.data_path(url, os.getenv('HOME') + '/datasets') # doctest:+SKIP

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
    destination = re.sub('/download', '', _url_to_local_path(url, op.join(path, 'MNE-limo-data')))
    limo_dir = op.split(destination)[0]

    # check if zip.-folders are contained
    is_zip = True if '.zip' in [x.lower() for x in op.splitext(destination)] else False

    # fetch the file from url
    do_unzip = False
    if not op.isfile(destination) or force_update:
        if op.isfile(destination):
            os.remove(destination)
        if not op.isdir(op.dirname(destination)):
            os.makedirs(op.dirname(destination))
        _fetch_file(url, destination, print_destination=False)
        do_unzip = True

    # decompress if necessary
    if is_zip and do_unzip:
        z1 = zipfile.ZipFile(destination)
        files = z1.namelist()
        if do_unzip:
            stdout.write('Decompressing %g files from\n'
                         '"%s" ...' % (len(files), destination))
            z1.extractall(limo_dir)
            stdout.write(' [done]\n')
        z1.close()
        # list files in .zip-folder and look for further .zip-folders to decompress
        extracted_files = [op.join(limo_dir, f) for f in files]
        sub_files = [e for e in extracted_files if ".zip" in e or ".tar" in e]

        if sub_files:
            stdout.write('Decompressing the extracted files ...\n')
            for i in sub_files:
                if '.zip' in i:
                    sub_dir = op.splitext(i)[0]
                    z2 = zipfile.ZipFile(i)
                    z2.extractall(limo_dir)
                    z2.close()
                    # continue decompressing if necessary
                    sub_sub_files = [j for j in os.listdir(sub_dir) if '.zip' in j]
                    for k in sub_sub_files:
                        z3 = zipfile.ZipFile(os.path.join(sub_dir, k))
                        z3.extractall(sub_dir)
                        z3.close()
                elif '.tar' in i:
                    t = tarfile.open(os.path.join(limo_dir, i))
                    t.extractall(limo_dir)
            stdout.write(' [done]\n')

    # update default if desired
    _do_path_update(path, update_path, key, name)
    return limo_dir


@verbose
def load_data(subject, path=None, interpolate=False, force_update=False, update_path=None,
              url=None, verbose=None):  # noqa: D301
    """Fetch subjects epochs data for the LIMO data set.
    
    Parameters
    ----------
    subject : int | str
        Subject to use. Can be in the range from 1 to 18.
        If string, must be 'S1', 'S2' etc.
    path : str
        Location of where to look for the LIMO data.  
        If None, the environment variable or config parameter
        ``MNE_DATASETS_LIMO_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used.
    interpolate : bool
        Whether to interpolate missing channels.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_LIMO_PATH in mne-python
        config to the given path. If None, the user is prompted.
    url : str
        The location from where the dataset should be downloaded, if not
        found on drive.
    %(verbose)s
    
    Returns
    -------
    epochs : MNE Epochs data structure
        The epochs.
    """  # noqa: E501
    if url is None:
        url = 'http://datashare.is.ed.ac.uk/download/DS_10283_2189.zip'

    # set limo path, download and decompress files if not found
    limo_path = data_path(url, path, force_update, update_path)

    # subject in question
    if isinstance(subject, int):
        subj = 'S%i' % subject
    elif isinstance(subject, str):
        if not subject.startswith('S'):
            raise ValueError('`subject` must start with `S`')
        subj = subject

    # -- 1) import .mat files
    # epochs info
    fname_info = op.join(limo_path, subj, 'LIMO.mat')
    data_info = scipy.io.loadmat(fname_info)
    # epochs data
    fname_eeg = op.join(limo_path, subj, 'Yr.mat')
    data = scipy.io.loadmat(fname_eeg)

    # -- 2) get epochs information from structure
    # sampling rate
    sfreq = data_info['LIMO']['data'][0][0][0][0]['sampling_rate'][0][0]
    # tmin and tmax
    tmin = data_info['LIMO']['data'][0][0][0][0]['start'][0][0]
    # number of epochs per condition
    design = data_info['LIMO']['design'][0][0]['X'][0][0]

    # create events matrix
    events = np.transpose(np.array([list(range(len(design))),
                                    np.zeros(len(design)),
                                    design[:, 1]]).astype(int))
    # event ids, such that Face B == 1
    event_id = {'Face A': 0, 'Face B': 1}

    # -- 3) extract channel labels from LIMO structure
    # get individual labels
    labels = data_info['LIMO']['data'][0][0][0][0]['chanlocs']['labels']
    labels = [ll[0] for ll in labels[0]]
    # get montage
    montage = mne.channels.read_montage('biosemi128')
    # add external electrodes (e.g., eogs)
    ch_names = montage.ch_names[:-3] + ['EXG1', 'EXG2', 'EXG3', 'EXG4']
    # match individual labels to labels in montage
    found_inds = [ii for ii, ch_name in enumerate(ch_names) if ch_name in labels]
    missing_chans = [ch for ch in ch_names if ch not in labels]
    assert labels == [ch_names[ii] for ii in found_inds]

    # -- 4) extract data from subjects Yr structure
    # data is stored as channels x time points x epochs
    # data['Yr'].shape  # <-- see here
    # transpose to epochs x channels time points
    obs_data = np.transpose(data['Yr'], (2, 0, 1))

    # initialize data in expected order
    all_data = np.empty((obs_data.shape[0],
                         len(ch_names),
                         obs_data.shape[2]))
    # copy over the non-missing data.
    for source, target in enumerate(found_inds):
        # avoid copy when fancy indexing.
        all_data[:, target, :] = obs_data[:, source, :]
    # data to V (to match MNE)
    obs_data = all_data / 1e6

    # create list containing channel types
    types = ["eog" if ch.startswith("EXG") else "eeg" for ch in ch_names]

    # -- 5) Create custom info for mne epochs structure
    # create info
    info = mne.create_info(ch_names=ch_names, ch_types=types, sfreq=sfreq)

    # get faces and noise variables from design matrix
    faces = ['A' if list(events[:, 2])[i] == 0 else 'B' for i in list(events[:, 2])]
    noise = list(design[:, 2])
    # create epochs metadata
    metadata = {'Face': faces,
                'Noise': noise}
    metadata = pd.DataFrame(metadata)

    # -- 6) Create custom epochs array
    epochs = mne.EpochsArray(obs_data, info, events, tmin, event_id, metadata=metadata)
    epochs.set_montage(montage=montage)
    epochs.info['bads'] = missing_chans  # missing channels are marked as bad.

    # -- 7) interpolate missing channels
    if interpolate is True and montage is not None:
        epochs.interpolate_bads(reset_bads=True)

    return epochs
