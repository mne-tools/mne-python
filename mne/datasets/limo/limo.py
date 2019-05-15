# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD (3-clause)

import os
import os.path as op
import re
import shutil
import zipfile
import tarfile
from sys import stdout

import scipy.io
import numpy as np
import pandas as pd

from ...channels import read_montage
from ...epochs import EpochsArray
from ...io.meas_info import create_info
from ...utils import _fetch_file, _url_to_local_path, verbose
from ..utils import _get_path, _do_path_update


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
    path : str
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
    limo_dir = op.join(path, 'MNE-limo-data')
    destination = re.sub('/download', '', _url_to_local_path(url, limo_dir))

    # fetch data from online repository if required
    if not op.isdir(limo_dir) or force_update:
        if op.isdir(limo_dir):
            shutil.rmtree(limo_dir)
        if not op.isdir(limo_dir):
            os.makedirs(limo_dir)
        _fetch_file(url, destination, print_destination=False)

        # check if download is a zip-folder
        if any(group.endswith(".zip") for group in op.splitext(destination)):
            with zipfile.ZipFile(destination) as z1:
                files = [op.join(limo_dir, file) for file in z1.namelist()]
                stdout.write('Decompressing %g files from\n'
                             '"%s" ...' % (len(files), destination))
                z1.extractall(limo_dir)
                stdout.write(' [done]\n')
                z1.close()
                os.remove(destination)

            # check if further .zip-folders are contained
            zips = [file for file in files if file.endswith('.zip')]
            if zips:
                for file in zips:
                    with zipfile.ZipFile(file) as z2:
                        in_zip = z2.namelist()
                        stdout.write('Decompressing .zip-files ...\n')
                        z2.extractall(limo_dir)
                        z2.close()
                        os.remove(file)
                    # continue decompressing if necessary
                    zfiles = [file for file in in_zip if file.endswith('.zip')]
                    if zfiles:
                        for zfile in zfiles:
                            zfile = op.join(limo_dir, zfile)
                            with zipfile.ZipFile(zfile) as z3:
                                z3.extractall(op.split(zfile)[0])
                                z3.close()
                                os.remove(zfile)

            # check if .tar-folders are contained
            tars = [file for file in files if file.endswith('.tar')]
            if tars:
                for file in tars:
                    with tarfile.open(op.join(limo_dir, file)) as tar:
                        stdout.write('Decompressing .tar-files ...\n')
                        tar.extractall(limo_dir)
                        tar.close()
                        os.remove(op.join(limo_dir, file))
            stdout.write(' [done]\n')

    # update path if desired
    _do_path_update(path, update_path, key, name)

    return limo_dir


@verbose
def load_data(subject, path=None, interpolate=False, force_update=False,
              update_path=None, url=None, verbose=None):  # noqa: D301
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
    if isinstance(subject, int) and 1 <= subject <= 18:
        subj = 'S%i' % subject
    else:
        raise ValueError('subject must be an int in the range from 1 to 18')

    # -- 1) import .mat files
    # epochs info
    fname_info = op.join(limo_path, subj, 'LIMO.mat')
    data_info = scipy.io.loadmat(fname_info)
    # number of epochs per condition
    design = data_info['LIMO']['design'][0][0]['X'][0][0]
    data_info = data_info['LIMO']['data'][0][0][0][0]
    # epochs data
    fname_eeg = op.join(limo_path, subj, 'Yr.mat')
    data = scipy.io.loadmat(fname_eeg)

    # -- 2) get epochs information from structure
    # sampling rate
    sfreq = data_info['sampling_rate'][0][0]
    # tmin and tmax
    tmin = data_info['start'][0][0]
    # create events matrix
    sample = list(range(len(design)))
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
    montage = read_montage('biosemi128')
    # add external electrodes (e.g., eogs)
    ch_names = montage.ch_names[:-3] + ['EXG1', 'EXG2', 'EXG3', 'EXG4']
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
    info = create_info(ch_names, sfreq, types, montage)
    # get faces and noise variables from design matrix
    event_list = list(events[:, 2])
    faces = ['A' if event == 0 else 'B' for event in event_list]
    noise = list(design[:, 2])
    # create epochs metadata
    metadata = {'Face': faces, 'Noise': noise}
    metadata = pd.DataFrame(metadata)

    # -- 6) Create custom epochs array
    epochs = EpochsArray(data, info, events, tmin, event_id, metadata=metadata)
    epochs.info['bads'] = missing_chans  # missing channels are marked as bad.

    # -- 7) interpolate missing channels
    if interpolate:
        epochs.interpolate_bads(reset_bads=True)

    return epochs
