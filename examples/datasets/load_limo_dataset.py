"""
========================================================
Import single subject from LIMO data set into MNE-Python
========================================================

Here we a define function to extract the eeg signal data
from the LIMO structures in the LIMO dataset, see [1]_ and:

    https://datashare.is.ed.ac.uk/handle/10283/2189?show=full

    https://github.com/LIMO-EEG-Toolbox

The code allows to:

    - Fetch single subjects epochs data for the LIMO data set.
    - Get the epochs information from the LIMO .mat files, such as
      sampling rate, number of epochs per condition, number and name
      of EEG channels per subject.
    - Create a custom info and epochs structure in MNE-Python.
    - Interpolate missing channels.
    - Plot the evoked response for experimental conditions.

.. note:: This example assumes that the LIMO data set has already
          been downloaded and stored in home directory.


References
----------
.. [1] Guillaume, Rousselet. (2016). LIMO EEG Dataset, [dataset].
       University of Edinburgh, Centre for Clinical Brain Sciences.
       https://doi.org/10.7488/ds/1556.
"""

# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
#
# License: BSD (3-clause)


# import necessary packages
import os.path as op
import scipy.io
import numpy as np
import mne

print(__doc__)


# --- define load limo epochs function ---


def load_limo_epochs(subject, path=None, interpolate=False):
    """Fetch subjects epochs data for the LIMO data set.
    Parameters
    ----------
    path : str
        The path to store the downloaded data. Defaults to home directory.
    subject : int | str
        Subject to use. Can be in the range from 2 to 18.
        If string, must be 'S1', 'S2' etc.
    interpolate : bool
        Whether to interpolate missing channels.
    Returns
    -------
    epochs : MNE Epochs data structure
        The epochs.
    """

    # set path to data
    if path is None:
        limo_path = op.expanduser('HOME/limo_dataset')
        if not op.exists(limo_path):
            raise ValueError('Could not find path to limo data set.')
    else:
        if isinstance(path, str):
            limo_path = path
        else:
            raise ValueError('"path" argument must be string or None.')

    # subject in question
    if isinstance(subject, int):
        sub = 'S%i' % subject
    elif isinstance(subject, str):
        if not subject.startswith('S'):
            raise ValueError('`subject` must start with `S`')
        sub = subject

    # -- 1) import .mat files
    # epochs info
    fname_info = op.join(limo_path, sub, 'LIMO.mat')
    data_info = scipy.io.loadmat(fname_info)
    # epochs data
    fname_eeg = op.join(limo_path, sub, 'Yr.mat')
    data = scipy.io.loadmat(fname_eeg)

    # -- 2) get epochs information from structure
    # sampling rate
    sfreq = data_info['LIMO']['data'][0][0][0][0]['sampling_rate'][0][0]
    # tmin and tmax
    tmin = data_info['LIMO']['data'][0][0][0][0]['start'][0][0]
    # number of epochs per condition
    design = data_info['LIMO']['design'][0][0]['X'][0][0]

    # create events matrix
    events = np.zeros(shape=(len(design), 3), dtype=int)
    events[:, 0] = list(range(len(design)))
    events[:, 2] = design[:, 1]
    # event ids, such that B == 1
    event_id = dict(A=0, B=1)

    # -- 3) extract channel labels from LIMO structure
    # get individual labels
    labels = data_info['LIMO']['data'][0][0][0][0]['chanlocs']['labels']
    labels = [ll[0] for ll in labels[0]]
    # get montage
    montage = mne.channels.read_montage('biosemi128')
    # add external electrodes (e.g., eogs)
    ch_names = montage.ch_names[:-3] + ['EXG1', 'EXG2', 'EXG3', 'EXG4']
    # match individual labels to labels in montage
    found_inds = [ii for ii, ee in enumerate(ch_names) if ee in labels]
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
    obs_data = all_data
    obs_data /= 1e6  # data to V (to match MNE)

    # create list containing channel types
    types = []
    for ch in ch_names:
        types.append(
            'eog' if ch in ('EXG1', 'EXG2', 'EXG3', 'EXG4') else 'eeg')

    # -- 5) Create custom info for mne epochs structure
    # create info
    info = mne.create_info(ch_names=ch_names, ch_types=types, sfreq=sfreq)

    # -- 6) Create custom epochs array
    epochs = mne.EpochsArray(obs_data, info, events, tmin, event_id)
    epochs.set_montage(montage=montage)
    epochs.info['bads'] = missing_chans  # missing channels are marked as bad.

    # -- 7) interpolate missing channels
    if interpolate is True and montage is not None:
        # code bad channels
        epochs.interpolate_bads(reset_bads=True)

    return epochs, design


# --- Use load_limo_epochs() function to fetch subject data ---

# fetch data from subject 1
data, design = load_limo_epochs(subject = 1, path=None)

# inpect epochs info
data.info

# plot evoked response for condition A
data['A'].average().plot_joint(times=[.09, .15])
