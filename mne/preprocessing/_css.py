# Author: John Samuelsson <johnsam@mit.edu>

import numpy as np
import mne
from mne.utils import logger
from numpy import linalg


def temp_proj(ref_2, ref_1, raw_data, n_proj=6):
    """Removes common signal subspace of ref_2 and ref_1 from raw_data.

    Parameters
    ----------
    ref_2 : np.ndarray of float, shape (n_sensors_2, n_times)
        The magnetometer data for CSS. Can use either all magnetometer data or
        a few selected sensors close to a region to be suppressed.
    ref_1 : np.ndarray of float, shape (n_sensors_1, n_times)
        The gradiometer data for CSS. Can use either all gradiometer data or
        a few selected sensors close to a region to be suppressed.
    raw_data : np.ndarray of float, shape (n_sensors_raw, n_times)
        The data to be filtered, typically the EEG data.
    n_proj : int
        The number of projection vectors.

    Returns
    -------
    filtered_data : np.ndarray of float, shape (n_sensors_raw, n_times)
        The filtered data.

    Notes
    -------
    This temporal projection procedure removes the common signal subspace
    between ref_2 and ref_1 from raw_data using n_proj number of
    projection vectors. Normally used for cortical signal suppression, where
    ref_1 is gradiometer data, ref_2 is magnetometer data and
    raw_data is EEG data.

    """
    # Orthonormalize gradiometer and magnetometer data by a QR decomposition
    Qg = linalg.qr(ref_1.T)[0]
    Qm = linalg.qr(ref_2.T)[0]

    # Calculate cross-correlation
    C = np.dot(Qg.T, Qm)

    # Channel weights for common temporal subspace by SVD of cross-correlation
    Y, S, Z = linalg.svd(C)

    # Get temporal signals from channel weights
    u = np.dot(Qg, Y)

    # Project out common subspace
    filtered_data = raw_data
    for i in range(n_proj):
        proj_vec = u[:, i].reshape(u.shape[0], 1)
        weights = np.dot(filtered_data, proj_vec)
        filtered_data = filtered_data - np.dot(weights, proj_vec.T)

    return filtered_data


def cortical_signal_suppression(evoked, n_proj=6):
    """Remove the cortical signals in EEG by cortical signal suppression (CSS).

    Parameters
    ----------
    evoked : evoked object
        The evoked object with the averaged epochs.
    n_proj : int
        The number of projection vectors.

    Returns
    -------
    evoked_subcortical : instance of Evoked
        The evoked object with filtered EEG data.

    Notes
    -----
    This method removes the common signal subspace between the magnetometer
    data and the gradiometer data from the EEG data. This is done by a temporal
    projection using n_proj number of projection vectors.

    """
    if not isinstance(evoked, mne.evoked.EvokedArray) and not \
            isinstance(evoked, mne.evoked.Evoked):
        raise ValueError('evoked needs to an instance of Evoked, however type \
                         {} was passed'.format(type(evoked)))

    evoked_subcortical = evoked.copy()

    # Load data if not preloaded
    if not evoked.preload:
        logger.info('Data not preloaded. Loading data now...')
        evoked.load_data()

    # Get data
    info = evoked.info
    mag_ind = mne.pick_types(info, meg='mag')
    grad_ind = mne.pick_types(info, meg='grad')
    eeg_ind = mne.pick_types(info, meg=False, eeg=True)
    all_data = evoked.data
    mag_data = all_data[mag_ind]
    grad_data = all_data[grad_ind]

    # Process data with temporal projection algorithm
    eeg_data = all_data[eeg_ind]
    eeg_subcortical = temp_proj(mag_data, grad_data, eeg_data, n_proj=n_proj)
    evoked_subcortical.data[eeg_ind, :] = eeg_subcortical

    return evoked_subcortical
