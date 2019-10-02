

import numpy as np
import mne
import matplotlib.pyplot as plt


def _css(mag_data, grad_data, filter_data, r=6):
    """Remove the cortical contribution to filter_data (typically
    magnetometer or EEG data) by means of a temporal projection.

    This procedure removes the common signal subspace between mag_data
    and grad_data from filter_data using r number of projection vectors.
    Note that the SNR needs to be relatively high for this to work; it
    therefore makes most sense using this on averaged data in an evoked
    object.

    Parameters
    ----------
    mag_data : np.ndarray of float, shape (n_mag_sensors,n_times)
        The magnetometer data. Can use either all magnetometer data or
        a few selected sensors close to a region you want to suppress.
    grad_data : np.ndarray of float, shape (n_grad_sensors,n_times)
        The gradiometer data. Can use either all gradiometer data or
        a few selected sensors close to a region you want to suppress.
    filter_data : np.ndarray of float, shape (n_grad_sensors,n_times)
        The data to be filtered, typically the magnetometer or EEG data.
    r : int
        The number of projection vectors.

    Returns
    -------
    filtered_data : np.ndarray of int, shape (n_features)
        The outlier indices.
    """
    # Orthonormalize gradiometer and magnetometer data by a QR decomposition
    Qg, rg = np.linalg.qr(grad_data.T)
    Qm, rm = np.linalg.qr(mag_data.T)

    # Calculate cross-correlation
    C = np.dot(Qg.T, Qm)

    # Channel weights for common temporal subspace by SVD of cross-correlation
    Y, S, Z = np.linalg.svd(C)

    # Get temporal signals from channel weights
    u = np.dot(Qg, Y)

    # Project out common subspace (r is number of projection vectors)
    filtered_data = filter_data
    for i in range(r):
        proj_vec = u[:, i].reshape(u.shape[0], 1)
        NpProj = np.dot(filtered_data, proj_vec)
        filtered_data = filtered_data - np.dot(NpProj, proj_vec.T)

    return filtered_data


def _plot_psd(evoked, evoked_subcortical, filt_inds, title):
    """Plots the total power spectral density of data in evoked and
    evoked_subcortical.

    Plots the sum of the power spectral density across channels filt_inds.

    Parameters
    ----------
    evoked : evoked object
        Unprocessed data.
    evoked_subcortical : evoked object
        Processed data.
    filt_inds : np.ndarray of int, shape (n_sensors)
        The indicies of the data that were filtered.
    title : string
        The title of the plot
    """
    def getFFT(y, Fs):
        Ft = np.fft.fft(y)
        Ft = np.abs(Ft[0:int(len(Ft) / 2 + 1)])**2
        Y = np.sqrt(Ft / (len(y) * Fs))
        f = Fs / 2 * np.linspace(0, 1, int(len(y) / 2 + 1))
        return [f, Y]

    def sum_fft(evoked, evoked_subcortical, mag_ind, chs):
        fft_length = int(evoked.times.shape[0] / 2 + 1)
        fft_unprocessed = np.zeros((fft_length))
        fft_processed = np.zeros((fft_length))
        for ch in chs:
            signal_unprocessed = evoked.data[mag_ind[ch], :]
            signal_processed = evoked_subcortical.data[mag_ind[ch], :]
            fft_unprocessed = fft_unprocessed + \
                getFFT(signal_unprocessed, evoked.info['sfreq'])[1]
            f, fft_processed = fft_processed + \
                getFFT(signal_processed, evoked.info['sfreq'])

        return fft_unprocessed, fft_processed, f

    fft_unprocessed, fft_processed, f = \
        sum_fft(evoked, evoked_subcortical, filt_inds,
                chs=range(len(filt_inds)))
    plt.figure()
    plt.plot(f, fft_unprocessed, label='raw data')
    plt.plot(f, fft_processed, label='processed')
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Spectral density')
    plt.legend()
    plt.tight_layout()
    plt.show()


def filter_cortical(evoked, projection_vectors=6, show_psd=False):
    """Remove the cortical contribution to evoked by cortical signal
    suppression (CSS).

    This method removes the common signal subspace between the magnetometer
    data and the gradiometer data from the magnetometer data, and EEG data if
    there is any. This is done by a temporal projection using a number of
    projection vectors equal to projection_vectors. If show_psd is True, then
    the total power spectral density of the unprocessed and processed data
    will be displayed.

    Parameters
    ----------
    evoked : evoked object
        The evoked object with the averaged epochs.
    projection_vectors : int
        The number of projection vectors.
    filter_data : boolean
        If true, plots PSD of unprocessed and processed data.

    Returns
    -------
    evoked_subcortical : np.ndarray of int, shape (n_features)
        The outlier indices.
    """
    evoked_subcortical = evoked.copy()

    # Load data if not preloaded
    if not evoked.preload:
        print('Data not preloaded. Loading data now...')
        evoked.load_data()

    # Get data
    info = evoked.info
    mag_ind = mne.pick_types(info, meg='mag')
    grad_ind = mne.pick_types(info, meg='grad')
    eeg_ind = mne.pick_types(info, meg=False, eeg=True)
    all_data = evoked.data
    mag_data = all_data[mag_ind]
    grad_data = all_data[grad_ind]

    # Process data with CSS
    mag_subcortical = _css(mag_data, grad_data, mag_data, r=projection_vectors)

    # Save processed data to raw file
    evoked_subcortical.data[mag_ind, :] = mag_subcortical

    # EEG
    if len(eeg_ind) == 0:
        print('No EEG data found. Filtering only magnetometer data.')
    else:
        eeg_data = all_data[eeg_ind]
        eeg_subcortical = _css(mag_data, grad_data, eeg_data,
                               r=projection_vectors)
        evoked_subcortical.data[eeg_ind, :] = eeg_subcortical

    if show_psd:
        _plot_psd(evoked, evoked_subcortical, filt_inds=mag_ind,
                  title='Magnetometers')
        if not len(eeg_ind) == 0:
            _plot_psd(evoked, evoked_subcortical, filt_inds=eeg_ind,
                      title='EEG')

    return evoked_subcortical
