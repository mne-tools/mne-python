

import numpy as np
import mne
import matplotlib.pyplot as plt


def _css(mag_data, grad_data, filter_data, r=6):
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
    for i in range(r):
        proj_vec = u[:, i].reshape(u.shape[0], 1)
        NpProj = np.dot(filter_data, proj_vec)
        filter_data = filter_data - np.dot(NpProj, proj_vec.T)

    return filter_data


def _plot_psd(evoked, evoked_subcortical, filt_inds, title):
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
    evoked_cortical = evoked.copy()

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
    evoked.data[mag_ind, :] = mag_subcortical

    # EEG
    if len(eeg_ind) == 0:
        print('No EEG data found. Filtering only magnetometer data.')
    else:
        eeg_data = all_data[eeg_ind]
        eeg_subcortical = _css(mag_data, grad_data, eeg_data,
                               r=projection_vectors)
        evoked.data[eeg_ind, :] = eeg_subcortical

    if show_psd:
        _plot_psd(evoked_cortical, evoked, filt_inds=mag_ind,
                  title='Magnetometers')
        if not len(eeg_ind) == 0:
            _plot_psd(evoked_cortical, evoked, filt_inds=eeg_ind, title='EEG')

    return evoked_cortical
