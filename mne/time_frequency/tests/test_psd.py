import numpy as np
import os.path as op
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true, assert_raises

from mne import io, pick_types, Epochs, read_events
from mne.utils import requires_version, slow_test
from mne.time_frequency import compute_raw_psd, compute_epochs_psd

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_fname = op.join(base_dir, 'test-eve.fif')


@requires_version('scipy', '0.12')
def test_psd():
    """Test PSD estimation
    """
    raw = io.Raw(raw_fname)

    exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053']  # bads + 2 more

    # picks MEG gradiometers
    picks = pick_types(raw.info, meg='mag', eeg=False, stim=False,
                       exclude=exclude)

    picks = picks[:2]

    tmin, tmax = 0, 10  # use the first 60s of data
    fmin, fmax = 2, 70  # look at frequencies between 5 and 70Hz

    n_fft = 128
    psds, freqs = compute_raw_psd(raw, tmin=tmin, tmax=tmax, fmin=fmin,
                                  fmax=fmax, proj=False, n_fft=n_fft,
                                  picks=picks, n_jobs=1)
    assert_true(psds.shape == (len(picks), len(freqs)))
    assert_true(np.sum(freqs < 0) == 0)
    assert_true(np.sum(psds < 0) == 0)

    n_fft = 2048  # the FFT size (n_fft). Ideally a power of 2
    psds, freqs = compute_raw_psd(raw, tmin=tmin, tmax=tmax, picks=picks,
                                  fmin=fmin, fmax=fmax, n_fft=n_fft, n_jobs=1,
                                  proj=False)
    psds_proj, freqs = compute_raw_psd(raw, tmin=tmin, tmax=tmax, picks=picks,
                                       fmin=fmin, fmax=fmax, n_fft=n_fft,
                                       n_jobs=1, proj=True)

    assert_array_almost_equal(psds, psds_proj)
    assert_true(psds.shape == (len(picks), len(freqs)))
    assert_true(np.sum(freqs < 0) == 0)
    assert_true(np.sum(psds < 0) == 0)


@requires_version('scipy', '0.12')
def test_psd_epochs():
    """Test PSD estimation on epochs
    """
    raw = io.Raw(raw_fname)

    exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053']  # bads + 2 more

    # picks MEG gradiometers
    picks = pick_types(raw.info, meg='mag', eeg=False, stim=False,
                       exclude=exclude)

    picks = picks[:2]
    n_fft = 512  # the FFT size (n_fft). Ideally a power of 2

    tmin, tmax, event_id = -0.5, 0.5, 1
    include = []
    raw.info['bads'] += ['MEG 2443']  # bads

    # picks MEG gradiometers
    picks = pick_types(raw.info, meg='grad', eeg=False, eog=True,
                       stim=False, include=include, exclude='bads')

    events = read_events(event_fname)

    epochs = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0),
                    reject=dict(grad=4000e-13, eog=150e-6), proj=False,
                    preload=True)

    tmin_full, tmax_full = -1, 1
    epochs_full = Epochs(raw, events[:10], event_id, tmax=tmax_full,
                         tmin=tmin_full, picks=picks,
                         baseline=(None, 0),
                         reject=dict(grad=4000e-13, eog=150e-6), proj=False,
                         preload=True)

    picks = pick_types(epochs.info, meg='grad', eeg=False, eog=True,
                       stim=False, include=include, exclude='bads')
    psds, freqs = compute_epochs_psd(epochs[:1], fmin=2, fmax=300,
                                     n_fft=n_fft, picks=picks, method='welch')

    psds_t, freqs_t = compute_epochs_psd(epochs_full[:1], fmin=2, fmax=300,
                                         tmin=tmin, tmax=tmax, method='welch',
                                         n_fft=n_fft, picks=picks)
    # this one will fail if you add for example 0.1 to tmin
    assert_array_almost_equal(psds, psds_t, 27)

    psds_proj, _ = compute_epochs_psd(epochs[:1].apply_proj(), fmin=2,
                                      fmax=300, n_fft=n_fft, picks=picks,
                                      method='welch')

    assert_array_almost_equal(psds, psds_proj)
    assert_true(psds.shape == (1, len(picks), len(freqs)))
    assert_true(np.sum(freqs < 0) == 0)
    assert_true(np.sum(psds < 0) == 0)

    # Multitaper
    psds_mt, freqs_mt = compute_epochs_psd(epochs_full[:1], fmin=2, fmax=300,
                                           tmin=tmin, tmax=tmax,
                                           method='multitaper', picks=picks)
    psds_mt_proj, freqs_mt_proj = compute_epochs_psd(
        epochs_full[:1].apply_proj(), fmin=2, fmax=300, tmin=tmin, tmax=tmax,
        method='multitaper', picks=picks)

    assert_array_almost_equal(psds_mt, psds_mt_proj)
    assert_true(psds_mt.shape == (1, len(picks), len(freqs_mt)))
    assert_true(np.sum(freqs_mt < 0) == 0)
    assert_true(np.sum(psds_mt < 0) == 0)

    # Passing arrays
    sfreq = epochs.info['sfreq']
    epochs_array = epochs.get_data()
    psds_arr, freqs_arr = compute_epochs_psd(epochs_array[:1], fmin=2,
                                             fmax=300, n_fft=n_fft,
                                             sfreq=sfreq, picks=picks)
    assert_raises(ValueError, compute_epochs_psd, epochs_array)
    assert_raises(ValueError, compute_epochs_psd, epochs_array[0], sfreq=sfreq)


@slow_test
@requires_version('scipy', '0.12')
def test_compares_psd():
    """Test PSD estimation on raw for plt.psd and scipy.signal.welch
    """
    raw = io.Raw(raw_fname)

    exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053']  # bads + 2 more

    # picks MEG gradiometers
    picks = pick_types(raw.info, meg='grad', eeg=False, stim=False,
                       exclude=exclude)[:2]

    tmin, tmax = 0, 10  # use the first 60s of data
    fmin, fmax = 2, 70  # look at frequencies between 5 and 70Hz
    n_fft = 2048

    # Compute psds with the new implementation using Welch
    psds_welch, freqs_welch = compute_raw_psd(raw, tmin=tmin, tmax=tmax,
                                              fmin=fmin, fmax=fmax,
                                              proj=False, picks=picks,
                                              n_fft=n_fft, n_jobs=1)

    # Compute psds with plt.psd
    start, stop = raw.time_as_index([tmin, tmax])
    data, times = raw[picks, start:(stop + 1)]
    from matplotlib.pyplot import psd
    out = [psd(d, Fs=raw.info['sfreq'], NFFT=n_fft) for d in data]
    freqs_mpl = out[0][1]
    psds_mpl = np.array([o[0] for o in out])

    mask = (freqs_mpl >= fmin) & (freqs_mpl <= fmax)
    freqs_mpl = freqs_mpl[mask]
    psds_mpl = psds_mpl[:, mask]

    assert_array_almost_equal(psds_welch, psds_mpl)
    assert_array_almost_equal(freqs_welch, freqs_mpl)

    assert_true(psds_welch.shape == (len(picks), len(freqs_welch)))
    assert_true(psds_mpl.shape == (len(picks), len(freqs_mpl)))

    assert_true(np.sum(freqs_welch < 0) == 0)
    assert_true(np.sum(freqs_mpl < 0) == 0)

    assert_true(np.sum(psds_welch < 0) == 0)
    assert_true(np.sum(psds_mpl < 0) == 0)
