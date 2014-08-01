import numpy as np
import os.path as op
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true

from mne import io, pick_types
from mne import Epochs
from mne import read_events
from mne.time_frequency import compute_raw_psd, compute_epochs_psd

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_fname = op.join(base_dir, 'test-eve.fif')


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
    n_fft = 128  # the FFT size (n_fft). Ideally a power of 2
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


def test_psd_epochs():
    """Test PSD estimation on epochs
    """
    raw = io.Raw(raw_fname)

    exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053']  # bads + 2 more

    # picks MEG gradiometers
    picks = pick_types(raw.info, meg='mag', eeg=False, stim=False,
                            exclude=exclude)

    picks = picks[:2]

    n_fft = 128  # the FFT size (n_fft). Ideally a power of 2

    tmin, tmax, event_id = -1, 1, 1
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

    picks = pick_types(epochs.info, meg='grad', eeg=False, eog=True,
                            stim=False, include=include, exclude='bads')
    psds, freqs = compute_epochs_psd(epochs[:1], fmin=2, fmax=300, n_fft=n_fft,
                                     picks=picks)
    psds_proj, _ = compute_epochs_psd(epochs[:1].apply_proj(), fmin=2,
                                      fmax=300, n_fft=n_fft, picks=picks)

    assert_array_almost_equal(psds, psds_proj)
    assert_true(psds.shape == (1, len(picks), len(freqs)))
    assert_true(np.sum(freqs < 0) == 0)
    assert_true(np.sum(psds < 0) == 0)
