import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true

from mne.datasets import sample
from mne import fiff, find_events, Epochs
from mne.label import read_label
from mne.minimum_norm.inverse import read_inverse_operator, \
                                     apply_inverse_epochs
from mne.minimum_norm.time_frequency import source_band_induced_power, \
                            source_induced_power, compute_source_psd, \
                            compute_source_psd_epochs


from mne.time_frequency import multitaper_psd

data_path = sample.data_path()
fname_inv = op.join(data_path, 'MEG', 'sample',
                                        'sample_audvis-meg-oct-6-meg-inv.fif')
fname_data = op.join(data_path, 'MEG', 'sample',
                                        'sample_audvis_raw.fif')
fname_label = op.join(data_path, 'MEG', 'sample', 'labels', 'Aud-lh.label')


def test_tfr_with_inverse_operator():
    """Test time freq with MNE inverse computation"""

    tmin, tmax, event_id = -0.2, 0.5, 1

    # Setup for reading the raw data
    raw = fiff.Raw(fname_data)
    events = find_events(raw, stim_channel='STI 014')
    inverse_operator = read_inverse_operator(fname_inv)

    raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more

    # picks MEG gradiometers
    picks = fiff.pick_types(raw.info, meg=True, eeg=False, eog=True,
                            stim=False, exclude='bads')

    # Load condition 1
    event_id = 1
    events3 = events[:3]  # take 3 events to keep the computation time low
    epochs = Epochs(raw, events3, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6),
                    preload=True)

    # Compute a source estimate per frequency band
    bands = dict(alpha=[10, 10])
    label = read_label(fname_label)

    stcs = source_band_induced_power(epochs, inverse_operator, bands,
                                     n_cycles=2, use_fft=False, pca=True,
                                     label=label)

    stc = stcs['alpha']
    assert_true(len(stcs) == len(bands.keys()))
    assert_true(np.all(stc.data > 0))
    assert_array_almost_equal(stc.times, epochs.times)

    stcs_no_pca = source_band_induced_power(epochs, inverse_operator, bands,
                                            n_cycles=2, use_fft=False,
                                            pca=False, label=label)

    assert_array_almost_equal(stcs['alpha'].data, stcs_no_pca['alpha'].data)

    # Compute a source estimate per frequency band
    epochs = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6),
                    preload=True)

    frequencies = np.arange(7, 30, 2)  # define frequencies of interest
    power, phase_lock = source_induced_power(epochs, inverse_operator,
                            frequencies, label, baseline=(-0.1, 0),
                            baseline_mode='percent', n_cycles=2, n_jobs=1)
    assert_true(np.all(phase_lock > 0))
    assert_true(np.all(phase_lock <= 1))
    assert_true(np.max(power) > 10)


def test_source_psd():
    """Test source PSD computation in label"""
    raw = fiff.Raw(fname_data)
    inverse_operator = read_inverse_operator(fname_inv)
    label = read_label(fname_label)
    tmin, tmax = 0, 20  # seconds
    fmin, fmax = 55, 65  # Hz
    NFFT = 2048
    stc = compute_source_psd(raw, inverse_operator, lambda2=1. / 9.,
                             method="dSPM", tmin=tmin, tmax=tmax,
                             fmin=fmin, fmax=fmax, pick_normal=True,
                             NFFT=NFFT, label=label, overlap=0.1)
    assert_true(stc.times[0] >= fmin * 1e-3)
    assert_true(stc.times[-1] <= fmax * 1e-3)
    # Time max at line frequency (60 Hz in US)
    assert_true(59e-3 <= stc.times[np.argmax(np.sum(stc.data, axis=0))]
                      <= 61e-3)


def test_source_psd_epochs():
    """Test multi-taper source PSD computation in label from epochs"""

    raw = fiff.Raw(fname_data)
    inverse_operator = read_inverse_operator(fname_inv)
    label = read_label(fname_label)

    event_id, tmin, tmax = 1, -0.2, 0.5
    lambda2, method = 1. / 9., 'dSPM'
    bandwidth = 8.
    fmin, fmax = 0, 100

    picks = fiff.pick_types(raw.info, meg=True, eeg=False, stim=True,
                            ecg=True, eog=True, include=['STI 014'],
                            exclude='bads')
    reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)

    events = find_events(raw, stim_channel='STI 014')
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=reject)

    # only look at one epoch
    epochs.drop_bad_epochs()
    one_epochs = epochs[:1]

    # return list
    stc_psd = compute_source_psd_epochs(one_epochs, inverse_operator,
                                        lambda2=lambda2, method=method,
                                        pick_normal=True, label=label,
                                        bandwidth=bandwidth,
                                        fmin=fmin, fmax=fmax)[0]

    # return generator
    stcs = compute_source_psd_epochs(one_epochs, inverse_operator,
                                     lambda2=lambda2, method=method,
                                     pick_normal=True, label=label,
                                     bandwidth=bandwidth,
                                     fmin=fmin, fmax=fmax,
                                     return_generator=True)

    for stc in stcs:
        stc_psd_gen = stc

    assert_array_almost_equal(stc_psd.data, stc_psd_gen.data)

    # compare with direct computation
    stc = apply_inverse_epochs(one_epochs, inverse_operator,
                               lambda2=lambda2, method=method,
                               pick_normal=True, label=label)[0]

    sfreq = epochs.info['sfreq']
    psd, freqs = multitaper_psd(stc.data, sfreq=sfreq, bandwidth=bandwidth,
                                fmin=fmin, fmax=fmax)

    assert_array_almost_equal(psd, stc_psd.data)
    assert_array_almost_equal(freqs, stc_psd.times)
