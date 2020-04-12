import os.path as op

import numpy as np
from numpy.testing import assert_allclose
import pytest

from mne.datasets import testing
from mne import find_events, Epochs, pick_types
from mne.io import read_raw_fif
from mne.io.constants import FIFF
from mne.utils import run_tests_if_main
from mne.label import read_label
from mne.minimum_norm import (read_inverse_operator, apply_inverse_epochs,
                              prepare_inverse_operator, INVERSE_METHODS)
from mne.minimum_norm.time_frequency import (source_band_induced_power,
                                             source_induced_power,
                                             compute_source_psd,
                                             compute_source_psd_epochs)


from mne.time_frequency.multitaper import psd_array_multitaper

data_path = testing.data_path(download=False)
fname_inv = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-4-meg-inv.fif')
fname_data = op.join(data_path, 'MEG', 'sample',
                     'sample_audvis_trunc_raw.fif')
fname_label = op.join(data_path, 'MEG', 'sample', 'labels', 'Aud-lh.label')


@testing.requires_testing_data
@pytest.mark.parametrize('method', INVERSE_METHODS)
def test_tfr_with_inverse_operator(method):
    """Test time freq with MNE inverse computation."""
    tmin, tmax, event_id = -0.2, 0.5, 1

    # Setup for reading the raw data
    raw = read_raw_fif(fname_data)
    events = find_events(raw, stim_channel='STI 014')
    inv = read_inverse_operator(fname_inv)
    inv = prepare_inverse_operator(inv, nave=1, lambda2=1. / 9., method=method)

    raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more

    # picks MEG gradiometers
    picks = pick_types(raw.info, meg=True, eeg=False, eog=True,
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

    # XXX someday we should refactor this so that you don't have to pass
    # method -- maybe `prepare_inverse_operator` should add a `method`
    # to it and when `prepared=True` the value passed in can be ignored
    # (or better, default method=None means "dSPM if unprepared" and if they
    # actually pass a value, we check against `inv['method']`)
    stcs = source_band_induced_power(epochs, inv, bands, method=method,
                                     n_cycles=2, use_fft=False, pca=True,
                                     label=label, prepared=True)

    stc = stcs['alpha']
    assert len(stcs) == len(list(bands.keys()))
    assert np.all(stc.data > 0)
    assert_allclose(stc.times, epochs.times, atol=1e-6)

    stcs_no_pca = source_band_induced_power(epochs, inv, bands, method=method,
                                            n_cycles=2, use_fft=False,
                                            pca=False, label=label,
                                            prepared=True)

    assert_allclose(stcs['alpha'].data, stcs_no_pca['alpha'].data)

    # Compute a source estimate per frequency band
    epochs = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6),
                    preload=True)

    freqs = np.arange(7, 30, 2)  # define frequencies of interest
    power, phase_lock = source_induced_power(
        epochs, inv, freqs, label, baseline=(-0.1, 0), baseline_mode='percent',
        n_cycles=2, n_jobs=1, method=method, prepared=True)
    assert np.all(phase_lock > 0)
    assert np.all(phase_lock <= 1)
    assert 5 < np.max(power) < 7


@testing.requires_testing_data
@pytest.mark.parametrize('method', INVERSE_METHODS)
@pytest.mark.parametrize('pick_ori', (None, 'normal'))  # XXX vector someday?
def test_source_psd(method, pick_ori):
    """Test source PSD computation from raw."""
    raw = read_raw_fif(fname_data)
    raw.crop(0, 5).load_data()
    inverse_operator = read_inverse_operator(fname_inv)
    fmin, fmax = 40, 65  # Hz
    n_fft = 512

    assert inverse_operator['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI

    stc, ev = compute_source_psd(
        raw, inverse_operator, lambda2=1. / 9., method=method,
        fmin=fmin, fmax=fmax, pick_ori=pick_ori, n_fft=n_fft,
        overlap=0., return_sensor=True, dB=True)

    assert ev.data.shape == (len(ev.info['ch_names']), len(stc.times))
    assert ev.times[0] >= fmin
    assert ev.times[-1] <= fmax
    # Time max at line frequency (60 Hz in US)
    assert 58 <= ev.times[np.argmax(np.sum(ev.data, axis=0))] <= 61
    assert ev.nave == 2

    assert stc.shape[0] == inverse_operator['nsource']
    assert stc.times[0] >= fmin
    assert stc.times[-1] <= fmax
    assert 58 <= stc.times[np.argmax(np.sum(stc.data, axis=0))] <= 61

    if method in ('sLORETA', 'dSPM'):
        stc_dspm = stc
        stc_mne, _ = compute_source_psd(
            raw, inverse_operator, lambda2=1. / 9., method='MNE',
            fmin=fmin, fmax=fmax, pick_ori=pick_ori, n_fft=n_fft,
            overlap=0., return_sensor=True, dB=True)
        # normalize each source point by its power after undoing the dB
        stc_dspm.data = 10 ** (stc_dspm.data / 10.)
        stc_dspm /= stc_dspm.mean()
        stc_mne.data = 10 ** (stc_mne.data / 10.)
        stc_mne /= stc_mne.mean()
        assert_allclose(stc_dspm.data, stc_mne.data, atol=1e-4)


@testing.requires_testing_data
@pytest.mark.parametrize('method', INVERSE_METHODS)
def test_source_psd_epochs(method):
    """Test multi-taper source PSD computation in label from epochs."""
    raw = read_raw_fif(fname_data)
    inverse_operator = read_inverse_operator(fname_inv)
    label = read_label(fname_label)

    event_id, tmin, tmax = 1, -0.2, 0.5
    lambda2 = 1. / 9.
    bandwidth = 8.
    fmin, fmax = 0, 100

    picks = pick_types(raw.info, meg=True, eeg=False, stim=True,
                       ecg=True, eog=True, include=['STI 014'],
                       exclude='bads')
    reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)

    events = find_events(raw, stim_channel='STI 014')
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=reject)

    # only look at one epoch
    epochs.drop_bad()
    one_epochs = epochs[:1]

    inv = prepare_inverse_operator(inverse_operator, nave=1,
                                   lambda2=1. / 9., method="dSPM")
    # return list
    stc_psd = compute_source_psd_epochs(one_epochs, inv,
                                        lambda2=lambda2, method=method,
                                        pick_ori="normal", label=label,
                                        bandwidth=bandwidth,
                                        fmin=fmin, fmax=fmax,
                                        prepared=True)[0]

    # return generator
    stcs = compute_source_psd_epochs(one_epochs, inv,
                                     lambda2=lambda2, method=method,
                                     pick_ori="normal", label=label,
                                     bandwidth=bandwidth,
                                     fmin=fmin, fmax=fmax,
                                     return_generator=True,
                                     prepared=True)

    for stc in stcs:
        stc_psd_gen = stc

    assert_allclose(stc_psd.data, stc_psd_gen.data, atol=1e-7)

    # compare with direct computation
    stc = apply_inverse_epochs(one_epochs, inv,
                               lambda2=lambda2, method=method,
                               pick_ori="normal", label=label,
                               prepared=True)[0]

    sfreq = epochs.info['sfreq']
    psd, freqs = psd_array_multitaper(stc.data, sfreq=sfreq,
                                      bandwidth=bandwidth, fmin=fmin,
                                      fmax=fmax)

    assert_allclose(psd, stc_psd.data, atol=1e-7)
    assert_allclose(freqs, stc_psd.times)

    # Check corner cases caused by tiny bandwidth
    with pytest.raises(ValueError, match='use a value of at least'):
        compute_source_psd_epochs(
            one_epochs, inv, lambda2=lambda2, method=method,
            pick_ori="normal", label=label, bandwidth=0.01, low_bias=True,
            fmin=fmin, fmax=fmax, return_generator=False, prepared=True)


run_tests_if_main()
