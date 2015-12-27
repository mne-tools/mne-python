import numpy as np
import warnings
import os.path as op
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true

from mne import io, pick_types, Epochs, read_events
from mne.utils import requires_version, slow_test
from mne.time_frequency import (compute_raw_psd, compute_epochs_psd,
                                psd_welch, psd_multitaper)

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_fname = op.join(base_dir, 'test-eve.fif')


@requires_version('scipy', '0.12')
def test_psd_deprecate():
    """Test PSD estimation
    """
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
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
                                      fmin=fmin, fmax=fmax, n_fft=n_fft,
                                      n_jobs=1, proj=False)
        psds_proj, freqs = compute_raw_psd(raw, tmin=tmin, tmax=tmax,
                                           picks=picks, fmin=fmin, fmax=fmax,
                                           n_fft=n_fft, n_jobs=1, proj=True)

        assert_array_almost_equal(psds, psds_proj)
        assert_true(psds.shape == (len(picks), len(freqs)))
        assert_true(np.sum(freqs < 0) == 0)
        assert_true(np.sum(psds < 0) == 0)


@requires_version('scipy', '0.12')
def test_psd():
    """Tests the welch and multitaper PSD
    """
    raw = io.Raw(raw_fname)
    exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053']  # bads + 2 more
    include = []

    tmin, tmax = 0, 10  # use the first 60s of data
    fmin, fmax = 2, 70  # look at frequencies between 5 and 70Hz
    n_fft = 128

    # picks MEG gradiometers
    picks = pick_types(raw.info, meg='grad', eeg=False, eog=True,
                       stim=False, include=include, exclude=exclude)

    # -- Raw --
    picks_mt = picks[:2]
    picks_wel = picks
    kws_psd = dict(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
    kws_welch = dict(picks=picks_wel, n_fft=n_fft)
    kws_mt = dict(picks=picks_mt, low_bias=True)
    funcs = {psd_welch: kws_welch, psd_multitaper: kws_mt}

    for func, kws in funcs.iteritems():
        kws = kws.copy()
        kws.update(kws_psd)
        psds, freqs = func(raw, proj=False, **kws)
        psds_proj, freqs_proj = func(raw, proj=True, **kws)

        assert_array_almost_equal(psds, psds_proj)
        assert_true(psds.shape == (len(kws['picks']), len(freqs)))
        assert_true(np.sum(freqs < 0) == 0)
        assert_true(np.sum(psds < 0) == 0)

    # -- Epochs/Evoked --
    events = read_events(event_fname)
    tmin, tmax, event_id = -0.5, 0.5, 1
    epochs = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0),
                    reject=dict(grad=4000e-13, eog=150e-6), proj=False,
                    preload=True)
    evoked = epochs.average()

    tmin_full, tmax_full = -1, 1
    epochs_full = Epochs(raw, events[:10], event_id, tmin_full, tmax_full,
                         picks=picks, baseline=(None, 0),
                         reject=dict(grad=4000e-13, eog=150e-6), proj=False,
                         preload=True)

    picks_wel = pick_types(epochs.info, meg=True, eeg=True, ref_meg=False,
                           exclude='bads')
    picks_mt = picks_wel[:2]

    kws_psd = dict(fmin=2, fmax=300)
    kws_welch = dict(picks=picks_wel, n_fft=512)
    kws_mt = dict(picks=picks_mt, low_bias=True)
    funcs = {psd_welch: kws_welch, psd_multitaper: kws_mt}

    for func, kws in funcs.iteritems():
        kws = kws.copy()
        kws.update(kws_psd)

        psds, freqs = func(
            epochs[:1], tmin=tmin, tmax=tmax, proj=False, **kws)
        psds_proj, freqs_proj = func(
            epochs[:1], tmin=tmin, tmax=tmax, proj=True, **kws)
        psds_f, freqs_f = func(
            epochs_full[:1], tmin=tmin, tmax=tmax, proj=False, **kws)

        psds_ev, freqs_ev = func(
            evoked, tmin=tmin, tmax=tmax, proj=False, **kws)
        psds_ev_proj, freqs_ev_proj = func(
            evoked, tmin=tmin, tmax=tmax, proj=True, **kws)

        # this one will fail if you add for example 0.1 to tmin
        assert_array_almost_equal(psds, psds_f, 27)
        assert_array_almost_equal(psds, psds_proj, 27)
        assert_array_almost_equal(psds_ev, psds_ev_proj, 27)

        assert_true(psds.shape == (1, len(kws['picks']), len(freqs)))
        assert_true(psds_ev.shape == (len(kws['picks']), len(freqs)))
        assert_true(np.sum(freqs < 0) == 0)
        assert_true(np.sum(psds < 0) == 0)


@requires_version('scipy', '0.12')
def test_psd_epochs_deprecate():
    """Test PSD estimation on epochs
    """
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
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
                             reject=dict(grad=4000e-13, eog=150e-6),
                             proj=False, preload=True)

        picks = pick_types(epochs.info, meg='grad', eeg=False, eog=True,
                           stim=False, include=include, exclude='bads')
        psds, freqs = compute_epochs_psd(epochs[:1], fmin=2, fmax=300,
                                         n_fft=n_fft, picks=picks)

        psds_t, freqs_t = compute_epochs_psd(epochs_full[:1], fmin=2, fmax=300,
                                             tmin=tmin, tmax=tmax,
                                             n_fft=n_fft, picks=picks)
        # this one will fail if you add for example 0.1 to tmin
        assert_array_almost_equal(psds, psds_t, 27)

        psds_proj, _ = compute_epochs_psd(epochs[:1].apply_proj(), fmin=2,
                                          fmax=300, n_fft=n_fft, picks=picks)

        assert_array_almost_equal(psds, psds_proj)
        assert_true(psds.shape == (1, len(picks), len(freqs)))
        assert_true(np.sum(freqs < 0) == 0)
        assert_true(np.sum(psds < 0) == 0)


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
    psds_welch, freqs_welch = psd_welch(raw, tmin=tmin, tmax=tmax,
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
