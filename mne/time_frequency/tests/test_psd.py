import numpy as np
import os.path as op
from numpy.testing import assert_array_almost_equal, assert_allclose
from scipy.signal import welch
import pytest

from mne import pick_types, Epochs, read_events
from mne.io import RawArray, read_raw_fif
from mne.utils import catch_logging
from mne.time_frequency import psd_welch, psd_multitaper, psd_array_welch

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_fname = op.join(base_dir, 'test-eve.fif')


def test_psd_nan():
    """Test handling of NaN in psd_array_welch."""
    n_samples, n_fft, n_overlap = 2048, 1024, 512
    x = np.random.RandomState(0).randn(1, n_samples)
    psds, freqs = psd_array_welch(x[:, :n_fft + n_overlap], float(n_fft),
                                  n_fft=n_fft, n_overlap=n_overlap)
    x[:, n_fft + n_overlap:] = np.nan  # what Raw.get_data() will give us
    psds_2, freqs_2 = psd_array_welch(x, float(n_fft), n_fft=n_fft,
                                      n_overlap=n_overlap)
    assert_allclose(freqs, freqs_2)
    assert_allclose(psds, psds_2)
    # 1-d
    psds_2, freqs_2 = psd_array_welch(
        x[0], float(n_fft), n_fft=n_fft, n_overlap=n_overlap)
    assert_allclose(freqs, freqs_2)
    assert_allclose(psds[0], psds_2)
    # defaults
    with catch_logging() as log:
        psd_array_welch(x, float(n_fft), verbose='debug')
    log = log.getvalue()
    assert 'using 256-point FFT on 256 samples with 0 overlap' in log
    assert 'hamming window' in log


def test_psd():
    """Tests the welch and multitaper PSD."""
    raw = read_raw_fif(raw_fname)
    picks_psd = [0, 1]

    # Populate raw with sinusoids
    rng = np.random.RandomState(40)
    data = 0.1 * rng.randn(len(raw.ch_names), raw.n_times)
    freqs_sig = [8., 50.]
    for ix, freq in zip(picks_psd, freqs_sig):
        data[ix, :] += 2 * np.sin(np.pi * 2. * freq * raw.times)
    first_samp = raw._first_samps[0]
    raw = RawArray(data, raw.info)

    tmin, tmax = 0, 20  # use a few seconds of data
    fmin, fmax = 2, 70  # look at frequencies between 2 and 70Hz
    n_fft = 128

    # -- Raw --
    kws_psd = dict(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax,
                   picks=picks_psd)  # Common to all
    kws_welch = dict(n_fft=n_fft)
    kws_mt = dict(low_bias=True)
    funcs = [(psd_welch, kws_welch),
             (psd_multitaper, kws_mt)]

    for func, kws in funcs:
        kws = kws.copy()
        kws.update(kws_psd)
        kws.update(verbose='debug')
        if func is psd_welch:
            kws.update(window='hann')
        with catch_logging() as log:
            psds, freqs = func(raw, proj=False, **kws)
        log = log.getvalue()
        if func is psd_welch:
            assert f'{n_fft}-point FFT on {n_fft} samples with 0 overl' in log
            assert 'hann window' in log
        psds_proj, freqs_proj = func(raw, proj=True, **kws)

        assert psds.shape == (len(kws['picks']), len(freqs))
        assert np.sum(freqs < 0) == 0
        assert np.sum(psds < 0) == 0

        # Is power found where it should be
        ixs_max = np.argmax(psds, axis=1)
        for ixmax, ifreq in zip(ixs_max, freqs_sig):
            # Find nearest frequency to the "true" freq
            ixtrue = np.argmin(np.abs(ifreq - freqs))
            assert (np.abs(ixmax - ixtrue) < 2)

        # Make sure the projection doesn't change channels it shouldn't
        assert_array_almost_equal(psds, psds_proj)
        # Array input shouldn't work
        pytest.raises(ValueError, func, raw[:3, :20][0])

    # test n_per_seg in psd_welch (and padding)
    psds1, freqs1 = psd_welch(raw, proj=False, n_fft=128, n_per_seg=128,
                              **kws_psd)
    psds2, freqs2 = psd_welch(raw, proj=False, n_fft=256, n_per_seg=128,
                              **kws_psd)
    assert (len(freqs1) == np.floor(len(freqs2) / 2.))
    assert (psds1.shape[-1] == np.floor(psds2.shape[-1] / 2.))

    kws_psd.update(dict(n_fft=tmax * 1.1 * raw.info['sfreq']))
    with pytest.raises(ValueError, match='n_fft is not allowed to be > n_tim'):
        psd_welch(raw, proj=False, n_per_seg=None,
                  **kws_psd)
    kws_psd.update(dict(n_fft=128, n_per_seg=64, n_overlap=90))
    with pytest.raises(ValueError, match='n_overlap cannot be greater'):
        psd_welch(raw, proj=False, **kws_psd)
    with pytest.raises(ValueError, match='No frequencies found'):
        psd_array_welch(np.zeros((1, 1000)), 1000., fmin=10, fmax=1)

    # -- Epochs/Evoked --
    events = read_events(event_fname)
    events[:, 0] -= first_samp
    tmin, tmax, event_id = -0.5, 0.5, 1
    epochs = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks_psd,
                    proj=False, preload=True, baseline=None)
    evoked = epochs.average()

    tmin_full, tmax_full = -1, 1
    epochs_full = Epochs(raw, events[:10], event_id, tmin_full, tmax_full,
                         picks=picks_psd, proj=False, preload=True,
                         baseline=None)
    kws_psd = dict(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax,
                   picks=picks_psd)  # Common to all
    funcs = [(psd_welch, kws_welch),
             (psd_multitaper, kws_mt)]

    for func, kws in funcs:
        kws = kws.copy()
        kws.update(kws_psd)

        psds, freqs = func(
            epochs[:1], proj=False, **kws)
        psds_proj, freqs_proj = func(
            epochs[:1], proj=True, **kws)
        psds_f, freqs_f = func(
            epochs_full[:1], proj=False, **kws)

        # this one will fail if you add for example 0.1 to tmin
        assert_array_almost_equal(psds, psds_f, 27)
        # Make sure the projection doesn't change channels it shouldn't
        assert_array_almost_equal(psds, psds_proj, 27)

        # Is power found where it should be
        ixs_max = np.argmax(psds.mean(0), axis=1)
        for ixmax, ifreq in zip(ixs_max, freqs_sig):
            # Find nearest frequency to the "true" freq
            ixtrue = np.argmin(np.abs(ifreq - freqs))
            assert (np.abs(ixmax - ixtrue) < 2)
        assert (psds.shape == (1, len(kws['picks']), len(freqs)))
        assert (np.sum(freqs < 0) == 0)
        assert (np.sum(psds < 0) == 0)

        # Array input shouldn't work
        pytest.raises(ValueError, func, epochs.get_data())

        # Testing evoked (doesn't work w/ compute_epochs_psd)
        psds_ev, freqs_ev = func(
            evoked, proj=False, **kws)
        psds_ev_proj, freqs_ev_proj = func(
            evoked, proj=True, **kws)

        # Is power found where it should be
        ixs_max = np.argmax(psds_ev, axis=1)
        for ixmax, ifreq in zip(ixs_max, freqs_sig):
            # Find nearest frequency to the "true" freq
            ixtrue = np.argmin(np.abs(ifreq - freqs_ev))
            assert (np.abs(ixmax - ixtrue) < 2)

        # Make sure the projection doesn't change channels it shouldn't
        assert_array_almost_equal(psds_ev, psds_ev_proj, 27)
        assert (psds_ev.shape == (len(kws['picks']), len(freqs)))


@pytest.mark.parametrize('kind', ('raw', 'epochs', 'evoked'))
def test_psd_welch_average_kwarg(kind):
    """Test `average` kwarg of psd_welch()."""
    raw = read_raw_fif(raw_fname)
    picks_psd = [0, 1]

    # Populate raw with sinusoids
    rng = np.random.RandomState(40)
    data = 0.1 * rng.randn(len(raw.ch_names), raw.n_times)
    freqs_sig = [8., 50.]
    for ix, freq in zip(picks_psd, freqs_sig):
        data[ix, :] += 2 * np.sin(np.pi * 2. * freq * raw.times)
    first_samp = raw._first_samps[0]
    raw = RawArray(data, raw.info)

    tmin, tmax = -0.5, 0.5
    fmin, fmax = 0, np.inf
    n_fft = 256
    n_per_seg = 128
    n_overlap = 0

    event_id = 2
    events = read_events(event_fname)
    events[:, 0] -= first_samp

    kws = dict(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, n_fft=n_fft,
               n_per_seg=n_per_seg, n_overlap=n_overlap, picks=picks_psd)

    if kind == 'raw':
        inst = raw
    elif kind == 'epochs':
        inst = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks_psd,
                      proj=False, preload=True, baseline=None)
    elif kind == 'evoked':
        inst = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks_psd,
                      proj=False, preload=True, baseline=None).average()
    else:
        raise ValueError('Unknown parametrization passed to test, check test '
                         'for typos.')

    psds_mean, freqs_mean = psd_welch(inst=inst, average='mean', **kws)
    psds_median, freqs_median = psd_welch(inst=inst, average='median', **kws)
    psds_unagg, freqs_unagg = psd_welch(inst=inst, average=None, **kws)

    # Frequencies should be equal across all "average" types, as we feed in
    # the exact same data.
    assert_allclose(freqs_mean, freqs_median)
    assert_allclose(freqs_mean, freqs_unagg)

    # For `average=None`, the last dimension contains the un-aggregated
    # segments.
    assert psds_mean.shape == psds_median.shape
    assert psds_mean.shape == psds_unagg.shape[:-1]
    assert_allclose(psds_mean, psds_unagg.mean(axis=-1))

    # Compare with manual median calculation
    assert_allclose(psds_median, np.median(psds_unagg, axis=-1))


@pytest.mark.slowtest
def test_compares_psd():
    """Test PSD estimation on raw for plt.psd and scipy.signal.welch."""
    raw = read_raw_fif(raw_fname)

    exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053']  # bads + 2 more

    # picks MEG gradiometers
    picks = pick_types(raw.info, meg='grad', eeg=False, stim=False,
                       exclude=exclude)[:2]

    tmin, tmax = 0, 10  # use the first 60s of data
    fmin, fmax = 2, 70  # look at frequencies between 5 and 70Hz
    n_fft = 2048

    # Compute psds with the new implementation using Welch
    psds_welch, freqs_welch = psd_welch(raw, tmin=tmin, tmax=tmax, fmin=fmin,
                                        fmax=fmax, proj=False, picks=picks,
                                        n_fft=n_fft, n_jobs=1)

    # Compute psds with plt.psd
    start, stop = raw.time_as_index([tmin, tmax])
    data, times = raw[picks, start:(stop + 1)]
    out = [welch(d, fs=raw.info['sfreq'], nperseg=n_fft, noverlap=0)
           for d in data]
    freqs_mpl = out[0][0]
    psds_mpl = np.array([o[1] for o in out])

    mask = (freqs_mpl >= fmin) & (freqs_mpl <= fmax)
    freqs_mpl = freqs_mpl[mask]
    psds_mpl = psds_mpl[:, mask]

    assert_array_almost_equal(psds_welch, psds_mpl)
    assert_array_almost_equal(freqs_welch, freqs_mpl)

    assert (psds_welch.shape == (len(picks), len(freqs_welch)))
    assert (psds_mpl.shape == (len(picks), len(freqs_mpl)))

    assert (np.sum(freqs_welch < 0) == 0)
    assert (np.sum(freqs_mpl < 0) == 0)

    assert (np.sum(psds_welch < 0) == 0)
    assert (np.sum(psds_mpl < 0) == 0)
