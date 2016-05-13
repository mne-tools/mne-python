import numpy as np
import warnings
import os.path as op
from numpy.testing import assert_array_almost_equal, assert_raises
from nose.tools import assert_true

from mne import io, pick_types, Epochs, read_events
from mne.io import RawArray
from mne.utils import requires_version, slow_test
from mne.time_frequency import (compute_raw_psd, compute_epochs_psd,
                                psd_welch, psd_multitaper)

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_fname = op.join(base_dir, 'test-eve.fif')


@requires_version('scipy', '0.12')
def test_psd():
    """Tests the welch and multitaper PSD
    """
    raw = io.read_raw_fif(raw_fname)
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
             (psd_multitaper, kws_mt),
             (compute_raw_psd, kws_welch)]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        for func, kws in funcs:
            kws = kws.copy()
            kws.update(kws_psd)
            psds, freqs = func(raw, proj=False, **kws)
            psds_proj, freqs_proj = func(raw, proj=True, **kws)

            assert_true(psds.shape == (len(kws['picks']), len(freqs)))
            assert_true(np.sum(freqs < 0) == 0)
            assert_true(np.sum(psds < 0) == 0)

            # Is power found where it should be
            ixs_max = np.argmax(psds, axis=1)
            for ixmax, ifreq in zip(ixs_max, freqs_sig):
                # Find nearest frequency to the "true" freq
                ixtrue = np.argmin(np.abs(ifreq - freqs))
                assert_true(np.abs(ixmax - ixtrue) < 2)

            # Make sure the projection doesn't change channels it shouldn't
            assert_array_almost_equal(psds, psds_proj)
            # Array input shouldn't work
            assert_raises(ValueError, func, raw[:3, :20][0])
        assert_true(len(w), 3)

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
             (psd_multitaper, kws_mt),
             (compute_epochs_psd, kws_welch)]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
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
                assert_true(np.abs(ixmax - ixtrue) < 2)
            assert_true(psds.shape == (1, len(kws['picks']), len(freqs)))
            assert_true(np.sum(freqs < 0) == 0)
            assert_true(np.sum(psds < 0) == 0)

            # Array input shouldn't work
            assert_raises(ValueError, func, epochs.get_data())

            if func is not compute_epochs_psd:
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
                    assert_true(np.abs(ixmax - ixtrue) < 2)

                # Make sure the projection doesn't change channels it shouldn't
                assert_array_almost_equal(psds_ev, psds_ev_proj, 27)
                assert_true(psds_ev.shape == (len(kws['picks']), len(freqs)))
        assert_true(len(w), 3)


@slow_test
@requires_version('scipy', '0.12')
def test_compares_psd():
    """Test PSD estimation on raw for plt.psd and scipy.signal.welch
    """
    raw = io.read_raw_fif(raw_fname)

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
