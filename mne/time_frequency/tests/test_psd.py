import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_allclose,
                           assert_array_equal)
from scipy.signal import welch
import pytest

from mne.utils import catch_logging
from mne.time_frequency import psd_array_welch, psd_array_multitaper
from mne.time_frequency.multitaper import _psd_from_mt
from mne.time_frequency.psd import _median_biases


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


def _make_psd_data():
    """Make noise data with sinusoids in 2 out of 7 channels."""
    rng = np.random.default_rng(0)
    n_chan, n_times, sfreq = 7, 8000, 1000
    data = 0.1 * rng.random((n_chan, n_times))
    times = np.arange(n_times) / sfreq
    sinusoid_freqs = [8., 50.]
    chs_with_sinusoids = [0, 1]
    for ix, freq in zip(chs_with_sinusoids, sinusoid_freqs):
        data[ix, :] += 2 * np.sin(np.pi * 2. * freq * times)
    return data, sfreq, sinusoid_freqs


@pytest.mark.parametrize(
    'psd_func, psd_kwargs',
    [(psd_array_welch, dict(n_fft=128, window='hann')),
     (psd_array_multitaper, dict(low_bias=True))])
def test_psd(psd_func, psd_kwargs):
    """Tests the welch and multitaper PSD."""
    data, sfreq, sinusoid_freqs = _make_psd_data()
    # prepare kwargs
    psd_kwargs.update(dict(fmin=2, fmax=70, verbose='debug'))
    # compute PSD and test basic conformity
    with catch_logging() as log:
        psds, freqs = psd_func(data, sfreq, **psd_kwargs)
    if psd_func is psd_array_welch:
        log = log.getvalue()
        n_fft = psd_kwargs['n_fft']
        assert f'{n_fft}-point FFT on {n_fft} samples with 0 overl' in log
        assert 'hann window' in log
    assert psds.shape == (data.shape[0], len(freqs))
    assert np.sum(freqs < 0) == 0
    assert np.sum(psds < 0) == 0
    # Is power found where it should be?
    ixs_max = np.argmax(psds, axis=1)
    for ixmax, ifreq in zip(ixs_max, sinusoid_freqs):
        # Find nearest frequency to the "true" freq
        ixtrue = np.argmin(np.abs(ifreq - freqs))
        assert (np.abs(ixmax - ixtrue) < 2)


def test_psd_array_welch_nperseg_kwarg():
    """Test n_per_seg and padding in psd_array_welch()."""
    data, sfreq, _ = _make_psd_data()
    # prepare kwargs
    kwargs = dict(fmin=2, fmax=70, n_per_seg=128)
    # test n_per_seg in psd_welch (and padding)
    psds1, freqs1 = psd_array_welch(data, sfreq, n_fft=128, **kwargs)
    psds2, freqs2 = psd_array_welch(data, sfreq, n_fft=256, **kwargs)
    assert len(freqs1) == np.floor(len(freqs2) / 2.)
    assert psds1.shape[-1] == np.floor(psds2.shape[-1] / 2.)
    # test bad n_fft
    with pytest.raises(ValueError, match='n_fft is not allowed to be > n_tim'):
        kwargs.update(n_per_seg=None)
        bad_n_fft = int(data.shape[-1] * 1.1)
        psd_array_welch(data, sfreq, n_fft=bad_n_fft, **kwargs)
    # test bad n_overlap
    with pytest.raises(ValueError, match='n_overlap cannot be greater'):
        kwargs.update(n_per_seg=64)
        psd_array_welch(data, sfreq, n_fft=128, n_overlap=90, **kwargs)
    # test bad fmin/fmax
    with pytest.raises(ValueError, match='No frequencies found'):
        psd_array_welch(data, sfreq, fmin=10, fmax=1)


def test_complex_multitaper():
    """Test complex-valued multitaper output."""
    data, sfreq, _ = _make_psd_data()
    psd_complex, freq_complex, weights = psd_array_multitaper(
        data[:4, :500], sfreq, output='complex')
    psd, freq = psd_array_multitaper(data[:4, :500], sfreq, output='power')
    assert_array_equal(freq_complex, freq)
    assert psd_complex.ndim == 3  # channels x tapers x freqs
    psd_from_complex = _psd_from_mt(psd_complex, weights)
    assert_allclose(psd_from_complex, psd)


# Copied from SciPy
def _median_bias(n):
    ii_2 = 2 * np.arange(1., (n - 1) // 2 + 1)
    return 1 + np.sum(1. / (ii_2 + 1) - 1. / ii_2)


@pytest.mark.parametrize('crop', (False, True))
def test_psd_welch_average_kwarg(crop):
    """Test `average` kwarg of psd_array_welch()."""
    data, sfreq, _ = _make_psd_data()
    # prepare kwargs
    n_per_seg = 32
    kwargs = dict(fmin=0, fmax=np.inf, n_fft=64, n_per_seg=n_per_seg,
                  n_overlap=0)
    # optionally crop data by n_per_seg so that we are sure to test both an
    # odd number and an even number of estimates (for median bias)
    if crop:
        data = data[..., :-n_per_seg]
    # run with average=mean/median/None
    psds_mean, freqs_mean = psd_array_welch(
        data, sfreq, average='mean', **kwargs)
    psds_median, freqs_median = psd_array_welch(
        data, sfreq, average='median', **kwargs)
    psds_unagg, freqs_unagg = psd_array_welch(
        data, sfreq, average=None, **kwargs)
    # Frequencies should be equal across all "average" types, as we feed in
    # the exact same data.
    assert_array_equal(freqs_mean, freqs_median)
    assert_array_equal(freqs_mean, freqs_unagg)
    # For `average=None`, the last dimension contains the un-aggregated
    # segments.
    assert psds_mean.shape == psds_median.shape
    assert psds_mean.shape == psds_unagg.shape[:-1]
    assert_array_equal(psds_mean, psds_unagg.mean(axis=-1))
    # Compare with manual median calculation (_median_bias copied from SciPy)
    bias = _median_bias(psds_unagg.shape[-1])
    assert_allclose(psds_median, np.median(psds_unagg, axis=-1) / bias)
    # check shape of unagg
    n_chan, n_times = data.shape
    n_freq = len(freqs_unagg)
    n_segs = np.ceil(n_times / n_per_seg).astype(int)
    assert n_segs % 2 == (1 if crop else 0)
    assert psds_unagg.shape == (n_chan, n_freq, n_segs)


@pytest.mark.parametrize('n', (2, 3, 5, 8, 12, 13, 14, 15))
def test_median_biases(n):
    """Test vectorization of median_biases."""
    want_biases = np.concatenate(
        ([1., 1.], [_median_bias(ii) for ii in range(2, n + 1)]))
    got_biases = _median_biases(n)
    assert_allclose(want_biases, got_biases)
    assert_allclose(got_biases[n], _median_bias(n))
    assert_allclose(got_biases[:3], 1.)


@pytest.mark.slowtest
def test_compares_psd():
    """Test PSD estimation on raw for plt.psd and scipy.signal.welch."""
    data, sfreq, _ = _make_psd_data()
    n_fft = 2048
    fmin, fmax = 2, 70
    # Compute PSD with psd_array_welch
    psds_mne, freqs_mne = psd_array_welch(
        data, sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft)
    # Compute psds with scipy.signal.welch
    freqs_scipy, psds_scipy = welch(
        data, fs=sfreq, nperseg=n_fft, noverlap=0, window='hamming')
    # restrict to the relevant frequencies
    mask = (freqs_scipy >= fmin) & (freqs_scipy <= fmax)
    freqs_scipy = freqs_scipy[mask]
    psds_scipy = psds_scipy[:, mask]
    # make sure they match
    assert_array_almost_equal(psds_mne, psds_scipy)
    assert_array_equal(freqs_mne, freqs_scipy)
    assert (psds_mne.shape == (data.shape[0], len(freqs_mne)))
    assert (psds_scipy.shape == (data.shape[0], len(freqs_scipy)))
    assert (np.sum(freqs_mne < 0) == 0)
    assert (np.sum(freqs_scipy < 0) == 0)
    assert (np.sum(psds_mne < 0) == 0)
    assert (np.sum(psds_scipy < 0) == 0)
