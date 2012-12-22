import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true

from mne.filter import band_pass_filter, high_pass_filter, low_pass_filter, \
                       band_stop_filter, resample, construct_iir_filter, \
                       notch_filter

def test_notch_filters():
    """Test notch filtering
    """
    Fs = 500.0
    sig_len_secs = 60
    t = np.arange(0, sig_len_secs * Fs) / Fs
    freqs = 60 * np.arange(1, 4)

    # make a "signal"
    a = np.random.randn(sig_len_secs * Fs)
    orig_power = np.sqrt(np.mean(a ** 2))
    # make line noise
    a += np.sum([np.sin(2 * np.pi * f * t) for f in freqs], axis=0)

    # test notch filtering
    methods = ['fft', 'fft', 'iir', 'spectrum_fit']
    filter_lengths = [None, 8192, None, None]
    tols = [2, 2, 1, 3]
    for meth, fl, tol in zip(methods, filter_lengths, tols):
        print meth
        b = notch_filter(a, Fs, freqs, filter_length=fl, method=meth)
        new_power = np.sqrt(np.mean(b ** 2))
        assert_almost_equal(new_power, orig_power, tol)


def test_filters():
    """Test low-, band-, high-pass, and band-stop filters"""
    Fs = 500
    sig_len_secs = 60

    # Filtering of short signals (filter length = len(a))
    a = np.random.randn(sig_len_secs * Fs)
    bp = band_pass_filter(a, Fs, 4, 8)
    bs = band_stop_filter(a, Fs, 4 - 0.5, 8 + 0.5)
    lp = low_pass_filter(a, Fs, 8)
    hp = high_pass_filter(lp, Fs, 4)
    assert_array_almost_equal(hp, bp, 2)
    assert_array_almost_equal(bp + bs, a, 1)

    # Overlap-add filtering with a fixed filter length
    filter_length = 8192
    bp_oa = band_pass_filter(a, Fs, 4, 8, filter_length)
    bs_oa = band_stop_filter(a, Fs, 4 - 0.5, 8 + 0.5, filter_length)
    lp_oa = low_pass_filter(a, Fs, 8, filter_length)
    hp_oa = high_pass_filter(lp_oa, Fs, 4, filter_length)
    assert_array_almost_equal(hp_oa, bp_oa, 2)
    assert_array_almost_equal(bp_oa + bs_oa, a, 2)

    # The two methods should give the same result
    # As filtering for short signals uses a circular convolution (FFT) and
    # the overlap-add filter implements a linear convolution, the signal
    # boundary will be slightly different and we ignore it
    n_edge_ignore = 0
    assert_array_almost_equal(hp[n_edge_ignore:-n_edge_ignore],
                              hp_oa[n_edge_ignore:-n_edge_ignore], 2)

    # and since these are low-passed, downsampling/upsampling should be close
    n_resamp_ignore = 10
    bp_up_dn = resample(resample(bp_oa, 2, 1), 1, 2)
    assert_array_almost_equal(bp_oa[n_resamp_ignore:-n_resamp_ignore],
                              bp_up_dn[n_resamp_ignore:-n_resamp_ignore], 2)
    # make sure we don't alias
    t = np.array(range(Fs*sig_len_secs))/float(Fs)
    # make sinusoid close to the Nyquist frequency
    sig = np.sin(2*np.pi*Fs/2.2*t)
    # signal should disappear with 2x downsampling
    sig_gone = resample(sig,1,2)[n_resamp_ignore:-n_resamp_ignore]
    assert_array_almost_equal(np.zeros_like(sig_gone), sig_gone, 2)

    # let's construct some filters
    iir_params = dict(ftype='cheby1', gpass=1, gstop=20)
    iir_params = construct_iir_filter(iir_params, 40, 80, 1000, 'low')
    # this should be a third order filter
    assert_true(iir_params['a'].size - 1 == 3)
    assert_true(iir_params['b'].size - 1 == 3)
    iir_params = dict(ftype='butter', order=4)
    iir_params = construct_iir_filter(iir_params, 40, None, 1000, 'low')
    assert_true(iir_params['a'].size - 1 == 4)
    assert_true(iir_params['b'].size - 1 == 4)


def test_detrend():
    """Test zeroth and first order detrending
    """
    x = np.arange(10)
    assert_array_almost_equal(detrend(x, 1), np.zeros_like(x))
    x = np.ones(10)
    assert_array_almost_equal(detrend(x, 0), np.zeros_like(x))
