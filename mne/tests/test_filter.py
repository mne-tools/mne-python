import os.path as op

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_almost_equal,
                           assert_array_equal, assert_allclose,
                           assert_array_less)
import pytest
from scipy.signal import resample as sp_resample, butter, freqz, sosfreqz

from mne import create_info
from mne.fixes import fft, fftfreq
from mne.io import RawArray, read_raw_fif
from mne.io.pick import _DATA_CH_TYPES_SPLIT
from mne.filter import (filter_data, resample, _resample_stim_channels,
                        construct_iir_filter, notch_filter, detrend,
                        _overlap_add_filter, _smart_pad, design_mne_c_filter,
                        estimate_ringing_samples, create_filter)

from mne.utils import (sum_squared, run_tests_if_main,
                       catch_logging, requires_mne, run_subprocess)


def test_filter_array():
    """Test filtering an array."""
    for data in (np.zeros((11, 1, 10)), np.zeros((9, 1, 10))):
        filter_data(data, 512., 8, 12, method='iir',
                    iir_params=dict(ftype='butterworth', order=2))


@requires_mne
def test_mne_c_design(tmpdir):
    """Test MNE-C filter design."""
    tempdir = str(tmpdir)
    temp_fname = op.join(tempdir, 'test_raw.fif')
    out_fname = op.join(tempdir, 'test_c_raw.fif')
    x = np.zeros((1, 10001))
    x[0, 5000] = 1.
    time_sl = slice(5000 - 4096, 5000 + 4097)
    sfreq = 1000.
    RawArray(x, create_info(1, sfreq, 'eeg')).save(temp_fname)

    tols = dict(rtol=1e-4, atol=1e-4)
    cmd = ('mne_process_raw', '--projoff', '--raw', temp_fname,
           '--save', out_fname)
    run_subprocess(cmd)
    h = design_mne_c_filter(sfreq, None, 40)
    h_c = read_raw_fif(out_fname)[0][0][0][time_sl]
    assert_allclose(h, h_c, **tols)

    run_subprocess(cmd + ('--highpass', '5', '--highpassw', '2.5'))
    h = design_mne_c_filter(sfreq, 5, 40, 2.5)
    h_c = read_raw_fif(out_fname)[0][0][0][time_sl]
    assert_allclose(h, h_c, **tols)

    run_subprocess(cmd + ('--lowpass', '1000', '--highpass', '10'))
    h = design_mne_c_filter(sfreq, 10, None, verbose=True)
    h_c = read_raw_fif(out_fname)[0][0][0][time_sl]
    assert_allclose(h, h_c, **tols)


def test_estimate_ringing():
    """Test our ringing estimation function."""
    # Actual values might differ based on system, so let's be approximate
    for kind in ('ba', 'sos'):
        for thresh, lims in ((0.1, (30, 60)),  # 47
                             (0.01, (300, 600)),  # 475
                             (0.001, (3000, 6000)),  # 4758
                             (0.0001, (30000, 60000))):  # 37993
            n_ring = estimate_ringing_samples(butter(3, thresh, output=kind))
            assert lims[0] <= n_ring <= lims[1], (
                '%s %s: %s <= %s <= %s'
                % (kind, thresh, lims[0], n_ring, lims[1]))
    with pytest.warns(RuntimeWarning, match='properly estimate'):
        assert estimate_ringing_samples(butter(4, 0.00001)) == 100000


def test_1d_filter():
    """Test our private overlap-add filtering function."""
    # make some random signals and filters
    rng = np.random.RandomState(0)
    for n_signal in (1, 2, 3, 5, 10, 20, 40):
        x = rng.randn(n_signal)
        for n_filter in (1, 2, 3, 5, 10, 11, 20, 21, 40, 41, 100, 101):
            for filter_type in ('identity', 'random'):
                if filter_type == 'random':
                    h = rng.randn(n_filter)
                else:  # filter_type == 'identity'
                    h = np.concatenate([[1.], np.zeros(n_filter - 1)])
                # ensure we pad the signal the same way for both filters
                n_pad = n_filter - 1
                x_pad = _smart_pad(x, (n_pad, n_pad))
                for phase in ('zero', 'linear', 'zero-double'):
                    # compute our expected result the slow way
                    if phase == 'zero':
                        # only allow zero-phase for odd-length filters
                        if n_filter % 2 == 0:
                            pytest.raises(RuntimeError, _overlap_add_filter,
                                          x[np.newaxis], h, phase=phase)
                            continue
                        shift = (len(h) - 1) // 2
                        x_expected = np.convolve(x_pad, h)
                        x_expected = x_expected[shift:len(x_expected) - shift]
                    elif phase == 'zero-double':
                        shift = len(h) - 1
                        x_expected = np.convolve(x_pad, h)
                        x_expected = np.convolve(x_expected[::-1], h)[::-1]
                        x_expected = x_expected[shift:len(x_expected) - shift]
                        shift = 0
                    else:
                        shift = 0
                        x_expected = np.convolve(x_pad, h)
                        x_expected = x_expected[:len(x_expected) - len(h) + 1]
                    # remove padding
                    if n_pad > 0:
                        x_expected = x_expected[n_pad:len(x_expected) - n_pad]
                    assert len(x_expected) == len(x)
                    # make sure we actually set things up reasonably
                    if filter_type == 'identity':
                        out = x_pad.copy()
                        out = out[shift + n_pad:]
                        out = out[:len(x)]
                        out = np.concatenate((out, np.zeros(max(len(x) -
                                                                len(out), 0))))
                        assert len(out) == len(x)
                        assert_allclose(out, x_expected)
                    assert len(x_expected) == len(x)

                    # compute our version
                    for n_fft in (None, 32, 128, 129, 1023, 1024, 1025, 2048):
                        # need to use .copy() b/c signal gets modified inplace
                        x_copy = x[np.newaxis, :].copy()
                        min_fft = 2 * n_filter - 1
                        if phase == 'zero-double':
                            min_fft = 2 * min_fft - 1
                        if n_fft is not None and n_fft < min_fft:
                            pytest.raises(ValueError, _overlap_add_filter,
                                          x_copy, h, n_fft, phase=phase)
                        else:
                            x_filtered = _overlap_add_filter(
                                x_copy, h, n_fft, phase=phase)[0]
                            assert_allclose(x_filtered, x_expected, atol=1e-13)


def test_iir_stability():
    """Test IIR filter stability check."""
    sig = np.random.RandomState(0).rand(1000)
    sfreq = 1000
    # This will make an unstable filter, should throw RuntimeError
    pytest.raises(RuntimeError, filter_data, sig, sfreq, 0.6, None,
                  method='iir', iir_params=dict(ftype='butter', order=8,
                                                output='ba'))
    # This one should work just fine
    filter_data(sig, sfreq, 0.6, None, method='iir',
                iir_params=dict(ftype='butter', order=8, output='sos'))
    # bad system type
    pytest.raises(ValueError, filter_data, sig, sfreq, 0.6, None, method='iir',
                  iir_params=dict(ftype='butter', order=8, output='foo'))
    # missing ftype
    pytest.raises(RuntimeError, filter_data, sig, sfreq, 0.6, None,
                  method='iir', iir_params=dict(order=8, output='sos'))
    # bad ftype
    pytest.raises(RuntimeError, filter_data, sig, sfreq, 0.6, None,
                  method='iir',
                  iir_params=dict(order=8, ftype='foo', output='sos'))
    # missing gstop
    pytest.raises(RuntimeError, filter_data, sig, sfreq, 0.6, None,
                  method='iir', iir_params=dict(gpass=0.5, output='sos'))
    # can't pass iir_params if method='fft'
    pytest.raises(ValueError, filter_data, sig, sfreq, 0.1, None,
                  method='fft', iir_params=dict(ftype='butter', order=2,
                                                output='sos'))
    # method must be string
    pytest.raises(TypeError, filter_data, sig, sfreq, 0.1, None,
                  method=1)
    # unknown method
    pytest.raises(ValueError, filter_data, sig, sfreq, 0.1, None,
                  method='blah')
    # bad iir_params
    pytest.raises(TypeError, filter_data, sig, sfreq, 0.1, None,
                  method='iir', iir_params='blah')
    pytest.raises(ValueError, filter_data, sig, sfreq, 0.1, None,
                  method='fir', iir_params=dict())

    # should pass because default trans_bandwidth is not relevant
    iir_params = dict(ftype='butter', order=2, output='sos')
    x_sos = filter_data(sig, 250, 0.5, None, method='iir',
                        iir_params=iir_params)
    iir_params_sos = construct_iir_filter(iir_params, f_pass=0.5, sfreq=250,
                                          btype='highpass')
    x_sos_2 = filter_data(sig, 250, 0.5, None, method='iir',
                          iir_params=iir_params_sos)
    assert_allclose(x_sos[100:-100], x_sos_2[100:-100])
    x_ba = filter_data(sig, 250, 0.5, None, method='iir',
                       iir_params=dict(ftype='butter', order=2, output='ba'))
    # Note that this will fail for higher orders (e.g., 6) showing the
    # hopefully decreased numerical error of SOS
    assert_allclose(x_sos[100:-100], x_ba[100:-100])


def test_notch_filters():
    """Test notch filters."""
    # let's use an ugly, prime sfreq for fun
    rng = np.random.RandomState(0)
    sfreq = 487.0
    sig_len_secs = 20
    t = np.arange(0, int(sig_len_secs * sfreq)) / sfreq
    freqs = np.arange(60, 241, 60)

    # make a "signal"
    a = rng.randn(int(sig_len_secs * sfreq))
    orig_power = np.sqrt(np.mean(a ** 2))
    # make line noise
    a += np.sum([np.sin(2 * np.pi * f * t) for f in freqs], axis=0)

    # only allow None line_freqs with 'spectrum_fit' mode
    pytest.raises(ValueError, notch_filter, a, sfreq, None, 'fft')
    pytest.raises(ValueError, notch_filter, a, sfreq, None, 'iir')
    methods = ['spectrum_fit', 'spectrum_fit', 'fft', 'fft', 'iir']
    filter_lengths = ['auto', 'auto', 'auto', 8192, 'auto']
    line_freqs = [None, freqs, freqs, freqs, freqs]
    tols = [2, 1, 1, 1]
    for meth, lf, fl, tol in zip(methods, line_freqs, filter_lengths, tols):
        with catch_logging() as log_file:
            with pytest.warns(None):
                b = notch_filter(a, sfreq, lf, fl, method=meth,
                                 fir_design='firwin', verbose=True)
        if lf is None:
            out = log_file.getvalue().split('\n')[:-1]
            if len(out) != 2 and len(out) != 3:  # force_serial: len(out) == 3
                raise ValueError('Detected frequencies not logged properly')
            out = np.fromstring(out[-1], sep=', ')
            assert_array_almost_equal(out, freqs)
        new_power = np.sqrt(sum_squared(b) / b.size)
        assert_almost_equal(new_power, orig_power, tol)


def test_resample():
    """Test resampling."""
    rng = np.random.RandomState(0)
    x = rng.normal(0, 1, (10, 10, 10))
    x_rs = resample(x, 1, 2, 10)
    assert x.shape == (10, 10, 10)
    assert x_rs.shape == (10, 10, 5)

    x_2 = x.swapaxes(0, 1)
    x_2_rs = resample(x_2, 1, 2, 10)
    assert_array_equal(x_2_rs.swapaxes(0, 1), x_rs)

    x_3 = x.swapaxes(0, 2)
    x_3_rs = resample(x_3, 1, 2, 10, 0)
    assert_array_equal(x_3_rs.swapaxes(0, 2), x_rs)

    # make sure we cast to array if necessary
    assert_array_equal(resample([0., 0.], 2, 1), [0., 0., 0., 0.])


def test_resample_scipy():
    """Test resampling against SciPy."""
    n_jobs_test = (1, 'cuda')
    for window in ('boxcar', 'hann'):
        for N in (100, 101, 102, 103):
            x = np.arange(N).astype(float)
            err_msg = '%s: %s' % (N, window)
            x_2_sp = sp_resample(x, 2 * N, window=window)
            for n_jobs in n_jobs_test:
                x_2 = resample(x, 2, 1, 0, window=window, n_jobs=n_jobs)
                assert_allclose(x_2, x_2_sp, atol=1e-12, err_msg=err_msg)
            new_len = int(round(len(x) * (1. / 2.)))
            x_p5_sp = sp_resample(x, new_len, window=window)
            for n_jobs in n_jobs_test:
                x_p5 = resample(x, 1, 2, 0, window=window, n_jobs=n_jobs)
                assert_allclose(x_p5, x_p5_sp, atol=1e-12, err_msg=err_msg)


@pytest.mark.parametrize('n_jobs', (2, 'cuda'))
def test_n_jobs(n_jobs):
    """Test resampling against SciPy."""
    x = np.random.RandomState(0).randn(4, 100)
    y1 = resample(x, 2, 1, n_jobs=1)
    y2 = resample(x, 2, 1, n_jobs=n_jobs)
    assert_allclose(y1, y2)
    y1 = filter_data(x, 100., 0, 40, n_jobs=1)
    y2 = filter_data(x, 100., 0, 40, n_jobs=n_jobs)
    assert_allclose(y1, y2)


def test_resamp_stim_channel():
    """Test resampling of stim channels."""
    # Downsampling
    assert_array_equal(
        _resample_stim_channels([1, 0, 0, 0, 2, 0, 0, 0], 1, 2),
        [[1, 0, 2, 0]])
    assert_array_equal(
        _resample_stim_channels([1, 0, 0, 0, 2, 0, 0, 0], 1, 1.5),
        [[1, 0, 0, 2, 0]])
    assert_array_equal(
        _resample_stim_channels([1, 0, 0, 1, 2, 0, 0, 1], 1, 2),
        [[1, 1, 2, 1]])

    # Upsampling
    assert_array_equal(
        _resample_stim_channels([1, 2, 3], 2, 1), [[1, 1, 2, 2, 3, 3]])
    assert_array_equal(
        _resample_stim_channels([1, 2, 3], 2.5, 1), [[1, 1, 1, 2, 2, 3, 3, 3]])

    # Proper number of samples in stim channel resampling from io/base.py
    data_chunk = np.zeros((1, 315600))
    for new_data_len in (52598, 52599, 52600, 52601, 315599, 315600):
        new_data = _resample_stim_channels(data_chunk, new_data_len,
                                           data_chunk.shape[1])
        assert new_data.shape[1] == new_data_len


def test_resample_raw():
    """Test resampling using RawArray."""
    x = np.zeros((1, 1001))
    sfreq = 2048.
    raw = RawArray(x, create_info(1, sfreq, 'eeg'))
    raw.resample(128, npad=10)
    data = raw.get_data()
    assert data.shape == (1, 63)


@pytest.mark.slowtest
def test_filters():
    """Test low-, band-, high-pass, and band-stop filters plus resampling."""
    rng = np.random.RandomState(0)
    sfreq = 100
    sig_len_secs = 15

    a = rng.randn(2, sig_len_secs * sfreq)

    # let's test our catchers
    for fl in ['blah', [0, 1], 1000.5, '10ss', '10']:
        pytest.raises(ValueError, filter_data, a, sfreq, 4, 8, None, fl,
                      1.0, 1.0, fir_design='firwin')
    for nj in ['blah', 0.5]:
        pytest.raises(ValueError, filter_data, a, sfreq, 4, 8, None, 1000,
                      1.0, 1.0, n_jobs=nj, phase='zero', fir_design='firwin')
    pytest.raises(ValueError, filter_data, a, sfreq, 4, 8, None, 100,
                  1., 1., fir_window='foo')
    pytest.raises(ValueError, filter_data, a, sfreq, 4, 8, None, 10,
                  1., 1., fir_design='firwin')  # too short
    # > Nyq/2
    pytest.raises(ValueError, filter_data, a, sfreq, 4, sfreq / 2., None,
                  100, 1.0, 1.0, fir_design='firwin')
    pytest.raises(ValueError, filter_data, a, sfreq, -1, None, None,
                  100, 1.0, 1.0, fir_design='firwin')
    # these should work
    create_filter(None, sfreq, None, None)
    create_filter(a, sfreq, None, None, fir_design='firwin')
    create_filter(a, sfreq, None, None, method='iir')

    # check our short-filter warning:
    with pytest.warns(RuntimeWarning, match='attenuation'):
        # Warning for low attenuation
        filter_data(a, sfreq, 1, 8, filter_length=256, fir_design='firwin2')
    with pytest.warns(RuntimeWarning, match='Increase filter_length'):
        # Warning for too short a filter
        filter_data(a, sfreq, 1, 8, filter_length='0.5s', fir_design='firwin2')

    # try new default and old default
    freqs = fftfreq(a.shape[-1], 1. / sfreq)
    A = np.abs(fft(a))
    kwargs = dict(fir_design='firwin')
    for fl in ['auto', '10s', '5000ms', 1024, 1023]:
        bp = filter_data(a, sfreq, 4, 8, None, fl, 1.0, 1.0, **kwargs)
        bs = filter_data(a, sfreq, 8 + 1.0, 4 - 1.0, None, fl, 1.0, 1.0,
                         **kwargs)
        lp = filter_data(a, sfreq, None, 8, None, fl, 10, 1.0, n_jobs=2,
                         **kwargs)
        hp = filter_data(lp, sfreq, 4, None, None, fl, 1.0, 10, **kwargs)
        assert_allclose(hp, bp, rtol=1e-3, atol=2e-3)
        assert_allclose(bp + bs, a, rtol=1e-3, atol=1e-3)
        # Sanity check ttenuation
        mask = (freqs > 5.5) & (freqs < 6.5)
        assert_allclose(np.mean(np.abs(fft(bp)[:, mask]) / A[:, mask]),
                        1., atol=0.02)
        assert_allclose(np.mean(np.abs(fft(bs)[:, mask]) / A[:, mask]),
                        0., atol=0.2)
        # now the minimum-phase versions
        bp = filter_data(a, sfreq, 4, 8, None, fl, 1.0, 1.0,
                         phase='minimum', **kwargs)
        bs = filter_data(a, sfreq, 8 + 1.0, 4 - 1.0, None, fl, 1.0, 1.0,
                         phase='minimum', **kwargs)
        assert_allclose(np.mean(np.abs(fft(bp)[:, mask]) / A[:, mask]),
                        1., atol=0.11)
        assert_allclose(np.mean(np.abs(fft(bs)[:, mask]) / A[:, mask]),
                        0., atol=0.3)

    # and since these are low-passed, downsampling/upsampling should be close
    n_resamp_ignore = 10
    bp_up_dn = resample(resample(bp, 2, 1, n_jobs=2), 1, 2, n_jobs=2)
    assert_array_almost_equal(bp[n_resamp_ignore:-n_resamp_ignore],
                              bp_up_dn[n_resamp_ignore:-n_resamp_ignore], 2)
    # note that on systems without CUDA, this line serves as a test for a
    # graceful fallback to n_jobs=1
    bp_up_dn = resample(resample(bp, 2, 1, n_jobs='cuda'), 1, 2, n_jobs='cuda')
    assert_array_almost_equal(bp[n_resamp_ignore:-n_resamp_ignore],
                              bp_up_dn[n_resamp_ignore:-n_resamp_ignore], 2)
    # test to make sure our resamling matches scipy's
    bp_up_dn = sp_resample(sp_resample(bp, 2 * bp.shape[-1], axis=-1,
                                       window='boxcar'),
                           bp.shape[-1], window='boxcar', axis=-1)
    assert_array_almost_equal(bp[n_resamp_ignore:-n_resamp_ignore],
                              bp_up_dn[n_resamp_ignore:-n_resamp_ignore], 2)

    # make sure we don't alias
    t = np.array(list(range(sfreq * sig_len_secs))) / float(sfreq)
    # make sinusoid close to the Nyquist frequency
    sig = np.sin(2 * np.pi * sfreq / 2.2 * t)
    # signal should disappear with 2x downsampling
    sig_gone = resample(sig, 1, 2)[n_resamp_ignore:-n_resamp_ignore]
    assert_array_almost_equal(np.zeros_like(sig_gone), sig_gone, 2)

    # let's construct some filters
    iir_params = dict(ftype='cheby1', gpass=1, gstop=20, output='ba')
    iir_params = construct_iir_filter(iir_params, 40, 80, 1000, 'low')
    # this should be a third order filter
    assert iir_params['a'].size - 1 == 3
    assert iir_params['b'].size - 1 == 3
    iir_params = dict(ftype='butter', order=4, output='ba')
    iir_params = construct_iir_filter(iir_params, 40, None, 1000, 'low')
    assert iir_params['a'].size - 1 == 4
    assert iir_params['b'].size - 1 == 4
    iir_params = dict(ftype='cheby1', gpass=1, gstop=20)
    iir_params = construct_iir_filter(iir_params, 40, 80, 1000, 'low')
    # this should be a third order filter, which requires 2 SOS ((2, 6))
    assert iir_params['sos'].shape == (2, 6)
    iir_params = dict(ftype='butter', order=4, output='sos')
    iir_params = construct_iir_filter(iir_params, 40, None, 1000, 'low')
    assert iir_params['sos'].shape == (2, 6)

    # check that picks work for 3d array with one channel and picks=[0]
    a = rng.randn(5 * sfreq, 5 * sfreq)
    b = a[:, None, :]

    a_filt = filter_data(a, sfreq, 4, 8, None, 400, 2.0, 2.0,
                         fir_design='firwin')
    b_filt = filter_data(b, sfreq, 4, 8, [0], 400, 2.0, 2.0,
                         fir_design='firwin')

    assert_array_equal(a_filt[:, None, :], b_filt)

    # check for n-dimensional case
    a = rng.randn(2, 2, 2, 2)
    with pytest.warns(RuntimeWarning, match='longer'):
        pytest.raises(ValueError, filter_data, a, sfreq, 4, 8,
                      np.array([0, 1]), 100, 1.0, 1.0)

    # check corner case (#4693)
    h = create_filter(
        np.empty(10000), 1000., l_freq=None, h_freq=55.,
        h_trans_bandwidth=0.5, method='fir', phase='zero-double',
        fir_design='firwin', verbose=True)
    assert len(h) == 6601


def test_filter_auto():
    """Test filter auto parameters."""
    # test that our overlap-add filtering doesn't introduce strange
    # artifacts (from mne_analyze mailing list 2015/06/25)
    N = 300
    sfreq = 100.
    lp = 10.
    sine_freq = 1.
    x = np.ones(N)
    t = np.arange(N) / sfreq
    x += np.sin(2 * np.pi * sine_freq * t)
    x_orig = x.copy()
    for pad in ('reflect_limited', 'reflect', 'edge'):
        for fir_design in ('firwin2', 'firwin'):
            kwargs = dict(fir_design=fir_design, pad=pad)
            x = x_orig.copy()
            x_filt = filter_data(x, sfreq, None, lp, **kwargs)
            assert_array_equal(x, x_orig)
            n_edge = 10
            assert_allclose(x[n_edge:-n_edge], x_filt[n_edge:-n_edge],
                            atol=1e-2)
            assert_array_equal(x_filt, filter_data(x, sfreq, None, lp, None,
                                                   **kwargs))
            assert_array_equal(x, x_orig)
            assert_array_equal(x_filt, filter_data(x, sfreq, None, lp,
                                                   **kwargs))
            assert_array_equal(x, x_orig)
            assert_array_equal(x_filt, filter_data(x, sfreq, None, lp,
                                                   copy=False, **kwargs))
            assert_array_equal(x, x_filt)

    # degenerate conditions
    pytest.raises(ValueError, filter_data, x, -sfreq, 1, 10)
    pytest.raises(ValueError, filter_data, x, sfreq, 1, sfreq * 0.75)
    with pytest.raises(ValueError, match='Data to be filtered must be real'):
        filter_data(x.astype(np.float32), sfreq, None, 10)
    with pytest.raises(ValueError, match='Data to be filtered must be real'):
        filter_data(1j, 1000., None, 40.)


def test_cuda_fir():
    """Test CUDA-based filtering."""
    # Using `n_jobs='cuda'` on a non-CUDA system should be fine,
    # as it should fall back to using n_jobs=1.
    rng = np.random.RandomState(0)
    sfreq = 500
    sig_len_secs = 20
    a = rng.randn(sig_len_secs * sfreq)
    kwargs = dict(fir_design='firwin')

    with catch_logging() as log_file:
        for fl in ['auto', '10s', 2048]:
            args = [a, sfreq, 4, 8, None, fl, 1.0, 1.0]
            bp = filter_data(*args, **kwargs)
            bp_c = filter_data(*args, n_jobs='cuda', verbose='info', **kwargs)
            assert_array_almost_equal(bp, bp_c, 12)

            args = [a, sfreq, 8 + 1.0, 4 - 1.0, None, fl, 1.0, 1.0]
            bs = filter_data(*args, **kwargs)
            bs_c = filter_data(*args, n_jobs='cuda', verbose='info', **kwargs)
            assert_array_almost_equal(bs, bs_c, 12)

            args = [a, sfreq, None, 8, None, fl, 1.0]
            lp = filter_data(*args, **kwargs)
            lp_c = filter_data(*args, n_jobs='cuda', verbose='info', **kwargs)
            assert_array_almost_equal(lp, lp_c, 12)

            args = [lp, sfreq, 4, None, None, fl, 1.0]
            hp = filter_data(*args, **kwargs)
            hp_c = filter_data(*args, n_jobs='cuda', verbose='info', **kwargs)
            assert_array_almost_equal(hp, hp_c, 12)

    # check to make sure we actually used CUDA
    out = log_file.getvalue().split('\n')[:-1]
    # triage based on whether or not we actually expected to use CUDA
    from mne.cuda import _cuda_capable  # allow above funs to set it
    tot = 12 if _cuda_capable else 0
    assert sum(['Using CUDA for FFT FIR filtering' in o for o in out]) == tot
    if not _cuda_capable:
        pytest.skip('CUDA not enabled')


def test_cuda_resampling():
    """Test CUDA resampling."""
    rng = np.random.RandomState(0)
    for window in ('boxcar', 'triang'):
        for N in (997, 1000):  # one prime, one even
            a = rng.randn(2, N)
            for fro, to in ((1, 2), (2, 1), (1, 3), (3, 1)):
                a1 = resample(a, fro, to, n_jobs=1, npad='auto',
                              window=window)
                a2 = resample(a, fro, to, n_jobs='cuda', npad='auto',
                              window=window)
                assert_allclose(a1, a2, rtol=1e-7, atol=1e-14)
    assert_array_almost_equal(a1, a2, 14)
    assert_array_equal(resample(np.zeros(2), 2, 1, n_jobs='cuda'), np.zeros(4))


def test_detrend():
    """Test zeroth and first order detrending."""
    x = np.arange(10)
    assert_array_almost_equal(detrend(x, 1), np.zeros_like(x))
    x = np.ones(10)
    assert_array_almost_equal(detrend(x, 0), np.zeros_like(x))


@pytest.mark.parametrize('output', ('ba', 'sos'))
@pytest.mark.parametrize('ftype', ('butter', 'bessel', 'ellip'))
@pytest.mark.parametrize('btype', ('lowpass', 'bandpass'))
@pytest.mark.parametrize('order', (1, 4))
def test_reporting_iir(ftype, btype, order, output):
    """Test IIR filter reporting."""
    fs = 1000.
    l_freq = 1. if btype == 'bandpass' else None
    iir_params = dict(ftype=ftype, order=order, output=output)
    rs = 20 if order == 1 else 80
    if ftype == 'ellip':
        iir_params['rp'] = 3  # dB
        iir_params['rs'] = rs  # attenuation
        pass_tol = np.log10(iir_params['rp']) + 0.01
    else:
        pass_tol = 0.2
    with catch_logging() as log:
        x = create_filter(None, fs, l_freq, 40., method='iir',
                          iir_params=iir_params, verbose=True)
    order_eff = order * (1 + (btype == 'bandpass'))
    if output == 'ba':
        assert len(x['b']) == order_eff + 1
    log = log.getvalue()
    keys = [
        'IIR',
        'zero-phase',
        'two-pass forward and reverse',
        'non-causal',
        btype,
        ftype,
        'Filter order %d' % (order_eff * 2,),
        'Cutoff ' if btype == 'lowpass' else 'Cutoffs ',
    ]
    dB_decade = -27.74
    if ftype == 'ellip':
        dB_cutoff = -6.0
    elif order == 1 or ftype == 'butter':
        dB_cutoff = -6.02
    else:
        assert ftype == 'bessel'
        assert order == 4
        dB_cutoff = -15.16
    if btype == 'lowpass':
        keys += ['%0.2f dB' % (dB_cutoff,)]
    for key in keys:
        assert key.lower() in log.lower()
    # Verify some of the filter properties
    if output == 'ba':
        w, h = freqz(x['b'], x['a'], worN=10000)
    else:
        w, h = sosfreqz(x['sos'], worN=10000)
    w *= fs / (2 * np.pi)
    h = np.abs(h)
    # passband
    passes = [np.argmin(np.abs(w - 20))]
    # stopband
    decades = [np.argmin(np.abs(w - 400.))]  # one decade
    # transition
    edges = [np.argmin(np.abs(w - 40.))]
    # put these where they belong based on filter type
    assert w[0] == 0.
    idx_0p1 = np.argmin(np.abs(w - 0.1))
    idx_1 = np.argmin(np.abs(w - 1.))
    if btype == 'bandpass':
        edges += [idx_1]
        decades += [idx_0p1]
    else:
        passes += [idx_0p1, idx_1]

    edge_val = 10 ** (dB_cutoff / 40.)
    assert_allclose(h[edges], edge_val, atol=0.01)
    assert_allclose(h[passes], 1., atol=pass_tol)
    if ftype == 'butter' and btype == 'lowpass':
        attenuation = dB_decade * order
        assert_allclose(h[decades], 10 ** (attenuation / 20.), rtol=0.01)
    elif ftype == 'ellip':
        assert_array_less(h[decades], 10 ** (-rs / 20))


@pytest.mark.parametrize('phase', ('zero', 'zero-double', 'minimum'))
@pytest.mark.parametrize('fir_window', ('hamming', 'blackman'))
@pytest.mark.parametrize('btype', ('lowpass', 'bandpass'))
def test_reporting_fir(phase, fir_window, btype):
    """Test FIR filter reporting."""
    l_freq = 1. if btype == 'bandpass' else None
    fs = 1000.
    with catch_logging() as log:
        x = create_filter(None, fs, l_freq, 40, method='fir',
                          phase=phase, fir_window=fir_window, verbose=True)
    n_taps = len(x)
    log = log.getvalue()
    keys = ['FIR',
            btype,
            fir_window.capitalize(),
            'Filter length: %d samples' % (n_taps,),
            'passband ripple',
            'stopband attenuation',
            ]
    if phase == 'minimum':
        keys += [' causal ']
    else:
        keys += [' non-causal ', ' dB cutoff frequency: 45.00 Hz']
        if btype == 'bandpass':
            keys += [' dB cutoff frequency: 0.50 Hz']
    for key in keys:
        assert key in log
    if phase == 'zero':
        assert '-6 dB cutoff' in log
    elif phase == 'zero-double':
        assert '-12 dB cutoff' in log
    else:
        # XXX Eventually we should figure out where the resulting point is,
        # since the minimum-phase process will change it. For now we don't
        # report it.
        assert phase == 'minimum'
    # Verify some of the filter properties
    if phase == 'zero-double':
        x = np.convolve(x, x)  # effectively what happens
    w, h = freqz(x, worN=10000)
    w *= fs / (2 * np.pi)
    h = np.abs(h)
    # passband
    passes = [np.argmin(np.abs(w - f)) for f in (1, 20, 40)]
    # stopband
    stops = [np.argmin(np.abs(w - 50.))]
    # transition
    mids = [np.argmin(np.abs(w - 45.))]
    # put these where they belong based on filter type
    assert w[0] == 0.
    idx_0 = 0
    idx_0p5 = np.argmin(np.abs(w - 0.5))
    if btype == 'bandpass':
        stops += [idx_0]
        mids += [idx_0p5]
    else:
        passes += [idx_0, idx_0p5]
    assert_allclose(h[passes], 1., atol=0.01)
    attenuation = -20 if phase == 'minimum' else -50
    assert_allclose(h[stops], 0., atol=10 ** (attenuation / 20.))
    if phase != 'minimum':  # haven't worked out the math for this yet
        expected = 0.25 if phase == 'zero-double' else 0.5
        assert_allclose(h[mids], expected, atol=0.01)


def test_filter_picks():
    """Test filter picking."""
    data = np.random.RandomState(0).randn(3, 1000)
    fs = 1000.
    kwargs = dict(l_freq=None, h_freq=40.)
    filt = filter_data(data, fs, **kwargs)
    # don't include seeg or stim in this list because they are in the one below
    # to ensure default cases are treated properly
    for kind in ('eeg', 'grad', 'emg', 'misc'):
        for picks in (None, [-2], kind, 'k'):
            # With always at least one data channel
            info = create_info(['s', 'k', 't'], fs, ['seeg', kind, 'stim'])
            raw = RawArray(data.copy(), info)
            raw.filter(picks=picks, **kwargs)
            if picks is None:
                if kind in _DATA_CH_TYPES_SPLIT:  # should be included
                    want = np.concatenate((filt[:2], data[2:]))
                else:  # shouldn't
                    want = np.concatenate((filt[:1], data[1:]))
            else:  # just the kind of interest ([-2], kind, 'j' should be eq.)
                want = np.concatenate((data[:1], filt[1:2], data[2:]))
            assert_allclose(raw.get_data(), want)

            # Now with sometimes no data channels
            info = create_info(['k', 't'], fs, [kind, 'stim'])
            raw = RawArray(data[1:].copy(), info.copy())
            if picks is None and kind not in _DATA_CH_TYPES_SPLIT:
                with pytest.raises(ValueError, match='yielded no channels'):
                    raw.filter(picks=picks, **kwargs)
            else:
                raw.filter(picks=picks, **kwargs)
                want = want[1:]
                assert_allclose(raw.get_data(), want)


run_tests_if_main()
