import warnings

import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true, assert_raises

from mne.connectivity import spectral_connectivity
from mne.connectivity.spectral import _CohEst

from mne import SourceEstimate
from mne.utils import run_tests_if_main, slow_test
from mne.filter import filter_data

trans_bandwidth = 2.5
filt_kwargs = dict(filter_length='auto', fir_window='hamming', phase='zero',
                   l_trans_bandwidth=trans_bandwidth,
                   h_trans_bandwidth=trans_bandwidth)
warnings.simplefilter('always')


def _stc_gen(data, sfreq, tmin, combo=False):
    """Simulate a SourceEstimate generator"""
    vertices = [np.arange(data.shape[1]), np.empty(0)]
    for d in data:
        if not combo:
            stc = SourceEstimate(data=d, vertices=vertices,
                                 tmin=tmin, tstep=1 / float(sfreq))
            yield stc
        else:
            # simulate a combination of array and source estimate
            arr = d[0]
            stc = SourceEstimate(data=d[1:], vertices=vertices,
                                 tmin=tmin, tstep=1 / float(sfreq))
            yield (arr, stc)


@slow_test
def test_spectral_connectivity():
    """Test frequency-domain connectivity methods"""
    # Use a case known to have no spurious correlations (it would bad if
    # nosetests could randomly fail):
    np.random.seed(0)

    sfreq = 50.
    n_signals = 3
    n_epochs = 8
    n_times = 256

    tmin = 0.
    tmax = (n_times - 1) / sfreq
    data = np.random.randn(n_epochs, n_signals, n_times)
    times_data = np.linspace(tmin, tmax, n_times)
    # simulate connectivity from 5Hz..15Hz
    fstart, fend = 5.0, 15.0
    for i in range(n_epochs):
        data[i, 1, :] = filter_data(data[i, 0, :], sfreq, fstart, fend,
                                    **filt_kwargs)
        # add some noise, so the spectrum is not exactly zero
        data[i, 1, :] += 1e-2 * np.random.randn(n_times)

    # First we test some invalid parameters:
    assert_raises(ValueError, spectral_connectivity, data, method='notamethod')
    assert_raises(ValueError, spectral_connectivity, data,
                  mode='notamode')

    # test invalid fmin fmax settings
    assert_raises(ValueError, spectral_connectivity, data, fmin=10,
                  fmax=10 + 0.5 * (sfreq / float(n_times)))
    assert_raises(ValueError, spectral_connectivity, data, fmin=10, fmax=5)
    assert_raises(ValueError, spectral_connectivity, data, fmin=(0, 11),
                  fmax=(5, 10))
    assert_raises(ValueError, spectral_connectivity, data, fmin=(11,),
                  fmax=(12, 15))

    methods = ['coh', 'cohy', 'imcoh', ['plv', 'ppc', 'pli', 'pli2_unbiased',
               'wpli', 'wpli2_debiased', 'coh']]

    modes = ['multitaper', 'fourier', 'cwt_morlet']

    # define some frequencies for cwt
    cwt_frequencies = np.arange(3, 24.5, 1)

    for mode in modes:
        for method in methods:
            if method == 'coh' and mode == 'multitaper':
                # only check adaptive estimation for coh to reduce test time
                check_adaptive = [False, True]
            else:
                check_adaptive = [False]

            if method == 'coh' and mode == 'cwt_morlet':
                # so we also test using an array for num cycles
                cwt_n_cycles = 7. * np.ones(len(cwt_frequencies))
            else:
                cwt_n_cycles = 7.

            for adaptive in check_adaptive:

                if adaptive:
                    mt_bandwidth = 1.
                else:
                    mt_bandwidth = None

                con, freqs, times, n, _ = spectral_connectivity(
                    data, method=method, mode=mode, indices=None, sfreq=sfreq,
                    mt_adaptive=adaptive, mt_low_bias=True,
                    mt_bandwidth=mt_bandwidth, cwt_frequencies=cwt_frequencies,
                    cwt_n_cycles=cwt_n_cycles)

                assert_true(n == n_epochs)
                assert_array_almost_equal(times_data, times)

                if mode == 'multitaper':
                    upper_t = 0.95
                    lower_t = 0.5
                elif mode == 'fourier':
                    # other estimates have higher variance
                    upper_t = 0.8
                    lower_t = 0.75
                else:  # cwt_morlet
                    upper_t = 0.64
                    lower_t = 0.63

                # test the simulated signal
                if method == 'coh':
                    idx = np.searchsorted(freqs, (fstart + trans_bandwidth,
                                                  fend - trans_bandwidth))
                    # we see something for zero-lag
                    assert_true(np.all(con[1, 0, idx[0]:idx[1]] > upper_t),
                                con[1, 0, idx[0]:idx[1]].min())

                    if mode != 'cwt_morlet':
                        idx = np.searchsorted(freqs,
                                              (fstart - trans_bandwidth * 2,
                                               fend + trans_bandwidth * 2))
                        assert_true(np.all(con[1, 0, :idx[0]] < lower_t))
                        assert_true(np.all(con[1, 0, idx[1]:] < lower_t),
                                    con[1, 0, idx[1:]].max())
                elif method == 'cohy':
                    idx = np.searchsorted(freqs, (fstart + 1, fend - 1))
                    # imaginary coh will be zero
                    check = np.imag(con[1, 0, idx[0]:idx[1]])
                    assert_true(np.all(check < lower_t), check.max())
                    # we see something for zero-lag
                    assert_true(np.all(np.abs(con[1, 0, idx[0]:idx[1]]) >
                                upper_t))

                    idx = np.searchsorted(freqs, (fstart - trans_bandwidth * 2,
                                                  fend + trans_bandwidth * 2))
                    if mode != 'cwt_morlet':
                        assert_true(np.all(np.abs(con[1, 0, :idx[0]]) <
                                    lower_t))
                        assert_true(np.all(np.abs(con[1, 0, idx[1]:]) <
                                    lower_t))
                elif method == 'imcoh':
                    idx = np.searchsorted(freqs, (fstart + 1, fend - 1))
                    # imaginary coh will be zero
                    assert_true(np.all(con[1, 0, idx[0]:idx[1]] < lower_t))
                    idx = np.searchsorted(freqs, (fstart - 1, fend + 1))
                    assert_true(np.all(con[1, 0, :idx[0]] < lower_t))
                    assert_true(np.all(con[1, 0, idx[1]:] < lower_t),
                                con[1, 0, idx[1]:].max())

                # compute same connections using indices and 2 jobs
                indices = np.tril_indices(n_signals, -1)

                if not isinstance(method, list):
                    test_methods = (method, _CohEst)
                else:
                    test_methods = method

                stc_data = _stc_gen(data, sfreq, tmin)
                con2, freqs2, times2, n2, _ = spectral_connectivity(
                    stc_data, method=test_methods, mode=mode, indices=indices,
                    sfreq=sfreq, mt_adaptive=adaptive, mt_low_bias=True,
                    mt_bandwidth=mt_bandwidth, tmin=tmin, tmax=tmax,
                    cwt_frequencies=cwt_frequencies,
                    cwt_n_cycles=cwt_n_cycles, n_jobs=2)

                assert_true(isinstance(con2, list))
                assert_true(len(con2) == len(test_methods))

                if method == 'coh':
                    assert_array_almost_equal(con2[0], con2[1])

                if not isinstance(method, list):
                    con2 = con2[0]  # only keep the first method

                    # we get the same result for the probed connections
                    assert_array_almost_equal(freqs, freqs2)
                    assert_array_almost_equal(con[indices], con2)
                    assert_true(n == n2)
                    assert_array_almost_equal(times_data, times2)
                else:
                    # we get the same result for the probed connections
                    assert_true(len(con) == len(con2))
                    for c, c2 in zip(con, con2):
                        assert_array_almost_equal(freqs, freqs2)
                        assert_array_almost_equal(c[indices], c2)
                        assert_true(n == n2)
                        assert_array_almost_equal(times_data, times2)

                # compute same connections for two bands, fskip=1, and f. avg.
                fmin = (5., 15.)
                fmax = (15., 30.)
                con3, freqs3, times3, n3, _ = spectral_connectivity(
                    data, method=method, mode=mode, indices=indices,
                    sfreq=sfreq, fmin=fmin, fmax=fmax, fskip=1, faverage=True,
                    mt_adaptive=adaptive, mt_low_bias=True,
                    mt_bandwidth=mt_bandwidth, cwt_frequencies=cwt_frequencies,
                    cwt_n_cycles=cwt_n_cycles)

                assert_true(isinstance(freqs3, list))
                assert_true(len(freqs3) == len(fmin))
                for i in range(len(freqs3)):
                    assert_true(np.all((freqs3[i] >= fmin[i]) &
                                       (freqs3[i] <= fmax[i])))

                # average con2 "manually" and we get the same result
                if not isinstance(method, list):
                    for i in range(len(freqs3)):
                        freq_idx = np.searchsorted(freqs2, freqs3[i])
                        con2_avg = np.mean(con2[:, freq_idx], axis=1)
                        assert_array_almost_equal(con2_avg, con3[:, i])
                else:
                    for j in range(len(con2)):
                        for i in range(len(freqs3)):
                            freq_idx = np.searchsorted(freqs2, freqs3[i])
                            con2_avg = np.mean(con2[j][:, freq_idx], axis=1)
                            assert_array_almost_equal(con2_avg, con3[j][:, i])


run_tests_if_main()
