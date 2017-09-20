import warnings

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from nose.tools import assert_true, assert_raises

from mne.connectivity import spectral_connectivity
from mne.connectivity.spectral import _CohEst, _get_n_epochs

from mne import SourceEstimate
from mne.utils import run_tests_if_main
from mne.filter import filter_data

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


@pytest.mark.slowtest
def test_spectral_connectivity():
    """Test frequency-domain connectivity methods"""
    # Use a case known to have no spurious correlations (it would bad if
    # nosetests could randomly fail):
    rng = np.random.RandomState(0)
    trans_bandwidth = 2.

    sfreq = 50.
    n_signals = 3
    n_epochs = 8
    n_times = 256

    tmin = 0.
    tmax = (n_times - 1) / sfreq
    data = rng.randn(n_signals, n_epochs * n_times)
    times_data = np.linspace(tmin, tmax, n_times)
    # simulate connectivity from 5Hz..15Hz
    fstart, fend = 5.0, 15.0
    data[1, :] = filter_data(data[0, :], sfreq, fstart, fend,
                             filter_length='auto', fir_design='firwin2',
                             l_trans_bandwidth=trans_bandwidth,
                             h_trans_bandwidth=trans_bandwidth)
    # add some noise, so the spectrum is not exactly zero
    data[1, :] += 1e-2 * rng.randn(n_times * n_epochs)
    data = data.reshape(n_signals, n_epochs, n_times)
    data = np.transpose(data, [1, 0, 2])

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
    cwt_freqs = np.arange(3, 24.5, 1)

    for mode in modes:
        for method in methods:
            if method == 'coh' and mode == 'multitaper':
                # only check adaptive estimation for coh to reduce test time
                check_adaptive = [False, True]
            else:
                check_adaptive = [False]

            if method == 'coh' and mode == 'cwt_morlet':
                # so we also test using an array for num cycles
                cwt_n_cycles = 7. * np.ones(len(cwt_freqs))
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
                    mt_bandwidth=mt_bandwidth, cwt_freqs=cwt_freqs,
                    cwt_n_cycles=cwt_n_cycles)

                assert_true(n == n_epochs)
                assert_array_almost_equal(times_data, times)

                if mode == 'multitaper':
                    upper_t = 0.95
                    lower_t = 0.5
                else:  # mode == 'fourier' or mode == 'cwt_morlet'
                    # other estimates have higher variance
                    upper_t = 0.8
                    lower_t = 0.75

                # test the simulated signal
                gidx = np.searchsorted(freqs, (fstart, fend))
                bidx = np.searchsorted(freqs,
                                       (fstart - trans_bandwidth * 2,
                                        fend + trans_bandwidth * 2))
                if method == 'coh':
                    assert_true(np.all(con[1, 0, gidx[0]:gidx[1]] > upper_t),
                                con[1, 0, gidx[0]:gidx[1]].min())
                    # we see something for zero-lag
                    assert_true(np.all(con[1, 0, :bidx[0]] < lower_t))
                    assert_true(np.all(con[1, 0, bidx[1]:] < lower_t),
                                con[1, 0, bidx[1:]].max())
                elif method == 'cohy':
                    # imaginary coh will be zero
                    check = np.imag(con[1, 0, gidx[0]:gidx[1]])
                    assert_true(np.all(check < lower_t), check.max())
                    # we see something for zero-lag
                    assert_true(np.all(np.abs(con[1, 0, gidx[0]:gidx[1]]) >
                                upper_t))
                    assert_true(np.all(np.abs(con[1, 0, :bidx[0]]) <
                                lower_t))
                    assert_true(np.all(np.abs(con[1, 0, bidx[1]:]) <
                                lower_t))
                elif method == 'imcoh':
                    # imaginary coh will be zero
                    assert_true(np.all(con[1, 0, gidx[0]:gidx[1]] < lower_t))
                    assert_true(np.all(con[1, 0, :bidx[0]] < lower_t))
                    assert_true(np.all(con[1, 0, bidx[1]:] < lower_t),
                                con[1, 0, bidx[1]:].max())

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
                    cwt_freqs=cwt_freqs,
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
                    mt_bandwidth=mt_bandwidth, cwt_freqs=cwt_freqs,
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
    # test _get_n_epochs
    full_list = list(range(10))
    out_lens = np.array([len(x) for x in _get_n_epochs(full_list, 4)])
    assert_true((out_lens == np.array([4, 4, 2])).all())
    out_lens = np.array([len(x) for x in _get_n_epochs(full_list, 11)])
    assert_true(len(out_lens) > 0)
    assert_true(out_lens[0] == 10)

run_tests_if_main()
