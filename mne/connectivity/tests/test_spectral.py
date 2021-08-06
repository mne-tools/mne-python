import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import pytest

from mne import EpochsArray, SourceEstimate, create_info
from mne.connectivity import spectral_connectivity
from mne.connectivity.spectral import _CohEst, _get_n_epochs
from mne.filter import filter_data


def _stc_gen(data, sfreq, tmin, combo=False):
    """Simulate a SourceEstimate generator."""
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


@pytest.mark.parametrize('method', ['coh', 'cohy', 'imcoh', 'plv',
                                    ['ciplv', 'ppc', 'pli', 'pli2_unbiased',
                                     'wpli', 'wpli2_debiased', 'coh']])
@pytest.mark.parametrize('mode', ['multitaper', 'fourier', 'cwt_morlet'])
def test_spectral_connectivity(method, mode):
    """Test frequency-domain connectivity methods."""
    # Use a case known to have no spurious correlations (it would bad if
    # tests could randomly fail):
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
    pytest.raises(ValueError, spectral_connectivity, data, method='notamethod')
    pytest.raises(ValueError, spectral_connectivity, data,
                  mode='notamode')

    # test invalid fmin fmax settings
    pytest.raises(ValueError, spectral_connectivity, data, fmin=10,
                  fmax=10 + 0.5 * (sfreq / float(n_times)))
    pytest.raises(ValueError, spectral_connectivity, data, fmin=10, fmax=5)
    pytest.raises(ValueError, spectral_connectivity, data, fmin=(0, 11),
                  fmax=(5, 10))
    pytest.raises(ValueError, spectral_connectivity, data, fmin=(11,),
                  fmax=(12, 15))

    # define some frequencies for cwt
    cwt_freqs = np.arange(3, 24.5, 1)

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

        assert (n == n_epochs)
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
            assert np.all(con[1, 0, gidx[0]:gidx[1]] > upper_t), \
                con[1, 0, gidx[0]:gidx[1]].min()
            # we see something for zero-lag
            assert (np.all(con[1, 0, :bidx[0]] < lower_t))
            assert np.all(con[1, 0, bidx[1]:] < lower_t), \
                con[1, 0, bidx[1:]].max()
        elif method == 'cohy':
            # imaginary coh will be zero
            check = np.imag(con[1, 0, gidx[0]:gidx[1]])
            assert np.all(check < lower_t), check.max()
            # we see something for zero-lag
            assert np.all(np.abs(con[1, 0, gidx[0]:gidx[1]]) > upper_t)
            assert np.all(np.abs(con[1, 0, :bidx[0]]) < lower_t)
            assert np.all(np.abs(con[1, 0, bidx[1]:]) < lower_t)
        elif method == 'imcoh':
            # imaginary coh will be zero
            assert np.all(con[1, 0, gidx[0]:gidx[1]] < lower_t)
            assert np.all(con[1, 0, :bidx[0]] < lower_t)
            assert np.all(con[1, 0, bidx[1]:] < lower_t), \
                con[1, 0, bidx[1]:].max()

        # compute a subset of connections using indices and 2 jobs
        indices = (np.array([2, 1]), np.array([0, 0]))

        if not isinstance(method, list):
            test_methods = (method, _CohEst)
        else:
            test_methods = method

        stc_data = _stc_gen(data, sfreq, tmin)
        con2, freqs2, times2, n2, _ = spectral_connectivity(
            stc_data, method=test_methods, mode=mode, indices=indices,
            sfreq=sfreq, mt_adaptive=adaptive, mt_low_bias=True,
            mt_bandwidth=mt_bandwidth, tmin=tmin, tmax=tmax,
            cwt_freqs=cwt_freqs, cwt_n_cycles=cwt_n_cycles)

        assert isinstance(con2, list)
        assert len(con2) == len(test_methods)

        if method == 'coh':
            assert_array_almost_equal(con2[0], con2[1])

        if not isinstance(method, list):
            con2 = con2[0]  # only keep the first method

            # we get the same result for the probed connections
            assert_array_almost_equal(freqs, freqs2)
            assert_array_almost_equal(con[indices], con2)
            assert (n == n2)
            assert_array_almost_equal(times_data, times2)
        else:
            # we get the same result for the probed connections
            assert (len(con) == len(con2))
            for c, c2 in zip(con, con2):
                assert_array_almost_equal(freqs, freqs2)
                assert_array_almost_equal(c[indices], c2)
                assert (n == n2)
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

        assert (isinstance(freqs3, list))
        assert (len(freqs3) == len(fmin))
        for i in range(len(freqs3)):
            assert np.all((freqs3[i] >= fmin[i]) &
                          (freqs3[i] <= fmax[i]))

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
    assert ((out_lens == np.array([4, 4, 2])).all())
    out_lens = np.array([len(x) for x in _get_n_epochs(full_list, 11)])
    assert (len(out_lens) > 0)
    assert (out_lens[0] == 10)


@pytest.mark.parametrize('kind', ('epochs', 'ndarray', 'stc', 'combo'))
def test_epochs_tmin_tmax(kind):
    """Test spectral.spectral_connectivity with epochs and arrays."""
    rng = np.random.RandomState(0)
    n_epochs, n_chs, n_times, sfreq, f = 10, 2, 2000, 1000., 20.
    data = rng.randn(n_epochs, n_chs, n_times)
    sig = np.sin(2 * np.pi * f * np.arange(1000) / sfreq) * np.hanning(1000)
    data[:, :, 500:1500] += sig
    info = create_info(n_chs, sfreq, 'eeg')
    if kind == 'epochs':
        tmin = -1
        X = EpochsArray(data, info, tmin=tmin)
    elif kind == 'stc':
        tmin = -1
        X = [SourceEstimate(d, [[0], [0]], tmin, 1. / sfreq) for d in data]
    elif kind == 'combo':
        tmin = -1
        X = [(d[[0]], SourceEstimate(d[[1]], [[0], []], tmin, 1. / sfreq))
             for d in data]
    else:
        assert kind == 'ndarray'
        tmin = 0
        X = data
    want_times = np.arange(n_times) / sfreq + tmin

    # Parameters for computing connectivity
    fmin, fmax = f - 2, f + 2
    kwargs = {'method': 'coh', 'mode': 'multitaper', 'sfreq': sfreq,
              'fmin': fmin, 'fmax': fmax, 'faverage': True,
              'mt_adaptive': False, 'n_jobs': 1}

    # Check the entire interval
    conn = spectral_connectivity(X, **kwargs)
    assert 0.89 < conn[0][1, 0] < 0.91
    assert_allclose(conn[2], want_times)
    # Check a time interval before the sinusoid
    conn = spectral_connectivity(X, tmax=tmin + 0.5, **kwargs)
    assert 0 < conn[0][1, 0] < 0.15
    # Check a time during the sinusoid
    conn = spectral_connectivity(X, tmin=tmin + 0.5, tmax=tmin + 1.5, **kwargs)
    assert 0.93 < conn[0][1, 0] <= 0.94
    # Check a time interval after the sinusoid
    conn = spectral_connectivity(X, tmin=tmin + 1.5, tmax=tmin + 1.9, **kwargs)
    assert 0 < conn[0][1, 0] < 0.15

    # Check for warning if tmin, tmax is outside of the time limits of data
    with pytest.warns(RuntimeWarning, match='start time tmin'):
        spectral_connectivity(X, **kwargs, tmin=tmin - 0.1)

    with pytest.warns(RuntimeWarning, match='stop time tmax'):
        spectral_connectivity(X, **kwargs, tmax=tmin + 2.5)

    # make one with mismatched times
    if kind != 'combo':
        return
    X = [(SourceEstimate(d[[0]], [[0], []], tmin - 1, 1. / sfreq),
          SourceEstimate(d[[1]], [[0], []], tmin, 1. / sfreq)) for d in data]
    with pytest.warns(RuntimeWarning, match='time scales of input') as w:
        spectral_connectivity(X, **kwargs)

    # increased to 2 to catch the DeprecationWarning
    assert len(w) == 2  # just one even though there were multiple epochs
