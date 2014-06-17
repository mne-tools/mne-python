import numpy as np
import scipy as sp
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true, assert_raises

from mne.fixes import tril_indices
from mne.connectivity import mvar_connectivity

from mne import SourceEstimate
from mne.filter import band_pass_filter


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


def _make_data(var_coef, n_samples, n_epochs):
    var_order = var_coef.shape[0]
    n_signals = var_coef.shape[1]

    x = np.random.randn(n_signals, n_epochs * n_samples + 10 * var_order)
    for i in range(var_order, x.shape[1]):
        for k in range(var_order):
            x[:, [i]] += np.dot(var_coef[k], x[:, [i-k-1]])

    x = x[:, -n_epochs * n_samples:]

    win = np.arange(0, n_samples)
    return [x[:, i + win] for i in range(0, n_epochs * n_samples, n_samples)]


def test_mvar_connectivity():
    """Test MVAR connectivity estimation"""
    # Use a case known to have no spurious correlations (it would bad if
    # nosetests could randomly fail):
    np.random.seed(0)

    sfreq = 50.
    n_sigs = 3
    n_epochs = 100
    n_samples = 500

    # test invalid fmin fmax settings
    assert_raises(ValueError, mvar_connectivity, [], 'S', 5, fmin=10, fmax=5)
    assert_raises(ValueError, mvar_connectivity, [], 'DTF', 1, fmin=(0, 11),
                  fmax=(5, 10))
    assert_raises(ValueError, mvar_connectivity, [], 'PDC', 99, fmin=(11,),
                  fmax=(12, 15))

    methods = ['S', 'COH', 'DTF', 'PDC', 'ffDTF', 'GPDC', 'GDTF', 'A']

    # generate data without connectivity
    var_coef = np.zeros((1, n_sigs, n_sigs))
    data = _make_data(var_coef, n_samples, n_epochs)

    con, freqs = mvar_connectivity(data, methods, order=5, sfreq=sfreq)
    con = dict((m, c) for m, c in zip(methods, con))

    assert_array_almost_equal(con['S'][:, :, 0], np.eye(n_sigs), decimal=2)
    assert_array_almost_equal(con['COH'][:, :, 0], np.eye(n_sigs), decimal=2)
    assert_array_almost_equal(con['COH'][:, :, 0].diagonal(), np.ones(n_sigs))
    assert_array_almost_equal(con['DTF'][:, :, 0], np.eye(n_sigs), decimal=2)
    assert_array_almost_equal(con['PDC'][:, :, 0], np.eye(n_sigs), decimal=2)
    assert_array_almost_equal(con['ffDTF'][:, :, 0] / np.sqrt(len(freqs[0])),
                              np.eye(n_sigs), decimal=2)
    assert_array_almost_equal(con['GPDC'][:, :, 0], np.eye(n_sigs), decimal=2)
    assert_array_almost_equal(con['GDTF'][:, :, 0], np.eye(n_sigs), decimal=2)

    # generate data with strong directed connectivity
    f = 1e3
    var_coef = np.zeros((1, n_sigs, n_sigs))
    var_coef[:, 1, 0] = f
    data = _make_data(var_coef, n_samples, n_epochs)

    con, freqs = mvar_connectivity(data, methods, order=3, sfreq=sfreq)
    con = dict((m, c) for m, c in zip(methods, con))

    h = var_coef.squeeze() + np.eye(n_sigs)

    assert_array_almost_equal(con['S'][:, :, 0] / f**2, np.dot(h, h.T) / f**2,
                              decimal=2)
    assert_array_almost_equal(con['COH'][:, :, 0], np.dot(h, h.T) > 0,
                              decimal=2)
    assert_array_almost_equal(con['DTF'][:, :, 0],
                              h / np.sum(h, 1, keepdims=True), decimal=2)
    assert_array_almost_equal(con['ffDTF'][:, :, 0] / np.sqrt(len(freqs[0])),
                              h / np.sum(h, 1, keepdims=True), decimal=2)
    assert_array_almost_equal(con['GDTF'][:, :, 0],
                              h / np.sum(h, 1, keepdims=True), decimal=2)
    assert_array_almost_equal(con['PDC'][:, :, 0],
                              h / np.sum(h, 0, keepdims=True), decimal=2)
    assert_array_almost_equal(con['GPDC'][:, :, 0],
                              h / np.sum(h, 0, keepdims=True), decimal=2)

    # generate data with strong cascaded directed connectivity
    f = 1e3
    var_coef = np.zeros((1, n_sigs, n_sigs))
    var_coef[:, 1, 0] = f
    var_coef[:, 2, 1] = f
    data = _make_data(var_coef, n_samples, n_epochs)

    con, freqs = mvar_connectivity(data, methods, order=3, sfreq=sfreq)
    con = dict((m, c) for m, c in zip(methods, con))

    assert_array_almost_equal(con['S'][:, :, 0] / f**4, [[f**-4, f**-3, f**-2],
                                                         [f**-3, f**-2, f**-1],
                                                         [f**-2, f**-1, f**0]],
                              decimal=2)
    assert_array_almost_equal(con['COH'][:, :, 0], np.ones((n_sigs, n_sigs)),
                              decimal=2)
    assert_array_almost_equal(con['DTF'][:, :, 0], [[1, 0, 0],
                                                    [1, 0, 0],
                                                    [1, 0, 0]], decimal=2)
    assert_array_almost_equal(con['ffDTF'][:, :, 0] / np.sqrt(len(freqs[0])),
                              [[1, 0, 0], [1, 0, 0], [1, 0, 0]], decimal=2)
    assert_array_almost_equal(con['GDTF'], con['DTF'], decimal=2)

    h = var_coef.squeeze() + np.eye(n_sigs)
    assert_array_almost_equal(con['PDC'][:, :, 0],
                              h / np.sum(h, 0, keepdims=True), decimal=2)
    assert_array_almost_equal(con['GPDC'], con['PDC'], decimal=2)


test_mvar_connectivity()