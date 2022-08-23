# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause

from functools import partial
import os

import numpy as np
from scipy import sparse, linalg, stats
from numpy.testing import (assert_equal, assert_array_equal,
                           assert_array_almost_equal, assert_allclose)
import pytest

from mne import (SourceEstimate, VolSourceEstimate, MixedSourceEstimate,
                 SourceSpaces)
from mne.stats import ttest_ind_no_p, combine_adjacency
from mne.stats.cluster_level import (permutation_cluster_test, f_oneway,
                                     permutation_cluster_1samp_test,
                                     spatio_temporal_cluster_test,
                                     spatio_temporal_cluster_1samp_test,
                                     ttest_1samp_no_p, summarize_clusters_stc)
from mne.utils import (catch_logging, check_version, requires_sklearn,
                       _record_warnings)


n_space = 50


def _get_conditions():
    noise_level = 20
    n_time_1 = 20
    n_time_2 = 13
    normfactor = np.hanning(20).sum()
    rng = np.random.RandomState(42)
    condition1_1d = rng.randn(n_time_1, n_space) * noise_level
    for c in condition1_1d:
        c[:] = np.convolve(c, np.hanning(20), mode="same") / normfactor

    condition2_1d = rng.randn(n_time_2, n_space) * noise_level
    for c in condition2_1d:
        c[:] = np.convolve(c, np.hanning(20), mode="same") / normfactor

    pseudoekp = 10 * np.hanning(25)[None, :]
    condition1_1d[:, 25:] += pseudoekp
    condition2_1d[:, 25:] -= pseudoekp

    condition1_2d = condition1_1d[:, :, np.newaxis]
    condition2_2d = condition2_1d[:, :, np.newaxis]
    return condition1_1d, condition2_1d, condition1_2d, condition2_2d


def test_thresholds(numba_conditional):
    """Test automatic threshold calculations."""
    # within subjects
    rng = np.random.RandomState(0)
    X = rng.randn(10, 1, 1) + 0.08
    want_thresh = -stats.t.ppf(0.025, len(X) - 1)
    assert 0.03 < stats.ttest_1samp(X[:, 0, 0], 0)[1] < 0.05
    my_fun = partial(ttest_1samp_no_p)
    with catch_logging() as log:
        with pytest.warns(RuntimeWarning, match='threshold is only valid'):
            out = permutation_cluster_1samp_test(
                X, stat_fun=my_fun, seed=0, verbose=True, out_type='mask')
    log = log.getvalue()
    assert str(want_thresh)[:6] in log
    assert len(out[1]) == 1  # 1 cluster
    assert_allclose(out[2], 0.033203, atol=1e-6)
    # between subjects
    Y = rng.randn(10, 1, 1)
    Z = rng.randn(10, 1, 1) - 0.7
    X = [X, Y, Z]
    want_thresh = stats.f.ppf(1. - 0.05, 2, sum(len(a) for a in X) - len(X))
    p = stats.f_oneway(*X)[1]
    assert 0.03 < p < 0.05
    my_fun = partial(f_oneway)  # just to make the check fail
    with catch_logging() as log:
        with pytest.warns(RuntimeWarning, match='threshold is only valid'):
            out = permutation_cluster_test(X, tail=1, stat_fun=my_fun,
                                           seed=0, verbose=True,
                                           out_type='mask')
    log = log.getvalue()
    assert str(want_thresh)[:6] in log
    assert len(out[1]) == 1  # 1 cluster
    assert_allclose(out[2], 0.041992, atol=1e-6)
    with pytest.warns(RuntimeWarning, match='Ignoring argument "tail"'):
        permutation_cluster_test(X, tail=0, out_type='mask')

    # nan handling in TFCE
    X = np.repeat(X[0], 2, axis=1)
    X[:, 1] = 0
    with pytest.warns(RuntimeWarning, match='invalid value'):  # NumPy
        out = permutation_cluster_1samp_test(
            X, seed=0, threshold=dict(start=0, step=0.1), out_type='mask')
    assert (out[2] < 0.05).any()
    assert not (out[2] < 0.05).all()
    X[:, 0] = 0
    with pytest.raises(RuntimeError, match='finite'):
        with np.errstate(invalid='ignore'):
            permutation_cluster_1samp_test(
                X, seed=0, threshold=dict(start=0, step=0.1),
                buffer_size=None, out_type='mask')


def test_cache_dir(tmp_path, numba_conditional):
    """Test use of cache dir."""
    tempdir = str(tmp_path)
    orig_dir = os.getenv('MNE_CACHE_DIR', None)
    orig_size = os.getenv('MNE_MEMMAP_MIN_SIZE', None)
    rng = np.random.RandomState(0)
    X = rng.randn(9, 2, 10)
    try:
        os.environ['MNE_MEMMAP_MIN_SIZE'] = '1K'
        os.environ['MNE_CACHE_DIR'] = tempdir
        # Fix error for #1507: in-place when memmapping
        with catch_logging() as log_file:
            permutation_cluster_1samp_test(
                X, buffer_size=None, n_jobs=2, n_permutations=1, seed=0,
                stat_fun=ttest_1samp_no_p, verbose=False, out_type='mask')
        assert 'independently' not in log_file.getvalue()
        # ensure that non-independence yields warning
        stat_fun = partial(ttest_1samp_no_p, sigma=1e-3)
        if check_version('numpy', '1.17'):
            random_state = np.random.default_rng(0)
        else:
            random_state = 0
        with pytest.warns(RuntimeWarning, match='independently'):
            permutation_cluster_1samp_test(
                X, buffer_size=10, n_jobs=2, n_permutations=1,
                seed=random_state, stat_fun=stat_fun, verbose=False,
                out_type='mask')
    finally:
        if orig_dir is not None:
            os.environ['MNE_CACHE_DIR'] = orig_dir
        else:
            del os.environ['MNE_CACHE_DIR']
        if orig_size is not None:
            os.environ['MNE_MEMMAP_MIN_SIZE'] = orig_size
        else:
            del os.environ['MNE_MEMMAP_MIN_SIZE']


def test_permutation_large_n_samples(numba_conditional):
    """Test that non-replacement works with large N."""
    X = np.random.RandomState(0).randn(72, 1) + 1
    for n_samples in (11, 72):
        tails = (0, 1) if n_samples <= 20 else (0,)
        for tail in tails:
            H0 = permutation_cluster_1samp_test(
                X[:n_samples], threshold=1e-4, tail=tail, out_type='mask')[-1]
            assert H0.shape == (1024,)
            assert len(np.unique(H0)) >= 1024 - (H0 == 0).sum()


def test_permutation_step_down_p(numba_conditional):
    """Test cluster level permutations with step_down_p."""
    rng = np.random.RandomState(0)
    # subjects, time points, spatial points
    X = rng.randn(9, 2, 10)
    # add some significant points
    X[:, 0:2, 0:2] += 2  # span two time points and two spatial points
    X[:, 1, 5:9] += 0.5  # span four time points with 4x smaller amplitude
    thresh = 2
    # make sure it works when we use ALL points in step-down
    t, clusters, p, H0 = \
        permutation_cluster_1samp_test(X, threshold=thresh,
                                       step_down_p=1.0, out_type='mask')
    # make sure using step-down will actually yield improvements sometimes
    t, clusters, p_old, H0 = \
        permutation_cluster_1samp_test(X, threshold=thresh,
                                       step_down_p=0.0, out_type='mask')
    assert_equal(np.sum(p_old < 0.05), 1)  # just spatial cluster
    p_min = np.min(p_old)
    assert_allclose(p_min, 0.003906, atol=1e-6)
    t, clusters, p_new, H0 = \
        permutation_cluster_1samp_test(X, threshold=thresh,
                                       step_down_p=0.05, out_type='mask')
    assert_equal(np.sum(p_new < 0.05), 2)  # time one rescued
    assert np.all(p_old >= p_new)
    p_next = p_new[(p_new > 0.004) & (p_new < 0.05)][0]
    assert_allclose(p_next, 0.015625, atol=1e-6)


def test_cluster_permutation_test(numba_conditional):
    """Test cluster level permutations tests."""
    condition1_1d, condition2_1d, condition1_2d, condition2_2d = \
        _get_conditions()
    for condition1, condition2 in zip((condition1_1d, condition1_2d),
                                      (condition2_1d, condition2_2d)):
        T_obs, clusters, cluster_p_values, hist = permutation_cluster_test(
            [condition1, condition2], n_permutations=100, tail=1, seed=1,
            buffer_size=None, out_type='mask')
        p_min = np.min(cluster_p_values)
        assert_equal(np.sum(cluster_p_values < 0.05), 1)
        assert_allclose(p_min, 0.01, atol=1e-6)

        # test with 2 jobs and buffer_size enabled
        buffer_size = condition1.shape[1] // 10
        T_obs, clusters, cluster_p_values_buff, hist =\
            permutation_cluster_test([condition1, condition2],
                                     n_permutations=100, tail=1, seed=1,
                                     n_jobs=2, buffer_size=buffer_size,
                                     out_type='mask')
        assert_array_equal(cluster_p_values, cluster_p_values_buff)

    def stat_fun(X, Y):
        return stats.f_oneway(X, Y)[0]

    with pytest.warns(RuntimeWarning, match='is only valid'):
        permutation_cluster_test([condition1, condition2], n_permutations=1,
                                 stat_fun=stat_fun, out_type='mask')


@pytest.mark.parametrize('stat_fun', [
    ttest_1samp_no_p,
    partial(ttest_1samp_no_p, sigma=1e-1)
])
def test_cluster_permutation_t_test(numba_conditional, stat_fun):
    """Test cluster level permutations T-test."""
    condition1_1d, condition2_1d, condition1_2d, condition2_2d = \
        _get_conditions()

    # use a very large sigma to make sure Ts are not independent
    for condition1, p in ((condition1_1d, 0.01),
                          (condition1_2d, 0.01)):
        # these are so significant we can get away with fewer perms
        T_obs, clusters, cluster_p_values, hist =\
            permutation_cluster_1samp_test(condition1, n_permutations=100,
                                           tail=0, seed=1, out_type='mask',
                                           buffer_size=None)
        assert_equal(np.sum(cluster_p_values < 0.05), 1)
        p_min = np.min(cluster_p_values)
        assert_allclose(p_min, p, atol=1e-6)

        T_obs_pos, c_1, cluster_p_values_pos, _ =\
            permutation_cluster_1samp_test(condition1, n_permutations=100,
                                           tail=1, threshold=1.67, seed=1,
                                           stat_fun=stat_fun, out_type='mask',
                                           buffer_size=None)

        T_obs_neg, _, cluster_p_values_neg, _ =\
            permutation_cluster_1samp_test(-condition1, n_permutations=100,
                                           tail=-1, threshold=-1.67,
                                           seed=1, stat_fun=stat_fun,
                                           buffer_size=None, out_type='mask')
        assert_array_equal(T_obs_pos, -T_obs_neg)
        assert_array_equal(cluster_p_values_pos < 0.05,
                           cluster_p_values_neg < 0.05)

        # test with 2 jobs and buffer_size enabled
        buffer_size = condition1.shape[1] // 10
        with _record_warnings():  # sometimes "independently"
            T_obs_neg_buff, _, cluster_p_values_neg_buff, _ = \
                permutation_cluster_1samp_test(
                    -condition1, n_permutations=100, tail=-1, out_type='mask',
                    threshold=-1.67, seed=1, n_jobs=2, stat_fun=stat_fun,
                    buffer_size=buffer_size)

        assert_array_equal(T_obs_neg, T_obs_neg_buff)
        assert_array_equal(cluster_p_values_neg, cluster_p_values_neg_buff)

        # Bad stat_fun
        with pytest.raises(TypeError, match='must be .* ndarray'):
            permutation_cluster_1samp_test(
                condition1, threshold=1, stat_fun=lambda x: None,
                out_type='mask')
        with pytest.raises(ValueError, match='not compatible'):
            permutation_cluster_1samp_test(
                condition1, threshold=1, stat_fun=lambda x: stat_fun(x)[:-1],
                out_type='mask')


@requires_sklearn
def test_cluster_permutation_with_adjacency(numba_conditional, monkeypatch):
    """Test cluster level permutations with adjacency matrix."""
    from sklearn.feature_extraction.image import grid_to_graph
    condition1_1d, condition2_1d, condition1_2d, condition2_2d = \
        _get_conditions()

    n_pts = condition1_1d.shape[1]
    # we don't care about p-values in any of these, so do fewer permutations
    args = dict(seed=None, max_step=1, exclude=None, out_type='mask',
                step_down_p=0, t_power=1, threshold=1.67,
                check_disjoint=False, n_permutations=50)

    did_warn = False
    for X1d, X2d, func, spatio_temporal_func in \
            [(condition1_1d, condition1_2d,
              permutation_cluster_1samp_test,
              spatio_temporal_cluster_1samp_test),
             ([condition1_1d, condition2_1d],
              [condition1_2d, condition2_2d],
              permutation_cluster_test,
              spatio_temporal_cluster_test)]:
        out = func(X1d, **args)
        adjacency = grid_to_graph(1, n_pts)
        out_adjacency = func(X1d, adjacency=adjacency, **args)
        assert_array_equal(out[0], out_adjacency[0])
        for a, b in zip(out_adjacency[1], out[1]):
            assert_array_equal(out[0][a], out[0][b])
            assert np.all(a[b])

        # test spatio-temporal w/o time adjacency (repeat spatial pattern)
        adjacency_2 = sparse.coo_matrix(
            linalg.block_diag(adjacency.asfptype().todense(),
                              adjacency.asfptype().todense()))
        # nesting here is time then space:
        adjacency_2a = combine_adjacency(np.eye(2), adjacency)
        assert_array_equal(adjacency_2.toarray().astype(bool),
                           adjacency_2a.toarray().astype(bool))

        if isinstance(X1d, list):
            X1d_2 = [np.concatenate((x, x), axis=1) for x in X1d]
        else:
            X1d_2 = np.concatenate((X1d, X1d), axis=1)

        out_adjacency_2 = func(X1d_2, adjacency=adjacency_2, **args)
        # make sure we were operating on the same values
        split = len(out[0])
        assert_array_equal(out[0], out_adjacency_2[0][:split])
        assert_array_equal(out[0], out_adjacency_2[0][split:])

        # make sure we really got 2x the number of original clusters
        n_clust_orig = len(out[1])
        assert len(out_adjacency_2[1]) == 2 * n_clust_orig

        # Make sure that we got the old ones back
        data_1 = {np.sum(out[0][b[:n_pts]]) for b in out[1]}
        data_2 = {np.sum(out_adjacency_2[0][a]) for a in
                  out_adjacency_2[1][:]}
        assert len(data_1.intersection(data_2)) == len(data_1)

        # now use the other algorithm
        if isinstance(X1d, list):
            X1d_3 = [np.reshape(x, (-1, 2, n_space)) for x in X1d_2]
        else:
            X1d_3 = np.reshape(X1d_2, (-1, 2, n_space))

        out_adjacency_3 = spatio_temporal_func(
            X1d_3, n_permutations=50, adjacency=adjacency,
            max_step=0, threshold=1.67, check_disjoint=True)
        # make sure we were operating on the same values
        split = len(out[0])
        assert_array_equal(out[0], out_adjacency_3[0][0])
        assert_array_equal(out[0], out_adjacency_3[0][1])

        # make sure we really got 2x the number of original clusters
        assert len(out_adjacency_3[1]) == 2 * n_clust_orig

        # Make sure that we got the old ones back
        data_1 = {np.sum(out[0][b[:n_pts]]) for b in out[1]}
        data_2 = {np.sum(out_adjacency_3[0][a[0], a[1]]) for a in
                  out_adjacency_3[1]}
        assert len(data_1.intersection(data_2)) == len(data_1)

        # test new versus old method
        out_adjacency_4 = spatio_temporal_func(
            X1d_3, n_permutations=50, adjacency=adjacency,
            max_step=2, threshold=1.67)
        out_adjacency_5 = spatio_temporal_func(
            X1d_3, n_permutations=50, adjacency=adjacency,
            max_step=1, threshold=1.67)

        # clusters could be in a different order
        sums_4 = [np.sum(out_adjacency_4[0][a])
                  for a in out_adjacency_4[1]]
        sums_5 = [np.sum(out_adjacency_4[0][a])
                  for a in out_adjacency_5[1]]
        sums_4 = np.sort(sums_4)
        sums_5 = np.sort(sums_5)
        assert_array_almost_equal(sums_4, sums_5)

        monkeypatch.delenv('MNE_FORCE_SERIAL', raising=False)
        with pytest.raises(ValueError, match='must not be less'):
            spatio_temporal_func(
                X1d_3, n_permutations=1, adjacency=adjacency,
                max_step=1, threshold=1.67, n_jobs=-1000)

        # not enough TFCE params
        with pytest.raises(KeyError, match='threshold, if dict, must have'):
            spatio_temporal_func(
                X1d_3, adjacency=adjacency, threshold=dict(me='hello'))

        # too extreme a start threshold
        with _record_warnings() as w:
            spatio_temporal_func(X1d_3, adjacency=adjacency,
                                 threshold=dict(start=10, step=1))
        if not did_warn:
            assert len(w) == 1
            did_warn = True

        with pytest.raises(ValueError, match='threshold.*<= 0 for tail == -1'):
            spatio_temporal_func(
                X1d_3, adjacency=adjacency, tail=-1,
                threshold=dict(start=1, step=-1))
        with pytest.warns(RuntimeWarning, match='threshold.* is more extreme'):
            spatio_temporal_func(
                X1d_3, adjacency=adjacency, tail=1,
                threshold=dict(start=100, step=1))
        bad_con = adjacency.todense()
        with pytest.raises(ValueError, match='must be a SciPy sparse matrix'):
            spatio_temporal_func(
                X1d_3, n_permutations=50, adjacency=bad_con,
                max_step=1, threshold=1.67)
        bad_con = adjacency.tocsr()[:-1, :-1].tocoo()
        with pytest.raises(ValueError, match='adjacency.*the correct size'):
            spatio_temporal_func(
                X1d_3, n_permutations=50, adjacency=bad_con,
                max_step=1, threshold=1.67)
        with pytest.raises(TypeError, match='must be a'):
            spatio_temporal_func(
                X1d_3, adjacency=adjacency, threshold=[])
        with pytest.raises(ValueError, match='Invalid value for the \'tail\''):
            # sometimes ignoring tail
            with _record_warnings():
                spatio_temporal_func(
                    X1d_3, adjacency=adjacency, tail=2)

        # make sure it actually found a significant point
        out_adjacency_6 = spatio_temporal_func(
            X1d_3, n_permutations=50, adjacency=adjacency, max_step=1,
            threshold=dict(start=1, step=1))
        assert np.min(out_adjacency_6[2]) < 0.05

        with pytest.raises(ValueError, match='not compatible'):
            with pytest.warns(RuntimeWarning, match='No clusters'):
                spatio_temporal_func(
                    X1d_3, n_permutations=50, adjacency=adjacency,
                    threshold=1e-3, stat_fun=lambda *x: f_oneway(*x)[:-1],
                    buffer_size=None)


@pytest.mark.parametrize('threshold', [
    0.1,
    pytest.param(dict(start=0., step=0.5), id='TFCE'),
])
@pytest.mark.parametrize('kind', ('1samp', 'ind'))
def test_permutation_cluster_signs(threshold, kind):
    """Test cluster signs."""
    # difference between two conditions for 3 subjects x 2 vertices x 2 times
    X = np.array([[[-10, 5], [-2, -7]],
                  [[-4, 5], [-8, -0]],
                  [[-6, 3], [-4, -2]]], float)
    want_signs = np.sign(np.mean(X, axis=0))
    n_permutations = 1
    if kind == '1samp':
        func = permutation_cluster_1samp_test
        stat_fun = ttest_1samp_no_p
        use_X = X
    else:
        assert kind == 'ind'
        func = permutation_cluster_test
        stat_fun = ttest_ind_no_p
        use_X = [X, np.random.RandomState(0).randn(*X.shape) * 0.1]
    tobs, clu, clu_pvalues, _ = func(
        use_X, n_permutations=n_permutations, threshold=threshold, tail=0,
        stat_fun=stat_fun, out_type='mask')
    clu_signs = np.zeros(X.shape[1:])
    used = np.zeros(X.shape[1:])
    assert len(clu) == len(clu_pvalues)
    for c, p in zip(clu, clu_pvalues):
        assert not used[c].any()
        assert len(np.unique(np.sign(tobs[c]))) == 1
        clu_signs[c] = np.sign(tobs[c])[0]
        used[c] = True
    assert used.all()
    assert clu_signs.all()
    assert_array_equal(np.sign(tobs), want_signs)
    assert_array_equal(clu_signs, want_signs)


@requires_sklearn
def test_permutation_adjacency_equiv(numba_conditional):
    """Test cluster level permutations with and without adjacency."""
    from sklearn.feature_extraction.image import grid_to_graph
    rng = np.random.RandomState(0)
    # subjects, time points, spatial points
    n_time = 2
    n_space = 4
    X = rng.randn(6, n_time, n_space)
    # add some significant points
    X[:, :, 0:2] += 10  # span two time points and two spatial points
    X[:, 1, 3] += 20  # span one time point
    max_steps = [1, 1, 1, 2, 1]
    # This will run full algorithm in two ways, then the ST-algorithm in 2 ways
    # All of these should give the same results
    adjs = [None,
            grid_to_graph(n_time, n_space),
            grid_to_graph(1, n_space),
            grid_to_graph(1, n_space),
            None]
    stat_map = None
    thresholds = [2, 2, 2, 2, dict(start=0.01, step=1.0)]
    sig_counts = [2, 2, 2, 2, 5]
    stat_fun = partial(ttest_1samp_no_p, sigma=1e-3)

    cs = None
    ps = None
    for thresh, count, max_step, adj in zip(thresholds, sig_counts,
                                            max_steps, adjs):
        t, clusters, p, H0 = \
            permutation_cluster_1samp_test(
                X, threshold=thresh, adjacency=adj, n_jobs=2,
                max_step=max_step, stat_fun=stat_fun, seed=0, out_type='mask')
        # make sure our output datatype is correct
        assert isinstance(clusters[0], np.ndarray)
        assert clusters[0].dtype == bool
        assert_array_equal(clusters[0].shape, X.shape[1:])

        # make sure all comparisons were done; for TFCE, no perm
        # should come up empty
        inds = np.where(p < 0.05)[0]
        assert_equal(len(inds), count)
        assert_allclose(p[inds], 0.03125, atol=1e-6)
        if isinstance(thresh, dict):
            assert_equal(len(clusters), n_time * n_space)
            assert np.all(H0 != 0)
            continue
        this_cs = [clusters[ii] for ii in inds]
        this_ps = p[inds]
        this_stat_map = np.zeros((n_time, n_space), dtype=bool)
        for ci, c in enumerate(this_cs):
            if isinstance(c, tuple):
                this_c = np.zeros((n_time, n_space), bool)
                for x, y in zip(c[0], c[1]):
                    this_stat_map[x, y] = True
                    this_c[x, y] = True
                this_cs[ci] = this_c
                c = this_c
            this_stat_map[c] = True
        if cs is None:
            ps = this_ps
            cs = this_cs
        if stat_map is None:
            stat_map = this_stat_map
        assert_array_equal(ps, this_ps)
        assert len(cs) == len(this_cs)
        for c1, c2 in zip(cs, this_cs):
            assert_array_equal(c1, c2)
        assert_array_equal(stat_map, this_stat_map)


@requires_sklearn
def test_spatio_temporal_cluster_adjacency(numba_conditional):
    """Test spatio-temporal cluster permutations."""
    from sklearn.feature_extraction.image import grid_to_graph
    condition1_1d, condition2_1d, condition1_2d, condition2_2d = \
        _get_conditions()

    rng = np.random.RandomState(0)
    noise1_2d = rng.randn(condition1_2d.shape[0], condition1_2d.shape[1], 10)
    data1_2d = np.transpose(np.dstack((condition1_2d, noise1_2d)), [0, 2, 1])

    noise2_d2 = rng.randn(condition2_2d.shape[0], condition2_2d.shape[1], 10)
    data2_2d = np.transpose(np.dstack((condition2_2d, noise2_d2)), [0, 2, 1])

    adj = grid_to_graph(data1_2d.shape[-1], 1)

    threshold = dict(start=4.0, step=2)
    T_obs, clusters, p_values_adj, hist = \
        spatio_temporal_cluster_test([data1_2d, data2_2d], adjacency=adj,
                                     n_permutations=50, tail=1, seed=1,
                                     threshold=threshold, buffer_size=None)

    buffer_size = data1_2d.size // 10
    T_obs, clusters, p_values_no_adj, hist = \
        spatio_temporal_cluster_test([data1_2d, data2_2d],
                                     n_permutations=50, tail=1, seed=1,
                                     threshold=threshold, n_jobs=2,
                                     buffer_size=buffer_size)

    assert_equal(np.sum(p_values_adj < 0.05), np.sum(p_values_no_adj < 0.05))

    # make sure results are the same without buffer_size
    T_obs, clusters, p_values2, hist2 = \
        spatio_temporal_cluster_test([data1_2d, data2_2d],
                                     n_permutations=50, tail=1, seed=1,
                                     threshold=threshold, n_jobs=2,
                                     buffer_size=None)
    assert_array_equal(p_values_no_adj, p_values2)
    pytest.raises(ValueError, spatio_temporal_cluster_test,
                  [data1_2d, data2_2d], tail=1, threshold=-2.)
    pytest.raises(ValueError, spatio_temporal_cluster_test,
                  [data1_2d, data2_2d], tail=-1, threshold=2.)
    pytest.raises(ValueError, spatio_temporal_cluster_test,
                  [data1_2d, data2_2d], tail=0, threshold=-1)


def ttest_1samp(X):
    """Return T-values."""
    return stats.ttest_1samp(X, 0)[0]


@pytest.mark.parametrize('kind', ('surface', 'volume', 'mixed'))
def test_summarize_clusters(kind):
    """Test cluster summary stcs."""
    src_surf = SourceSpaces(
        [dict(vertno=np.arange(10242), type='surf') for _ in range(2)])
    assert src_surf.kind == 'surface'
    src_vol = SourceSpaces(
        [dict(vertno=np.arange(10), type='vol')])
    assert src_vol.kind == 'volume'
    if kind == 'surface':
        src = src_surf
        klass = SourceEstimate
    elif kind == 'volume':
        src = src_vol
        klass = VolSourceEstimate
    else:
        assert kind == 'mixed'
        src = src_surf + src_vol
        klass = MixedSourceEstimate
    n_vertices = sum(len(s['vertno']) for s in src)
    clu = (np.random.random([1, n_vertices]),
           [(np.array([0]), np.array([0, 2, 4]))],
           np.array([0.02, 0.1]),
           np.array([12, -14, 30]))
    kwargs = dict()
    if kind == 'volume':
        with pytest.raises(ValueError, match='did not match'):
            summarize_clusters_stc(clu)
        assert len(src) == 1
        kwargs['vertices'] = [src[0]['vertno']]
    elif kind == 'mixed':
        kwargs['vertices'] = src
    stc_sum = summarize_clusters_stc(clu, **kwargs)
    assert isinstance(stc_sum, klass)
    assert stc_sum.data.shape[1] == 2
    clu[2][0] = 0.3
    with pytest.raises(RuntimeError, match='No significant'):
        summarize_clusters_stc(clu, **kwargs)


def test_permutation_test_H0(numba_conditional):
    """Test that H0 is populated properly during testing."""
    rng = np.random.RandomState(0)
    data = rng.rand(7, 10, 1) - 0.5
    with pytest.warns(RuntimeWarning, match='No clusters found'):
        t, clust, p, h0 = spatio_temporal_cluster_1samp_test(
            data, threshold=100, n_permutations=1024, seed=rng)
    assert_equal(len(h0), 0)

    for n_permutations in (1024, 65, 64, 63):
        t, clust, p, h0 = spatio_temporal_cluster_1samp_test(
            data, threshold=0.1, n_permutations=n_permutations, seed=rng)
        assert_equal(len(h0), min(n_permutations, 64))
        assert isinstance(clust[0], tuple)  # sets of indices
    for tail, thresh in zip((-1, 0, 1), (-0.1, 0.1, 0.1)):
        t, clust, p, h0 = spatio_temporal_cluster_1samp_test(
            data, threshold=thresh, seed=rng, tail=tail, out_type='mask')
        assert isinstance(clust[0], np.ndarray)  # bool mask
        # same as "128 if tail else 64"
        assert_equal(len(h0), 2 ** (7 - (tail == 0)))  # exact test


def test_tfce_thresholds(numba_conditional):
    """Test TFCE thresholds."""
    rng = np.random.RandomState(0)
    data = rng.randn(7, 10, 1) - 0.5

    # if tail==-1, step must also be negative
    with pytest.raises(ValueError, match='must be < 0 for tail == -1'):
        permutation_cluster_1samp_test(
            data, tail=-1, out_type='mask', threshold=dict(start=0, step=0.1))
    # this works (smoke test)
    permutation_cluster_1samp_test(data, tail=-1, out_type='mask',
                                   threshold=dict(start=0, step=-0.1))

    # thresholds must be monotonically increasing
    with pytest.raises(ValueError, match='must be monotonically increasing'):
        permutation_cluster_1samp_test(
            data, tail=1, out_type='mask', threshold=dict(start=1, step=-0.5))

    # Should work with 2D data too
    permutation_cluster_1samp_test(X=data[..., 0],
                                   threshold=dict(start=0, step=0.2))


# 1D gives slices, 2D+ gives boolean masks
@pytest.mark.parametrize('shape', ((11,), (11, 3), (11, 1, 2)))
@pytest.mark.parametrize('out_type', ('mask', 'indices'))
@pytest.mark.parametrize('adjacency', (None, 'sparse'))
def test_output_equiv(shape, out_type, adjacency):
    """Test equivalence of output types."""
    rng = np.random.RandomState(0)
    n_subjects = 10
    data = rng.randn(n_subjects, *shape)
    data -= data.mean(axis=0, keepdims=True)
    data[:, 2:4] += 2
    data[:, 6:9] += 2
    want_mask = np.zeros(shape, int)
    want_mask[2:4] = 1
    want_mask[6:9] = 2
    if adjacency is not None:
        assert adjacency == 'sparse'
        adjacency = combine_adjacency(*shape)
    clusters = permutation_cluster_1samp_test(
        X=data, n_permutations=1, adjacency=adjacency, out_type=out_type)[1]
    got_mask = np.zeros_like(want_mask)
    for n, clu in enumerate(clusters, 1):
        if out_type == 'mask':
            if len(shape) == 1 and adjacency is None:
                assert isinstance(clu, tuple)
                assert len(clu) == 1
                assert isinstance(clu[0], slice)
            else:
                assert isinstance(clu, np.ndarray)
                assert clu.dtype == bool
                assert clu.shape == shape
            got_mask[clu] = n
        else:
            assert isinstance(clu, tuple)
            for c in clu:
                assert isinstance(c, np.ndarray)
                assert c.dtype.kind == 'i'
            assert out_type == 'indices'
            got_mask[np.ix_(*clu)] = n
    assert_array_equal(got_mask, want_mask)
