# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

from functools import partial
import os
import warnings

import numpy as np
from scipy import sparse, linalg, stats
from numpy.testing import (assert_equal, assert_array_equal,
                           assert_array_almost_equal)
from nose.tools import assert_true, assert_raises

from mne.parallel import _force_serial
from mne.stats.cluster_level import (permutation_cluster_test,
                                     permutation_cluster_1samp_test,
                                     spatio_temporal_cluster_test,
                                     spatio_temporal_cluster_1samp_test,
                                     ttest_1samp_no_p, summarize_clusters_stc)
from mne.utils import run_tests_if_main, _TempDir, catch_logging

warnings.simplefilter('always')  # enable b/c these tests throw warnings


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


def test_cache_dir():
    """Test use of cache dir."""
    tempdir = _TempDir()
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
                X, buffer_size=None, n_jobs=2, n_permutations=1,
                seed=0, stat_fun=ttest_1samp_no_p, verbose=False)
            # ensure that non-independence yields warning
            stat_fun = partial(ttest_1samp_no_p, sigma=1e-3)
            assert_true('independently' not in log_file.getvalue())
            with warnings.catch_warnings(record=True):  # independently
                permutation_cluster_1samp_test(
                    X, buffer_size=10, n_jobs=2, n_permutations=1,
                    seed=0, stat_fun=stat_fun, verbose=False)
            assert_true('independently' in log_file.getvalue())
    finally:
        if orig_dir is not None:
            os.environ['MNE_CACHE_DIR'] = orig_dir
        else:
            del os.environ['MNE_CACHE_DIR']
        if orig_size is not None:
            os.environ['MNE_MEMMAP_MIN_SIZE'] = orig_size
        else:
            del os.environ['MNE_MEMMAP_MIN_SIZE']


def test_permutation_large_n_samples():
    """Test that non-replacement works with large N."""
    X = np.random.RandomState(0).randn(72, 1) + 1
    for n_samples in (11, 72):
        tails = (0, 1) if n_samples <= 20 else (0,)
        for tail in tails:
            H0 = permutation_cluster_1samp_test(
                X[:n_samples], threshold=1e-4, tail=tail)[-1]
            assert H0.shape == (1024,)
            assert len(np.unique(H0)) >= 1024 - (H0 == 0).sum()


def test_permutation_step_down_p():
    """Test cluster level permutations with step_down_p."""
    try:
        try:
            from sklearn.feature_extraction.image import grid_to_graph
        except ImportError:
            from scikits.learn.feature_extraction.image import grid_to_graph  # noqa: F401,E501 analysis:ignore
    except ImportError:
        return
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
                                       step_down_p=1.0)
    # make sure using step-down will actually yield improvements sometimes
    t, clusters, p_old, H0 = \
        permutation_cluster_1samp_test(X, threshold=thresh,
                                       step_down_p=0.0)
    assert_equal(np.sum(p_old < 0.05), 1)  # just spatial cluster
    t, clusters, p_new, H0 = \
        permutation_cluster_1samp_test(X, threshold=thresh,
                                       step_down_p=0.05)
    assert_equal(np.sum(p_new < 0.05), 2)  # time one rescued
    assert_true(np.all(p_old >= p_new))


def test_cluster_permutation_test():
    """Test cluster level permutations tests."""
    condition1_1d, condition2_1d, condition1_2d, condition2_2d = \
        _get_conditions()
    for condition1, condition2 in zip((condition1_1d, condition1_2d),
                                      (condition2_1d, condition2_2d)):
        T_obs, clusters, cluster_p_values, hist = permutation_cluster_test(
            [condition1, condition2], n_permutations=100, tail=1, seed=1,
            buffer_size=None)
        assert_equal(np.sum(cluster_p_values < 0.05), 1)

        T_obs, clusters, cluster_p_values, hist = permutation_cluster_test(
            [condition1, condition2], n_permutations=100, tail=0, seed=1,
            buffer_size=None)
        assert_equal(np.sum(cluster_p_values < 0.05), 1)

        # test with 2 jobs and buffer_size enabled
        buffer_size = condition1.shape[1] // 10
        T_obs, clusters, cluster_p_values_buff, hist =\
            permutation_cluster_test([condition1, condition2],
                                     n_permutations=100, tail=0, seed=1,
                                     n_jobs=2, buffer_size=buffer_size)
        assert_array_equal(cluster_p_values, cluster_p_values_buff)


def test_cluster_permutation_t_test():
    """Test cluster level permutations T-test."""
    condition1_1d, condition2_1d, condition1_2d, condition2_2d = \
        _get_conditions()

    # use a very large sigma to make sure Ts are not independent
    stat_funs = [ttest_1samp_no_p,
                 partial(ttest_1samp_no_p, sigma=1e-1)]

    for stat_fun in stat_funs:
        for condition1 in (condition1_1d, condition1_2d):
            # these are so significant we can get away with fewer perms
            T_obs, clusters, cluster_p_values, hist =\
                permutation_cluster_1samp_test(condition1, n_permutations=100,
                                               tail=0, seed=1,
                                               buffer_size=None)
            assert_equal(np.sum(cluster_p_values < 0.05), 1)

            T_obs_pos, c_1, cluster_p_values_pos, _ =\
                permutation_cluster_1samp_test(condition1, n_permutations=100,
                                               tail=1, threshold=1.67, seed=1,
                                               stat_fun=stat_fun,
                                               buffer_size=None)

            T_obs_neg, _, cluster_p_values_neg, _ =\
                permutation_cluster_1samp_test(-condition1, n_permutations=100,
                                               tail=-1, threshold=-1.67,
                                               seed=1, stat_fun=stat_fun,
                                               buffer_size=None)
            assert_array_equal(T_obs_pos, -T_obs_neg)
            assert_array_equal(cluster_p_values_pos < 0.05,
                               cluster_p_values_neg < 0.05)

            # test with 2 jobs and buffer_size enabled
            buffer_size = condition1.shape[1] // 10
            with warnings.catch_warnings(record=True):  # independently
                T_obs_neg_buff, _, cluster_p_values_neg_buff, _ = \
                    permutation_cluster_1samp_test(
                        -condition1, n_permutations=100, tail=-1,
                        threshold=-1.67, seed=1, n_jobs=2, stat_fun=stat_fun,
                        buffer_size=buffer_size)

            assert_array_equal(T_obs_neg, T_obs_neg_buff)
            assert_array_equal(cluster_p_values_neg, cluster_p_values_neg_buff)


def test_cluster_permutation_with_connectivity():
    """Test cluster level permutations with connectivity matrix."""
    try:
        try:
            from sklearn.feature_extraction.image import grid_to_graph
        except ImportError:
            from scikits.learn.feature_extraction.image import grid_to_graph
    except ImportError:
        return
    condition1_1d, condition2_1d, condition1_2d, condition2_2d = \
        _get_conditions()

    n_pts = condition1_1d.shape[1]
    # we don't care about p-values in any of these, so do fewer permutations
    args = dict(seed=None, max_step=1, exclude=None,
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
        connectivity = grid_to_graph(1, n_pts)
        out_connectivity = func(X1d, connectivity=connectivity, **args)
        assert_array_equal(out[0], out_connectivity[0])
        for a, b in zip(out_connectivity[1], out[1]):
            assert_array_equal(out[0][a], out[0][b])
            assert_true(np.all(a[b]))

        # test spatio-temporal w/o time connectivity (repeat spatial pattern)
        connectivity_2 = sparse.coo_matrix(
            linalg.block_diag(connectivity.asfptype().todense(),
                              connectivity.asfptype().todense()))

        if isinstance(X1d, list):
            X1d_2 = [np.concatenate((x, x), axis=1) for x in X1d]
        else:
            X1d_2 = np.concatenate((X1d, X1d), axis=1)

        out_connectivity_2 = func(X1d_2, connectivity=connectivity_2, **args)
        # make sure we were operating on the same values
        split = len(out[0])
        assert_array_equal(out[0], out_connectivity_2[0][:split])
        assert_array_equal(out[0], out_connectivity_2[0][split:])

        # make sure we really got 2x the number of original clusters
        n_clust_orig = len(out[1])
        assert_true(len(out_connectivity_2[1]) == 2 * n_clust_orig)

        # Make sure that we got the old ones back
        data_1 = set([np.sum(out[0][b[:n_pts]]) for b in out[1]])
        data_2 = set([np.sum(out_connectivity_2[0][a]) for a in
                     out_connectivity_2[1][:]])
        assert_true(len(data_1.intersection(data_2)) == len(data_1))

        # now use the other algorithm
        if isinstance(X1d, list):
            X1d_3 = [np.reshape(x, (-1, 2, n_space)) for x in X1d_2]
        else:
            X1d_3 = np.reshape(X1d_2, (-1, 2, n_space))

        out_connectivity_3 = spatio_temporal_func(X1d_3, n_permutations=50,
                                                  connectivity=connectivity,
                                                  max_step=0, threshold=1.67,
                                                  check_disjoint=True)
        # make sure we were operating on the same values
        split = len(out[0])
        assert_array_equal(out[0], out_connectivity_3[0][0])
        assert_array_equal(out[0], out_connectivity_3[0][1])

        # make sure we really got 2x the number of original clusters
        assert_true(len(out_connectivity_3[1]) == 2 * n_clust_orig)

        # Make sure that we got the old ones back
        data_1 = set([np.sum(out[0][b[:n_pts]]) for b in out[1]])
        data_2 = set([np.sum(out_connectivity_3[0][a[0], a[1]]) for a in
                     out_connectivity_3[1]])
        assert_true(len(data_1.intersection(data_2)) == len(data_1))

        # test new versus old method
        out_connectivity_4 = spatio_temporal_func(X1d_3, n_permutations=50,
                                                  connectivity=connectivity,
                                                  max_step=2, threshold=1.67)
        out_connectivity_5 = spatio_temporal_func(X1d_3, n_permutations=50,
                                                  connectivity=connectivity,
                                                  max_step=1, threshold=1.67)

        # clusters could be in a different order
        sums_4 = [np.sum(out_connectivity_4[0][a])
                  for a in out_connectivity_4[1]]
        sums_5 = [np.sum(out_connectivity_4[0][a])
                  for a in out_connectivity_5[1]]
        sums_4 = np.sort(sums_4)
        sums_5 = np.sort(sums_5)
        assert_array_almost_equal(sums_4, sums_5)

        if not _force_serial:
            assert_raises(ValueError, spatio_temporal_func, X1d_3,
                          n_permutations=1, connectivity=connectivity,
                          max_step=1, threshold=1.67, n_jobs=-1000)

        # not enough TFCE params
        assert_raises(KeyError, spatio_temporal_func, X1d_3,
                      connectivity=connectivity, threshold=dict(me='hello'))

        # too extreme a start threshold
        with warnings.catch_warnings(record=True) as w:
            spatio_temporal_func(X1d_3, connectivity=connectivity,
                                 threshold=dict(start=10, step=1))
        if not did_warn:
            assert_true(len(w) == 1)
            did_warn = True

        # too extreme a start threshold
        assert_raises(ValueError, spatio_temporal_func, X1d_3,
                      connectivity=connectivity, tail=-1,
                      threshold=dict(start=1, step=-1))
        assert_raises(ValueError, spatio_temporal_func, X1d_3,
                      connectivity=connectivity, tail=-1,
                      threshold=dict(start=-1, step=1))
        # Make sure connectivity has to be sparse
        assert_raises(ValueError, spatio_temporal_func, X1d_3,
                      n_permutations=50, connectivity=connectivity.todense(),
                      max_step=1, threshold=1.67)

        # wrong type for threshold
        assert_raises(TypeError, spatio_temporal_func, X1d_3,
                      connectivity=connectivity, threshold=[])

        # wrong value for tail
        assert_raises(ValueError, spatio_temporal_func, X1d_3,
                      connectivity=connectivity, tail=2)

        # make sure it actually found a significant point
        out_connectivity_6 = spatio_temporal_func(X1d_3, n_permutations=50,
                                                  connectivity=connectivity,
                                                  max_step=1,
                                                  threshold=dict(start=1,
                                                                 step=1))
        assert_true(np.min(out_connectivity_6[2]) < 0.05)


def test_permutation_connectivity_equiv():
    """Test cluster level permutations with and without connectivity."""
    try:
        try:
            from sklearn.feature_extraction.image import grid_to_graph
        except ImportError:
            from scikits.learn.feature_extraction.image import grid_to_graph
    except ImportError:
        return
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
    conns = [None,
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
    for thresh, count, max_step, conn in zip(thresholds, sig_counts,
                                             max_steps, conns):
        t, clusters, p, H0 = \
            permutation_cluster_1samp_test(
                X, threshold=thresh, connectivity=conn, n_jobs=2,
                max_step=max_step, stat_fun=stat_fun)
        # make sure our output datatype is correct
        assert_true(isinstance(clusters[0], np.ndarray))
        assert_true(clusters[0].dtype == bool)
        assert_array_equal(clusters[0].shape, X.shape[1:])

        # make sure all comparisons were done; for TFCE, no perm
        # should come up empty
        inds = np.where(p < 0.05)[0]
        assert_equal(len(inds), count)
        if isinstance(thresh, dict):
            assert_equal(len(clusters), n_time * n_space)
            assert_true(np.all(H0 != 0))
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
        assert_true(len(cs) == len(this_cs))
        for c1, c2 in zip(cs, this_cs):
            assert_array_equal(c1, c2)
        assert_array_equal(stat_map, this_stat_map)


def test_spatio_temporal_cluster_connectivity():
    """Test spatio-temporal cluster permutations."""
    try:
        try:
            from sklearn.feature_extraction.image import grid_to_graph
        except ImportError:
            from scikits.learn.feature_extraction.image import grid_to_graph
    except ImportError:
        return
    condition1_1d, condition2_1d, condition1_2d, condition2_2d = \
        _get_conditions()

    rng = np.random.RandomState(0)
    noise1_2d = rng.randn(condition1_2d.shape[0], condition1_2d.shape[1], 10)
    data1_2d = np.transpose(np.dstack((condition1_2d, noise1_2d)), [0, 2, 1])

    noise2_d2 = rng.randn(condition2_2d.shape[0], condition2_2d.shape[1], 10)
    data2_2d = np.transpose(np.dstack((condition2_2d, noise2_d2)), [0, 2, 1])

    conn = grid_to_graph(data1_2d.shape[-1], 1)

    threshold = dict(start=4.0, step=2)
    T_obs, clusters, p_values_conn, hist = \
        spatio_temporal_cluster_test([data1_2d, data2_2d], connectivity=conn,
                                     n_permutations=50, tail=1, seed=1,
                                     threshold=threshold, buffer_size=None)

    buffer_size = data1_2d.size // 10
    T_obs, clusters, p_values_no_conn, hist = \
        spatio_temporal_cluster_test([data1_2d, data2_2d],
                                     n_permutations=50, tail=1, seed=1,
                                     threshold=threshold, n_jobs=2,
                                     buffer_size=buffer_size)

    assert_equal(np.sum(p_values_conn < 0.05), np.sum(p_values_no_conn < 0.05))

    # make sure results are the same without buffer_size
    T_obs, clusters, p_values2, hist2 = \
        spatio_temporal_cluster_test([data1_2d, data2_2d],
                                     n_permutations=50, tail=1, seed=1,
                                     threshold=threshold, n_jobs=2,
                                     buffer_size=None)
    assert_array_equal(p_values_no_conn, p_values2)
    assert_raises(ValueError, spatio_temporal_cluster_test,
                  [data1_2d, data2_2d], tail=1, threshold=-2.)
    assert_raises(ValueError, spatio_temporal_cluster_test,
                  [data1_2d, data2_2d], tail=-1, threshold=2.)
    assert_raises(ValueError, spatio_temporal_cluster_test,
                  [data1_2d, data2_2d], tail=0, threshold=-1)


def ttest_1samp(X):
    """Return T-values."""
    return stats.ttest_1samp(X, 0)[0]


def test_summarize_clusters():
    """Test cluster summary stcs."""
    clu = (np.random.random([1, 20484]),
           [(np.array([0]), np.array([0, 2, 4]))],
           np.array([0.02, 0.1]),
           np.array([12, -14, 30]))
    stc_sum = summarize_clusters_stc(clu)
    assert_true(stc_sum.data.shape[1] == 2)
    clu[2][0] = 0.3
    assert_raises(RuntimeError, summarize_clusters_stc, clu)


def test_permutation_test_H0():
    """Test that H0 is populated properly during testing."""
    rng = np.random.RandomState(0)
    data = rng.rand(7, 10, 1) - 0.5
    with warnings.catch_warnings(record=True) as w:
        t, clust, p, h0 = spatio_temporal_cluster_1samp_test(
            data, threshold=100, n_permutations=1024, seed=rng)
    assert_equal(len(w), 1)
    assert_true('No clusters found' in str(w[0].message))
    assert_equal(len(h0), 0)

    for n_permutations in (1024, 65, 64, 63):
        t, clust, p, h0 = spatio_temporal_cluster_1samp_test(
            data, threshold=0.1, n_permutations=n_permutations, seed=rng)
        assert_equal(len(h0), min(n_permutations, 64))
        assert_true(isinstance(clust[0], tuple))  # sets of indices
    for tail, thresh in zip((-1, 0, 1), (-0.1, 0.1, 0.1)):
        with warnings.catch_warnings(record=True) as w:
            t, clust, p, h0 = spatio_temporal_cluster_1samp_test(
                data, threshold=thresh, seed=rng, tail=tail, out_type='mask')
        assert_equal(len(w), 0)
        assert_true(isinstance(clust[0], np.ndarray))  # bool mask
        # same as "128 if tail else 64"
        assert_equal(len(h0), 2 ** (7 - (tail == 0)))  # exact test


run_tests_if_main()
