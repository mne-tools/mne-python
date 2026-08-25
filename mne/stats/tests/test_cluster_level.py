# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
from functools import partial

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)
from scipy import linalg, sparse, stats

from mne import (
    EpochsArray,
    EvokedArray,
    MixedSourceEstimate,
    SourceEstimate,
    SourceSpaces,
    VolSourceEstimate,
    create_info,
)
from mne.stats import cluster_test, combine_adjacency, ttest_ind_no_p
from mne.stats.cluster_level import (
    ClusterResult,
    _find_clusters,
    _TTestReordered,
    f_oneway,
    permutation_cluster_1samp_test,
    permutation_cluster_test,
    spatio_temporal_cluster_1samp_test,
    spatio_temporal_cluster_test,
    summarize_clusters_stc,
    ttest_1samp_no_p,
)
from mne.stats.parametric import f_mway_rm, f_threshold_mway_rm
from mne.time_frequency import AverageTFRArray, BaseTFR, EpochsTFRArray
from mne.utils import GetEpochsMixin, _record_warnings, catch_logging

n_space = 50


def _get_conditions():
    noise_level = 20
    n_time_1 = 20
    n_time_2 = 13
    normfactor = np.hanning(20).sum()
    rng = np.random.default_rng(42)
    condition1_1d = rng.normal(scale=noise_level, size=(n_time_1, n_space))
    for c in condition1_1d:
        c[:] = np.convolve(c, np.hanning(20), mode="same") / normfactor

    condition2_1d = rng.normal(scale=noise_level, size=(n_time_2, n_space))
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
    # seed chosen so both the 1-sample and between-subjects data are only
    # marginally significant (0.03 < p < 0.05), as the asserts below require
    rng = np.random.default_rng(426)
    X = rng.normal(loc=0.08, size=(10, 1, 1))
    want_thresh = -stats.t.ppf(0.025, len(X) - 1)
    assert 0.03 < stats.ttest_1samp(X[:, 0, 0], 0)[1] < 0.05
    my_fun = partial(ttest_1samp_no_p)
    with catch_logging() as log:
        with pytest.warns(RuntimeWarning, match="threshold is only valid"):
            out = permutation_cluster_1samp_test(
                X, stat_fun=my_fun, seed=0, verbose=True, out_type="mask"
            )
    log = log.getvalue()
    assert str(want_thresh)[:6] in log
    assert len(out[1]) == 1  # 1 cluster
    assert_allclose(out[2], 0.046875, atol=1e-6)
    # between subjects
    Y = rng.standard_normal((10, 1, 1))
    Z = rng.normal(loc=-0.7, size=(10, 1, 1))
    X = [X, Y, Z]
    want_thresh = stats.f.ppf(1.0 - 0.05, 2, sum(len(a) for a in X) - len(X))
    p = stats.f_oneway(*X)[1]
    assert 0.03 < p < 0.05
    my_fun = partial(f_oneway)  # just to make the check fail
    with catch_logging() as log:
        with pytest.warns(RuntimeWarning, match="threshold is only valid"):
            out = permutation_cluster_test(
                X, tail=1, stat_fun=my_fun, seed=0, verbose=True, out_type="mask"
            )
    log = log.getvalue()
    assert str(want_thresh)[:6] in log
    assert len(out[1]) == 1  # 1 cluster
    assert_allclose(out[2], 0.031250, atol=1e-6)
    with pytest.warns(RuntimeWarning, match='Ignoring argument "tail"'):
        permutation_cluster_test(X, tail=0, out_type="mask")

    # nan handling in TFCE
    X = np.repeat(X[0], 2, axis=1)
    X[:, 1] = 0
    with (
        _record_warnings(),
        pytest.warns(RuntimeWarning, match="invalid value"),
    ):  # NumPy
        out = permutation_cluster_1samp_test(
            X, seed=0, threshold=dict(start=0, step=0.1), out_type="mask"
        )
    assert (out[2] < 0.05).any()
    assert not (out[2] < 0.05).all()
    X[:, 0] = 0
    with pytest.raises(RuntimeError, match="finite"):
        with np.errstate(invalid="ignore"):
            permutation_cluster_1samp_test(
                X,
                seed=0,
                threshold=dict(start=0, step=0.1),
                buffer_size=None,
                out_type="mask",
            )


def test_cache_dir(tmp_path, numba_conditional):
    """Test use of cache dir."""
    tempdir = str(tmp_path)
    orig_dir = os.getenv("MNE_CACHE_DIR", None)
    orig_size = os.getenv("MNE_MEMMAP_MIN_SIZE", None)
    # seed chosen so clusters are actually found
    rng = np.random.default_rng(4)
    X = rng.standard_normal((9, 2, 10))
    try:
        os.environ["MNE_MEMMAP_MIN_SIZE"] = "1K"
        os.environ["MNE_CACHE_DIR"] = tempdir
        # Fix error for #1507: in-place when memmapping
        with catch_logging() as log_file:
            permutation_cluster_1samp_test(
                X,
                buffer_size=None,
                n_jobs=2,
                n_permutations=1,
                seed=0,
                stat_fun=ttest_1samp_no_p,
                verbose=False,
                out_type="mask",
            )
        assert "independently" not in log_file.getvalue()
        # ensure that non-independence yields warning
        stat_fun = partial(ttest_1samp_no_p, sigma=1e-3)
        random_state = np.random.default_rng(0)
        with _record_warnings(), pytest.warns(RuntimeWarning, match="independently"):
            permutation_cluster_1samp_test(
                X,
                buffer_size=10,
                n_jobs=2,
                n_permutations=1,
                seed=random_state,
                stat_fun=stat_fun,
                verbose=False,
                out_type="mask",
            )
    finally:
        if orig_dir is not None:
            os.environ["MNE_CACHE_DIR"] = orig_dir
        else:
            del os.environ["MNE_CACHE_DIR"]
        if orig_size is not None:
            os.environ["MNE_MEMMAP_MIN_SIZE"] = orig_size
        else:
            del os.environ["MNE_MEMMAP_MIN_SIZE"]


def test_permutation_large_n_samples(numba_conditional):
    """Test that non-replacement works with large N."""
    X = np.random.default_rng(0).normal(loc=1, size=(72, 1))
    for n_samples in (11, 72):
        tails = (0, 1) if n_samples <= 20 else (0,)
        for tail in tails:
            H0 = permutation_cluster_1samp_test(
                X[:n_samples], threshold=1e-4, tail=tail, seed=0, out_type="mask"
            )[-1]
            assert H0.shape == (1024,)
            assert len(np.unique(H0)) >= 1024 - (H0 == 0).sum()


def test_permutation_step_down_p(numba_conditional):
    """Test cluster level permutations with step_down_p."""
    # seed chosen so step-down yields the improvement asserted below
    rng = np.random.default_rng(11)
    # subjects, time points, spatial points
    X = rng.standard_normal((9, 2, 10))
    # add some significant points
    X[:, 0:2, 0:2] += 2  # span two time points and two spatial points
    X[:, 1, 5:9] += 0.5  # span four time points with 4x smaller amplitude
    thresh = 2
    # make sure it works when we use ALL points in step-down
    t, clusters, p, H0 = permutation_cluster_1samp_test(
        X, threshold=thresh, step_down_p=1.0, out_type="mask"
    )
    # make sure using step-down will actually yield improvements sometimes
    t, clusters, p_old, H0 = permutation_cluster_1samp_test(
        X, threshold=thresh, step_down_p=0.0, out_type="mask"
    )
    assert_equal(np.sum(p_old < 0.05), 1)  # just spatial cluster
    p_min = np.min(p_old)
    assert_allclose(p_min, 0.003906, atol=1e-6)
    t, clusters, p_new, H0 = permutation_cluster_1samp_test(
        X, threshold=thresh, step_down_p=0.05, out_type="mask"
    )
    assert_equal(np.sum(p_new < 0.05), 2)  # time one rescued
    assert np.all(p_old >= p_new)
    p_next = p_new[(p_new > 0.004) & (p_new < 0.05)][0]
    assert_allclose(p_next, 0.015625, atol=1e-6)


def test_cluster_permutation_test(numba_conditional):
    """Test cluster level permutations tests."""
    condition1_1d, condition2_1d, condition1_2d, condition2_2d = _get_conditions()
    for condition1, condition2 in zip(
        (condition1_1d, condition1_2d), (condition2_1d, condition2_2d)
    ):
        T_obs, clusters, cluster_p_values, hist = permutation_cluster_test(
            [condition1, condition2],
            n_permutations=100,
            tail=1,
            seed=1,
            buffer_size=None,
            out_type="mask",
        )
        p_min = np.min(cluster_p_values)
        assert_equal(np.sum(cluster_p_values < 0.05), 1)
        assert_allclose(p_min, 0.01, atol=1e-6)

        # test with 2 jobs and buffer_size enabled
        buffer_size = condition1.shape[1] // 10
        T_obs, clusters, cluster_p_values_buff, hist = permutation_cluster_test(
            [condition1, condition2],
            n_permutations=100,
            tail=1,
            seed=1,
            n_jobs=2,
            buffer_size=buffer_size,
            out_type="mask",
        )
        assert_array_equal(cluster_p_values, cluster_p_values_buff)

    def stat_fun(X, Y):
        return stats.f_oneway(X, Y)[0]

    with pytest.warns(RuntimeWarning, match="is only valid"):
        permutation_cluster_test(
            [condition1, condition2],
            n_permutations=1,
            stat_fun=stat_fun,
            out_type="mask",
        )


@pytest.mark.parametrize(
    "stat_fun", [ttest_1samp_no_p, partial(ttest_1samp_no_p, sigma=1e-1)]
)
def test_cluster_permutation_t_test(numba_conditional, stat_fun):
    """Test cluster level permutations T-test."""
    condition1_1d, _, condition1_2d, _ = _get_conditions()

    # use a very large sigma to make sure Ts are not independent
    for condition1, p in ((condition1_1d, 0.01), (condition1_2d, 0.01)):
        # these are so significant we can get away with fewer perms
        T_obs, clusters, cluster_p_values, hist = permutation_cluster_1samp_test(
            condition1,
            n_permutations=100,
            tail=0,
            seed=1,
            out_type="mask",
            buffer_size=None,
        )
        assert_equal(np.sum(cluster_p_values < 0.05), 1)
        p_min = np.min(cluster_p_values)
        assert_allclose(p_min, p, atol=1e-6)

        T_obs_pos, _, cluster_p_values_pos, _ = permutation_cluster_1samp_test(
            condition1,
            n_permutations=100,
            tail=1,
            threshold=1.67,
            seed=1,
            stat_fun=stat_fun,
            out_type="mask",
            buffer_size=None,
        )

        T_obs_neg, _, cluster_p_values_neg, _ = permutation_cluster_1samp_test(
            -condition1,
            n_permutations=100,
            tail=-1,
            threshold=-1.67,
            seed=1,
            stat_fun=stat_fun,
            buffer_size=None,
            out_type="mask",
        )
        assert_array_equal(T_obs_pos, -T_obs_neg)
        assert_array_equal(cluster_p_values_pos < 0.05, cluster_p_values_neg < 0.05)

        # test with 2 jobs and buffer_size enabled
        buffer_size = condition1.shape[1] // 10
        with _record_warnings():  # sometimes "independently"
            (
                T_obs_neg_buff,
                _,
                cluster_p_values_neg_buff,
                _,
            ) = permutation_cluster_1samp_test(
                -condition1,
                n_permutations=100,
                tail=-1,
                out_type="mask",
                threshold=-1.67,
                seed=1,
                n_jobs=2,
                stat_fun=stat_fun,
                buffer_size=buffer_size,
            )

        assert_array_equal(T_obs_neg, T_obs_neg_buff)
        assert_array_equal(cluster_p_values_neg, cluster_p_values_neg_buff)

        # Bad stat_fun
        with pytest.raises(TypeError, match="must be .* ndarray"):
            permutation_cluster_1samp_test(
                condition1, threshold=1, stat_fun=lambda x: None, out_type="mask"
            )
        with pytest.raises(ValueError, match="not compatible"):
            permutation_cluster_1samp_test(
                condition1,
                threshold=1,
                stat_fun=lambda x: stat_fun(x)[:-1],
                out_type="mask",
            )


def test_cluster_permutation_with_adjacency(numba_conditional, monkeypatch):
    """Test cluster level permutations with adjacency matrix."""
    pytest.importorskip("sklearn")
    from sklearn.feature_extraction.image import grid_to_graph

    condition1_1d, condition2_1d, _, _ = _get_conditions()

    n_pts = condition1_1d.shape[1]
    # we don't care about p-values in any of these, so do fewer permutations
    args = dict(
        seed=None,
        max_step=1,
        exclude=None,
        out_type="mask",
        step_down_p=0,
        t_power=1,
        threshold=1.67,
        check_disjoint=False,
        n_permutations=50,
    )

    did_warn = False
    for X1d, func, spatio_temporal_func in [
        (
            condition1_1d,
            permutation_cluster_1samp_test,
            spatio_temporal_cluster_1samp_test,
        ),
        (
            [condition1_1d, condition2_1d],
            permutation_cluster_test,
            spatio_temporal_cluster_test,
        ),
    ]:
        out = func(X1d, **args)
        adjacency = grid_to_graph(1, n_pts)
        out_adjacency = func(X1d, adjacency=adjacency, **args)
        assert_array_equal(out[0], out_adjacency[0])
        for a, b in zip(out_adjacency[1], out[1]):
            assert_array_equal(out[0][a], out[0][b])
            assert np.all(a[b])

        # test spatio-temporal w/o time adjacency (repeat spatial pattern)
        adjacency_2 = sparse.coo_array(
            linalg.block_diag(
                adjacency.asfptype().todense(), adjacency.asfptype().todense()
            )
        )
        # nesting here is time then space:
        adjacency_2a = combine_adjacency(sparse.eye_array(2), adjacency)
        assert_array_equal(
            adjacency_2.toarray().astype(bool), adjacency_2a.toarray().astype(bool)
        )

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
        data_2 = {np.sum(out_adjacency_2[0][a]) for a in out_adjacency_2[1][:]}
        assert len(data_1.intersection(data_2)) == len(data_1)

        # now use the other algorithm
        if isinstance(X1d, list):
            X1d_3 = [np.reshape(x, (-1, 2, n_space)) for x in X1d_2]
        else:
            X1d_3 = np.reshape(X1d_2, (-1, 2, n_space))

        out_adjacency_3 = spatio_temporal_func(
            X1d_3,
            n_permutations=50,
            adjacency=adjacency,
            max_step=0,
            threshold=1.67,
            check_disjoint=True,
        )
        # make sure we were operating on the same values
        split = len(out[0])
        assert_array_equal(out[0], out_adjacency_3[0][0])
        assert_array_equal(out[0], out_adjacency_3[0][1])

        # make sure we really got 2x the number of original clusters
        assert len(out_adjacency_3[1]) == 2 * n_clust_orig

        # Make sure that we got the old ones back
        data_1 = {np.sum(out[0][b[:n_pts]]) for b in out[1]}
        data_2 = {np.sum(out_adjacency_3[0][a[0], a[1]]) for a in out_adjacency_3[1]}
        assert len(data_1.intersection(data_2)) == len(data_1)

        # test new versus old method
        out_adjacency_4 = spatio_temporal_func(
            X1d_3, n_permutations=50, adjacency=adjacency, max_step=2, threshold=1.67
        )
        out_adjacency_5 = spatio_temporal_func(
            X1d_3, n_permutations=50, adjacency=adjacency, max_step=1, threshold=1.67
        )

        # clusters could be in a different order
        sums_4 = [np.sum(out_adjacency_4[0][a]) for a in out_adjacency_4[1]]
        sums_5 = [np.sum(out_adjacency_4[0][a]) for a in out_adjacency_5[1]]
        sums_4 = np.sort(sums_4)
        sums_5 = np.sort(sums_5)
        assert_array_almost_equal(sums_4, sums_5)

        monkeypatch.delenv("MNE_FORCE_SERIAL", raising=False)
        with pytest.raises(ValueError, match="must not be less"):
            spatio_temporal_func(
                X1d_3,
                n_permutations=1,
                adjacency=adjacency,
                max_step=1,
                threshold=1.67,
                n_jobs=-1000,
            )

        # not enough TFCE params
        with pytest.raises(KeyError, match="threshold, if dict, must have"):
            spatio_temporal_func(X1d_3, adjacency=adjacency, threshold=dict(me="hello"))

        # too extreme a start threshold
        with _record_warnings() as w:
            spatio_temporal_func(
                X1d_3, adjacency=adjacency, threshold=dict(start=10, step=1)
            )
        if not did_warn:
            messages = [str(ww.message) for ww in w]
            assert any("is more extreme" in message for message in messages), messages
            did_warn = True

        with pytest.raises(ValueError, match="threshold.*<= 0 for tail == -1"):
            spatio_temporal_func(
                X1d_3, adjacency=adjacency, tail=-1, threshold=dict(start=1, step=-1)
            )
        with pytest.warns(RuntimeWarning, match="threshold.* is more extreme"):
            spatio_temporal_func(
                X1d_3, adjacency=adjacency, tail=1, threshold=dict(start=100, step=1)
            )
        bad_con = adjacency.todense()
        with pytest.raises(ValueError, match="must be a SciPy sparse matrix"):
            spatio_temporal_func(
                X1d_3, n_permutations=50, adjacency=bad_con, max_step=1, threshold=1.67
            )
        bad_con = adjacency.tocsr()[:-1, :-1].tocoo()
        with pytest.raises(ValueError, match="adjacency.*the correct size"):
            spatio_temporal_func(
                X1d_3, n_permutations=50, adjacency=bad_con, max_step=1, threshold=1.67
            )
        with pytest.raises(TypeError, match="must be a"):
            spatio_temporal_func(X1d_3, adjacency=adjacency, threshold=[])
        with pytest.raises(ValueError, match="Invalid value for the 'tail'"):
            # sometimes ignoring tail
            with _record_warnings():
                spatio_temporal_func(X1d_3, adjacency=adjacency, tail=2)

        # make sure it actually found a significant point
        out_adjacency_6 = spatio_temporal_func(
            X1d_3,
            n_permutations=50,
            adjacency=adjacency,
            max_step=1,
            threshold=dict(start=1, step=1),
        )
        assert np.min(out_adjacency_6[2]) < 0.05

        with pytest.raises(ValueError, match="not compatible"):
            with _record_warnings():
                spatio_temporal_func(
                    X1d_3,
                    n_permutations=50,
                    adjacency=adjacency,
                    threshold=1e-3,
                    stat_fun=lambda *x: f_oneway(*x)[:-1],
                    buffer_size=None,
                )


@pytest.mark.parametrize(
    "threshold",
    [
        0.1,
        pytest.param(dict(start=0.0, step=0.5), id="TFCE"),
    ],
)
@pytest.mark.parametrize("kind", ("1samp", "ind"))
def test_permutation_cluster_signs(threshold, kind):
    """Test cluster signs."""
    # difference between two conditions for 3 subjects x 2 vertices x 2 times
    X = np.array(
        [[[-10, 5], [-2, -7]], [[-4, 5], [-8, -0]], [[-6, 3], [-4, -2]]], float
    )
    want_signs = np.sign(np.mean(X, axis=0))
    n_permutations = 1
    if kind == "1samp":
        func = permutation_cluster_1samp_test
        stat_fun = ttest_1samp_no_p
        use_X = X
    else:
        assert kind == "ind"
        func = permutation_cluster_test
        stat_fun = ttest_ind_no_p
        use_X = [X, np.random.default_rng(0).normal(scale=0.1, size=X.shape)]
    tobs, clu, clu_pvalues, _ = func(
        use_X,
        n_permutations=n_permutations,
        threshold=threshold,
        tail=0,
        stat_fun=stat_fun,
        out_type="mask",
    )
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


def test_permutation_adjacency_equiv(numba_conditional):
    """Test cluster level permutations with and without adjacency."""
    pytest.importorskip("sklearn")
    from sklearn.feature_extraction.image import grid_to_graph

    rng = np.random.default_rng(0)
    # subjects, time points, spatial points
    n_time = 2
    n_space = 4
    X = rng.standard_normal((6, n_time, n_space))
    # add some significant points
    X[:, :, 0:2] += 10  # span two time points and two spatial points
    X[:, 1, 3] += 20  # span one time point
    max_steps = [1, 1, 1, 2, 1]
    # This will run full algorithm in two ways, then the ST-algorithm in 2 ways
    # All of these should give the same results
    adjs = [
        None,
        grid_to_graph(n_time, n_space),
        grid_to_graph(1, n_space),
        grid_to_graph(1, n_space),
        None,
    ]
    stat_map = None
    thresholds = [2, 2, 2, 2, dict(start=0.01, step=1.0)]
    sig_counts = [2, 2, 2, 2, 5]
    stat_fun = partial(ttest_1samp_no_p, sigma=1e-3)

    cs = None
    ps = None
    for thresh, count, max_step, adj in zip(thresholds, sig_counts, max_steps, adjs):
        t, clusters, p, H0 = permutation_cluster_1samp_test(
            X,
            threshold=thresh,
            adjacency=adj,
            n_jobs=2,
            max_step=max_step,
            stat_fun=stat_fun,
            seed=0,
            out_type="mask",
        )
        # make sure our output datatype is correct
        assert isinstance(clusters[0], np.ndarray)
        assert clusters[0].dtype == np.dtype(bool)
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


def test_spatio_temporal_cluster_chain_merge():
    """Test that a chain of spatio-temporal merges combines into one cluster."""
    # Regression test: joining these active points into one cluster requires
    # a chain of 3 merges alternating between spatial and temporal adjacency
    # (t0's {2, 3, 4} - t1's {4} - t0's {0} - t1's {0, 1, 5}); that chain used
    # to get broken, incorrectly splitting off {(1, 2)} as its own cluster.
    # seed chosen to produce the cluster-merge chain described above
    rng = np.random.default_rng(1)
    n_subjects, n_times, n_space = 3, 2, 6
    X = rng.normal(scale=0.01, size=(n_subjects, n_times, n_space))
    active = [(0, 0), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 4), (1, 5)]
    for t, s in active:
        X[:, t, s] += 10
    # path graph: 0-1-5-4-3-2
    row = np.array([0, 1, 2, 3, 4])
    col = np.array([1, 5, 3, 4, 5])
    adjacency = sparse.coo_array((np.ones(5), (row, col)), shape=(n_space, n_space))

    _, clusters, cluster_pv, _ = permutation_cluster_1samp_test(
        X,
        threshold=5.0,
        tail=1,
        adjacency=adjacency,
        max_step=1,
        n_permutations=20,
        out_type="indices",
        seed=0,
        verbose=False,
    )
    assert len(clusters) == 1
    t_idx, s_idx = clusters[0]
    assert_equal(sorted(zip(t_idx.tolist(), s_idx.tolist())), active)
    assert_allclose(cluster_pv, [1 / 4])


def test_ttest_reordered():
    """Test that _TTestReordered matches ttest_1samp_no_p under sign flips."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((9, 5))
    X_orig = X.copy()
    stat_fun = _TTestReordered(X)
    for _ in range(5):
        signs = rng.choice([-1, 1], size=9)
        want = ttest_1samp_no_p(X * signs[:, None])
        assert_allclose(stat_fun(X * signs[:, None]), want)
        # from_signs should match, and must not mutate X
        assert_allclose(stat_fun.from_signs(signs.astype(float), X), want)
        assert_array_equal(X, X_orig)
    # degenerate (zero-variance) columns should give 0, not nan/inf
    X_deg = np.ones((9, 1))
    assert_allclose(_TTestReordered(X_deg)(X_deg), 0.0)


@pytest.mark.parametrize("t_power", (1, 2))
@pytest.mark.parametrize("kind", ("no_adjacency", "global", "spatio_temporal"))
def test_find_clusters_sums_only(kind, t_power):
    """Test that sums_only=True matches the full-clusters path."""
    rng = np.random.default_rng(0)
    n_space = 8
    kwargs = dict(threshold=0.0, tail=0, t_power=t_power)
    if kind == "spatio_temporal":
        x = rng.standard_normal(3 * n_space)  # 3 timepoints
        row, col = np.array([0, 1, 2, 3, 4]), np.array([1, 5, 3, 4, 5])
        adj = sparse.coo_array((np.ones(5), (row, col)), shape=(n_space, n_space))
        # spatio-temporal adjacency is always CSR (see _setup_adjacency)
        kwargs["adjacency"] = (adj + adj.transpose()).tocsr()
    elif kind == "global":
        x = rng.standard_normal(n_space)
        row, col = np.arange(n_space - 1), np.arange(1, n_space)
        kwargs["adjacency"] = sparse.coo_array(
            (np.ones(n_space - 1), (row, col)), shape=(n_space, n_space)
        )
    else:
        x = rng.standard_normal(n_space)
        kwargs["adjacency"] = False

    clusters, sums_full = _find_clusters(x, **kwargs)
    assert clusters is not None
    clusters_none, sums_only = _find_clusters(x, sums_only=True, **kwargs)
    assert clusters_none is None
    assert_allclose(sorted(sums_only), sorted(sums_full))


def test_spatio_temporal_cluster_adjacency(numba_conditional):
    """Test spatio-temporal cluster permutations."""
    pytest.importorskip("sklearn")
    from sklearn.feature_extraction.image import grid_to_graph

    condition1_1d, condition2_1d, condition1_2d, condition2_2d = _get_conditions()

    rng = np.random.default_rng(0)
    noise1_2d = rng.standard_normal(
        (condition1_2d.shape[0], condition1_2d.shape[1], 10)
    )
    data1_2d = np.transpose(np.dstack((condition1_2d, noise1_2d)), [0, 2, 1])

    noise2_d2 = rng.standard_normal(
        (condition2_2d.shape[0], condition2_2d.shape[1], 10)
    )
    data2_2d = np.transpose(np.dstack((condition2_2d, noise2_d2)), [0, 2, 1])

    adj = grid_to_graph(data1_2d.shape[-1], 1)

    threshold = dict(start=4.0, step=2)
    T_obs, clusters, p_values_adj, hist = spatio_temporal_cluster_test(
        [data1_2d, data2_2d],
        adjacency=adj,
        n_permutations=50,
        tail=1,
        seed=1,
        threshold=threshold,
        buffer_size=None,
    )

    buffer_size = data1_2d.size // 10
    T_obs, clusters, p_values_no_adj, hist = spatio_temporal_cluster_test(
        [data1_2d, data2_2d],
        n_permutations=50,
        tail=1,
        seed=1,
        threshold=threshold,
        n_jobs=2,
        buffer_size=buffer_size,
    )

    assert_equal(np.sum(p_values_adj < 0.05), np.sum(p_values_no_adj < 0.05))

    # make sure results are the same without buffer_size
    T_obs, clusters, p_values2, _ = spatio_temporal_cluster_test(
        [data1_2d, data2_2d],
        n_permutations=50,
        tail=1,
        seed=1,
        threshold=threshold,
        n_jobs=2,
        buffer_size=None,
    )
    assert_array_equal(p_values_no_adj, p_values2)
    pytest.raises(
        ValueError,
        spatio_temporal_cluster_test,
        [data1_2d, data2_2d],
        tail=1,
        threshold=-2.0,
    )
    pytest.raises(
        ValueError,
        spatio_temporal_cluster_test,
        [data1_2d, data2_2d],
        tail=-1,
        threshold=2.0,
    )
    pytest.raises(
        ValueError,
        spatio_temporal_cluster_test,
        [data1_2d, data2_2d],
        tail=0,
        threshold=-1,
    )


def ttest_1samp(X):
    """Return T-values."""
    return stats.ttest_1samp(X, 0)[0]


@pytest.mark.parametrize("kind", ("surface", "volume", "mixed"))
def test_summarize_clusters(kind):
    """Test cluster summary stcs."""
    src_surf = SourceSpaces(
        [dict(vertno=np.arange(10242), type="surf") for _ in range(2)]
    )
    assert src_surf.kind == "surface"
    src_vol = SourceSpaces([dict(vertno=np.arange(10), type="vol")])
    assert src_vol.kind == "volume"
    if kind == "surface":
        src = src_surf
        klass = SourceEstimate
    elif kind == "volume":
        src = src_vol
        klass = VolSourceEstimate
    else:
        assert kind == "mixed"
        src = src_surf + src_vol
        klass = MixedSourceEstimate
    n_vertices = sum(len(s["vertno"]) for s in src)
    rng = np.random.default_rng(0)
    clu = (
        rng.random([1, n_vertices]),
        [(np.array([0]), np.array([0, 2, 4]))],
        np.array([0.02, 0.1]),
        np.array([12, -14, 30]),
    )
    kwargs = dict()
    if kind == "volume":
        with pytest.raises(ValueError, match="did not match"):
            summarize_clusters_stc(clu)
        assert len(src) == 1
        kwargs["vertices"] = [src[0]["vertno"]]
    elif kind == "mixed":
        kwargs["vertices"] = src
    stc_sum = summarize_clusters_stc(clu, **kwargs)
    assert isinstance(stc_sum, klass)
    assert stc_sum.data.shape[1] == 2
    clu[2][0] = 0.3
    with pytest.raises(RuntimeError, match="No significant"):
        summarize_clusters_stc(clu, **kwargs)


def test_permutation_test_H0(numba_conditional):
    """Test that H0 is populated properly during testing."""
    rng = np.random.default_rng(0)
    data = rng.random((7, 10, 1)) - 0.5
    with pytest.warns(RuntimeWarning, match="No clusters found"):
        t, clust, p, h0 = spatio_temporal_cluster_1samp_test(
            data, threshold=100, n_permutations=1024, seed=rng
        )
    assert_equal(len(h0), 0)

    for n_permutations in (1024, 65, 64, 63):
        t, clust, p, h0 = spatio_temporal_cluster_1samp_test(
            data, threshold=0.1, n_permutations=n_permutations, seed=rng
        )
        assert_equal(len(h0), min(n_permutations, 64))
        assert isinstance(clust[0], tuple)  # sets of indices
    for tail, thresh in zip((-1, 0, 1), (-0.1, 0.1, 0.1)):
        t, clust, p, h0 = spatio_temporal_cluster_1samp_test(
            data, threshold=thresh, seed=rng, tail=tail, out_type="mask"
        )
        assert isinstance(clust[0], np.ndarray)  # bool mask
        # same as "128 if tail else 64"
        assert_equal(len(h0), 2 ** (7 - (tail == 0)))  # exact test


def test_tfce_thresholds(numba_conditional):
    """Test TFCE thresholds."""
    rng = np.random.default_rng(0)
    data = rng.normal(loc=-0.5, size=(7, 10, 1))

    # if tail==-1, step must also be negative
    with pytest.raises(ValueError, match="must be < 0 for tail == -1"):
        permutation_cluster_1samp_test(
            data, tail=-1, out_type="mask", threshold=dict(start=0, step=0.1)
        )
    # this works (smoke test)
    permutation_cluster_1samp_test(
        data, tail=-1, out_type="mask", threshold=dict(start=0, step=-0.1)
    )

    # thresholds must be monotonically increasing
    with pytest.raises(ValueError, match="must be monotonically increasing"):
        permutation_cluster_1samp_test(
            data, tail=1, out_type="mask", threshold=dict(start=1, step=-0.5)
        )

    # Should work with 2D data too
    permutation_cluster_1samp_test(X=data[..., 0], threshold=dict(start=0, step=0.2))


# 1D gives slices, 2D+ gives boolean masks
@pytest.mark.parametrize("shape", ((11,), (11, 3), (11, 1, 2)))
@pytest.mark.parametrize("out_type", ("mask", "indices"))
@pytest.mark.parametrize("adjacency", (None, "sparse"))
@pytest.mark.parametrize("threshold", (None, dict(start=0, step=0.1)))
def test_output_equiv(shape, out_type, adjacency, threshold):
    """Test equivalence of output types."""
    rng = np.random.default_rng(0)
    n_subjects = 10
    data = rng.standard_normal((n_subjects, *shape))
    data -= data.mean(axis=0, keepdims=True)
    data[:, 2:4] += 2
    data[:, 6:9] += 2
    tfce = isinstance(threshold, dict)
    want_mask = np.zeros(shape, int)
    if not tfce:
        want_mask[2:4] = 1
        want_mask[6:9] = 2
    else:
        want_mask = np.arange(want_mask.size).reshape(shape) + 1
    if adjacency is not None:
        assert adjacency == "sparse"
        adjacency = combine_adjacency(*shape)
    clusters = permutation_cluster_1samp_test(
        X=data,
        n_permutations=1,
        adjacency=adjacency,
        out_type=out_type,
        threshold=threshold,
    )[1]
    got_mask = np.zeros_like(want_mask)
    for n, clu in enumerate(clusters, 1):
        if out_type == "mask":
            if len(shape) == 1 and adjacency is None:
                assert isinstance(clu, tuple)
                assert len(clu) == 1
                assert isinstance(clu[0], slice)
            else:
                assert isinstance(clu, np.ndarray)
                assert clu.dtype == np.dtype(bool)
                assert clu.shape == shape
            got_mask[clu] = n
        else:
            assert isinstance(clu, tuple)
            for c in clu:
                assert isinstance(c, np.ndarray)
                assert c.dtype.kind == "i"
            assert out_type == "indices"
            got_mask[np.ix_(*clu)] = n
    assert_array_equal(got_mask, want_mask)


def test_cluster_test_one_sample():
    """Test cluster_test with a single-group (1-sample) design."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("formulaic")  # required for cluster_test API
    condition1_1d, _, _, _ = _get_conditions()
    df = pd.DataFrame(dict(data=[condition1_1d], group=["only"]))
    kwargs = dict(n_permutations=100, tail=0, seed=1, buffer_size=None)
    T_obs, clusters, cluster_pvals, H0 = permutation_cluster_1samp_test(
        condition1_1d, **kwargs
    )
    result = cluster_test(df, "data ~ group", **kwargs)
    assert result.stat_name == "paired T-statistic"
    assert_array_equal(result.H0, H0)
    assert_array_equal(result.stat_obs, T_obs)
    assert_array_equal(result.cluster_p_values, cluster_pvals)
    assert len(result.clusters) == len(clusters)
    for clu1, clu2 in zip(result.clusters, clusters):
        assert_array_equal(clu1, clu2)


def test_compare_old_and_new_cluster_api():
    """Test for same results from old and new APIs."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("formulaic")

    condition1_1d, condition2_1d, condition1_2d, condition2_2d = _get_conditions()
    df_1d = pd.DataFrame(
        dict(
            data=[condition1_1d, condition2_1d],
            condition=["a", "b"],
        )
    )
    kwargs = dict(n_permutations=100, tail=1, seed=1, buffer_size=None, out_type="mask")
    F_obs, clusters, cluster_pvals, H0 = permutation_cluster_test(
        [condition1_1d, condition2_1d], **kwargs
    )
    formula = "data ~ condition"
    cluster_result = cluster_test(df_1d, formula, **kwargs)
    assert_array_equal(cluster_result.H0, H0)
    assert_array_equal(cluster_result.stat_obs, F_obs)
    assert_array_equal(cluster_result.cluster_p_values, cluster_pvals)
    assert cluster_result.clusters == clusters


@pytest.mark.parametrize(
    "Inst", (EpochsArray, EvokedArray, EpochsTFRArray, AverageTFRArray)
)
@pytest.mark.filterwarnings('ignore:Ignoring argument "tail":RuntimeWarning')
def test_new_cluster_api(Inst):
    """Test handling different MNE objects in the cluster API."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("formulaic")

    rng = np.random.default_rng(seed=8675309)
    is_epo = GetEpochsMixin in Inst.__mro__
    is_tfr = BaseTFR in Inst.__mro__

    n_epo, n_chan, n_freq, n_times = 6, 3, 4, 5

    # prepare the dimensions of the simulated data, then simulate
    size = (n_chan,)
    if is_epo:
        size = (n_epo, *size)
    if is_tfr:
        size = (*size, n_freq)
    size = (*size, n_times)
    data = rng.normal(size=size)

    # construct the instance
    info = create_info(ch_names=n_chan, sfreq=1000, ch_types="eeg")
    kw = dict(times=np.arange(n_times), freqs=np.arange(n_freq)) if is_tfr else dict()
    cond_a = Inst(data=data, info=info, **kw)
    cond_b = cond_a.copy()
    # introduce a significant difference in a specific region, time, and frequency
    ch_start, ch_end = 0, 2  # 2 channels
    t_start, t_end = 2, 4  # 2 times
    f_start, f_end = 2, 4  # 2 freqs
    if is_tfr:
        cond_b._data[..., ch_start:ch_end, f_start:f_end, t_start:t_end] += 2
    else:
        cond_b._data[..., ch_start:ch_end, t_start:t_end] += 2
    # for Evokeds/AverageTFRs, we create fake "subjects" as our observations within each
    # condition. We add a bit of noise while we do so.
    if not is_epo:
        insts = list()
        for cond in cond_a, cond_b:
            for _n in range(n_epo):
                if not _n:
                    insts.append(cond)
                    continue
                _cond = cond.copy()
                _cond.data += rng.normal(scale=0.1, size=_cond.data.shape)
                insts.append(_cond)
        conds = np.repeat(["a", "b"], n_epo).tolist()
    else:
        # For Epochs(TFR)Array, each epoch is an observation and they're already
        # noisy/non-identical, so no duplication / noise-addition necessary.
        insts = [cond_a, cond_b]
        conds = ["a", "b"]

    # run new clustering API
    df = pd.DataFrame(dict(data=insts, condition=conds))
    kwargs = dict(
        n_permutations=100, seed=42, tail=1, buffer_size=None, out_type="mask"
    )
    result_new_api = cluster_test(df, "data~condition", **kwargs)

    # make sure channels are last dimension for old API
    if is_epo:
        axes = (0, 3, 2, 1) if is_tfr else (0, 2, 1)
        X = [cond_a.get_data().transpose(*axes), cond_b.get_data().transpose(*axes)]
    else:
        axes = (2, 1, 0) if is_tfr else (1, 0)
        Xa = list()
        Xb = list()
        for inst, cond in zip(insts, conds):
            container = Xa if cond == "a" else Xb
            container.append(inst.get_data().transpose(*axes))
        X = [np.stack(Xa), np.stack(Xb)]

    F_obs, clusters, cluster_pvals, H0 = permutation_cluster_test(X, **kwargs)
    assert_array_almost_equal(result_new_api.H0, H0)
    assert_array_almost_equal(result_new_api.stat_obs, F_obs)
    assert_array_almost_equal(result_new_api.cluster_p_values, cluster_pvals)
    assert len(result_new_api.clusters) == len(clusters)
    for clu1, clu2 in zip(result_new_api.clusters, clusters):
        assert_array_equal(clu1, clu2)


@pytest.mark.filterwarnings('ignore:Ignoring argument "tail":RuntimeWarning')
def test_cluster_test_rm_anova():
    """Test the interaction-formula (repeated-measures ANOVA) branch of cluster_test."""
    pd = pytest.importorskip("pandas")

    rng = np.random.default_rng(seed=0)
    n_subjects, n_channels, n_times = 8, 3, 6
    info = create_info(n_channels, sfreq=100.0, ch_types="eeg")
    factor_levels = [2, 2]
    conditions = ["a1b1", "a1b2", "a2b1", "a2b2"]
    data = {
        cond: rng.normal(size=(n_subjects, n_channels, n_times)) for cond in conditions
    }
    # inject an interaction effect (crossover pattern) in the first 2 channels
    data["a1b1"][:, :2] += 3
    data["a2b2"][:, :2] += 3
    data["a1b2"][:, :2] -= 3
    data["a2b1"][:, :2] -= 3

    # reference: old-style call with a hand-rolled f_mway_rm stat_fun, exactly as
    # done in tutorials/stats-sensor-space/70_cluster_rmANOVA_time_freq.py
    def stat_fun(*args):
        return f_mway_rm(
            np.swapaxes(np.asarray(args), 1, 0),
            factor_levels=factor_levels,
            effects="A:B",
            return_pvals=False,
        )[0]

    f_thresh = f_threshold_mway_rm(
        n_subjects, factor_levels, effects="A:B", pvalue=0.001
    )
    # channels last, as required by permutation_cluster_test
    X_old = [data[cond].transpose(0, 2, 1) for cond in conditions]
    kwargs = dict(
        n_permutations=100,
        tail=1,
        seed=3,
        buffer_size=None,
        out_type="mask",
        threshold=f_thresh,
    )
    F_obs, clusters, cluster_pvals, H0 = permutation_cluster_test(
        X_old, stat_fun=stat_fun, **kwargs
    )

    # new API: one row per (subject, condition), with an EvokedArray holding that
    # subject's data for that condition
    rows = list()
    for cond in conditions:
        for subj in range(n_subjects):
            rows.append(
                dict(
                    data=EvokedArray(data[cond][subj], info, tmin=0),
                    modality=cond[1],
                    location=cond[3],
                    subject=subj,
                )
            )
    df = pd.DataFrame(rows)
    result = cluster_test(df, "data ~ modality:location", within_id="subject", **kwargs)

    assert result.stat_name == "F-statistic (repeated-measures ANOVA)"
    assert_array_almost_equal(result.stat_obs, F_obs)
    assert_array_almost_equal(result.H0, H0)
    assert_array_almost_equal(result.cluster_p_values, cluster_pvals)
    assert len(result.clusters) == len(clusters)
    for clu1, clu2 in zip(result.clusters, clusters):
        assert_array_equal(clu1, clu2)


def test_cluster_test_formula_validation():
    """Test that cluster_test raises clear errors for unsupported formulas."""
    pd = pytest.importorskip("pandas")

    condition1_1d, condition2_1d, _, _ = _get_conditions()
    df = pd.DataFrame(dict(data=[condition1_1d, condition2_1d], a=["x", "y"]))
    df["b"] = "z"

    # multi-term right-hand side ("a+b") is not a single effect
    with pytest.raises(ValueError, match="single term"):
        cluster_test(df, "data ~ a+b")

    # interaction effect requires within_id
    with pytest.raises(ValueError, match="repeated-measures"):
        cluster_test(df, "data ~ a:b")

    # unbalanced repeated-measures design (subject missing an observation)
    rows = [
        dict(data=condition1_1d, a="x", b="p", subject=0),
        dict(data=condition2_1d, a="x", b="q", subject=0),
        dict(data=condition1_1d, a="y", b="p", subject=0),
        # subject 0 is missing the "y"/"q" combination
        dict(data=condition1_1d, a="x", b="p", subject=1),
        dict(data=condition2_1d, a="x", b="q", subject=1),
        dict(data=condition1_1d, a="y", b="p", subject=1),
        dict(data=condition2_1d, a="y", b="q", subject=1),
    ]
    df_unbalanced = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="must have exactly"):
        cluster_test(df_unbalanced, "data ~ a:b", within_id="subject")


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive.*:UserWarning")
def test_cluster_test_plot_cluster_time_frequency():
    """Test ClusterResult.plot_cluster_time_frequency."""
    import matplotlib.pyplot as plt

    pd = pytest.importorskip("pandas")
    pytest.importorskip("formulaic")

    rng = np.random.default_rng(seed=0)
    n_subjects, n_channels, n_freqs, n_times = 6, 4, 3, 5
    ch_names = ["Fz", "Cz", "Pz", "Oz"]
    info = create_info(ch_names, sfreq=100.0, ch_types="eeg")
    info.set_montage("colin27_1020")
    freqs = np.arange(n_freqs)
    times = np.arange(n_times) / 10.0

    def make_tfr(bump):
        data = rng.normal(size=(n_channels, n_freqs, n_times))
        if bump:
            data[:2, 1:, 2:] += 4
        return AverageTFRArray(info=info, data=data, times=times, freqs=freqs)

    rows = list()
    for _ in range(n_subjects):
        rows.append(dict(data=make_tfr(False), condition="a"))
        rows.append(dict(data=make_tfr(True), condition="b"))
    df = pd.DataFrame(rows)
    result = cluster_test(
        df,
        "data ~ condition",
        n_permutations=100,
        tail=1,
        seed=1,
        buffer_size=None,
        out_type="indices",
    )
    assert result.stat_obs.ndim == 3  # (time, freq, channel)
    assert result.cluster_masses.shape == result.cluster_p_values.shape
    assert (result.cluster_masses > 0).all()  # F-statistic is non-negative
    result.plot_cluster_time_frequency(df["data"].iloc[0])
    plt.close("all")


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive.*:UserWarning")
def test_cluster_test_plot_cluster_time_frequency_disjoint_clusters():
    """cluster_idx must select between clusters, ranked by mass not p-value."""
    import matplotlib.pyplot as plt

    pd = pytest.importorskip("pandas")

    rng = np.random.default_rng(seed=0)
    n_subjects, n_channels, n_freqs, n_times = 10, 4, 3, 6
    ch_names = ["Fz", "Cz", "Pz", "Oz"]
    info = create_info(ch_names, sfreq=100.0, ch_types="eeg")
    info.set_montage("colin27_1020")
    freqs = np.arange(n_freqs)
    times = np.arange(n_times) / 10.0

    rows = list()
    for _ in range(n_subjects):
        data = rng.normal(scale=0.1, size=(n_channels, n_freqs, n_times))
        data[:2, :, :2] += 4  # clear positive effect, early times, Fz/Cz
        data[2:, :, 3:] -= 4  # clear negative effect, later times, Pz/Oz
        rows.append(
            dict(
                data=AverageTFRArray(info=info, data=data, times=times, freqs=freqs),
                group="only",
            )
        )
    df = pd.DataFrame(rows)
    result = cluster_test(
        df, "data ~ group", n_permutations=100, tail=0, seed=1, buffer_size=None
    )
    signs = [np.sign(result.stat_obs[c].mean()) for c in result.clusters]
    assert 1 in signs and -1 in signs  # sanity check on the test setup itself
    sig = np.where(result.cluster_p_values < 0.05)[0]
    assert len(sig) >= 2  # at least the two planted effects are significant
    order = sig[np.argsort(-np.abs(result.cluster_masses[sig]))]

    def spectrogram_channel(fig):
        return fig.axes[2].get_title().split("(")[1].split(")")[0]

    # the chosen channel must belong to the cluster at that mass rank, and
    # (since the two clusters here don't share any channel) different ranks
    # must pick different channels -- proving cluster_idx switches clusters
    chosen_channels = list()
    for rank, cluster_idx in enumerate((0, 1)):
        result.plot_cluster_time_frequency(df["data"].iloc[0], cluster_idx=cluster_idx)
        assert len(plt.get_fignums()) == 1
        ch_name = spectrogram_channel(plt.gcf())
        chosen_channels.append(ch_name)
        cluster_chs = np.unique(result.clusters[order[rank]][-1])
        expected_channels = {ch_names[c] for c in cluster_chs}
        assert ch_name in expected_channels
        plt.close("all")
    assert chosen_channels[0] != chosen_channels[1]

    with pytest.raises(ValueError, match="cluster_idx=2 is out of range"):
        result.plot_cluster_time_frequency(df["data"].iloc[0], cluster_idx=2)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive.*:UserWarning")
def test_cluster_test_plot_cluster_time_frequency_selection():
    """cluster_idx/p_accept must select/filter by mass, and overlay shared clusters."""
    import matplotlib.pyplot as plt

    ch_names = ["Fz", "Cz", "Pz"]
    info = create_info(ch_names, sfreq=100.0, ch_types="eeg")
    info.set_montage("colin27_1020")
    n_times, n_freqs, n_channels = 6, 3, 3
    times, freqs = np.arange(n_times) / 10.0, np.arange(n_freqs)

    stat_obs = np.zeros((n_times, n_freqs, n_channels))
    stat_obs[0:2, :, 0] = 4.0  # cluster 0: "Fz"
    stat_obs[0:2, :, 1] = 5.0  # cluster 0: "Cz" (peak)
    stat_obs[3:6, :, 1] = -2.0  # cluster 1: "Cz" (shared channel, not peak)
    stat_obs[3:6, :, 2] = -3.0  # cluster 1: "Pz" (peak)
    mask0 = np.zeros_like(stat_obs, dtype=bool)
    mask0[0:2, :, 0:2] = True
    mask1 = np.zeros_like(stat_obs, dtype=bool)
    mask1[3:6, :, 1:3] = True
    clusters = [tuple(np.where(mask0)), tuple(np.where(mask1))]
    # the lower-p-value cluster (1) has the smaller mass, to prove selection
    # ranks by mass, not by p-value
    cluster_p_values = np.array([0.03, 0.01])
    inst = AverageTFRArray(
        info=info,
        data=np.zeros((n_channels, n_freqs, n_times)),
        times=times,
        freqs=freqs,
    )
    result = ClusterResult(
        stat_obs, clusters, cluster_p_values, np.array([0.0]), ttest_1samp_no_p
    )
    assert_array_almost_equal(result.cluster_masses, [54.0, -45.0])

    def spectrogram_channel(fig):
        return fig.axes[2].get_title().split("(")[1].split(")")[0]

    def overlay_values(fig):
        arr = np.ma.getdata(fig.axes[2].get_images()[1].get_array())
        return arr[np.isfinite(arr)]

    # default: cluster 0 has the larger |mass| despite the higher p-value
    result.plot_cluster_time_frequency(inst)
    assert spectrogram_channel(plt.gcf()) == "Cz"
    vals = overlay_values(plt.gcf())
    assert (vals > 0).any() and (vals < 0).any()  # cluster 1 shares "Cz", shown too
    plt.close("all")

    # cluster_idx=1: its peak channel ("Pz") isn't touched by cluster 0, so the
    # overlay reduces to cluster 1 alone
    result.plot_cluster_time_frequency(inst, cluster_idx=1)
    assert spectrogram_channel(plt.gcf()) == "Pz"
    assert (overlay_values(plt.gcf()) < 0).all()
    plt.close("all")

    with pytest.raises(ValueError, match="cluster_idx=2 is out of range"):
        result.plot_cluster_time_frequency(inst, cluster_idx=2)

    # p_accept excludes the larger-mass cluster -> cluster_idx=0 now resolves
    # to the other cluster
    result.plot_cluster_time_frequency(inst, p_accept=0.02)
    assert spectrogram_channel(plt.gcf()) == "Pz"
    plt.close("all")
    with pytest.raises(ValueError, match="cluster_idx=1 is out of range"):
        result.plot_cluster_time_frequency(inst, cluster_idx=1, p_accept=0.02)

    with pytest.raises(ValueError, match="No clusters have"):
        result.plot_cluster_time_frequency(inst, p_accept=0.005)

    empty_result = ClusterResult(
        stat_obs, [], np.array([]), np.array([0.0]), ttest_1samp_no_p
    )
    assert empty_result.cluster_masses.shape == (0,)
    with pytest.raises(ValueError, match="found no clusters"):
        empty_result.plot_cluster_time_frequency(inst)


def test_cluster_test_plot_cluster_time_frequency_wrong_dim():
    """Test plot_cluster_time_frequency rejects 2D (time x channel) clusters."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("formulaic")

    condition1_1d, condition2_1d, _, _ = _get_conditions()
    df = pd.DataFrame(dict(data=[condition1_1d, condition2_1d], condition=["a", "b"]))
    result = cluster_test(
        df,
        "data ~ condition",
        n_permutations=100,
        tail=1,
        seed=1,
        buffer_size=None,
        out_type="indices",
    )
    with pytest.raises(ValueError, match="requires 3D"):
        result.plot_cluster_time_frequency(None)
