# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mne import EpochsArray, EvokedArray, create_info
from mne.stats import (
    cluster_test,
    f_mway_rm,
    f_threshold_mway_rm,
    permutation_cluster_1samp_test,
    permutation_cluster_test,
)
from mne.time_frequency import AverageTFRArray, BaseTFR, EpochsTFRArray
from mne.utils import GetEpochsMixin

pd = pytest.importorskip("pandas")
pytest.importorskip("formulaic")  # required for cluster_test API


def _convert_cluster_slices_to_arrays(clusters, stat_obs_shape):
    """Old API sometimes returns slices, we always want masked arrays."""
    cluster_masks = list()
    for clust in clusters:
        clust_mask = np.zeros(stat_obs_shape, bool)
        clust_mask[clust] = True
        cluster_masks.append(clust_mask)
    return cluster_masks


def test_cluster_test_one_sample(stat_conditions):
    """Test cluster_test with a single-group (1-sample) design."""
    condition1_1d, _, _, _ = stat_conditions
    df = pd.DataFrame(dict(data=[condition1_1d], group=["only"]))
    kwargs = dict(n_permutations=100, tail=0, rng=1, buffer_size=None)
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


def test_compare_old_and_new_cluster_api(stat_conditions):
    """Test for same results from old and new APIs."""
    condition1_1d, condition2_1d, condition1_2d, condition2_2d = stat_conditions
    df_1d = pd.DataFrame(
        dict(
            data=[condition1_1d, condition2_1d],
            condition=["a", "b"],
        )
    )
    kwargs = dict(n_permutations=100, tail=1, rng=1, buffer_size=None, out_type="mask")
    F_obs, clusters, cluster_pvals, H0 = permutation_cluster_test(
        [condition1_1d, condition2_1d], **kwargs
    )
    formula = "data ~ condition"
    cluster_result = cluster_test(df_1d, formula, **kwargs)

    for clust in cluster_result.clusters:
        assert clust.shape == cluster_result.stat_obs.shape
    assert_array_equal(cluster_result.H0, H0)
    assert_array_equal(cluster_result.stat_obs, F_obs)
    assert_array_equal(cluster_result.cluster_p_values, cluster_pvals)

    assert len(clusters) == len(cluster_result.clusters)
    # Convert slices to masked arrays
    cluster_masks = _convert_cluster_slices_to_arrays(clusters, F_obs.shape)
    for cluster, res_clust in zip(cluster_masks, cluster_result.clusters):
        # bool_clust = np.zeros(F_obs.shape, bool)
        # bool_clust[cluster] = True
        np.testing.assert_array_equal(cluster.T, res_clust)


@pytest.mark.parametrize(
    "Inst", (EpochsArray, EvokedArray, EpochsTFRArray, AverageTFRArray)
)
@pytest.mark.filterwarnings('ignore:Ignoring argument "tail":RuntimeWarning')
def test_new_cluster_api(Inst):
    """Test handling different MNE objects in the cluster API."""
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
    kwargs = dict(n_permutations=100, rng=42, tail=1, buffer_size=None, out_type="mask")
    result_new_api = cluster_test(df, "data~condition", **kwargs)
    # make sure channels are last dimension for old API
    if is_epo:
        assert result_new_api.stat_obs.shape == df["data"][0].get_data()[0, ...].shape
        axes = (0, 3, 2, 1) if is_tfr else (0, 2, 1)
        X = [cond_a.get_data().transpose(*axes), cond_b.get_data().transpose(*axes)]
    else:
        assert result_new_api.stat_obs.shape == df["data"][0].get_data().shape
        axes = (2, 1, 0) if is_tfr else (1, 0)
        Xa = list()
        Xb = list()
        for inst, cond in zip(insts, conds):
            container = Xa if cond == "a" else Xb
            container.append(inst.get_data().transpose(*axes))
        X = [np.stack(Xa), np.stack(Xb)]

    F_obs, clusters, cluster_pvals, H0 = permutation_cluster_test(X, **kwargs)

    for clust in result_new_api.clusters:
        assert clust.shape == result_new_api.stat_obs.shape

    assert_array_almost_equal(result_new_api.H0, H0)
    assert_array_almost_equal(result_new_api.stat_obs, F_obs.T)
    assert_array_almost_equal(result_new_api.cluster_p_values, cluster_pvals)
    assert len(result_new_api.clusters) == len(clusters)
    for clu1, clu2 in zip(result_new_api.clusters, clusters):
        assert_array_equal(clu1, clu2.T)


@pytest.mark.filterwarnings('ignore:Ignoring argument "tail":RuntimeWarning')
def test_cluster_test_rm_anova():
    """Test the interaction-formula (repeated-measures ANOVA) branch of cluster_test."""
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
        rng=3,
        buffer_size=None,
        out_type="indices",
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
    assert_array_almost_equal(result.stat_obs, F_obs.T)
    assert len(result.clusters) == len(clusters)
    for clu1, clu2 in zip(result.clusters, clusters):
        assert_array_equal(clu1, tuple(reversed(clu2)))
    # the observed stat and clusters match the legacy API, but the null differs
    # by design: cluster_test permutes repeated-measures data within subject
    # only, whereas the legacy API shuffles rows across the whole design
    assert result.H0.shape == H0.shape
    assert not np.allclose(result.H0, H0)
    assert result.cluster_p_values.shape == cluster_pvals.shape


def test_cluster_test_formula_validation(stat_conditions):
    """Test that cluster_test raises clear errors for unsupported formulas."""
    condition1_1d, condition2_1d, _, _ = stat_conditions
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


@pytest.mark.filterwarnings('ignore:Ignoring argument "tail":RuntimeWarning')
@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:No clusters found:RuntimeWarning")
def test_cluster_test_reduce(stat_conditions):
    """Reduce multiple observations for paired t-test."""
    # TODO: parametrize this test for Epochs, AveragedTFR etc.

    condition1_1d, _, _, _ = stat_conditions
    # For this test we need equal sized arrays
    condition2_1d = condition1_1d.copy()
    rng = np.random.default_rng(0)
    rng.shuffle(condition2_1d)

    info = create_info(
        ch_names=[f"ch_{ii}" for ii in range(condition1_1d.shape[0])],
        sfreq=10,
        ch_types="eeg",
    )
    data = [EvokedArray(arr, info) for arr in [condition1_1d, condition2_1d]]
    df = pd.DataFrame(dict(data=data, a=["x", "y"]))
    df["b"] = 1

    df_2 = df.copy()
    df_2["b"] = 2
    df = pd.concat([df, df_2])
    del df_2

    df["c"] = "foo"

    df_2 = df.copy()
    df_2["c"] = "bar"
    df = pd.concat([df, df_2])
    del df_2
    # This should not raise
    cluster_test(df, formula="data ~ a", within_id="c")
