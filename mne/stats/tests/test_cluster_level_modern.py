import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mne import EpochsArray, EvokedArray, create_info
from mne.stats import (
    ClusterResult,
    cluster_test,
    f_mway_rm,
    f_threshold_mway_rm,
    permutation_cluster_1samp_test,
    permutation_cluster_test,
    ttest_1samp_no_p,
)
from mne.time_frequency import AverageTFRArray, BaseTFR, EpochsTFRArray
from mne.utils import GetEpochsMixin


def test_cluster_test_one_sample(stat_conditions):
    """Test cluster_test with a single-group (1-sample) design."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("formulaic")  # required for cluster_test API
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
    pd = pytest.importorskip("pandas")
    pytest.importorskip("formulaic")

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
    kwargs = dict(n_permutations=100, rng=42, tail=1, buffer_size=None, out_type="mask")
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
        rng=3,
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


def test_cluster_test_formula_validation(stat_conditions):
    """Test that cluster_test raises clear errors for unsupported formulas."""
    pd = pytest.importorskip("pandas")

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
        rng=1,
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
        df, "data ~ group", n_permutations=100, tail=0, rng=1, buffer_size=None
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


def test_cluster_test_plot_cluster_time_frequency_wrong_dim(stat_conditions):
    """Test plot_cluster_time_frequency rejects 2D (time x channel) clusters."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("formulaic")

    condition1_1d, condition2_1d, _, _ = stat_conditions
    df = pd.DataFrame(dict(data=[condition1_1d, condition2_1d], condition=["a", "b"]))
    result = cluster_test(
        df,
        "data ~ condition",
        n_permutations=100,
        tail=1,
        rng=1,
        buffer_size=None,
        out_type="indices",
    )
    with pytest.raises(ValueError, match="requires 3D"):
        result.plot_cluster_time_frequency(None)
