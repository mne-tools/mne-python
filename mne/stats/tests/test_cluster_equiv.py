"""FieldTrip equivalence tests for :func:`mne.stats.cluster_test`.

Reference values were computed with FieldTrip (fieldtrip-20260812) in MATLAB R2026a:
https://gist.github.com/larsoner/5c99b464bccf67f5641c1a2babc2c84e
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from scipy import sparse, stats

import mne
from mne.stats import cluster_test

pd = pytest.importorskip("pandas")
pytest.importorskip("formulaic")


def _bump(n_times, sl, amp):
    """Return a length-n_times signal with a Hann bump of peak ``amp`` in ``sl``."""
    out = np.zeros(n_times)
    width = sl.stop - sl.start
    out[sl] = amp * np.hanning(width + 2)[1:-1]  # no zero-amplitude edge samples
    return out


def _within_rows(rng, cond_signals, n_subjects, cols=("condition",), offsets=True):
    """Make one df row per subject and condition (with random intercepts)."""
    offs = rng.standard_normal(n_subjects) if offsets else np.zeros(n_subjects)
    rows = []
    for si in range(n_subjects):
        for cond, sig in cond_signals.items():
            data = sig + offs[si] + rng.standard_normal(sig.shape)
            row = dict(data=data[np.newaxis], subject=si)
            row.update(zip(cols, (cond,) if isinstance(cond, str) else cond))
            rows.append(row)
    return pd.DataFrame(rows)


def _between_rows(rng, group_signals, group_ns):
    """Make ``group_ns[group]`` df rows per independent group."""
    rows = [
        dict(data=(sig + rng.standard_normal(sig.shape))[np.newaxis], group=group)
        for group, sig in group_signals.items()
        for _ in range(group_ns[group])
    ]
    return pd.DataFrame(rows)


def _kwargs(**overrides):
    """Paired-test defaults; n_permutations > 2**(n-1) -> exact sign flips."""
    out = dict(
        formula="data ~ condition",
        within_id="subject",
        tail=0,
        n_permutations=4096,
        rng=0,
    )
    out.update(overrides)
    return out


# Scenario builders return (df, kwargs-for-cluster_test); the gist's .mat
# exporter imports these same builders (via SCENARIOS) for the FieldTrip runs.


def scenario_one_sample():
    """Pre-subtracted diffs, one pos bump; FT: depsamplesT vs. all-zero partner."""
    rng = np.random.default_rng(1)
    n_subjects, n_times = 12, 30
    signals = {"diff": _bump(n_times, slice(8, 15), 1.2)}
    df = _within_rows(rng, signals, n_subjects, offsets=False)
    return df, _kwargs(threshold=stats.t.ppf(1 - 0.025, n_subjects - 1))


def scenario_paired():
    """Paired t, 2 conditions, one pos + one neg bump; FT: depsamplesT."""
    rng = np.random.default_rng(2)
    n_subjects, n_times = 12, 30
    signals = {
        "a": _bump(n_times, slice(5, 12), 1.5),  # a > b early
        "b": _bump(n_times, slice(18, 25), 1.5),  # a < b late
    }
    df = _within_rows(rng, signals, n_subjects)
    return df, _kwargs(threshold=stats.t.ppf(1 - 0.025, n_subjects - 1))


def scenario_between_t():
    """2 independent groups, unequal n, opposite-sign group differences.

    FT: indepsamplesF, and indepsamplesT (MNE f_oneway F == FT t**2).
    """
    rng = np.random.default_rng(3)
    n_times, group_ns = 30, {"ctrl": 10, "pat": 12}
    signals = {
        "ctrl": _bump(n_times, slice(4, 11), 1.8),  # ctrl > pat early
        "pat": _bump(n_times, slice(17, 24), 1.8),  # pat > ctrl late
    }
    df = _between_rows(rng, signals, group_ns)
    threshold = stats.f.ppf(1 - 0.05, 1, sum(group_ns.values()) - 2)
    return df, _kwargs(
        formula="data ~ group",
        within_id=None,
        tail=1,
        threshold=threshold,
        n_permutations=5000,
    )


def scenario_between_anova():
    """3 independent groups, unequal n, 2 bumps with different orderings.

    FT: indepsamplesF.
    """
    rng = np.random.default_rng(4)
    n_times, group_ns = 30, {"g1": 8, "g2": 9, "g3": 10}
    means_1 = {"g1": 1.8, "g2": 0.0, "g3": -1.8}  # bump 1 group means
    means_2 = {"g1": 0.0, "g2": -1.8, "g3": 1.8}  # bump 2 group means
    signals = {
        g: _bump(n_times, slice(3, 10), means_1[g])
        + _bump(n_times, slice(17, 24), means_2[g])
        for g in group_ns
    }
    df = _between_rows(rng, signals, group_ns)
    threshold = stats.f.ppf(1 - 0.05, 2, sum(group_ns.values()) - 3)
    return df, _kwargs(
        formula="data ~ group",
        within_id=None,
        tail=1,
        threshold=threshold,
        n_permutations=5000,
    )


def scenario_rm_anova_interaction():
    """2x2 repeated-measures ANOVA interaction, two opposite-sign bumps.

    FT: depsamplesT on the double difference (a1b1 - a1b2) - (a2b1 - a2b2);
    the interaction F equals that t**2.
    """
    rng = np.random.default_rng(5)
    n_subjects, n_times, c = 12, 30, 0.8
    # pure interaction pattern: sign of the bumps per (a, b) cell
    signals = {
        (a, b): _bump(n_times, slice(6, 13), sgn) + _bump(n_times, slice(19, 26), -sgn)
        for (a, b), sgn in {
            ("a1", "b1"): +c,
            ("a1", "b2"): -c,
            ("a2", "b1"): -c,
            ("a2", "b2"): +c,
        }.items()
    }
    df = _within_rows(rng, signals, n_subjects, cols=("a", "b"))
    threshold = stats.f.ppf(1 - 0.05, 1, n_subjects - 1)
    return df, _kwargs(
        formula="data ~ a:b", tail=1, threshold=threshold, n_permutations=1024
    )


def scenario_spatiotemporal_paired():
    """Paired t on 4-channel Evoked data with chain adjacency A1-A2-A3-A4.

    Three bumps in the a-minus-b difference: positive on A1+A2 (adjacent ->
    must merge) and on A4 (same time window, but not adjacent to A2 because A3
    is clean -> must stay separate), negative on A3 only, later.
    FT: depsamplesT with cfg.neighbours encoding the same chain.
    """
    rng = np.random.default_rng(6)
    n_subjects, n_channels, n_times = 12, 4, 20
    sig_a = np.zeros((n_channels, n_times))
    for ch in (0, 1, 3):
        sig_a[ch] += _bump(n_times, slice(3, 9), 1.5)
    sig_b = np.zeros((n_channels, n_times))
    sig_b[2] += _bump(n_times, slice(12, 18), 1.5)  # a - b negative on A3
    info = mne.create_info([f"A{n + 1}" for n in range(n_channels)], 1000.0, "eeg")
    df = _within_rows(rng, {"a": sig_a, "b": sig_b}, n_subjects)
    df["data"] = df["data"].map(lambda d: mne.EvokedArray(d[0], info, tmin=0.0))
    adjacency = sparse.coo_array(
        np.diag(np.ones(n_channels - 1), 1) + np.diag(np.ones(n_channels - 1), -1)
    )
    threshold = stats.t.ppf(1 - 0.025, n_subjects - 1)
    return df, _kwargs(threshold=threshold, adjacency=adjacency)


def scenario_rm_3level():
    """3-level within factor: one-way rm ANOVA; FT: depsamplesFunivariate."""
    rng = np.random.default_rng(7)
    n_subjects, n_times = 12, 30
    means = {"c1": 1.0, "c2": 0.0, "c3": -1.0}
    signals = {c: _bump(n_times, slice(10, 17), amp) for c, amp in means.items()}
    df = _within_rows(rng, signals, n_subjects)
    threshold = stats.f.ppf(1 - 0.05, 2, 2 * (n_subjects - 1))
    return df, _kwargs(tail=1, threshold=threshold, n_permutations=1024)


SCENARIOS = {
    "one_sample": scenario_one_sample,
    "paired": scenario_paired,
    "between_t": scenario_between_t,
    "between_anova": scenario_between_anova,
    "rm_anova_interaction": scenario_rm_anova_interaction,
    "spatiotemporal_paired": scenario_spatiotemporal_paired,
    "rm_3level": scenario_rm_3level,
}

# FieldTrip reference values (see gist for the full cfg): ``critval`` is FT's
# cluster-forming threshold (== ``threshold``; squared when ``square=True``,
# i.e. FT ran a signed t where MNE runs the equivalent F = t**2 -- masses are
# then sums of the squared FT t map). ``clusters`` holds (member indices,
# cluster mass, FT prob); ``p_slack`` is tight for exhaustive sign-flip tests,
# loose for Monte Carlo and for differing permutation groups (2x2 interaction).
FT_REF = {
    "one_sample": dict(  # cfg.statistic='depsamplesT' (data vs. zeros)
        stat_name="paired T-statistic",
        critval=2.20098516009,
        p_slack=0.001,
        stat_max=8.30997573613,
        stat_sum=15.2992735697,
        clusters=[
            ([9, 10, 11, 12], 24.215180039, 0),
            ([8], -2.80238763033, 0.218017578125),
            ([0], 2.42485110412, 0.376220703125),
            ([4], 2.39754436956, 0.38916015625),
            ([2], -2.39390450562, 0.39013671875),
            ([17], -2.29386021535, 0.4423828125),
        ],
    ),
    "paired": dict(  # cfg.statistic='depsamplesT'
        stat_name="paired T-statistic",
        critval=2.20098516009,
        p_slack=0.001,
        stat_max=4.17741541515,
        stat_sum=-10.4491353518,
        clusters=[
            ([19, 20, 21, 22, 23], -20.5133161849, 0),
            ([7, 8], 6.48082672584, 0.00830078125),
        ],
    ),
    "between_t": dict(  # cfg.statistic='indepsamplesF'
        stat_name="F-statistic",
        critval=4.35124350333,
        p_slack=0.03,
        stat_max=21.5844596996,
        stat_sum=106.323938106,
        clusters=[
            ([6, 7, 8, 9], 44.1166473563, 0.0001999900005),
            ([19, 20, 21], 32.99796735, 0.00109994500275),
            ([17], 9.91559364543, 0.183390830458),
        ],
    ),
    "between_anova": dict(  # cfg.statistic='indepsamplesF'
        stat_name="F-statistic",
        critval=3.40282610535,
        p_slack=0.03,
        stat_max=42.9850709813,
        stat_sum=251.00244401,
        clusters=[
            ([18, 19, 20, 21, 22], 113.413482696, 4.9997500125e-05),
            ([5, 6, 7, 8], 111.934649709, 4.9997500125e-05),
            ([12], 4.11422783286, 0.566721663917),
        ],
    ),
    # cfg.statistic='depsamplesT' on the per-subject double difference
    # (a1b1 - a1b2) - (a2b1 - a2b2) vs. zeros; interaction F = t**2
    "rm_anova_interaction": dict(
        stat_name="F-statistic (repeated-measures ANOVA)",
        critval=2.20098516009,
        square=True,
        p_slack=0.05,
        stat_max=35.751136081,
        stat_sum=227.529952225,
        clusters=[
            ([20, 21, 22, 23], 104.752717043, 0),
            ([8, 9, 10, 11], 83.9317992007, 0.00048828125),
            ([6], 9.38758796849, 0.144287109375),
        ],
    ),
    # cfg.statistic='depsamplesT', cfg.neighbours = chain A1-A2-A3-A4
    "spatiotemporal_paired": dict(
        stat_name="paired T-statistic",
        critval=2.20098516009,
        p_slack=0.001,
        stat_max=5.28186348184,
        stat_sum=27.7385987319,
        clusters=[
            ([(4, 0), (5, 0), (5, 1), (6, 0), (6, 1), (7, 1)], 18.6062044319, 0),
            ([(14, 2), (15, 2), (16, 2)], -11.5475200021, 0.001708984375),
            ([(5, 3), (6, 3)], 7.58354029468, 0.013671875),
            ([(1, 2)], -2.91124038589, 0.4404296875),
            ([(16, 0)], 2.83280102942, 0.477783203125),
            ([(10, 3)], 2.38278255024, 0.7353515625),
        ],
    ),
    "rm_3level": dict(  # cfg.statistic='depsamplesFunivariate'
        stat_name="F-statistic (repeated-measures ANOVA)",
        critval=3.44335677937,
        p_slack=0.03,
        stat_max=24.7722284367,
        stat_sum=83.3268364113,
        clusters=[
            ([12, 13, 14], 50.6420452705, 4.9997500125e-05),
        ],
    ),
}
# cfg.statistic='indepsamplesT' on the between_t data: same clusters and (via
# F = t**2) masses as indepsamplesF, but a t critval and per-tail probs
_t_probs = (4.9997500125e-05, 0.00029998500075, 0.0896955152242)
FT_REF["between_t_T"] = dict(
    FT_REF["between_t"],
    critval=2.08596344727,
    square=True,
    clusters=[(*c[:2], p) for c, p in zip(FT_REF["between_t"]["clusters"], _t_probs)],
)


def _mne_clusters(result):
    """Map frozenset of cluster members -> (mass, p) for a ClusterResult."""
    out = {}
    for ci, cl in enumerate(result.clusters):
        cl = cl if isinstance(cl, tuple) else (cl,)
        if len(cl) == 1:
            members = frozenset(int(i) for i in cl[0])
        else:
            # reveersal here to deal with transpoe added after FT reulsts were generated
            members = frozenset(zip(*cl[::-1]))
        out[members] = (result.cluster_masses[ci], result.cluster_p_values[ci])
    return out


def _assert_ft_equiv(result, ref, threshold):
    """Assert a ClusterResult matches one FieldTrip reference run."""
    square = ref.get("square", False)
    assert result.stat_name == ref["stat_name"]  # check test-type routing
    # the threshold each scenario passes must equal FT's parametric
    # cfg.clustercritval (from cfg.clusteralpha=0.05), or its square
    np.testing.assert_allclose(threshold, ref["critval"] ** (1 + square), rtol=1e-9)
    np.testing.assert_allclose(result.stat_obs.max(), ref["stat_max"], rtol=1e-6)
    np.testing.assert_allclose(result.stat_obs.sum(), ref["stat_sum"], rtol=1e-6)
    got = _mne_clusters(result)
    expected = {
        frozenset(members): (mass, prob) for members, mass, prob in ref["clusters"]
    }
    assert set(got) == set(expected), (sorted(map(sorted, got)),)
    ft_is_t = square or "T-statistic" in ref["stat_name"]
    slack = ref["p_slack"]
    for members, (ft_mass, ft_prob) in expected.items():
        mass, p = got[members]
        np.testing.assert_allclose(mass, ft_mass, rtol=1e-6)
        if ft_is_t:  # MNE null pools both tails; FT prob is per-tail
            assert ft_prob - slack <= p <= 2 * ft_prob + slack, (p, ft_prob)
        else:  # same one-sided F null on both sides
            assert abs(p - ft_prob) <= slack, (p, ft_prob)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_fieldtrip_equivalence(name):
    """Compare cluster_test output against FieldTrip reference values."""
    df, kwargs = SCENARIOS[name]()
    result = cluster_test(df, **kwargs, verbose="error")
    _assert_ft_equiv(result, FT_REF[name], kwargs["threshold"])
    if name == "between_t":  # also equivalent to FT's independent-samples t
        _assert_ft_equiv(result, FT_REF["between_t_T"], kwargs["threshold"])
