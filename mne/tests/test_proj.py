# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import copy as cp
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_equal
from scipy import linalg

from mne import (
    Epochs,
    EvokedArray,
    compute_proj_epochs,
    compute_proj_evoked,
    compute_proj_raw,
    compute_raw_covariance,
    convert_forward_solution,
    create_info,
    pick_types,
    read_events,
    read_forward_solution,
    read_source_estimate,
    sensitivity_map,
)
from mne._fiff.proj import (
    _EEG_AVREF_PICK_DICT,
    Projection,
    _needs_eeg_average_ref_proj,
    activate_proj,
    make_projector,
    setup_proj,
)
from mne.cov import compute_whitener, regularize
from mne.datasets import testing
from mne.io import RawArray, read_raw_fif
from mne.preprocessing import maxwell_filter
from mne.proj import (
    _has_eeg_average_ref_proj,
    make_eeg_average_ref_proj,
    read_proj,
    write_proj,
)
from mne.rank import _compute_rank_int
from mne.utils import _record_warnings

base_dir = Path(__file__).parents[1] / "io" / "tests" / "data"
raw_fname = base_dir / "test_raw.fif"
event_fname = base_dir / "test-eve.fif"
proj_fname = base_dir / "test-proj.fif"
proj_gz_fname = base_dir / "test-proj.fif.gz"
bads_fname = base_dir / "test_bads.txt"
sample_path = testing.data_path(download=False) / "MEG" / "sample"
fwd_fname = sample_path / "sample_audvis_trunc-meg-eeg-oct-4-fwd.fif"
sensmap_fname = sample_path / "sample_audvis_trunc-%s-oct-4-fwd-sensmap-%s.w"
eog_fname = sample_path / "sample_audvis_eog-proj.fif"
ecg_fname = sample_path / "sample_audvis_ecg-proj.fif"


def _make_test_proj(ch_names, vector, desc):
    """Make a projector for selection tests."""
    vector = np.atleast_2d(np.asarray(vector, float))
    return Projection(
        data=dict(
            col_names=ch_names,
            row_names=None,
            nrow=len(vector),
            ncol=len(ch_names),
            data=vector,
        ),
        desc=desc,
    )


def _make_selection_raw():
    """Make mixed EEG/MEG data with three attached projectors."""
    ch_names = [f"EEG {ii:03d}" for ii in range(3)]
    ch_names += [f"MEG {ii:03d}" for ii in range(3)]
    info = create_info(ch_names, 100.0, ["eeg"] * 3 + ["mag"] * 3)
    data = np.random.default_rng(0).standard_normal((6, 600))
    raw = RawArray(data, info, verbose=False)
    projs = [
        _make_test_proj(ch_names[:3], [1.0, 1.0, 0.0], "EEG-a"),
        _make_test_proj(ch_names[:3], [0.0, 1.0, 1.0], "EEG-b"),
        _make_test_proj(ch_names[3:], [1.0, 1.0, 0.0], "MEG"),
    ]
    raw.add_proj(projs, verbose=False)
    return raw, projs


def _active_projs(inst):
    """Return the active state of attached projectors."""
    return [proj["active"] for proj in inst.info["projs"]]


def _make_selection_epochs(raw, *, preload=False, proj=False, reject=None, events=None):
    """Make Epochs for projector selection tests."""
    if events is None:
        events = (100, 300, 500)
    events = np.column_stack(
        [events, np.zeros(len(events), int), np.ones(len(events), int)]
    )
    return Epochs(
        raw,
        events,
        1,
        0.0,
        0.2,
        baseline=None,
        reject=reject,
        proj=proj,
        preload=preload,
        verbose=False,
    )


def _make_reconstruct_evoked():
    """Make small mixed MEG/EEG data with reconstruction projectors."""
    raw = read_raw_fif(raw_fname, preload=True, verbose=False).crop(0, 0.02)
    mag = pick_types(raw.info, meg="mag", exclude="bads")[:8]
    eeg = pick_types(raw.info, eeg=True, exclude="bads")[:8]
    raw.pick([raw.ch_names[pick] for pick in np.concatenate([mag, eeg])]).del_proj()
    mag_names, eeg_names = raw.ch_names[:8], raw.ch_names[8:]
    vector_a = np.zeros(8)
    vector_a[:2] = 1.0
    vector_b = np.zeros(8)
    vector_b[1:3] = 1.0
    projs = [
        _make_test_proj(mag_names, vector_a, "A"),
        _make_test_proj(mag_names, vector_b, "B"),
        _make_test_proj(eeg_names, vector_a, "EEG"),
    ]
    evoked = EvokedArray(raw.get_data(), raw.info, tmin=0.0)
    evoked.add_proj(projs, verbose=False)
    return evoked, projs, np.arange(8), np.arange(8, 16)


def test_apply_proj_selection():
    """Test selecting attached projectors when applying projections."""
    raw, projs = _make_selection_raw()
    data = raw.get_data()
    legacy = raw.copy().apply_proj(verbose=False)
    default = raw.copy().apply_proj(projs=None, verbose=False)
    assert_allclose(default.get_data(), legacy.get_data())
    assert _active_projs(default) == [True] * 3

    references = []
    for selection, active in (
        (projs[0], [True, False, False]),
        (projs[1], [False, True, False]),
        (projs[:2], [True, True, False]),
    ):
        passed = cp.deepcopy(selection)
        got = raw.copy().apply_proj(projs=passed, verbose=False)
        reference = raw.copy().del_proj().add_proj(selection, verbose=False)
        reference.apply_proj(verbose=False)
        references.append(reference)
        assert_allclose(got.get_data(), reference.get_data())
        assert_allclose(got._projector, reference._projector)
        assert _active_projs(got) == active
        assert not got.proj
        assert_allclose(got.get_data()[3:], data[3:])  # MEG is unaffected
        if isinstance(passed, list):
            assert not any(proj["active"] for proj in passed)
        else:
            assert not passed["active"]

    meg = raw.copy().apply_proj(projs=projs[2], verbose=False)
    assert_allclose(meg.get_data()[:3], data[:3])  # EEG is unaffected
    assert _active_projs(meg) == [False, False, True]

    cumulative = raw.copy().apply_proj(projs=projs[0], verbose=False)
    cumulative.apply_proj(projs=projs[1], verbose=False)
    assert_allclose(cumulative.get_data(), references[2].get_data())
    assert_allclose(cumulative._projector, references[2]._projector)
    assert _active_projs(cumulative) == [True, True, False]
    before = cumulative.get_data().copy()
    projector = cumulative._projector.copy()
    cumulative.apply_proj(projs=projs[0], verbose=False)
    assert_allclose(cumulative.get_data(), before)
    assert_allclose(cumulative._projector, projector)
    assert _active_projs(cumulative) == [True, True, False]
    cumulative.apply_proj(verbose=False)
    assert_allclose(cumulative.get_data(), legacy.get_data())
    assert_allclose(cumulative._projector, legacy._projector)
    assert cumulative.proj

    copied = cp.deepcopy(raw.info["projs"][0])
    raw.copy().apply_proj(projs=copied, verbose=False)
    unattached = _make_test_proj(raw.ch_names[:3], [1.0, 0.0, 1.0], "other")
    unchanged = raw.copy()
    with pytest.raises(ValueError, match="does not match"):
        unchanged.apply_proj(projs=unattached, verbose=False)
    assert_allclose(unchanged.get_data(), data)
    assert unchanged._projector is None
    assert not any(_active_projs(unchanged))

    ambiguous = raw.copy()
    ambiguous.info["projs"].append(cp.deepcopy(ambiguous.info["projs"][0]))
    with pytest.raises(ValueError, match="matches multiple"):
        ambiguous.apply_proj(projs=projs[0], verbose=False)

    evoked = EvokedArray(data[:, :10], raw.info, tmin=0.0)
    evoked.apply_proj(projs=projs[0], verbose=False)
    assert_allclose(evoked.data, references[0]._projector @ data[:, :10])
    assert_allclose(evoked._projector, references[0]._projector)


def test_apply_proj_selection_support():
    """Test selection with unsupported and all-bad projector channels."""
    raw, projs = _make_selection_raw()
    unsupported = _make_test_proj(["missing"], [1.0], "unsupported")
    raw.add_proj(unsupported, verbose=False)
    before = raw.copy()
    raw.apply_proj(projs=[], verbose=False)
    assert_allclose(raw.get_data(), before.get_data())
    assert raw._projector is None
    raw.apply_proj(projs=unsupported, verbose=False)
    assert_allclose(raw.get_data(), before.get_data())
    assert raw._projector is None
    assert not any(_active_projs(raw))

    partial = before.copy().apply_proj(projs=projs[0], verbose=False)
    data = partial.get_data().copy()
    projector = partial._projector.copy()
    partial.apply_proj(projs=unsupported, verbose=False)
    assert_allclose(partial.get_data(), data)
    assert_allclose(partial._projector, projector)
    assert _active_projs(partial) == [True, False, False, False]
    partial_support = _make_test_proj(
        [*raw.ch_names[:3], "missing"], np.full(4, 0.5), "partial"
    )
    partial.add_proj(partial_support, verbose=False)
    with pytest.warns(RuntimeWarning, match="reduced") as records:
        partial.apply_proj(projs=partial_support, verbose=False)
    assert len(records) == 1

    delayed_raw = before.copy()
    delayed_raw.info["bads"] = projs[0]["data"]["col_names"]
    selected = delayed_raw.info["projs"][0]
    delayed = _make_selection_epochs(delayed_raw, proj="delayed", events=(100, 300))
    delayed.apply_proj(projs=selected, verbose=False)
    assert delayed._do_delayed_proj
    assert not delayed.preload
    assert delayed._projector is not None  # all projectors remain for delayed SSP
    assert not any(_active_projs(delayed))

    raw = before
    raw.info["bads"] = projs[0]["data"]["col_names"]
    raw.apply_proj(projs=projs[0], verbose=False)
    assert raw._projector is None
    assert not any(_active_projs(raw))


def test_apply_proj_selection_eeg_reference():
    """Test that EEG reference and artifact projector selection stay separate."""
    raw, projs = _make_selection_raw()
    raw.del_proj()
    car = make_eeg_average_ref_proj(raw.info)
    raw.add_proj([projs[0], car], verbose=False)
    data = raw.get_data()

    raw.apply_proj(projs=projs[0], verbose=False)
    artifact_matrix = make_projector([projs[0]], raw.ch_names)[0]
    assert_allclose(raw.get_data(), artifact_matrix @ data)
    assert _active_projs(raw) == [True, False]

    raw.apply_proj(projs=car, verbose=False)
    joint_matrix = make_projector([projs[0], car], raw.ch_names)[0]
    assert_allclose(raw.get_data(), joint_matrix @ data, atol=1e-15)
    assert_allclose(raw._projector, joint_matrix)
    assert _active_projs(raw) == [True, True]


@pytest.mark.parametrize("preload", [False, True])
@pytest.mark.parametrize("proj", [False, "delayed"])
def test_apply_proj_selection_epochs(preload, proj):
    """Test selected projection for lazy, preloaded, and delayed Epochs."""
    raw, projs = _make_selection_raw()
    epochs = _make_selection_epochs(raw, preload=preload, proj=proj)
    data = epochs.get_data(copy=True)
    matrix = make_projector([projs[0]], epochs.ch_names)[0]
    epochs.apply_proj(projs=projs[0], verbose=False)
    assert epochs.preload
    assert not epochs._do_delayed_proj
    assert_allclose(epochs.get_data(copy=True), np.matmul(matrix, data))
    assert_allclose(epochs._projector, matrix)
    assert _active_projs(epochs) == [True, False, False]


def test_apply_proj_selection_delayed_rejection():
    """Test that unrelated projectors do not affect delayed rejection."""
    ch_names = [f"EEG {ii:03d}" for ii in range(3)]
    info = create_info(ch_names, 100.0, "eeg")
    data = np.zeros((3, 500))
    vector = np.array([0.0, 1.0, 1.0])
    data[:, 100:121] = vector[:, np.newaxis] * np.linspace(-2.0, 2.0, 21)
    raw = RawArray(data, info, verbose=False)
    projs = [
        _make_test_proj(ch_names, [1.0, 1.0, 0.0], "selected"),
        _make_test_proj(ch_names, vector, "unrelated"),
    ]
    raw.add_proj(projs, verbose=False)
    epochs = _make_selection_epochs(
        raw,
        reject=dict(eeg=1.0),
        proj="delayed",
        events=(100, 300),
    )
    epochs.apply_proj(projs=projs[0], verbose=False)
    assert len(epochs) == 1
    assert epochs.selection.tolist() == [1]
    assert _active_projs(epochs) == [True, False]


def test_apply_proj_selection_raw_lazy(tmp_path):
    """Test that selected projection of lazy Raw stays lazy."""
    raw, _ = _make_selection_raw()
    fname = tmp_path / "selection_raw.fif"
    raw.save(fname)
    lazy = read_raw_fif(fname, preload=False)
    data = lazy.get_data()
    selected = lazy.info["projs"][0]
    matrix = make_projector([selected], lazy.ch_names)[0]
    lazy.apply_proj(projs=selected, verbose=False)
    assert not lazy.preload
    assert_allclose(lazy.get_data(), matrix @ data, atol=1e-7)
    assert_allclose(lazy._projector, matrix)
    assert _active_projs(lazy) == [True, False, False]


def test_reconstruct_proj_selection():
    """Test selecting projectors for projection reconstruction."""
    evoked, projs, meg, eeg = _make_reconstruct_evoked()
    data = evoked.data.copy()

    default = evoked.copy()._reconstruct_proj(mode="fast")
    explicit_none = evoked.copy()._reconstruct_proj(projs=None, mode="fast")
    assert_allclose(explicit_none.data, default.data)
    assert _active_projs(explicit_none) == [True, True, True]
    with pytest.raises(TypeError):
        evoked.copy()._reconstruct_proj(projs[0])

    reconstructed = []
    for selection, active in (
        (projs[0], [True, False, False]),
        (projs[1], [False, True, False]),
        (projs[:2], [True, True, False]),
    ):
        got = evoked.copy()._reconstruct_proj(projs=selection, mode="fast")
        reference = evoked.copy().del_proj().add_proj(selection, verbose=False)
        reference._reconstruct_proj(mode="fast")
        assert_allclose(got.data[meg], reference.data[meg])
        assert np.array_equal(got.data[eeg], data[eeg])
        assert _active_projs(got) == active
        reconstructed.append(got.data[meg].copy())
    assert not np.array_equal(reconstructed[0], reconstructed[1])

    got = evoked.copy()._reconstruct_proj(projs=projs[2], mode="fast")
    reference = evoked.copy().del_proj().add_proj(projs[2], verbose=False)
    reference._reconstruct_proj(mode="fast")
    assert_allclose(got.data[eeg], reference.data[eeg])
    assert np.array_equal(got.data[meg], data[meg])
    assert _active_projs(got) == [False, False, True]


def test_reconstruct_proj_state():
    """Test active projector and EEG reference reconstruction state."""
    evoked, projs, meg, eeg = _make_reconstruct_evoked()
    data = evoked.data.copy()

    active = evoked.copy().apply_proj(projs=projs[0], verbose=False)
    got = active.copy()._reconstruct_proj(projs=projs[1], mode="fast")
    reference = active.copy().del_proj(2)._reconstruct_proj(mode="fast")
    assert_allclose(got.data[meg], reference.data[meg])
    assert np.array_equal(got.data[eeg], data[eeg])
    assert _active_projs(got) == [True, True, False]

    evoked.del_proj()
    car = make_eeg_average_ref_proj(evoked.info)
    evoked.add_proj([projs[2], car], verbose=False)
    inactive_car = evoked.copy()._reconstruct_proj(projs=projs[2], mode="fast")
    reference = evoked.copy().del_proj().add_proj(projs[2], verbose=False)
    reference._reconstruct_proj(mode="fast")
    assert_allclose(inactive_car.data[eeg], reference.data[eeg])
    assert np.array_equal(inactive_car.data[meg], data[meg])
    assert not np.allclose(inactive_car.data[eeg].mean(axis=0), 0.0, atol=1e-12)
    assert _active_projs(inactive_car) == [True, False]

    active_car = evoked.copy().apply_proj(projs=car, verbose=False)
    got = active_car.copy()._reconstruct_proj(projs=projs[2], mode="fast")
    reference = active_car.copy()._reconstruct_proj(mode="fast")
    assert_allclose(got.data[eeg], reference.data[eeg])
    assert_allclose(got.data[meg], active_car.data[meg], atol=1e-20)
    assert_allclose(got.data[eeg].mean(axis=0), 0.0, atol=1e-12)
    assert _active_projs(got) == [True, True]


def test_reconstruct_proj_noop():
    """Test empty and unsupported reconstruction selections."""
    evoked, projs, _, _ = _make_reconstruct_evoked()
    data = evoked.data.copy()

    evoked._reconstruct_proj(projs=[])
    assert np.array_equal(evoked.data, data)
    assert not any(_active_projs(evoked))

    unsupported = _make_test_proj(["missing"], [1.0], "unsupported")
    evoked.add_proj(unsupported, verbose=False)
    evoked._reconstruct_proj(projs=unsupported)
    assert np.array_equal(evoked.data, data)
    assert not any(_active_projs(evoked))

    active = evoked.copy().apply_proj(projs=projs[0], verbose=False)
    data = active.data.copy()
    active._reconstruct_proj(projs=unsupported)
    assert np.array_equal(active.data, data)
    assert _active_projs(active) == [True, False, False, False]

    partial, _, meg, _ = _make_reconstruct_evoked()
    partial_names = [partial.ch_names[pick] for pick in meg[:3]]
    partial_proj = _make_test_proj(
        [*partial_names, "missing"], np.full(4, 0.5), "partial"
    )
    partial.add_proj(partial_proj, verbose=False)
    with pytest.warns(RuntimeWarning, match="reduced") as records:
        partial._reconstruct_proj(projs=partial_proj, mode="fast")
    assert len(records) == 2  # apply_proj and actual mapping, but not the probe


def test_bad_proj():
    """Test dealing with bad projection application."""
    raw = read_raw_fif(raw_fname, preload=True)
    events = read_events(event_fname)
    picks = pick_types(
        raw.info, meg=True, stim=False, ecg=False, eog=False, exclude="bads"
    )
    picks = picks[2:18:3]
    _check_warnings(raw, events, picks)
    # still bad
    raw.pick([raw.ch_names[ii] for ii in picks])
    _check_warnings(raw, events)
    # "fixed"
    raw.info.normalize_proj()  # avoid projection warnings
    _check_warnings(raw, events, count=0)
    # eeg avg ref is okay
    raw = read_raw_fif(raw_fname, preload=True).pick(picks="eeg")
    raw.set_eeg_reference(projection=True)
    _check_warnings(raw, events, count=0)
    raw.info["bads"] = raw.ch_names[:10]
    _check_warnings(raw, events, count=0)

    raw = read_raw_fif(raw_fname)
    pytest.raises(ValueError, raw.del_proj, "foo")
    n_proj = len(raw.info["projs"])
    raw.del_proj(0)
    assert_equal(len(raw.info["projs"]), n_proj - 1)
    raw.del_proj()
    assert_equal(len(raw.info["projs"]), 0)

    # Ensure we deal with newer-style Neuromag projs properly, were getting:
    #
    #     Projection vector "PCA-v2" has magnitude 1.00 (should be unity),
    #     applying projector with 101/306 of the original channels available
    #     may be dangerous.
    raw = read_raw_fif(raw_fname).crop(0, 1)
    raw.set_eeg_reference(projection=True)
    raw.info["bads"] = ["MEG 0111"]
    meg_picks = pick_types(raw.info, meg=True, exclude=())
    ch_names = [raw.ch_names[pick] for pick in meg_picks]
    for p in raw.info["projs"][:-1]:
        data = np.zeros((1, len(ch_names)))
        idx = [ch_names.index(ch_name) for ch_name in p["data"]["col_names"]]
        data[:, idx] = p["data"]["data"]
        p["data"].update(ncol=len(meg_picks), col_names=ch_names, data=data)
    # smoke test for no warnings during reg
    regularize(compute_raw_covariance(raw, verbose="error"), raw.info)


def _check_warnings(raw, events, picks=None, count=3):
    """Count warnings."""
    with _record_warnings() as w:
        Epochs(
            raw,
            events,
            dict(aud_l=1, vis_l=3),
            -0.2,
            0.5,
            picks=picks,
            preload=True,
            proj=True,
        )
    assert len(w) == count
    assert all("dangerous" in str(ww.message) for ww in w)


@testing.requires_testing_data
def test_sensitivity_maps():
    """Test sensitivity map computation."""
    fwd = read_forward_solution(fwd_fname)
    fwd = convert_forward_solution(fwd, surf_ori=True)
    projs = read_proj(eog_fname)
    projs.extend(read_proj(ecg_fname))
    decim = 6
    for ch_type in ["eeg", "grad", "mag"]:
        w = read_source_estimate(Path(str(sensmap_fname) % (ch_type, "lh"))).data
        stc = sensitivity_map(
            fwd, projs=None, ch_type=ch_type, mode="free", exclude="bads"
        )
        assert_array_almost_equal(stc.data, w, decim)
        assert stc.subject == "sample"
        # let's just make sure the others run
        if ch_type == "grad":
            # fixed (2)
            w = read_source_estimate(str(sensmap_fname) % (ch_type, "2-lh")).data
            stc = sensitivity_map(
                fwd, projs=None, mode="fixed", ch_type=ch_type, exclude="bads"
            )
            assert_array_almost_equal(stc.data, w, decim)
        if ch_type == "mag":
            # ratio (3)
            w = read_source_estimate(str(sensmap_fname) % (ch_type, "3-lh")).data
            stc = sensitivity_map(
                fwd, projs=None, mode="ratio", ch_type=ch_type, exclude="bads"
            )
            assert_array_almost_equal(stc.data, w, decim)
        if ch_type == "eeg":
            # radiality (4), angle (5), remaining (6), and  dampening (7)
            modes = ["radiality", "angle", "remaining", "dampening"]
            ends = ["4-lh", "5-lh", "6-lh", "7-lh"]
            for mode, end in zip(modes, ends):
                w = read_source_estimate(str(sensmap_fname) % (ch_type, end)).data
                stc = sensitivity_map(
                    fwd, projs=projs, mode=mode, ch_type=ch_type, exclude="bads"
                )
                assert_array_almost_equal(stc.data, w, decim)

    # test corner case for EEG
    stc = sensitivity_map(
        fwd,
        projs=[make_eeg_average_ref_proj(fwd["info"])],
        ch_type="eeg",
        exclude="bads",
    )
    # test corner case for projs being passed but no valid ones (#3135)
    pytest.raises(ValueError, sensitivity_map, fwd, projs=None, mode="angle")
    pytest.raises(RuntimeError, sensitivity_map, fwd, projs=[], mode="angle")
    # test volume source space
    fname = sample_path / "sample_audvis_trunc-meg-vol-7-fwd.fif"
    fwd = read_forward_solution(fname)
    sensitivity_map(fwd)


def test_compute_proj_epochs(tmp_path):
    """Test SSP computation on epochs."""
    event_id, tmin, tmax = 1, -0.2, 0.3

    raw = read_raw_fif(raw_fname, preload=True)
    events = read_events(event_fname)
    bad_ch = "MEG 2443"
    picks = pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False, exclude=[])
    epochs = Epochs(
        raw, events, event_id, tmin, tmax, picks=picks, baseline=None, proj=False
    )

    evoked = epochs.average()
    projs = compute_proj_epochs(epochs, n_grad=1, n_mag=1, n_eeg=0)
    write_proj(tmp_path / "test-proj.fif.gz", projs)
    for p_fname in [proj_fname, proj_gz_fname, tmp_path / "test-proj.fif.gz"]:
        projs2 = read_proj(p_fname)
        assert len(projs) == len(projs2)

        for p1, p2 in zip(projs, projs2):
            assert p1["desc"] == p2["desc"]
            assert p1["data"]["col_names"] == p2["data"]["col_names"]
            assert p1["active"] == p2["active"]
            # compare with sign invariance
            p1_data = p1["data"]["data"] * np.sign(p1["data"]["data"][0, 0])
            p2_data = p2["data"]["data"] * np.sign(p2["data"]["data"][0, 0])
            if bad_ch in p1["data"]["col_names"]:
                bad = p1["data"]["col_names"].index("MEG 2443")
                mask = np.ones(p1_data.size, dtype=bool)
                mask[bad] = False
                p1_data = p1_data[:, mask]
                p2_data = p2_data[:, mask]
            corr = np.corrcoef(p1_data, p2_data)[0, 1]
            assert_array_almost_equal(corr, 1.0, 5)
            if p2["explained_var"]:
                assert isinstance(p2["explained_var"], float)
                assert_array_almost_equal(p1["explained_var"], p2["explained_var"])

    # test that you can compute the projection matrix
    projs = activate_proj(projs)
    proj, nproj, U = make_projector(projs, epochs.ch_names, bads=[])

    assert nproj == 2
    assert U.shape[1] == 2

    # test that you can save them
    with epochs.info._unlock():
        epochs.info["projs"] += projs
    evoked = epochs.average()
    evoked.save(tmp_path / "foo-ave.fif")

    projs = read_proj(proj_fname)

    projs_evoked = compute_proj_evoked(evoked, n_grad=1, n_mag=1, n_eeg=0)
    assert len(projs_evoked) == 2
    # XXX : test something

    # test parallelization
    projs = compute_proj_epochs(
        epochs, n_grad=1, n_mag=1, n_eeg=0, desc_prefix="foobar"
    )
    assert all("foobar" in x["desc"] for x in projs)
    projs = activate_proj(projs)
    proj_par, _, _ = make_projector(projs, epochs.ch_names, bads=[])
    assert_allclose(proj, proj_par, rtol=1e-8, atol=1e-16)

    # test warnings on bad filenames
    proj_badname = tmp_path / "test-bad-name.fif.gz"
    with pytest.warns(RuntimeWarning, match="-proj.fif"):
        write_proj(proj_badname, projs)
    with pytest.warns(RuntimeWarning, match="-proj.fif"):
        read_proj(proj_badname)

    # bad inputs
    fname = tmp_path / "out-proj.fif"
    with pytest.raises(TypeError, match="projs"):
        write_proj(fname, "foo")
    with pytest.raises(TypeError, match=r"projs\[0\] must be .*"):
        write_proj(fname, ["foo"], overwrite=True)


@pytest.mark.slowtest
def test_compute_proj_raw(tmp_path):
    """Test SSP computation on raw."""
    # Test that the raw projectors work
    raw_time = 2.5  # Do shorter amount for speed
    raw = read_raw_fif(raw_fname).crop(0, raw_time)
    raw.load_data()
    for ii in (0.25, 0.5, 1, 2):
        with pytest.warns(RuntimeWarning, match="Too few samples"):
            projs = compute_proj_raw(
                raw, duration=ii - 0.1, stop=raw_time, n_grad=1, n_mag=1, n_eeg=0
            )

        # test that you can compute the projection matrix
        projs = activate_proj(projs)
        proj, nproj, U = make_projector(projs, raw.ch_names, bads=[])

        assert nproj == 2
        assert U.shape[1] == 2

        # test that you can save them
        with raw.info._unlock():
            raw.info["projs"] += projs
        raw.save(tmp_path / f"foo_{ii}_raw.fif", overwrite=True)

    # Test that purely continuous (no duration) raw projection works
    with pytest.warns(RuntimeWarning, match="Too few samples"):
        projs = compute_proj_raw(
            raw, duration=None, stop=raw_time, n_grad=1, n_mag=1, n_eeg=0
        )

    # test that you can compute the projection matrix
    projs = activate_proj(projs)
    proj, nproj, U = make_projector(projs, raw.ch_names, bads=[])

    assert nproj == 2
    assert U.shape[1] == 2

    # test that you can save them
    with raw.info._unlock():
        raw.info["projs"] += projs
    raw.save(tmp_path / "foo_rawproj_continuous_raw.fif")

    # test resampled-data projector, upsampling instead of downsampling
    # here to save an extra filtering (raw would have to be LP'ed to be equiv)
    raw_resamp = cp.deepcopy(raw)
    raw_resamp.resample(raw.info["sfreq"] * 2, n_jobs=2, npad="auto")
    projs = compute_proj_raw(
        raw_resamp, duration=None, stop=raw_time, n_grad=1, n_mag=1, n_eeg=0
    )
    projs = activate_proj(projs)
    proj_new, _, _ = make_projector(projs, raw.ch_names, bads=[])
    assert_array_almost_equal(proj_new, proj, 4)

    # test with bads
    raw.load_bad_channels(bads_fname)  # adds 2 bad mag channels
    with pytest.warns(RuntimeWarning, match="Too few samples"):
        projs = compute_proj_raw(raw, n_grad=0, n_mag=0, n_eeg=1)
    assert len(projs) == 1

    # test that bad channels can be excluded, and empty support
    for projs_ in (projs, []):
        proj, nproj, U = make_projector(projs_, raw.ch_names, bads=raw.ch_names)
        assert_array_almost_equal(proj, np.eye(len(raw.ch_names)))
        assert nproj == 0  # all channels excluded
        assert U.shape == (len(raw.ch_names), nproj)


@pytest.mark.parametrize("duration", [1, np.pi / 2.0])
@pytest.mark.parametrize("sfreq", [600.614990234375, 1000.0])
def test_proj_raw_duration(duration, sfreq):
    """Test equivalence of `duration` options."""
    n_ch, n_dim = 30, 3
    rng = np.random.default_rng(0)
    signals = rng.standard_normal((n_dim, 10000))
    mixing = rng.normal(loc=[0, 1, 2], size=(n_ch, n_dim))
    data = np.dot(mixing, signals)
    raw = RawArray(data, create_info(n_ch, sfreq, "eeg"))
    raw.set_eeg_reference(projection=True)
    n_eff = int(round(raw.info["sfreq"] * duration))
    # crop to an even "duration" number of epochs
    stop = ((len(raw.times) // n_eff) * n_eff - 1) / raw.info["sfreq"]
    raw.crop(0, stop)
    proj_def = compute_proj_raw(raw, n_eeg=n_dim)
    proj_dur = compute_proj_raw(raw, duration=duration, n_eeg=n_dim)
    proj_none = compute_proj_raw(raw, duration=None, n_eeg=n_dim)
    assert len(proj_dur) == len(proj_none) == len(proj_def) == n_dim
    # proj_def is not in here because it does not necessarily evenly divide
    # the signal length:
    for pu, pn in zip(proj_dur, proj_none):
        assert_allclose(pu["data"]["data"], pn["data"]["data"])
    # but we can test it here since it should still be a small subspace angle:
    for proj in (proj_dur, proj_none, proj_def):
        computed = np.concatenate([p["data"]["data"] for p in proj], 0)
        angle = np.rad2deg(linalg.subspace_angles(computed.T, mixing)[0])
        assert angle < 1e-5


def test_make_eeg_average_ref_proj():
    """Test EEG average reference projection."""
    raw = read_raw_fif(raw_fname, preload=True)
    eeg = pick_types(raw.info, meg=False, eeg=True)

    # No average EEG reference
    assert not np.all(raw._data[eeg].mean(axis=0) < 1e-19)

    # Apply average EEG reference
    car = make_eeg_average_ref_proj(raw.info)
    reref = raw.copy()
    reref.add_proj(car)
    reref.apply_proj()
    assert_array_almost_equal(reref._data[eeg].mean(axis=0), 0, decimal=18)

    # Error when custom reference has already been applied
    with raw.info._unlock():
        raw.info["custom_ref_applied"] = True
    pytest.raises(RuntimeError, make_eeg_average_ref_proj, raw.info)

    # test that an average EEG ref is not added when doing proj
    raw.set_eeg_reference(projection=True)
    assert _has_eeg_average_ref_proj(raw.info)
    raw.del_proj(idx=-1)
    assert not _has_eeg_average_ref_proj(raw.info)
    raw.apply_proj()
    assert not _has_eeg_average_ref_proj(raw.info)


@pytest.mark.parametrize("ch_type", tuple(_EEG_AVREF_PICK_DICT) + ("all",))
def test_has_eeg_average_ref_proj(ch_type):
    """Test checking whether an (i)EEG average reference exists."""
    all_ref_ch_types = list(_EEG_AVREF_PICK_DICT)
    if ch_type == "all":
        ch_types = all_ref_ch_types
        set_eeg_ref_ch_type = all_ref_ch_types
    else:
        ch_types = [ch_type] * len(all_ref_ch_types)
        set_eeg_ref_ch_type = ch_type
    empty_info = create_info(len(all_ref_ch_types), 1000.0, ch_types)
    assert not _has_eeg_average_ref_proj(empty_info)

    raw = read_raw_fif(raw_fname)
    raw.del_proj()
    raw.load_data()
    picks = pick_types(raw.info, eeg=True)
    assert len(picks) == 60
    # repeat `ch_types` over and over again
    ch_types = sum([ch_types] * (len(picks) // len(ch_types) + 1), [])[: len(picks)]
    raw.set_channel_types(
        {raw.ch_names[pick]: ch_type for pick, ch_type in zip(picks, ch_types)}
    )
    raw._data[picks] = 1.0  # set all to unity
    # For individual channel types, set them and return from the test early
    if ch_type != "all":
        for ch_type in ("auto", set_eeg_ref_ch_type):
            raw.set_eeg_reference(projection=True, ch_type=ch_type)
            assert len(raw.info["projs"]) == 1
            assert _has_eeg_average_ref_proj(raw.info)
            assert not _needs_eeg_average_ref_proj(raw.info)
            if ch_type == "auto":
                with pytest.warns(RuntimeWarning, match="added.*untouched"):
                    raw.set_eeg_reference(projection=True, ch_type=ch_type)
                raw.del_proj()
        desc = raw.info["projs"][0]["desc"].lower()
        assert ch_type in desc
        data = raw.copy().apply_proj()[picks][0]
        assert_allclose(data, 0.0, atol=1e-12)  # zeroed out
        return

    # Now for ch_type == 'all', ensure that we can make one proj or
    # len(all_ref_ch_types) projs

    # One big joint proj
    raw.set_eeg_reference(projection=True, ch_type=set_eeg_ref_ch_type)
    assert len(raw.info["projs"]) == len(all_ref_ch_types)
    raw.del_proj()
    raw.set_eeg_reference(projection=True, ch_type=set_eeg_ref_ch_type, joint=True)
    assert len(raw.info["projs"]) == 1
    assert _has_eeg_average_ref_proj(raw.info)
    assert not _needs_eeg_average_ref_proj(raw.info)
    desc = raw.info["projs"][0]["desc"].lower()
    assert all(ch_type in desc for ch_type in all_ref_ch_types)
    for ch_type in all_ref_ch_types + [all_ref_ch_types]:
        with pytest.warns(RuntimeWarning, match="already added.*untouch"):
            raw.set_eeg_reference(projection=True, ch_type=ch_type)
    data = raw.copy().apply_proj()[picks][0]
    assert_allclose(data, 0.0, atol=1e-12)  # zeroed out
    raw.del_proj()

    # len(all_kinds) separate projs, with data for each channel type that
    # is a different non-zero integer (EEG=1, SEEG=2, ...)
    for ci, ch_type in enumerate(all_ref_ch_types):
        raw._data[pick_types(raw.info, **{ch_type: True})] = ci + 1
    for ci, ch_type in enumerate(all_ref_ch_types):
        raw.set_eeg_reference(projection=True, ch_type=ch_type)
        assert len(raw.info["projs"]) == ci + 1
        if ci < len(all_ref_ch_types) < 1:
            assert not _has_eeg_average_ref_proj(raw.info)
            assert _needs_eeg_average_ref_proj(raw.info)
    assert len(raw.info["projs"]) == len(all_ref_ch_types)
    assert _has_eeg_average_ref_proj(raw.info)
    assert not _needs_eeg_average_ref_proj(raw.info)
    descs = [p["desc"].lower() for p in raw.info["projs"]]
    assert len(descs) == len(all_ref_ch_types)
    for desc, ch_type in zip(descs, all_ref_ch_types):
        assert ch_type in desc
    assert_allclose(raw[picks][0], raw[all_ref_ch_types][0], atol=1e-12)
    data = raw.copy().apply_proj()[picks][0]
    assert len(data) == len(picks)
    assert_allclose(data, 0.0, atol=1e-12)  # zeroed out
    # a single joint proj will *not* zero out given these integer 1/2/3...
    # values per channel type assuming we have an even number of channels.
    # If this changes we'll have to make this check better
    assert len(all_ref_ch_types) % 2 == 0
    data_nz = (
        raw.del_proj()
        .set_eeg_reference(projection=True, ch_type=all_ref_ch_types, joint=True)
        .apply_proj()[picks][0]
    )
    assert not np.isclose(data_nz, 0.0).any()


def test_needs_eeg_average_ref_proj():
    """Test checking whether a recording needs an EEG average reference."""
    raw = read_raw_fif(raw_fname)
    assert _needs_eeg_average_ref_proj(raw.info)

    raw.set_eeg_reference(projection=True)
    assert not _needs_eeg_average_ref_proj(raw.info)

    # No EEG channels
    raw = read_raw_fif(raw_fname, preload=True)
    eeg = [raw.ch_names[c] for c in pick_types(raw.info, meg=False, eeg=True)]
    raw.drop_channels(eeg)
    assert not _needs_eeg_average_ref_proj(raw.info)

    # Custom ref flag set
    raw = read_raw_fif(raw_fname)
    with raw.info._unlock():
        raw.info["custom_ref_applied"] = True
    assert not _needs_eeg_average_ref_proj(raw.info)


def test_sss_proj():
    """Test `meg` proj option."""
    raw = read_raw_fif(raw_fname)
    raw.crop(0, 1.0).load_data().pick(picks="meg")
    raw.pick(raw.ch_names[:51]).del_proj()
    raw_sss = maxwell_filter(raw, int_order=5, ext_order=2)
    sss_rank = 21  # really low due to channel picking
    assert len(raw_sss.info["projs"]) == 0
    for meg, n_proj, want_rank in (
        ("separate", 6, sss_rank),
        ("combined", 3, sss_rank - 3),
    ):
        proj = compute_proj_raw(raw_sss, n_grad=3, n_mag=3, meg=meg, verbose="error")
        this_raw = raw_sss.copy().add_proj(proj).apply_proj()
        assert len(this_raw.info["projs"]) == n_proj
        sss_proj_rank = _compute_rank_int(this_raw)
        cov = compute_raw_covariance(this_raw, verbose="error")
        W, ch_names, rank = compute_whitener(cov, this_raw.info, return_rank=True)
        assert ch_names == this_raw.ch_names
        assert want_rank == sss_proj_rank == rank  # proper reduction
        if meg == "combined":
            assert this_raw.info["projs"][0]["data"]["col_names"] == ch_names
        else:
            mag_names = ch_names[2::3]
            assert this_raw.info["projs"][3]["data"]["col_names"] == mag_names


def test_eq_ne():
    """Test == and != between projectors."""
    raw = read_raw_fif(raw_fname, preload=False)
    assert len(raw.info["projs"]) == 3
    raw.set_eeg_reference(projection=True)

    pca1 = cp.deepcopy(raw.info["projs"][0])
    pca2 = cp.deepcopy(raw.info["projs"][1])
    car = cp.deepcopy(raw.info["projs"][3])

    assert pca1 != pca2
    assert pca1 != car
    assert pca2 != car
    assert pca1 == raw.info["projs"][0]
    assert pca2 == raw.info["projs"][1]
    assert car == raw.info["projs"][3]


def test_setup_proj():
    """Test setup_proj."""
    raw = read_raw_fif(raw_fname)
    assert _needs_eeg_average_ref_proj(raw.info)
    raw.del_proj()
    setup_proj(raw.info)


@testing.requires_testing_data
def test_compute_proj_explained_variance():
    """Test computation based on the explained variance."""
    raw = read_raw_fif(sample_path / "sample_audvis_trunc_raw.fif", preload=False)
    raw.crop(0, 5).load_data()
    raw.del_proj("all")
    with pytest.warns(RuntimeWarning, match="Too few samples"):
        projs = compute_proj_raw(raw, duration=None, n_grad=0.7, n_mag=0.9, n_eeg=0.8)
    for type_, n_vector_ in (("planar", 0.7), ("axial", 0.9), ("eeg", 0.8)):
        explained_var = [
            proj["explained_var"]
            for proj in projs
            if proj["desc"].split("-")[0] == type_
        ]
        assert n_vector_ <= sum(explained_var)
