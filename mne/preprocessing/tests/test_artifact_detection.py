# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mne import Annotations, events_from_annotations
from mne.chpi import read_head_pos
from mne.datasets import testing
from mne.io import read_raw_fif
from mne.preprocessing import (
    annotate_break,
    annotate_movement,
    annotate_muscle_zscore,
    compute_average_dev_head_t,
)
from mne.tests.test_annotations import _assert_annotations_equal
from mne.transforms import _angle_dist_between_rigid, quat_to_rot, rot_to_quat

data_path = testing.data_path(download=False)
sss_path = data_path / "SSS"
pos_fname = sss_path / "test_move_anon_raw.pos"
raw_fname = sss_path / "test_move_anon_raw.fif"


@testing.requires_testing_data
@pytest.mark.parametrize("meas_date", (None, "orig"))
def test_movement_annotation_head_correction(meas_date):
    """Test correct detection movement artifact and dev_head_t."""
    raw = read_raw_fif(raw_fname, allow_maxshield="yes").load_data()
    pos = read_head_pos(pos_fname)
    if meas_date is None:
        raw.set_meas_date(None)
    else:
        assert meas_date == "orig"
    raw_unannot = raw.copy()

    # Check 5 rotation segments are detected
    annot_rot, [] = annotate_movement(raw, pos, rotation_velocity_limit=5)
    assert annot_rot.orig_time == raw.info["meas_date"]
    assert annot_rot.duration.size == 5

    # Check 2 translation vel. segments are detected
    annot_tra, [] = annotate_movement(raw, pos, translation_velocity_limit=0.05)
    assert annot_tra.duration.size == 2

    # Check 1 movement distance segment is detected
    annot_dis, _ = annotate_movement(raw, pos, mean_distance_limit=0.02)
    assert annot_dis.duration.size == 1

    # Check correct trans mat
    annot_all_2 = annotate_movement(
        raw,
        pos,
        rotation_velocity_limit=5,
        translation_velocity_limit=0.05,
        mean_distance_limit=0.02,
    )[0]
    assert (
        annot_rot.orig_time
        == annot_tra.orig_time
        == annot_dis.orig_time
        == raw.info["meas_date"]
    )
    annot_all = annot_rot + annot_tra + annot_dis
    _assert_annotations_equal(annot_all_2, annot_all)
    assert annot_all.orig_time == raw.info["meas_date"]
    raw.set_annotations(annot_all)
    dev_head_t = compute_average_dev_head_t(raw, pos)["trans"]

    dev_head_t_ori = np.array(
        [
            [0.9957292, -0.08688804, 0.03120615, 0.00698271],
            [0.09020767, 0.9875856, -0.12859731, -0.0159098],
            [-0.01964518, 0.1308631, 0.99120578, 0.07258289],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert_allclose(dev_head_t_ori, dev_head_t, rtol=1e-5, atol=0)

    with pytest.raises(ValueError, match="Number of .* must match .*"):
        compute_average_dev_head_t([raw], [pos] * 2)
    # Using two identical ones should be identical ...
    dev_head_t_double = compute_average_dev_head_t([raw] * 2, [pos] * 2)["trans"]
    assert_allclose(dev_head_t, dev_head_t_double)
    # ... unannotated and annotated versions differ ...
    dev_head_t_unannot = compute_average_dev_head_t(raw_unannot, pos)["trans"]
    rot_tol = 1.5e-3
    mov_tol = 1e-3
    assert not np.allclose(
        dev_head_t_unannot[:3, :3],
        dev_head_t[:3, :3],
        atol=rot_tol,
        rtol=0,
    )
    assert not np.allclose(
        dev_head_t_unannot[:3, 3],
        dev_head_t[:3, 3],
        atol=mov_tol,
        rtol=0,
    )
    # ... and Averaging the two is close to (but not identical!) to operating on the two
    # files. Note they shouldn't be identical because there are more time points
    # included in the unannotated version!
    dev_head_t_naive = np.eye(4)
    dev_head_t_naive[:3, :3] = quat_to_rot(
        np.mean(
            rot_to_quat(np.array([dev_head_t[:3, :3], dev_head_t_unannot[:3, :3]])),
            axis=0,
        )
    )
    dev_head_t_naive[:3, 3] = np.mean(
        [dev_head_t[:3, 3], dev_head_t_unannot[:3, 3]], axis=0
    )
    dev_head_t_combo = compute_average_dev_head_t([raw, raw_unannot], [pos] * 2)[
        "trans"
    ]
    unit_kw = dict(distance_units="mm", angle_units="deg")
    deg_annot_combo, mm_annot_combo = _angle_dist_between_rigid(
        dev_head_t,
        dev_head_t_combo,
        **unit_kw,
    )
    deg_unannot_combo, mm_unannot_combo = _angle_dist_between_rigid(
        dev_head_t_unannot,
        dev_head_t_combo,
        **unit_kw,
    )
    deg_annot_unannot, mm_annot_unannot = _angle_dist_between_rigid(
        dev_head_t,
        dev_head_t_unannot,
        **unit_kw,
    )
    deg_combo_naive, mm_combo_naive = _angle_dist_between_rigid(
        dev_head_t_combo,
        dev_head_t_naive,
        **unit_kw,
    )
    # combo<->naive closer than combo<->annotated closer than annotated<->unannotated
    assert 0.05 < deg_combo_naive < deg_annot_combo < deg_annot_unannot < 1.5
    assert 0.1 < mm_combo_naive < mm_annot_combo < mm_annot_unannot < 2
    # combo<->naive closer than combo<->unannotated closer than annotated<->unannotated
    assert 0.05 < deg_combo_naive < deg_unannot_combo < deg_annot_unannot < 1.5
    assert 0.12 < mm_combo_naive < mm_unannot_combo < mm_annot_unannot < 2.0

    # Smoke test skipping time due to previous annotations.
    raw.set_annotations(Annotations([raw.times[0]], 0.1, "bad"))
    annot_dis, _ = annotate_movement(raw, pos, mean_distance_limit=0.02)
    assert annot_dis.duration.size == 1

    # really far should warn
    pos[:, 4] += 5
    with pytest.warns(RuntimeWarning, match="Implausible head position"):
        compute_average_dev_head_t(raw, pos)


@testing.requires_testing_data
@pytest.mark.parametrize("meas_date", (None, "orig"))
def test_muscle_annotation(meas_date, events):
    """Test correct detection muscle artifacts."""
    raw = read_raw_fif(raw_fname, allow_maxshield="yes").load_data()
    if meas_date is None:
        raw.set_meas_date(None)
    raw.notch_filter([50, 110, 150])
    # Check 2 muscle segments are detected
    annot_muscle, scores = annotate_muscle_zscore(raw, ch_type="mag", threshold=10)
    assert annot_muscle.orig_time == raw.info["meas_date"]
    onset = annot_muscle.onset * raw.info["sfreq"]
    if meas_date is not None:
        onset -= raw.first_samp
    onset = onset.astype(int)
    assert_array_equal(scores[onset].astype(int), np.array([23, 10]))
    assert annot_muscle.duration.size == 2
    raw.set_annotations(annot_muscle)


@testing.requires_testing_data
@pytest.mark.parametrize("meas_date", (None, "orig"))
def test_muscle_annotation_without_meeg_data(meas_date):
    """Call annotate_muscle_zscore with data without meg or eeg."""
    raw = read_raw_fif(raw_fname, allow_maxshield="yes")
    if meas_date is None:
        raw.set_meas_date(None)
    raw.crop(0, 0.1).load_data()
    raw.pick("stim")
    with pytest.raises(ValueError, match="No M/EEG channel types found"):
        annotate_muscle_zscore(raw, threshold=10)


@pytest.mark.parametrize("meas_date", (None, "orig"))
@testing.requires_testing_data
def test_annotate_breaks(meas_date):
    """Test annotate_breaks."""
    raw = read_raw_fif(raw_fname, allow_maxshield="yes")
    if meas_date is None:
        raw.set_meas_date(None)

    annots = Annotations(
        onset=[12, 15, 16, 20, 21],
        duration=[1, 1, 1, 2, 0.5],
        description=["test"],
        orig_time=raw.info["meas_date"],
    )

    if raw.info["meas_date"] is None:
        annots.onset -= raw.first_time

    raw.set_annotations(annots)

    min_break_duration = 0.5
    t_start_after_previous = 0.1
    t_stop_before_next = 0.1

    expected_onsets = np.array(
        [
            raw.first_time,
            13 + t_start_after_previous,
            17 + t_start_after_previous,
            22 + t_start_after_previous,
        ]
    )

    if raw.info["meas_date"] is None:
        expected_onsets -= raw.first_time

    expected_durations = np.array(
        [
            12 - raw.first_time - t_stop_before_next,
            15 - 13 - t_start_after_previous - t_stop_before_next,
            20 - 17 - t_start_after_previous - t_stop_before_next,
            raw._last_time - 22 - t_start_after_previous,
        ]
    )

    break_annots = annotate_break(
        raw=raw,
        min_break_duration=min_break_duration,
        t_start_after_previous=t_start_after_previous,
        t_stop_before_next=t_stop_before_next,
    )

    assert break_annots.orig_time == raw.info["meas_date"]
    assert_allclose(break_annots.onset, expected_onsets)
    assert_allclose(break_annots.duration, expected_durations)
    assert all(description == "BAD_break" for description in break_annots.description)

    # try setting the annotations, this should not omit anything
    raw.set_annotations(break_annots)
    current_annotations = raw.annotations
    if raw.info["meas_date"] is None:
        current_annotations.onset -= raw.first_time
    raw.set_annotations(current_annotations + break_annots)

    # reset before next test
    raw.set_annotations(annots)

    # `ignore` parameter should be respected
    raw.annotations.description[0] = "BAD_"
    break_annots = annotate_break(
        raw=raw,
        min_break_duration=min_break_duration,
        t_start_after_previous=t_start_after_previous,
        t_stop_before_next=t_stop_before_next,
    )

    assert_allclose(break_annots.onset, expected_onsets[[True, False, True, True]])
    assert_allclose(
        break_annots.duration,
        [15 - raw.first_time - t_stop_before_next] + list(expected_durations[2:]),
    )

    # try setting the annotations, this should not omit anything
    raw.set_annotations(break_annots)
    current_annotations = raw.annotations
    if raw.info["meas_date"] is None:
        current_annotations.onset -= raw.first_time
    raw.set_annotations(current_annotations + break_annots)

    # Restore annotations for next test
    raw.set_annotations(annots)
    raw.annotations.description[0] = "test"

    # Test with events
    events, _ = events_from_annotations(raw=raw)
    raw.set_annotations(None)

    expected_onsets = np.array(
        [
            raw.first_time,
            12 + t_start_after_previous,
            15 + t_start_after_previous,
            16 + t_start_after_previous,
            20 + t_start_after_previous,
            21 + t_start_after_previous,
        ]
    )

    expected_durations = np.array(
        [
            12 - raw.first_time - t_stop_before_next,
            15 - 12 - t_start_after_previous - t_stop_before_next,
            16 - 15 - t_start_after_previous - t_stop_before_next,
            20 - 16 - t_start_after_previous - t_stop_before_next,
            21 - 20 - t_start_after_previous - t_stop_before_next,
            raw._last_time - 21 - t_start_after_previous,
        ]
    )

    break_annots = annotate_break(
        raw=raw,
        events=events,
        min_break_duration=min_break_duration,
        t_start_after_previous=t_start_after_previous,
        t_stop_before_next=t_stop_before_next,
    )

    if raw.info["meas_date"] is None:
        expected_onsets -= raw.first_time

    assert_allclose(break_annots.onset, expected_onsets)
    assert_allclose(break_annots.duration, expected_durations)

    # try setting the annotations, this should not omit anything
    raw.set_annotations(break_annots)
    current_annotations = raw.annotations
    if raw.info["meas_date"] is None:
        current_annotations.onset -= raw.first_time
    raw.set_annotations(current_annotations + break_annots)

    # reset before next test
    raw.set_annotations(annots)

    # Not finding any break periods
    break_annots = annotate_break(
        raw=raw,
        events=events,
        min_break_duration=1000,
    )

    assert len(break_annots) == 0

    # Implausible parameters (would produce break annot of duration < 0)
    with pytest.raises(ValueError, match="must be greater than 0"):
        annotate_break(
            raw=raw,
            min_break_duration=5,
            t_start_after_previous=5,
            t_stop_before_next=5,
        )

    # Empty events array
    with pytest.raises(ValueError, match="events array must not be empty"):
        annotate_break(raw=raw, events=np.array([]))

    # Invalid `ignore` value
    with pytest.raises(TypeError, match="must be an instance of str"):
        annotate_break(raw=raw, ignore=("foo", 1))

    # No annotations to work with
    raw.set_annotations(None)
    with pytest.raises(ValueError, match="Could not find.*annotations"):
        annotate_break(raw=raw)
