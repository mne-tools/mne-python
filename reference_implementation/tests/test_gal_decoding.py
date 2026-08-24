"""Tests for the local GAL decoding tutorial helpers."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from ..gal_decoding import (
    AdaptiveDiagonalLDA,
    MATLABGALReference,
    compare_with_matlab,
    compute_gal_scores,
    create_hydrocel_info,
    legacy_gal_inference,
    load_erp_data,
    load_matlab_gal_result,
    prepare_gal_data,
    subject_splits,
)


def _condition_data(offset):
    """Make a small 129-channel, two-time-point condition array."""
    return np.full((129, 2, 4), offset, float)


def test_prepare_gal_data_concatenates_conditions_with_time_as_features():
    """GAL input is trials by time features by selected sensors."""
    pleasant = _condition_data(1)
    unpleasant = _condition_data(-1)
    groups = np.array([1, 1, 2, 2])

    gal_data = prepare_gal_data(pleasant, unpleasant, groups, groups)

    assert gal_data.X.shape == (8, 2, 125)
    assert_array_equal(gal_data.y, [0, 0, 0, 0, 1, 1, 1, 1])
    assert_array_equal(gal_data.groups, [1, 1, 2, 2, 1, 1, 2, 2])
    assert_array_equal(gal_data.channel_indices, [*range(124), 128])
    assert_array_equal(gal_data.X[:4], 1)
    assert_array_equal(gal_data.X[4:], -1)


def test_create_hydrocel_info_matches_the_selected_original_sensors():
    """The topomap uses E1--E124 and Cz from the standard 129-channel montage."""
    info = create_hydrocel_info()

    assert info["sfreq"] == 500
    assert len(info["ch_names"]) == 125
    assert info["ch_names"][:2] == ["E1", "E2"]
    assert info["ch_names"][-2:] == ["E124", "Cz"]


def test_tutorial_script_can_be_invoked_from_the_repository_root():
    """The documented direct-file command resolves the local helper module."""
    root = Path(__file__).parents[2]
    result = subprocess.run(
        [sys.executable, "reference_implementation/tutorial_gal_decoding.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Replicate the decoding portion" in result.stdout


def test_documentation_tutorial_computes_the_paper_scale_result_from_raw_data():
    """The tutorial uses MNE's supported visual-object dataset for GAL."""
    root = Path(__file__).parents[2]
    tutorial = (root / "tutorials/machine-learning/60_gal_decoding.py").read_text()

    assert "visual_92_categories" in tutorial
    assert "StratifiedKFold" in tutorial
    assert "GeneralizingEstimator" in tutorial
    assert "time series at one sensor as a feature vector" in tutorial
    assert "MNE_GAL_DATA_PATH" not in tutorial
    assert "n_jobs=1" in tutorial
    assert 'LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")' in tutorial
    assert "Relation to representational similarity analysis" in tutorial


def test_prepare_gal_data_rejects_unmatched_subject_sets():
    """Each condition must contain trials from the same participants."""
    with pytest.raises(ValueError, match="same participants"):
        prepare_gal_data(
            _condition_data(1),
            _condition_data(-1),
            np.array([1, 1, 2, 2]),
            np.array([1, 1, 3, 3]),
        )


def test_load_erp_data_reads_the_original_aggregate_variable_names(tmp_path):
    """The loader accepts the condition files emitted by groupCondition.m."""
    groups = np.array([[1], [1], [2], [2]])
    savemat(
        tmp_path / "Condition_Pleasant.mat",
        {
            "Data_Condition_Pleasant": _condition_data(1),
            "List_Condition_Pleasant": groups,
        },
    )
    savemat(
        tmp_path / "Condition_Unpleasant.mat",
        {
            "Data_Condition_Unpleasant": _condition_data(-1),
            "List_Condition_Unpleasant": groups,
        },
    )

    gal_data = load_erp_data(tmp_path)

    assert gal_data.X.shape == (8, 2, 125)
    assert_array_equal(gal_data.groups, [1, 1, 2, 2, 1, 1, 2, 2])


def test_load_matlab_gal_result_extracts_decoding_and_masks(tmp_path):
    """MATLAB's nested timeGALoutput struct is converted to plain arrays."""
    savemat(
        tmp_path / "result.mat",
        {
            "timeGALoutput": {
                "GeneralizationMatrix": {
                    "GAL": np.full((2, 2, 2), 0.75),
                    "GALmaskPos": np.array([[1, 0], [0, 1]]),
                    "GALmaskNeg": np.array([[0, 1], [1, 0]]),
                },
                "Parameters": {"ListOfSubjects": np.array([1, 2])},
            }
        },
    )

    reference = load_matlab_gal_result(tmp_path / "result.mat")

    assert_array_equal(reference.scores, np.full((2, 2, 2), 0.75))
    assert_array_equal(reference.positive_mask, [[True, False], [False, True]])
    assert_array_equal(reference.negative_mask, [[False, True], [True, False]])
    assert_array_equal(reference.subjects, [1, 2])


def test_compare_with_matlab_allows_one_held_out_trial_per_cell():
    """Parity tolerance is expressed in the resolution of each fold's accuracy."""
    scores = np.array(
        [
            [[0.74, 0.74], [0.74, 0.74]],
            [[0.75, 0.75], [0.75, 0.75]],
        ]
    )
    inference = legacy_gal_inference(scores)
    reference = MATLABGALReference(
        scores=scores + np.array([[[0.5, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
        positive_mask=inference.positive,
        negative_mask=inference.negative,
        subjects=np.array([1, 2]),
    )

    report = compare_with_matlab(
        scores, inference, reference, np.array([1, 1, 2, 2])
    )

    assert report.scores_match
    assert report.positive_masks_match
    assert report.negative_masks_match


def test_compare_with_matlab_rejects_incompatible_participant_sets():
    """A reference generated from a different subject set is not comparable."""
    scores = np.array(
        [
            [[0.74, 0.74], [0.74, 0.74]],
            [[0.75, 0.75], [0.75, 0.75]],
        ]
    )
    inference = legacy_gal_inference(scores)
    reference = MATLABGALReference(
        scores=scores,
        positive_mask=inference.positive,
        negative_mask=inference.negative,
        subjects=np.array([1, 3]),
    )

    with pytest.raises(ValueError, match="participant identifiers"):
        compare_with_matlab(scores, inference, reference, np.array([1, 1, 2, 2]))


def test_subject_splits_hold_out_one_complete_participant():
    """Every LOSO fold leaves all and only one participant out."""
    groups = np.array([10, 10, 20, 20, 30, 30])

    splits = subject_splits(groups)

    assert len(splits) == 3
    for train, test in splits:
        assert len(set(groups[train]) & set(groups[test])) == 0
        assert len(np.unique(groups[test])) == 1


def test_compute_gal_scores_generalizes_temporal_decoder_across_sensors():
    """The score axes are held-out participant by train sensor by test sensor."""
    labels_per_subject = np.array([0, 0, 1, 1])
    y = np.tile(labels_per_subject, 3)
    groups = np.repeat([1, 2, 3], len(labels_per_subject))
    waveform = np.where(y == 0, -1.0, 1.0) + np.tile([-0.1, 0.1, -0.1, 0.1], 3)
    X = np.empty((len(y), 2, 2))
    X[:, :, 0] = waveform[:, np.newaxis]
    X[:, :, 1] = waveform[:, np.newaxis]

    scores = compute_gal_scores(X, y, groups)

    assert scores.shape == (3, 2, 2)
    assert_array_equal(scores, np.ones((3, 2, 2)))


def test_adaptive_diagonal_lda_regularizes_a_singular_covariance():
    """The temporal classifier stays usable with more features than trials."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(8, 20))
    X[:4] -= 3
    X[4:] += 3
    y = np.array([0] * 4 + [1] * 4)

    estimator = AdaptiveDiagonalLDA(rank_tolerance=1e-8).fit(X, y)

    assert estimator.gamma_ > 0
    assert_array_equal(estimator.predict(X), y)


def test_legacy_gal_inference_has_separate_positive_and_negative_masks():
    """The paper's two-sided inference uses alpha divided by sensor count."""
    scores = np.array(
        [
            [[0.60, 0.40], [0.60, 0.40]],
            [[0.61, 0.39], [0.61, 0.39]],
            [[0.59, 0.41], [0.59, 0.41]],
            [[0.60, 0.40], [0.60, 0.40]],
            [[0.61, 0.39], [0.61, 0.39]],
            [[0.59, 0.41], [0.59, 0.41]],
        ]
    )

    inference = legacy_gal_inference(scores, alpha=0.05)

    assert inference.alpha_corrected == pytest.approx(0.025)
    assert_array_equal(inference.positive, [[True, False], [True, False]])
    assert_array_equal(inference.negative, [[False, True], [False, True]])
    assert_array_equal(inference.signed_mask, [[1, -1], [1, -1]])


def assert_array_equal(actual, desired):
    """Keep NumPy's helpful comparison error in test output."""
    np.testing.assert_array_equal(actual, desired)
