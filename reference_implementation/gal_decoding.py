"""Helpers for replicating temporal-feature GAL decoding with MNE."""

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class AdaptiveDiagonalLDA(ClassifierMixin, BaseEstimator):
    """Linear discriminant analysis with numerical diagonal regularization.

    The high-dimensional temporal features in the Time-GAL analysis make the
    empirical covariance singular. This estimator applies the smallest
    diagonal regularization suggested by the numerical rank tolerance of the
    correlation matrix. It approximates MATLAB's undocumented ``MinGamma``
    choice while using its documented covariance form,
    ``(1 - gamma) * covariance + gamma * diag``.

    Parameters
    ----------
    rank_tolerance : float | None
        Relative numerical rank tolerance. By default, use the conventional
        ``n_features * eps`` tolerance.
    """

    def __init__(self, rank_tolerance=None):
        self.rank_tolerance = rank_tolerance

    def fit(self, X, y):
        """Fit the shared-covariance discriminant model."""
        from scipy.linalg import solve

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must have shape (samples, features).")
        if y.ndim != 1 or len(X) != len(y):
            raise ValueError("y must provide one label per sample.")

        self.classes_, inverse = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("AdaptiveDiagonalLDA supports exactly two classes.")
        self.n_features_in_ = X.shape[1]
        counts = np.bincount(inverse, minlength=len(self.classes_))
        if np.any(counts < 2):
            raise ValueError("Each class must contain at least two samples.")

        self.priors_ = counts / len(y)
        self.means_ = np.array([X[inverse == index].mean(axis=0) for index in range(2)])
        residuals = X - self.means_[inverse]
        standard_deviation = np.sqrt(np.mean(residuals**2, axis=0))
        if np.any(standard_deviation == 0):
            raise ValueError(
                "Temporal features must have non-zero within-class variance."
            )

        standardized = residuals / standard_deviation
        gram = standardized @ standardized.T / len(standardized)
        maximum_eigenvalue = np.linalg.eigvalsh(gram)[-1]
        tolerance = self.rank_tolerance
        if tolerance is None:
            tolerance = self.n_features_in_ * np.finfo(float).eps
        self.gamma_ = tolerance * maximum_eigenvalue
        self.gamma_ /= 1 + self.gamma_

        covariance = residuals.T @ residuals / len(residuals)
        covariance *= 1 - self.gamma_
        covariance.flat[:: self.n_features_in_ + 1] += (
            self.gamma_ * np.diag(covariance) / (1 - self.gamma_)
        )
        coefficients = solve(covariance, self.means_.T, assume_a="sym").T
        self.coef_ = coefficients
        self.intercept_ = (
            -0.5 * np.sum(self.means_ * coefficients, axis=1) + np.log(self.priors_)
        )
        return self

    def decision_function(self, X):
        """Return one discriminant score per class."""
        X = np.asarray(X, dtype=float)
        return X @ self.coef_.T + self.intercept_

    def predict(self, X):
        """Predict the class with the largest discriminant score."""
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def score(self, X, y):
        """Return the mean classification accuracy."""
        return np.mean(self.predict(X) == np.asarray(y))

_CHANNEL_INDICES = np.concatenate([np.arange(124), [128]])


@dataclass(frozen=True)
class GALData:
    """Data arranged for generalization across sensor locations."""

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    channel_indices: np.ndarray


@dataclass(frozen=True)
class GALInference:
    """Legacy Time-GAL group-level inference results."""

    positive: np.ndarray
    negative: np.ndarray
    signed_mask: np.ndarray
    alpha_corrected: float


@dataclass(frozen=True)
class MATLABGALReference:
    """Decoding outputs extracted from a MATLAB ``timeGALoutput`` struct."""

    scores: np.ndarray
    positive_mask: np.ndarray
    negative_mask: np.ndarray
    subjects: np.ndarray


@dataclass(frozen=True)
class ParityReport:
    """Numerical and inferential agreement with a MATLAB GAL result."""

    scores_match: bool
    positive_masks_match: bool
    negative_masks_match: bool
    maximum_score_difference: float


def create_hydrocel_info():
    """Create MNE sensor metadata for the 125 channels used in the paper."""
    import mne

    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    ch_names = [montage.ch_names[index] for index in _CHANNEL_INDICES]
    info = mne.create_info(ch_names, sfreq=500, ch_types="eeg")
    info.set_montage(montage)
    return info


def load_erp_data(data_dir):
    """Load the aggregate ERP condition files produced by ``groupCondition.m``."""
    from pathlib import Path

    data_dir = Path(data_dir)
    pleasant = _load_mat_file(data_dir / "Condition_Pleasant.mat")
    unpleasant = _load_mat_file(data_dir / "Condition_Unpleasant.mat")
    try:
        return prepare_gal_data(
            pleasant["Data_Condition_Pleasant"],
            unpleasant["Data_Condition_Unpleasant"],
            np.asarray(pleasant["List_Condition_Pleasant"]).squeeze(),
            np.asarray(unpleasant["List_Condition_Unpleasant"]).squeeze(),
        )
    except KeyError as err:
        raise ValueError(
            "Condition files must contain the Data_Condition_* and "
            "List_Condition_* variables created by groupCondition.m."
        ) from err


def load_matlab_gal_result(path):
    """Extract GAL scores and inference masks from a MATLAB result file."""
    result = _load_mat_file(path)
    try:
        output = result["timeGALoutput"]
        generalization = output["GeneralizationMatrix"]
        parameters = output["Parameters"]
        scores = np.asarray(generalization["GAL"])
        if scores.shape[-2:] == (129, 129):
            scores = scores[:, _CHANNEL_INDICES][:, :, _CHANNEL_INDICES]
        return MATLABGALReference(
            scores=scores,
            positive_mask=np.asarray(generalization["GALmaskPos"], dtype=bool),
            negative_mask=np.asarray(generalization["GALmaskNeg"], dtype=bool),
            subjects=np.asarray(parameters["ListOfSubjects"]).squeeze(),
        )
    except (KeyError, TypeError) as err:
        raise ValueError(
            "MATLAB result does not contain a valid timeGALoutput struct."
        ) from err


def compare_with_matlab(scores, inference, reference, groups):
    """Compare a GAL run with MATLAB using one held-out trial as tolerance."""
    scores = np.asarray(scores)
    groups = np.asarray(groups)
    subjects = np.unique(groups)
    if not np.array_equal(subjects, reference.subjects):
        raise ValueError(
            "MATLAB reference and current data use different participant identifiers; "
            "their GAL scores cannot be compared."
        )
    if scores.shape != reference.scores.shape:
        raise ValueError(
            "MATLAB reference and current GAL scores have different shapes: "
            f"{reference.scores.shape} != {scores.shape}."
        )
    if inference.positive.shape != reference.positive_mask.shape:
        raise ValueError(
            "MATLAB reference and current positive masks have different shapes."
        )
    if inference.negative.shape != reference.negative_mask.shape:
        raise ValueError(
            "MATLAB reference and current negative masks have different shapes."
        )

    held_out_trials = np.array([np.sum(groups == subject) for subject in subjects])
    tolerance = (1 / held_out_trials)[:, np.newaxis, np.newaxis]
    difference = np.abs(scores - reference.scores)
    return ParityReport(
        scores_match=bool(np.all(difference <= tolerance)),
        positive_masks_match=bool(
            np.array_equal(inference.positive, reference.positive_mask)
        ),
        negative_masks_match=bool(
            np.array_equal(inference.negative, reference.negative_mask)
        ),
        maximum_score_difference=float(difference.max()),
    )


def prepare_gal_data(pleasant, unpleasant, pleasant_groups, unpleasant_groups):
    """Arrange two condition arrays as trials by time by sensors.

    Parameters
    ----------
    pleasant, unpleasant : ndarray, shape (129, n_times, n_trials)
        Preprocessed ERP data from the original Time-GAL dataset.
    pleasant_groups, unpleasant_groups : ndarray, shape (n_trials,)
        Participant identifiers for the corresponding condition trials.

    Returns
    -------
    gal_data : GALData
        Data with time as features and sensors as generalization tasks.
    """
    pleasant = _validate_condition(pleasant, pleasant_groups, "pleasant")
    unpleasant = _validate_condition(unpleasant, unpleasant_groups, "unpleasant")
    if pleasant.shape[1] != unpleasant.shape[1]:
        raise ValueError("Both conditions must have the same number of time samples.")

    pleasant_groups = np.asarray(pleasant_groups)
    unpleasant_groups = np.asarray(unpleasant_groups)
    if set(pleasant_groups) != set(unpleasant_groups):
        raise ValueError(
            "Both conditions must contain trials from the same participants."
        )

    pleasant_data = pleasant[_CHANNEL_INDICES].transpose(2, 1, 0)
    unpleasant_data = unpleasant[_CHANNEL_INDICES].transpose(2, 1, 0)
    X = np.concatenate([pleasant_data, unpleasant_data])
    y = np.concatenate(
        [
            np.zeros(len(pleasant_groups), dtype=int),
            np.ones(len(unpleasant_groups), dtype=int),
        ]
    )
    groups = np.concatenate([pleasant_groups, unpleasant_groups])
    return GALData(X=X, y=y, groups=groups, channel_indices=_CHANNEL_INDICES.copy())


def subject_splits(groups):
    """Return leave-one-participant-out train/test index pairs."""
    from sklearn.model_selection import LeaveOneGroupOut

    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) == 0:
        raise ValueError("groups must be a non-empty one-dimensional array.")
    splitter = LeaveOneGroupOut()
    return list(splitter.split(np.empty((len(groups), 1)), groups=groups))


def compute_gal_scores(X, y, groups, n_jobs=None):
    """Compute leave-one-subject-out temporal-decoder GAL accuracies."""
    from sklearn.model_selection import LeaveOneGroupOut

    from mne.decoding import GeneralizingEstimator, cross_val_multiscore

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if X.ndim != 3:
        raise ValueError("X must have shape (trials, time_samples, sensors).")
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y, and groups must contain the same number of trials.")

    estimator = GeneralizingEstimator(
        AdaptiveDiagonalLDA(),
        scoring="accuracy",
        n_jobs=n_jobs,
        verbose=False,
    )
    return cross_val_multiscore(
        estimator,
        X,
        y,
        groups=groups,
        cv=LeaveOneGroupOut(),
        n_jobs=1,
    )


def legacy_gal_inference(scores, alpha=0.05):
    """Apply the Time-GAL toolbox's two-sided channel-count correction."""
    from scipy.stats import ttest_1samp

    scores = np.asarray(scores)
    if scores.ndim != 3:
        raise ValueError(
            "scores must have shape (participants, train_sensors, test_sensors)."
        )
    alpha_corrected = alpha / scores.shape[-1]
    positive = (
        ttest_1samp(scores, 0.5, axis=0, alternative="greater").pvalue
        < alpha_corrected
    )
    negative = (
        ttest_1samp(scores, 0.5, axis=0, alternative="less").pvalue
        < alpha_corrected
    )
    signed_mask = positive.astype(int) - negative.astype(int)
    return GALInference(
        positive=positive,
        negative=negative,
        signed_mask=signed_mask,
        alpha_corrected=alpha_corrected,
    )


def _validate_condition(condition, groups, name):
    """Validate the original ERP condition-array contract."""
    condition = np.asarray(condition)
    groups = np.asarray(groups)
    if condition.ndim != 3:
        raise ValueError(
            f"{name} data must have shape (channels, time_samples, trials)."
        )
    if condition.shape[0] != 129:
        raise ValueError(f"{name} data must contain 129 HydroCel channels.")
    if groups.ndim != 1 or len(groups) != condition.shape[-1]:
        raise ValueError(
            f"{name} groups must provide one participant identifier per trial."
        )
    return condition


def _load_mat_file(path):
    """Read a MATLAB v5 file, or a simple numeric MATLAB v7.3 file."""
    from scipy.io import loadmat

    try:
        return loadmat(path, simplify_cells=True)
    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as h5_file:
            return {
                name: np.asarray(value).transpose()
                for name, value in h5_file.items()
                if isinstance(value, h5py.Dataset)
            }
