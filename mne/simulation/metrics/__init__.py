# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Metrics module for compute stc-based metrics."""

from .metrics import (
    cosine_score,
    region_localization_error,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    peak_position_error,
    source_estimate_quantification,
    spatial_deviation_error,
    _thresholding,
    _check_threshold,
    _uniform_stc,
)
