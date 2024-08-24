"""Bad channel detection using Local Outlier Factor (LOF)."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from .._fiff.pick import _picks_to_idx
from ..io.base import BaseRaw
from ..utils import _soft_import, _validate_type, logger, verbose


@verbose
def find_bad_channels_lof(
    raw,
    n_neighbors=20,
    *,
    picks=None,
    metric="euclidean",
    threshold=1.5,
    return_scores=False,
    verbose=None,
):
    """Find bad channels using Local Outlier Factor (LOF) algorithm.

    Parameters
    ----------
    raw : instance of Raw
        Raw data to process.
    n_neighbors : int
        Number of neighbors defining the local neighborhood (default is 20).
        Smaller values will lead to higher LOF scores.
    %(picks_good_data)s
    metric : str
        Metric to use for distance computation. Default is “euclidean”,
        see :func:`sklearn.metrics.pairwise.distance_metrics` for details.
    threshold : float
        Threshold to define outliers. Theoretical threshold ranges anywhere
        between 1.0 and any positive integer. Default: 1.5
        It is recommended to consider this as an hyperparameter to optimize.
    return_scores : bool
        If ``True``, return a dictionary with LOF scores for each
        evaluated channel. Default is ``False``.
    %(verbose)s

    Returns
    -------
    noisy_chs : list
        List of bad M/EEG channels that were automatically detected.
    scores : ndarray, shape (n_picks,)
        Only returned when ``return_scores`` is ``True``. It contains the
        LOF outlier score for each channel in ``picks``.

    See Also
    --------
    maxwell_filter
    annotate_amplitude

    Notes
    -----
    See :footcite:`KumaravelEtAl2022` and :footcite:`BreunigEtAl2000` for background on
    choosing ``threshold``.

    .. versionadded:: 1.7

    References
    ----------
    .. footbibliography::
    """  # noqa: E501
    _soft_import("sklearn", "using LOF detection", strict=True)
    from sklearn.neighbors import LocalOutlierFactor

    _validate_type(raw, BaseRaw, "raw")
    # Get the channel types
    channel_types = raw.get_channel_types()
    picks = _picks_to_idx(raw.info, picks=picks, none="data", exclude="bads")
    picked_ch_types = set(channel_types[p] for p in picks)

    # Check if there are different channel types
    if len(picked_ch_types) != 1:
        raise ValueError(
            f"Need exactly one channel type in picks, got {sorted(picked_ch_types)}"
        )
    ch_names = [raw.ch_names[pick] for pick in picks]
    data = raw.get_data(picks=picks)
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, metric=metric)
    clf.fit_predict(data)
    scores_lof = clf.negative_outlier_factor_
    bad_channel_indices = [
        i for i, v in enumerate(np.abs(scores_lof)) if v >= threshold
    ]
    bads = [ch_names[idx] for idx in bad_channel_indices]
    logger.info(f"LOF: Detected bad channel(s): {bads}")
    if return_scores:
        return bads, scores_lof
    else:
        return bads
