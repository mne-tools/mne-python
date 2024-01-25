# Authors: Velu Prabhakar Kumaravel <vpr.kumaravel@gmail.com>
# License: BSD-3-Clause


import numpy as np

from ..io.base import BaseRaw
from ..utils import _validate_type, logger, verbose


@verbose
def find_bad_channels_lof(
    raw,
    n_neighbors=20,
    metric="euclidean",
    threshold=1.5,
    return_scores=False,
    verbose=None,
):
    r"""Find bad channels using Local Outlier Factor (LOF) algorithm.

    See :footcite:`KumaravelEtAl2022` for background on choosing
    `threshold``.

    Parameters
    ----------
    raw : instance of Raw
        Raw data to process
    n_neighbours : int
        Number of neighbours defining the local neighbourhood (default is 20).
        Smaller values will lead to higher LOF scores.
    metric: str in {'euclidean', 'nan_euclidean', 'cosine',
                    'cityblock', 'manhattan'}
        Metric to use for distance computation. Default is “euclidean”.
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
    scores : dict
        A dictionary with information produced by the scoring algorithms.
        Only returned when ``return_scores`` is ``True``. It contains the
        following keys:

        - ``ch_names`` : ndarray, shape (n_meeg,)
            The names of the M/EEG channels.
        - ``ch_types`` : ndarray, shape (n_meeg,)
            The types of the M/EEG channels in ``ch_names`` (``'mag'``,
            ``'grad'``, ``'eeg'``).
        - ``scores_lof`` : ndarray, shape (n_meeg,)
            LOF outlier score for each channel.

    See Also
    --------
    maxwell_filter
    annotate_amplitude

    References
    ----------
        KumaravelEtAl2022, BreunigEtAl2000

    footbibliography:
    """
    try:
        from sklearn.neighbors import LocalOutlierFactor
    except ImportError:
        print(
            "scikit-learn is not installed. "
            "Please install it by running 'pip install scikit-learn'."
        )

    _validate_type(raw, BaseRaw, "raw")
    # Get the channel types
    channel_types = set(raw.get_channel_types())

    # Check if there are different channel types
    if len(channel_types) > 1:
        # Print the channel types
        print("Channel types:", channel_types)
        raise ValueError(
            "Multiple channel types passed for LOF."
            "Please pick only one kind of data (e.g., 'grad')"
        )

    noisy_chs = raw.info["bads"]
    ch_names = raw.ch_names
    ch_types = raw.get_channel_types()
    data = raw.get_data()
    print(data.shape)
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, metric=metric)
    clf.fit_predict(data)
    scores_lof = clf.negative_outlier_factor_
    bad_channel_indices = [
        i for i, v in enumerate(np.abs(scores_lof)) if v >= threshold
    ]

    for elem in bad_channel_indices:
        logger.info(
            "LOF: Marking channel %s as " "bad" % raw.info["chs"][elem]["ch_name"]
        )
        noisy_chs.append(raw.info["chs"][elem]["ch_name"])

    if return_scores:
        scores = dict(ch_names=ch_names, ch_types=ch_types, scores_lof=scores_lof)
        return noisy_chs, scores
    else:
        return noisy_chs
