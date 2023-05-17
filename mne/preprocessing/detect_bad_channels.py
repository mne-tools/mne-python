# Authors: Velu Prabhakar Kumaravel <vkumaravel@fbk.eu>      
# License: BSD-3-Clause


import numpy as np

from ..io.base import BaseRaw
from .. import Transform
from ..utils import (logger, verbose, _validate_type)


@verbose
def mark_bad_channels_lof(raw, n_neighbors = 8, metric = 'sqeuclidean', lof_threshold = 1.5):
    """ Detects and marks bad channels using Local Outlier Factor (LOF) algorithm.

    Detects bad (a.k.a. anamolous) channels using a local density based algorithm:LOF.
    By default, n_neighbors parameter is set to 8, which will work in most cases

    See :footcite:`kumaravel2022` for background on choosing
    ``metric`` and ``lof_threshold``.

    Parameters
    ----------
    raw : instance of Raw
        The raw data.

    n_neighbors : int
        Number of neighbors for computing local density (default: 8).

    metric : 'euclidean' | 'sqeuclidean' | 'cityblock' | 'cosine' | 'l1' | 'l2' | 'manhattan' | 'nan_euclidean'
        The metric used for distance compuation. Default: 'sqeuclidean' as it performed better than 'euclidean' in MATLAB implementation.
        One could consider this as an hyperparameter to optimize.

    lof_threshold : float
        Decision threshold for outlier/bad channels. Default: 1.5. If you deal with noisier data like newborns EEG, you might use a higher value (e.g., 2.5)
        One could consider this as an hyperparameter to optimize.
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        The raw instance with updated 'info'

    References
    ----------
    footbibliography: Kumaravel et al. 2022. "Adaptable and Robust EEG Bad Channel Detection Using Local Outlier Factor (LOF)" Sensors 22, no. 19: 7314. https://doi.org/10.3390/s22197314
    """

    from sklearn.neighbors import LocalOutlierFactor

    _validate_type(raw, BaseRaw, 'raw')
    raw_copy = raw.copy()

    clf = LocalOutlierFactor(n_neighbors)
    data = raw_copy.get_data()
    clf.fit_predict(data)
    lof_scores = clf.negative_outlier_factor_
    bad_channel_indices = [i for i,v in enumerate(np.abs(lof_scores)) if v >= lof_threshold]

    for elem in bad_channel_indices:
        logger.info('LOF: Marking channel %s as bad'
                % raw_copy.info['chs'][elem]["ch_name"])
        raw_copy.info['bads'].append(raw_copy.info['chs'][elem]["ch_name"])
   

    return raw_copy
