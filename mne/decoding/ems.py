# Author: Denis Engemann <denis.engemann@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np

from ..utils import logger, verbose
from ..fixes import Counter
from ..parallel import parallel_func
from .. import pick_types, pick_info


@verbose
def compute_ems(epochs, conditions=None, picks=None, verbose=None, n_jobs=1):
    """Compute event-matched spatial filter on epochs

    This version operates on the entire time course. No time window needs to
    be specified. The result is a spatial filter at each time point and a
    corresponding time course. Intuitively, the result gives the similarity
    between the filter at each time point and the data vector (sensors) at
    that time point.

    References
    ----------
    [1] Aaron Schurger, Sebastien Marti, and Stanislas Dehaene, "Reducing
        multi-sensor data to a single time course that reveals experimental
        effects", BMC Neuroscience 2013, 14:122

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs.
    conditions : list of str | None
        If a list of strings, strings must match the
        epochs.event_id's key as well as the number of conditions supported
        by the objective_function. If None keys in epochs.event_id are used.
    picks : array-like of int | None
        Channels to be included. If None only good data channels are used.
        Defaults to None
    n_jobs : int
        Number of jobs to run in parallel.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        Defaults to self.verbose.

    Returns
    -------
    surrogate_trials : ndarray, shape (trials, n_trials, n_time_points)
        The trial surrogates.
    mean_spatial_filter : ndarray, shape (n_channels, n_times)
        The set of spatial filters.
    conditions : ndarray, shape (n_epochs,)
        The conditions used. Values correspond to original event ids.
    """
    logger.info('...computing surrogate time series. This can take some time')
    if picks is None:
        picks = pick_types(epochs.info, meg=True, eeg=True)

    if not len(set(Counter(epochs.events[:, 2]).values())) == 1:
        raise ValueError('The same number of epochs is required by '
                         'this function. Please consider '
                         '`epochs.equalize_event_counts`')

    if conditions is None:
        conditions = epochs.event_id.keys()
        epochs = epochs.copy()
    else:
        epochs = epochs[conditions]

    epochs.drop_bad_epochs()

    if len(conditions) != 2:
        raise ValueError('Currently this function expects exactly 2 '
                         'conditions but you gave me %i' %
                         len(conditions))

    ev = epochs.events[:, 2]
    # special care to avoid path dependant mappings and orders
    conditions = list(sorted(conditions))
    cond_idx = [np.where(ev == epochs.event_id[k])[0] for k in conditions]

    info = pick_info(epochs.info, picks)
    data = epochs.get_data()[:, picks]

    # Scale (z-score) the data by channel type
    for ch_type in ['mag', 'grad', 'eeg']:
        if ch_type in epochs:
            if ch_type == 'eeg':
                this_picks = pick_types(info, meg=False, eeg=True)
            else:
                this_picks = pick_types(info, meg=ch_type, eeg=False)
            data[:, this_picks] /= np.std(data[:, this_picks])

    from sklearn.cross_validation import LeaveOneOut

    parallel, p_func, _ = parallel_func(_run_ems, n_jobs=n_jobs)
    out = parallel(p_func(_ems_diff, data, cond_idx, train, test)
                   for train, test in LeaveOneOut(len(data)))

    surrogate_trials, spatial_filter = zip(*out)
    surrogate_trials = np.array(surrogate_trials)
    spatial_filter = np.mean(spatial_filter, axis=0)

    return surrogate_trials, spatial_filter, epochs.events[:, 2]


def _ems_diff(data0, data1):
    """default diff objective function"""
    return np.mean(data0, axis=0) - np.mean(data1, axis=0)


def _run_ems(objective_function, data, cond_idx, train, test):
    d = objective_function(*(data[np.intersect1d(c, train)] for c in cond_idx))
    d /= np.sqrt(np.sum(d ** 2, axis=0))[None, :]
    # compute surrogates
    return np.sum(data[test[0]] * d, axis=0), d
