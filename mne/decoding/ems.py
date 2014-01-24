import numpy as np
from scipy.linalg import norm
from ..externals.six import string_types
from ..utils import logger, verbose



@verbose
def compute_ems(epochs, conditions=None, objective_function=None,
                picks=None, verbose=None):
    """Compute event-matched spatial filter on epochs

    This version operates on the entire timecourse. No time window needs to
    be specified. The result is a spatial filter at each time point and a
    corresponding timecourse. Intuitively, the result gives the similarity
    between the filter at each time point and the data vector (sensors) at
    that timepoint.

    References
    ----------
    [1] Aaron Schurger, Sebastien Marti, and Stanislas Dehaene, "Reducing
        multi-sensor data to a single time course that reveals experimental
        effects", BMC Neuroscience 2013, 14:122

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs.
    conditions : list-like | list of str
        Either a list or an array of indices or bool arrays or a list of
        strings. If a list of strings, strings must match the
        epochs.event_id's key as well as the number of conditios supported
        by the objective_function.
    objective_function : callable
        The objective function to maximize. Must comply with the following
        API:

        def objective_function(data, conditions, **kwargs):
            ...
            return numpy.ndarray (n_channels, n_times)

        If None, the difference function as described in [1]
    verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.

    Returns
    -------
    surrogate_trials : numpy.ndarray (trials, n_trials, n_time_points)
        The trial surrogates.
    mean_spatial_filter : instance of numpy.ndarray (n_channels, n_times)
        The set of spatial filters.
    """

    data = epochs.get_data()
    if picks is not None:
        data = data[:, picks]

    if conditions is None:
        conditions = list(sorted(epochs.event_id.keys()))

    if (isinstance(conditions, list) and
       any(isinstance(k, string_types) for k in conditions)):
        if not all([k in epochs.event_id for k in conditions]):
            raise ValueError('Not all condition-keys present, pleas check')
        conditions = np.array([epochs.events[:, 2] == epochs.event_id[k] for
                               k in conditions])

    if not isinstance(conditions, np.ndarray):
        conditions = np.array(conditions)

    # make sure indices are bool if they're not positional
    if tuple(np.unique(conditions)) == (0, 1):
        conditions = conditions.astype(bool)

    return _compute_ems(data, conditions,
                        objective_function=objective_function, verbose=None)


def _compute_ems(data, conditions, objective_function=None, verbose=None):
    """Aux function
    """

    n_epochs, n_channels, n_times = data.shape
    spatial_filter = np.zeros((n_channels, n_times))
    surrogate_trials = np.zeros((n_epochs, n_times))

    if objective_function is None:
        objective_function = _ems_diff

    from sklearn.cross_validation import LeaveOneOut

    for train_indices, epoch_idx in LeaveOneOut(n_epochs):
        logger.info('.. processing epoch %i' % epoch_idx)
        d = objective_function(data, conditions, train_indices)
        # take norm over channels (matlab uses 2-norm)
        for time_idx in np.arange(n_times):
            d[:, time_idx] /= norm(d[:, time_idx], ord=2)

        # update spatial filter
        spatial_filter += d

        # compute surrogates
        surrogate_trials[epoch_idx] = np.nansum(np.squeeze(data[epoch_idx])
                                                * d, axis=0)

    spatial_filter /= n_epochs

    return surrogate_trials, spatial_filter


def _ems_diff(data, conditions, train):
    """defaut diff objective function
    """

    data_a, data_b = [data[conditions[i]] for i in [0, 1]]
    sum1, sum2 = np.nansum(data_a, axis=0), np.nansum(data_b, axis=0)
    m1 = (sum1 - np.nansum(data[train], axis=0)) / (len(data_b) - len(train))
    m2 = (sum2 - np.nansum(data[train], axis=0)) / (len(data_b) - len(train))

    return m1 - m2
