import operator
import numpy as np
from scipy.linalg import norm

from ..externals.six import string_types
from ..utils import logger, verbose
from ..fixes import partial
from ..fiff import pick_types
from collections import Counter


@verbose
def compute_ems(epochs, conditions=None,
                picks=None, verbose=None):
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
    conditions : list-like | list of str | None
        Either a list or an array of indices or bool arrays or a list of
        strings. If a list of strings, strings must match the
        epochs.event_id's key as well as the number of conditions supported
        by the objective_function. If None keys in epochs.event_id are used.
    picks : array-like | None
        Channels to be included. If None only good data channels are used.
        Defaults to None
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        Defaults to self.verbose.

    Returns
    -------
    surrogate_trials : numpy.ndarray (trials, n_trials, n_time_points)
        The trial surrogates.
    mean_spatial_filter : instance of numpy.ndarray (n_channels, n_times)
        The set of spatial filters.
    conditions : ndarray (n_epochs)
        The conditions used. Values correpsond to original event ids.
    """
    if picks is None:
        picks = pick_types(epochs.info, meg=True, eeg=True)

    if not len(set(Counter(epochs.events[:, 2]).values())) == 1:
        raise ValueError('the same number of epochs is required by '
                         'this function. Please consider '
                         '`epochs.equalize_event_counts`')

    conditions_ = _check_conditions(epochs, conditions)

    epochs = epochs[reduce(operator.add, conditions_)]
    conditions_ = _check_conditions(epochs, conditions)

    epochs = epochs.copy()
    data = epochs.get_data()
    epochs.drop_channels([epochs.ch_names[i] for i in np.arange(data.shape[1])
                          if i not in picks])
    scaling = None
    if sum([k in epochs for k in ['mag', 'grad', 'eeg']]) > 1:
        scaling = np.atleast_1d(np.std(data)) ** -1

    data = data[:, picks]

    if scaling is not None:
        data *= scaling
        logger.info('...multiple sensor types found, rescaling data. '
                    'This will result in arbitrary units')

    n_epochs = np.sum(conditions_)
    n_channels, n_times = data.shape[1:]
    spatial_filter = np.zeros((n_channels, n_times))
    surrogate_trials = np.zeros((n_epochs, n_times))

    extra_args = {}
    objective_function = None  # implement later
    if objective_function is None:
        objective_function = _ems_diff
        if len(conditions_) != 2:
            raise ValueError('Currently this function expects exactly 2 '
                             'conditions but you gave me %i' %
                             len(conditions_))

        data_a, data_b = [data[conditions_[i]] for i in [0, 1]]
        sum1, sum2 = np.sum(data_a, axis=0), np.sum(data_b, axis=0)
        extra_args.update({'data_a': data_a,  'data_b': data_b,
                           'sum1': sum1, 'sum2': sum2})
        objective_function = partial(objective_function, **extra_args)

    from sklearn.cross_validation import LeaveOneOut

    for train_indices, epoch_idx in LeaveOneOut(n_epochs):
        logger.info('.. processing epoch %i' % epoch_idx)
        d = objective_function(data[train_indices], conditions_)
        # take norm over channels (matlab uses 2-norm)
        for time_idx in np.arange(n_times):
            d[:, time_idx] /= norm(d[:, time_idx], ord=2)

        # update spatial filter
        spatial_filter += d

        # compute surrogates
        surrogate_trials[epoch_idx] = np.sum(data[epoch_idx[0]] * d, axis=0)

    spatial_filter /= n_epochs

    # create updated conditions indices for sorting
    values = zip(*list(sorted(epochs.event_id.items())))[1]
    conditions_ = conditions_.astype(int)
    for ii, val in enumerate(values):
        this_cond = conditions_[ii]
        this_cond[this_cond.nonzero()] = val

    return surrogate_trials, spatial_filter, conditions_.sum(axis=0)


def _ems_diff(data, conditions, **kwargs):
    """defaut diff objective function"""

    p = kwargs
    m1 = (p['sum1'] - np.sum(data, axis=0)) / (len(p['data_b']) - 1)
    m2 = (p['sum2'] - np.sum(data, axis=0)) / (len(p['data_b']) - 1)
    return m1 - m2


def _check_conditions(epochs, conditions):
    """Aux function"""

    if conditions is None:
        conditions = list(sorted(epochs.event_id.keys()))

    if (isinstance(conditions, list) and
       any(isinstance(k, string_types) for k in conditions)):
        if not all([k in epochs.event_id for k in conditions]):
            raise ValueError('Not all condition-keys present, pleas check')
        events = epochs.events
        # special care to avoid path dependant mappings and orders
        conditions = list(sorted([k for k in epochs.event_id if k in
                                  conditions]))
        conditions = [events[:, 2] == epochs.event_id[k] for k in conditions]

    if not isinstance(conditions, np.ndarray):
        conditions = np.array(conditions)

    # make sure indices are bool if they're not positional
    if tuple(np.unique(conditions)) == (0, 1):
        conditions = conditions.astype(bool)

    if not conditions.dtype == bool:
        conditions_ = np.zeros(len(epochs)).astype(bool)
        for c in conditions:
            conditions_[c] = True
        conditions = conditions_

    return conditions
