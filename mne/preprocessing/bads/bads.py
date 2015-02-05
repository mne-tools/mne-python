# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Marijn van Vliet <w.m.vanvliet@gmail.com>
# License: BSD (3-clause)

import numpy as np
from ...utils import verbose
from ...io.pick import pick_info
from ...io.pick import pick_types
from ...defaults import _handle_default
from . import faster_ as _faster


@verbose
def find_bad_channels(epochs, picks=None, method='faster', method_params=None,
                      return_by_metric=False, verbose=None):
    """Implements the first step of the FASTER algorithm.

    This function attempts to automatically mark bad EEG channels by performing
    outlier detection. It operated on epoched data, to make sure only relevant
    data is analyzed.

    Parameters
    ----------
    epochs : Instance of Epochs
        The epochs for which bad channels need to be marked
    picks : list of int | None
        Channels to operate on. Defaults to EEG channels.
    method : {'faster'}
        The detection algorithm.
    method_params : dict | None
        The method parameters in a dict.

        If ``method`` equals 'faster', and ``method_params``is None,
        defaults to the following parameters. Partial updates are supported.
        use_metrics : list of str
            List of metrics to use. Can be any combination of:
                'variance', 'correlation', 'hurst', 'kurtosis', 'line_noise'
            Defaults to all of them.
        thresh : float
            The threshold value, in standard deviations, to apply. A channel
            crossing this threshold value is marked as bad. Defaults to 3.
        max_iter : int
            The maximum number of iterations performed during outlier detection
            (defaults to 1, as in the original FASTER paper).

    return_by_metric : bool
        Whether to return the bad channels as a flat list (False, default) or
        as a dictionary with the names of the used metrics as keys and the
        bad channels found by this metric as values. Is ignored if not
        supported by method.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        Defaults to self.verbose.

    Returns
    -------
    bads : list of str
        The names of the bad EEG channels.
    """
    if picks is None:
        picks = pick_types(epochs.info, meg=True, eeg=True, exclude=[])
    _method_params = _handle_default('bads' + '_' + method, method_params)
    if method == 'faster':
        bads = _faster._find_bad_channels(epochs, picks, **_method_params)
    else:
        raise NotImplementedError(
            'Come back later, for now there is only "FASTER"')

    if return_by_metric:
        return bads
    else:
        return _combine_indices(bads)


@verbose
def find_bad_epochs(epochs, picks=None, return_by_metric=False,
                    method='faster', method_params=None, verbose=None):
    """Implements the second step of the FASTER algorithm.

    This function attempts to automatically mark bad epochs by performing
    outlier detection.

    Parameters
    ----------
    epochs : Instance of Epochs
        The epochs to analyze.
    picks : list of int | None
        Channels to operate on. Defaults to EEG channels.
    method : {'faster'}
        The detection algorithm.
    method_params : dict | None
        The method parameters in a dict.

        If ``method`` equals 'faster', and ``method_params``is None,
        defaults to the following parameters. Partial updates are supported.
        use_metrics : list of str
            List of metrics to use. Can be any combination of:
            'amplitude', 'variance', 'deviation'. Defaults to all of them.
        thresh : float
            The threshold value, in standard deviations, to apply. A channel
            crossing this threshold value is marked as bad. Defaults to 3.
        max_iter : int
            The maximum number of iterations performed during outlier detection
            (defaults to 1, as in the original FASTER paper).

    return_by_metric : bool
        Whether to return the bad channels as a flat list (False, default) or
        as a dictionary with the names of the used metrics as keys and the
        bad channels found by this metric as values. Is ignored if not
        supported by method.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        Defaults to self.verbose.

    Returns
    -------
    bads : list of int
        The indices of the bad epochs.
    """
    if picks is None:
        picks = pick_types(epochs.info, meg=True, eeg=True, exclude='bads')
    _method_params = _handle_default('bads' + '_' + method, method_params)
    if method == 'faster':
        bads = _faster._find_bad_epochs(epochs, picks, **_method_params)
    else:
        raise NotImplementedError(
            'Come back later, for now there is only "FASTER"')

    if return_by_metric:
        return bads
    else:
        return _combine_indices(bads)


@verbose
def find_bad_components(ica, epochs, return_by_metric=False,
                        method='faster', method_params=None, verbose=None):
    """Implements the third step of the FASTER algorithm.

    This function attempts to automatically mark bad ICA components by
    performing outlier detection.

    Parameters
    ----------
    ica : Instance of ICA
        The ICA operator, already fitted to the supplied Epochs object.
    epochs : Instance of Epochs
        The untransformed epochs to analyze.
    method : {'faster'}
        The detection algorithm.
    method_params : dict | None
        The method parameters in a dict.

        If ``method`` equals 'faster', and ``method_params``is None,
        defaults to the following parameters. Partial updates are supported.
        use_metrics : list of str
            List of metrics to use. Can be any combination of:
            'eog_correlation', 'kurtosis', 'power_gradient', 'hurst',
            'median_gradient'. Defaults to all of them.
        thresh : float
            The threshold value, in standard deviations, to apply. A channel
            crossing this threshold value is marked as bad. Defaults to 3.
        max_iter : int
            The maximum number of iterations performed during outlier detection
            (defaults to 1, as in the original FASTER paper).

    return_by_metric : bool
        Whether to return the bad channels as a flat list (False, default) or
        as a dictionary with the names of the used metrics as keys and the
        bad channels found by this metric as values. Is ignored if not
        supported by method.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        Defaults to self.verbose.

    Returns
    -------
    bads : list of int
        The indices of the bad components.

    See also
    --------
    ICA.find_bads_ecg
    ICA.find_bads_eog
    """
    _method_params = _handle_default('bads' + '_' + method, method_params)
    if method == 'faster':
        if 'power_gradient_range' not in _method_params:
            _method_params['power_gradient_range'] = (10, 50)
        bads = _faster._find_bad_components(ica, epochs, **_method_params)
    else:
        raise NotImplementedError(
            'Come back later, for now there is only "FASTER"')

    if return_by_metric:
        return bads
    else:
        return _combine_indices(bads)


@verbose
def find_bad_channels_in_epochs(epochs, picks=None, method='faster',
                                method_params=None, return_by_metric=False):
    """Implements the fourth step of the FASTER algorithm.

    This function attempts to automatically mark bad channels in each epochs by
    performing outlier detection.

    Parameters
    ----------
    epochs : Instance of Epochs
        The epochs to analyze.
    picks : list of int | None
        Channels to operate on. Defaults to EEG channels.
    method : {'faster'}
        The detection algorithm.
    method_params : dict | None
        The method parameters in a dict.

        If ``method`` equals 'faster', and ``method_params``is None,
        defaults to the following parameters. Partial updates are supported.
        use_metrics : list of str
            List of metrics to use. Can be any combination of:
            'amplitude', 'variance', 'deviation', 'median_gradient'
            Defaults to all of them.
        thresh : float
            The threshold value, in standard deviations, to apply. A channel
            crossing this threshold value is marked as bad. Defaults to 3.
        max_iter : int
            The maximum number of iterations performed during outlier detection
            (defaults to 1, as in the original FASTER paper).

    return_by_metric : bool
        Whether to return the bad channels as a flat list (False, default) or
        as a dictionary with the names of the used metrics as keys and the
        bad channels found by this metric as values. Is ignored if not
        supported by method.

    Returns
    -------
    bads : list of lists of int
        For each epoch, the indices of the bad channels.
    """
    if picks is None:
        picks = pick_types(epochs.info, meg=True, eeg=True, exclude=[])

    _method_params = _handle_default('bads' + '_' + method, method_params)
    if method == 'faster':
        bads = _faster._find_bad_channels_in_epochs(epochs, picks,
                                                    **_method_params)
    else:
        raise NotImplementedError(
            'Come back later, for now there is only "FASTER"')

    info = pick_info(epochs.info, picks, copy=True)
    if return_by_metric:
        bads = dict((m, _bad_mask_to_names(info, v)) for m, v in bads.items())
    else:
        bads = np.sum(bads.values(), axis=0).astype(bool)
        bads = _bad_mask_to_names(info, bads)

    return bads


def _bad_mask_to_names(info, bad_mask):
    """Remap mask to ch names"""
    bad_idx = [np.where(m)[0] for m in bad_mask]
    return [[info['ch_names'][k] for k in epoch] for epoch in bad_idx]


def _combine_indices(bads):
    """summarize indices"""
    return list(set(v for val in bads.values() if len(val) > 0 for v in val))
