# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import numpy as np

from ._peak_finder import peak_finder
from .. import pick_types, pick_channels
from ..utils import logger, verbose, _pl, _validate_type
from ..utils.check import _check_preload
from ..filter import filter_data
from ..epochs import BaseEpochs, Epochs
from ..io.pick import _picks_to_idx


@verbose
def find_eog_events(raw, event_id=998, l_freq=1, h_freq=10,
                    filter_length='10s', ch_name=None, tstart=0,
                    reject_by_annotation=False, thresh=None, verbose=None):
    """Locate EOG artifacts.

    .. note:: To control true-positive and true-negative detection rates, you
              may adjust the ``thresh`` parameter.

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    event_id : int
        The index to assign to found events.
    l_freq : float
        Low cut-off frequency to apply to the EOG channel in Hz.
    h_freq : float
        High cut-off frequency to apply to the EOG channel in Hz.
    filter_length : str | int | None
        Number of taps to use for filtering.
    %(ch_name_eog)s
    tstart : float
        Start detection after tstart seconds.
    reject_by_annotation : bool
        Whether to omit data that is annotated as bad.
    thresh : float | None
        Threshold to trigger the detection of an EOG event. This controls the
        thresholding of the underlying peak-finding algorithm. Larger values
        mean that fewer peaks (i.e., fewer EOG events) will be detected.
        If ``None``, use the default of ``(max(eog) - min(eog)) / 4``,
        with ``eog`` being the filtered EOG signal.
    %(verbose)s

    Returns
    -------
    eog_events : array
        Events.

    See Also
    --------
    create_eog_epochs
    compute_proj_eog
    """
    # Getting EOG Channel
    eog_inds = _get_eog_channel_index(ch_name, raw)
    logger.info('EOG channel index for this subject is: %s' % eog_inds)

    # Reject bad segments.
    reject_by_annotation = 'omit' if reject_by_annotation else None
    eog, times = raw.get_data(picks=eog_inds,
                              reject_by_annotation=reject_by_annotation,
                              return_times=True)
    times = times * raw.info['sfreq'] + raw.first_samp

    eog_events = _find_eog_events(eog, event_id=event_id, l_freq=l_freq,
                                  h_freq=h_freq,
                                  sampling_rate=raw.info['sfreq'],
                                  first_samp=raw.first_samp,
                                  filter_length=filter_length,
                                  tstart=tstart, thresh=thresh,
                                  verbose=verbose)
    # Map times to corresponding samples.
    eog_events[:, 0] = np.round(times[eog_events[:, 0] -
                                      raw.first_samp]).astype(int)
    return eog_events


@verbose
def _find_eog_events(eog, event_id, l_freq, h_freq, sampling_rate, first_samp,
                     filter_length='10s', tstart=0., thresh=None,
                     verbose=None):
    """Find EOG events."""
    logger.info('Filtering the data to remove DC offset to help '
                'distinguish blinks from saccades')

    # filtering to remove dc offset so that we know which is blink and saccades
    # hardcode verbose=False to suppress filter param messages (since this
    # filter is not under user control)
    fmax = np.minimum(45, sampling_rate / 2.0 - 0.75)  # protect Nyquist
    filteog = np.array([filter_data(
        x, sampling_rate, 2, fmax, None, filter_length, 0.5, 0.5,
        phase='zero-double', fir_window='hann', fir_design='firwin2',
        verbose=False) for x in eog])
    temp = np.sqrt(np.sum(filteog ** 2, axis=1))

    indexmax = np.argmax(temp)

    # easier to detect peaks with filtering.
    filteog = filter_data(
        eog[indexmax], sampling_rate, l_freq, h_freq, None,
        filter_length, 0.5, 0.5, phase='zero-double', fir_window='hann',
        fir_design='firwin2')

    # detecting eog blinks and generating event file

    logger.info('Now detecting blinks and generating corresponding events')

    temp = filteog - np.mean(filteog)
    n_samples_start = int(sampling_rate * tstart)
    if np.abs(np.max(temp)) > np.abs(np.min(temp)):
        eog_events, _ = peak_finder(filteog[n_samples_start:],
                                    thresh, extrema=1)
    else:
        eog_events, _ = peak_finder(filteog[n_samples_start:],
                                    thresh, extrema=-1)

    eog_events += n_samples_start
    n_events = len(eog_events)
    logger.info(f'Number of EOG events detected: {n_events}')
    eog_events = np.array([eog_events + first_samp,
                           np.zeros(n_events, int),
                           event_id * np.ones(n_events, int)]).T

    return eog_events


def _get_eog_channel_index(ch_name, inst):
    """Get EOG channel indices."""
    _validate_type(ch_name, types=(None, str, list), item_name='ch_name')

    if ch_name is None:
        eog_inds = pick_types(inst.info, meg=False, eeg=False, stim=False,
                              eog=True, ecg=False, emg=False, ref_meg=False,
                              exclude='bads')
        if eog_inds.size == 0:
            raise RuntimeError('No EOG channel(s) found')
        ch_names = [inst.ch_names[i] for i in eog_inds]
    elif isinstance(ch_name, str):
        ch_names = [ch_name]
    else:  # it's a list
        ch_names = ch_name.copy()

    # ensure the specified channels are present in the data
    if ch_name is not None:
        not_found = [ch_name for ch_name in ch_names
                     if ch_name not in inst.ch_names]
        if not_found:
            raise ValueError(f'The specified EOG channel{_pl(not_found)} '
                             f'cannot be found: {", ".join(not_found)}')

        eog_inds = pick_channels(inst.ch_names, include=ch_names)

    logger.info(f'Using EOG channel{_pl(ch_names)}: {", ".join(ch_names)}')
    return eog_inds


@verbose
def create_eog_epochs(raw, ch_name=None, event_id=998, picks=None, tmin=-0.5,
                      tmax=0.5, l_freq=1, h_freq=10, reject=None, flat=None,
                      baseline=None, preload=True, reject_by_annotation=True,
                      thresh=None, decim=1, verbose=None):
    """Conveniently generate epochs around EOG artifact events.

    %(create_eog_epochs)s

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    %(ch_name_eog)s
    event_id : int
        The index to assign to found events.
    %(picks_all)s
    tmin : float
        Start time before event.
    tmax : float
        End time after event.
    l_freq : float
        Low pass frequency to apply to the EOG channel while finding events.
    h_freq : float
        High pass frequency to apply to the EOG channel while finding events.
    reject : dict | None
        Rejection parameters based on peak-to-peak amplitude.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'.
        If reject is None then no rejection is done. Example::

            reject = dict(grad=4000e-13, # T / m (gradiometers)
                          mag=4e-12, # T (magnetometers)
                          eeg=40e-6, # V (EEG channels)
                          eog=250e-6 # V (EOG channels)
                          )

    flat : dict | None
        Rejection parameters based on flatness of signal.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg', and values
        are floats that set the minimum acceptable peak-to-peak amplitude.
        If flat is None then no rejection is done.
    baseline : tuple or list of length 2, or None
        The time interval to apply rescaling / baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal to (None, None) all the time
        interval is used. If None, no correction is applied.
    preload : bool
        Preload epochs or not.
    %(reject_by_annotation_epochs)s

        .. versionadded:: 0.14.0
    thresh : float
        Threshold to trigger EOG event.
    %(decim)s

        .. versionadded:: 0.21.0
    %(verbose)s

    Returns
    -------
    eog_epochs : instance of Epochs
        Data epoched around EOG events.

    See Also
    --------
    find_eog_events
    compute_proj_eog

    Notes
    -----
    Filtering is only applied to the EOG channel while finding events.
    The resulting ``eog_epochs`` will have no filtering applied (i.e., have
    the same filter properties as the input ``raw`` instance).
    """
    events = find_eog_events(raw, ch_name=ch_name, event_id=event_id,
                             l_freq=l_freq, h_freq=h_freq,
                             reject_by_annotation=reject_by_annotation,
                             thresh=thresh)

    # create epochs around EOG events
    eog_epochs = Epochs(raw, events=events, event_id=event_id, tmin=tmin,
                        tmax=tmax, proj=False, reject=reject, flat=flat,
                        picks=picks, baseline=baseline, preload=preload,
                        reject_by_annotation=reject_by_annotation,
                        decim=decim)
    return eog_epochs


def eog_regression(inst, eog_evokeds=None, eog_channels=None, picks=None):
    """Remove EOG signals from the EEG channels by regression.

    Employs linear regression to remove EOG signals from other channels, as
    described in [1]_. Optionally, one may chose to use evoked blink/saccade
    data to obtain the regression weights as described in [2]_.

    The operation is performed in-place.

    Parameters
    ----------
    inst : Raw | Epochs
        The data on which the EOG correction produce should be performed.
    eog_evokeds : Evoked | list of Evoked | None
        Optional recordings of averaged eye movements. For example averaged
        blinks and/or saccades (use a list to supply different kinds of
        movements). When specified, regression weights will be fitted on this
        data, before being applied to the data given as ``inst`` parameter.
        Regression weights obtained from averaged data might be more robust
        [2]_.
    eog_channels : str | list of str | None
        The names of the EOG channels to use in the regression. By default, all
        EOG channels are used.
    %(picks_all_data)s

    Returns
    -------
    inst : Raw | Epochs
        The version of the data with the EOG removed.
    weights : ndarray, shape (n_eog_channels, n_channels)
        The regression weights. For each EOG channel, the fitted regression
        weight to each data channel. The ordering of the weights matches the
        ordering of the channels of the object given as ``inst`` parameter.
    intercept : ndarray, shape (n_channels,)
        For each data channel, the fitted intercept. The ordering of the
        intercepts matches the ordering of the channels of the object given as
        ``inst`` parameter.

    References
    ----------
    .. [1] Gratton, G. and Coles, M. G. H. and Donchin, E. (1983). A new method
           for off-line removal of ocular artifact. Electroencephalography and
           Clinical Neurophysiology, 468-484.
           https://doi.org/10.1016/0013-4694(83)90135-9
    .. [2] Croft, R. J. and Barry, R. J. (1998). EOG correction: a new
           aligned-artifact average solution. Clinical Neurophysiology, 107(6),
           395-401. http://doi.org/10.1016/s0013-4694(98)00087-x
    """
    # Handle defaults for EOG channels parameter
    eog_inds = _get_eog_channel_index(eog_channels, inst)
    if len(eog_inds) == 0:
        raise RuntimeError('No EOG channels found in given data instance. '
                           'Make sure channel types are marked properly.')
    picks = _picks_to_idx(inst.info, picks, none='data')

    # This is the data from which the EOG should be removed. When operating on
    # epochs, concatenate all the epochs into a channels x time matrix.
    _check_preload(inst, 'EOG regression')
    data = inst._data
    if isinstance(inst, BaseEpochs):
        n_epochs, n_channels, n_samples = data.shape
        data = data.transpose(1, 0, 2).reshape(n_channels, -1)

    # This is the data on which to perform the regression. When eog_evokeds is
    # not provided, this is the same as the data from which the EOG should be
    # removed.
    if eog_evokeds is not None:
        # Make sure the channels of `eog_evokeds` are in the same order as
        # those in `inst`.
        try:
            ev_eog_inds = [eog_evokeds.ch_names.index(inst.ch_names[ch])
                           for ch in eog_inds]
            ev_picks = [eog_evokeds.ch_names.index(inst.ch_names[ch])
                        for ch in picks]
        except ValueError:
            raise RuntimeError('Cannot obtain all required regression weights '
                               'from the given eog_evokeds, as some channels '
                               'are missing.')
        eog_data = eog_evokeds.data[ev_eog_inds]
        reg_data = eog_evokeds.data[ev_picks]
    else:
        eog_data = data[eog_inds, :]
        reg_data = data[picks, :]

    # Calculate EOG weights. Add a row of ones to also fit the intercept.
    eog_data = np.vstack((np.ones(eog_data.shape[1]), eog_data)).T
    weights = np.linalg.lstsq(eog_data, reg_data.T, rcond=None)[0]
    intercept = weights[0]
    weights = weights[1:]

    # Remove EOG from data.
    data[picks, :] -= weights.T @ data[eog_inds, :] + intercept[:, None]
    if isinstance(inst, BaseEpochs):
        # Reshape the data back into epochs x channels x samples
        data = data.reshape(n_channels, n_epochs, n_samples).transpose(1, 0, 2)
    inst._data = data

    return inst, weights, intercept
