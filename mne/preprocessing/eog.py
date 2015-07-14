# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#          Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from scipy.linalg import lstsq

from .peak_finder import peak_finder
from .. import pick_types, pick_channels
from ..utils import logger, verbose
from ..filter import band_pass_filter
from ..epochs import Epochs


@verbose
def find_eog_events(raw, event_id=998, l_freq=1, h_freq=10,
                    filter_length='10s', ch_name=None, tstart=0,
                    verbose=None):
    """Locate EOG artifacts

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    event_id : int
        The index to assign to found events.
    l_freq : float
        Low cut-off frequency in Hz.
    h_freq : float
        High cut-off frequency in Hz.
    filter_length : str | int | None
        Number of taps to use for filtering.
    ch_name: str | None
        If not None, use specified channel(s) for EOG
    tstart : float
        Start detection after tstart seconds.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    eog_events : array
        Events.
    """

    # Getting EOG Channel
    eog_inds = _get_eog_channel_index(ch_name, raw)
    logger.info('EOG channel index for this subject is: %s' % eog_inds)

    eog, _ = raw[eog_inds, :]

    eog_events = _find_eog_events(eog, event_id=event_id, l_freq=l_freq,
                                  h_freq=h_freq,
                                  sampling_rate=raw.info['sfreq'],
                                  first_samp=raw.first_samp,
                                  filter_length=filter_length,
                                  tstart=tstart)

    return eog_events


def _find_eog_events(eog, event_id, l_freq, h_freq, sampling_rate, first_samp,
                     filter_length='10s', tstart=0.):
    """Helper function"""

    logger.info('Filtering the data to remove DC offset to help '
                'distinguish blinks from saccades')

    # filtering to remove dc offset so that we know which is blink and saccades
    fmax = np.minimum(45, sampling_rate / 2.0 - 0.75)  # protect Nyquist
    filteog = np.array([band_pass_filter(x, sampling_rate, 2, fmax,
                                         filter_length=filter_length)
                        for x in eog])
    temp = np.sqrt(np.sum(filteog ** 2, axis=1))

    indexmax = np.argmax(temp)

    # easier to detect peaks with filtering.
    filteog = band_pass_filter(eog[indexmax], sampling_rate, l_freq, h_freq,
                               filter_length=filter_length)

    # detecting eog blinks and generating event file

    logger.info('Now detecting blinks and generating corresponding events')

    temp = filteog - np.mean(filteog)
    n_samples_start = int(sampling_rate * tstart)
    if np.abs(np.max(temp)) > np.abs(np.min(temp)):
        eog_events, _ = peak_finder(filteog[n_samples_start:], extrema=1)
    else:
        eog_events, _ = peak_finder(filteog[n_samples_start:], extrema=-1)

    eog_events += n_samples_start
    n_events = len(eog_events)
    logger.info("Number of EOG events detected : %d" % n_events)
    eog_events = np.array([eog_events + first_samp,
                           np.zeros(n_events, int),
                           event_id * np.ones(n_events, int)]).T

    return eog_events


def _get_eog_channel_index(ch_name, inst):
    if isinstance(ch_name, str):
        # Check if multiple EOG Channels
        if ',' in ch_name:
            ch_name = ch_name.split(',')
        else:
            ch_name = [ch_name]

        eog_inds = pick_channels(inst.ch_names, include=ch_name)

        if len(eog_inds) == 0:
            raise ValueError('%s not in channel list' % ch_name)
        else:
            logger.info('Using channel %s as EOG channel%s' % (
                        " and ".join(ch_name),
                        '' if len(eog_inds) < 2 else 's'))
    elif ch_name is None:

        eog_inds = pick_types(inst.info, meg=False, eeg=False, stim=False,
                              eog=True, ecg=False, emg=False, ref_meg=False,
                              exclude='bads')

        if len(eog_inds) == 0:
            logger.info('No EOG channels found')
            logger.info('Trying with EEG 061 and EEG 062')
            eog_inds = pick_channels(inst.ch_names,
                                     include=['EEG 061', 'EEG 062'])
            if len(eog_inds) != 2:
                raise RuntimeError('EEG 61 or EEG 62 channel not found !!')

    else:
        raise ValueError('Could not find EOG channel.')
    return eog_inds


@verbose
def create_eog_epochs(raw, ch_name=None, event_id=998, picks=None,
                      tmin=-0.5, tmax=0.5, l_freq=1, h_freq=10,
                      reject=None, flat=None,
                      baseline=None, verbose=None):
    """Conveniently generate epochs around EOG artifact events

    Parameters
    ----------
    raw : instance of Raw
        The raw data
    ch_name : str
        The name of the channel to use for EOG peak detection.
        The argument is mandatory if the dataset contains no EOG channels.
    event_id : int
        The index to assign to found events
    picks : array-like of int | None (default)
        Indices of channels to include (if None, all channels
        are used).
    tmin : float
        Start time before event.
    tmax : float
        End time after event.
    l_freq : float
        Low pass frequency.
    h_freq : float
        High pass frequency.
    reject : dict | None
        Rejection parameters based on peak-to-peak amplitude.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'.
        If reject is None then no rejection is done. Example::

            reject = dict(grad=4000e-13, # T / m (gradiometers)
                          mag=4e-12, # T (magnetometers)
                          eeg=40e-6, # uV (EEG channels)
                          eog=250e-6 # uV (EOG channels)
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
        If baseline is equal ot (None, None) all the time
        interval is used. If None, no correction is applied.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    eog_epochs : instance of Epochs
        Data epoched around EOG events.
    """
    events = find_eog_events(raw, ch_name=ch_name, event_id=event_id,
                             l_freq=l_freq, h_freq=h_freq)

    # create epochs around EOG events
    eog_epochs = Epochs(raw, events=events, event_id=event_id,
                        tmin=tmin, tmax=tmax, proj=False, reject=reject,
                        flat=flat, picks=picks, baseline=baseline,
                        preload=True)
    return eog_epochs


def eog_regression(raw, blink_epochs, saccade_epochs=None, reog=None,
                   picks=None, copy=False):
    """Remove EOG signals from the EEG channels by regression.

    It employes the RAAA (recommended aligned-artifact average) procedure
    described by Croft & Barry [1].

    Parameters
    ----------
    raw : Instance of Raw
        The raw data on which the EOG correction produce should be performed.
    blink_epochs : Instance of Epochs
        Epochs cut around blink events. We recommend cutting a window from -0.5
        to 0.5 seconds relative to the onset of the blink.
    saccade_epochs : Instance of Epochs | None
        Epochs cut around saccade events. We recommend cutting a window from -1
        to 1.5 seconds relative to the onset of the saccades, and providing
        separate events for "up", "down", "left" and "right" saccades.
        By default, no saccade information is taken into account.
    reog : str | None
        The name of the rEOG channel, if present. If an rEOG channel is
        available as well as saccade data, the accuracy of the estimation of
        the weights can be improved. By default, no rEOG channel is assumed to
        be present.
    picks : list of int | None
        Indices of the channels in the Raw instance for which to apply the EOG
        correction procedure. By default, the correction is applied to EEG
        channels only.
    copy : bool
        If True, a copy of the Raw instance will be made before applying the
        EOG correction procedure. Defaults to False, which will perform the
        operation in-place.


    References
    ----------
    [1] Croft, R. J., & Barry, R. J. (2000). Removal of ocular artifact from
    the EEG: a review. Clinical Neurophysiology, 30(1), 5-19.
    http://doi.org/10.1016/S0987-7053(00)00055-1
    """

    if picks is None:
        picks = pick_types(raw.info, meg=False, ref_meg=False, eeg=True)

    if copy:
        raw = raw.copy()

    # Compute channel indices for the EOG channels
    raw_eog_ind = pick_types(raw.info, meg=False, ref_meg=False, eog=True)
    ev_eog_ind = pick_types(blink_epochs.info, meg=False, ref_meg=False,
                            eog=True)

    blink_evoked = [
        blink_epochs[cl].average(range(blink_epochs.info['nchan']))
        for cl in blink_epochs.event_id.keys()
    ]
    blink_data = np.hstack([ev.data for ev in blink_evoked])

    if saccade_epochs is None:
        # Calculate EOG weights
        v = np.vstack((
            np.ones(blink_data.shape[1]),
            blink_data[ev_eog_ind]
        )).T
        weights = lstsq(v, blink_data.T)[0][1:]
    else:
        saccade_evoked = [
            saccade_epochs[cl].average(range(saccade_epochs.info['nchan']))
            for cl in saccade_epochs.event_id.keys()
        ]
        saccade_data = np.hstack([ev.data for ev in saccade_evoked])

        if reog is None:
            # If no rEOG data is present, just concatenate the saccade data
            # to the blink data and treat it as one
            blink_sac_data = np.c_[blink_data, saccade_data]
            v = np.vstack((
                np.ones(blink_sac_data.shape[1]),
                blink_sac_data[np.r_[ev_eog_ind]]
            )).T
            weights = lstsq(v, blink_sac_data.T)[0][1:]
        else:
            # If rEOG data is present, use the saccade data to compute the
            # weights for all non-rEOG channels. The blink data will be used
            # for the rEOG channel weight.

            # Isolate the rEOG channel from the other EOG channels
            if reog is not None:
                raw_reog_ind = raw.ch_names.index(reog)
                raw_non_reog_ind = np.setdiff1d(raw_eog_ind, raw_reog_ind)
                ev_reog_ind = blink_epochs.ch_names.index(reog)
                ev_non_reog_ind = np.setdiff1d(ev_eog_ind, ev_reog_ind)

            # Compute non-rEOG weights on the saccade data
            v1 = np.vstack((
                np.ones(saccade_data.shape[1]),
                saccade_data[ev_non_reog_ind, :],
            )).T
            weights_sac = lstsq(v1, saccade_data.T)[0][1:]

            # Remove saccades from blink data
            blink_data -= weights_sac.T.dot(blink_data[ev_non_reog_ind, :])

            # Compute rEOG weights on the blink data
            v2 = np.vstack((
                np.ones(blink_data.shape[1]),
                blink_data[ev_reog_ind, :]
            )).T
            weights_blink = lstsq(v2, blink_data.T)[0][[1]]

            # Remove non-EOG channels from rEOG channel
            raw._data[raw_reog_ind, :] -= np.dot(
                weights_sac[:, ev_reog_ind].T, raw._data[raw_non_reog_ind, :])

            # Compile the EOG weights
            weights = np.vstack((weights_sac, weights_blink))

    # Create a mapping between the picked channels of the raw instance and the
    # EOG weights
    weight_names = blink_epochs.ch_names
    weight_ch_ind = [weight_names.index(raw.ch_names[ch]) for ch in picks]

    # Remove EOG from raw channels
    raw._data[picks, :] -= np.dot(weights[:, weight_ch_ind].T,
                                  raw._data[raw_eog_ind, :])

    return raw, weights
