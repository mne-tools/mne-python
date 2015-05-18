# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import warnings
from ..externals.six import string_types
import numpy as np

from .. import pick_types, pick_channels
from ..utils import logger, verbose, sum_squared
from ..filter import band_pass_filter
from ..epochs import Epochs, _BaseEpochs
from ..io.base import _BaseRaw
from ..evoked import Evoked


def qrs_detector(sfreq, ecg, thresh_value=0.6, levels=2.5, n_thresh=3,
                 l_freq=5, h_freq=35, tstart=0, filter_length='10s'):
    """Detect QRS component in ECG channels.

    QRS is the main wave on the heart beat.

    Parameters
    ----------
    sfreq : float
        Sampling rate
    ecg : array
        ECG signal
    thresh_value : float | str
        qrs detection threshold. Can also be "auto" for automatic
        selection of threshold.
    levels : float
        number of std from mean to include for detection
    n_thresh : int
        max number of crossings
    l_freq : float
        Low pass frequency
    h_freq : float
        High pass frequency
    tstart : float
        Start detection after tstart seconds.
    filter_length : str | int | None
        Number of taps to use for filtering.

    Returns
    -------
    events : array
        Indices of ECG peaks
    """
    win_size = int(round((60.0 * sfreq) / 120.0))

    filtecg = band_pass_filter(ecg, sfreq, l_freq, h_freq,
                               filter_length=filter_length)

    ecg_abs = np.abs(filtecg)
    init = int(sfreq)

    n_samples_start = int(sfreq * tstart)
    ecg_abs = ecg_abs[n_samples_start:]

    n_points = len(ecg_abs)

    maxpt = np.empty(3)
    maxpt[0] = np.max(ecg_abs[:init])
    maxpt[1] = np.max(ecg_abs[init:init * 2])
    maxpt[2] = np.max(ecg_abs[init * 2:init * 3])

    init_max = np.mean(maxpt)

    if thresh_value == 'auto':
        thresh_runs = np.arange(0.3, 1.1, 0.05)
    elif isinstance(thresh_value, string_types):
        raise ValueError('threshold value must be "auto" or a float')
    else:
        thresh_runs = [thresh_value]

    # Try a few thresholds (or just one)
    clean_events = list()
    for thresh_value in thresh_runs:
        thresh1 = init_max * thresh_value
        numcross = list()
        time = list()
        rms = list()
        ii = 0
        while ii < (n_points - win_size):
            window = ecg_abs[ii:ii + win_size]
            if window[0] > thresh1:
                max_time = np.argmax(window)
                time.append(ii + max_time)
                nx = np.sum(np.diff(((window > thresh1).astype(np.int) ==
                                     1).astype(int)))
                numcross.append(nx)
                rms.append(np.sqrt(sum_squared(window) / window.size))
                ii += win_size
            else:
                ii += 1

        if len(rms) == 0:
            rms.append(0.0)
            time.append(0.0)
        time = np.array(time)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        rms_thresh = rms_mean + (rms_std * levels)
        b = np.where(rms < rms_thresh)[0]
        a = np.array(numcross)[b]
        ce = time[b[a < n_thresh]]

        ce += n_samples_start
        clean_events.append(ce)

    # pick the best threshold; first get effective heart rates
    rates = np.array([60. * len(cev) / (len(ecg) / float(sfreq))
                      for cev in clean_events])

    # now find heart rates that seem reasonable (infant thru adult athlete)
    idx = np.where(np.logical_and(rates <= 160., rates >= 40.))[0]
    if len(idx) > 0:
        ideal_rate = np.median(rates[idx])  # get close to the median
    else:
        ideal_rate = 80.  # get close to a reasonable default
    idx = np.argmin(np.abs(rates - ideal_rate))
    clean_events = clean_events[idx]
    return clean_events


@verbose
def find_ecg_events(raw, event_id=999, ch_name=None, tstart=0.0,
                    l_freq=5, h_freq=35, qrs_threshold='auto',
                    filter_length='10s', verbose=None):
    """Find ECG peaks

    Parameters
    ----------
    raw : instance of Raw
        The raw data
    event_id : int
        The index to assign to found events
    ch_name : None | str
        The name of the channel to use for ECG peak detection.
        If None (default), a synthetic ECG channel is created from
        cross channel average. Synthetic channel can only be created from
        'meg' channels.
    tstart : float
        Start detection after tstart seconds. Useful when beginning
        of run is noisy.
    l_freq : float
        Low pass frequency.
    h_freq : float
        High pass frequency.
    qrs_threshold : float | str
        Between 0 and 1. qrs detection threshold. Can also be "auto" to
        automatically choose the threshold that generates a reasonable
        number of heartbeats (40-160 beats / min).
    filter_length : str | int | None
        Number of taps to use for filtering.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    ecg_events : array
        Events.
    ch_ecg : string
        Name of channel used.
    average_pulse : float
        Estimated average pulse.
    """
    idx_ecg = _get_ecg_channel_index(ch_name, raw)
    if idx_ecg is not None:
        logger.info('Using channel %s to identify heart beats.'
                    % raw.ch_names[idx_ecg])
        ecg, times = raw[idx_ecg, :]
    else:
        ecg, times = _make_ecg(raw, None, None, verbose)

    # detecting QRS and generating event file
    ecg_events = qrs_detector(raw.info['sfreq'], ecg.ravel(), tstart=tstart,
                              thresh_value=qrs_threshold, l_freq=l_freq,
                              h_freq=h_freq, filter_length=filter_length)

    n_events = len(ecg_events)
    average_pulse = n_events * 60.0 / (times[-1] - times[0])
    logger.info("Number of ECG events detected : %d (average pulse %d / "
                "min.)" % (n_events, average_pulse))

    ecg_events = np.c_[ecg_events + raw.first_samp, np.zeros(n_events),
                       event_id * np.ones(n_events)]
    return ecg_events, idx_ecg, average_pulse


def _get_ecg_channel_index(ch_name, inst):
    """Geting ECG channel index. If no channel found returns None."""
    if ch_name is None:
        ecg_idx = pick_types(inst.info, meg=False, eeg=False, stim=False,
                             eog=False, ecg=True, emg=False, ref_meg=False,
                             exclude='bads')
    else:
        if ch_name not in inst.ch_names:
            raise ValueError('%s not in channel list (%s)' %
                             (ch_name, inst.ch_names))
        ecg_idx = pick_channels(inst.ch_names, include=[ch_name])

    if len(ecg_idx) == 0:
        return None
        # raise RuntimeError('No ECG channel found. Please specify ch_name '
        #                    'parameter e.g. MEG 1531')

    if len(ecg_idx) > 1:
        warnings.warn('More than one ECG channel found. Using only %s.'
                      % inst.ch_names[ecg_idx[0]])

    return ecg_idx[0]


@verbose
def create_ecg_epochs(raw, ch_name=None, event_id=999, picks=None,
                      tmin=-0.5, tmax=0.5, l_freq=8, h_freq=16, reject=None,
                      flat=None, baseline=None, verbose=None):
    """Conveniently generate epochs around ECG artifact events


    Parameters
    ----------
    raw : instance of Raw
        The raw data
    ch_name : None | str
        The name of the channel to use for ECG peak detection.
        If None (default), a synthetic ECG channel is created from
        cross channel average. Synthetic channel can only be created from
        'meg' channels.
    event_id : int
        The index to assign to found events
    picks : array-like of int | None (default)
        Indices of channels to include (if None, all channels are used).
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
    ecg_epochs : instance of Epochs
        Data epoched around ECG r-peaks.
    """

    events, _, _ = find_ecg_events(raw, ch_name=ch_name, event_id=event_id,
                                   l_freq=l_freq, h_freq=h_freq,
                                   verbose=verbose)
    if picks is None:
        picks = pick_types(raw.info, meg=True, eeg=True, ref_meg=False)

    # create epochs around ECG events and baseline (important)
    ecg_epochs = Epochs(raw, events=events, event_id=event_id,
                        tmin=tmin, tmax=tmax, proj=False,
                        picks=picks, reject=reject, baseline=baseline,
                        verbose=verbose, preload=True)
    return ecg_epochs


@verbose
def _make_ecg(inst, start, stop, verbose=None):
    """Create ECG signal from cross channel average
    """
    if not any(c in inst for c in ['mag', 'grad']):
        raise ValueError('Unable to generate artifical ECG channel')
    for ch in ['mag', 'grad']:
        if ch in inst:
            break
    logger.info('Reconstructing ECG signal from {0}'
                .format({'mag': 'Magnetometers',
                         'grad': 'Gradiometers'}[ch]))
    picks = pick_types(inst.info, meg=ch, eeg=False, ref_meg=False)
    if isinstance(inst, _BaseRaw):
        ecg, times = inst[picks, start:stop]
    elif isinstance(inst, _BaseEpochs):
        ecg = np.hstack(inst.crop(start, stop, copy=True).get_data())
        times = inst.times
    elif isinstance(inst, Evoked):
        ecg = inst.data
        times = inst.times
    return ecg.mean(0), times
